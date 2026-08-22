# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate one session per resolution, then compare what they look like.

Lowering resolution is the only lever that moves this model's chunk latency
enough to reach realtime, so whether it may be pulled is a question about
pixels, not milliseconds. This produces the pixels.

Two subcommands, deliberately separate:

``generate``
    Needs a device, a checkpoint and one engine per resolution -- resolution
    is an engine construction parameter, so switching it means a fresh engine.
    Writes PNG frames and a manifest per run and computes nothing.

``compare``
    Reads those directories and prints the report. No device, no checkpoint.
    Splitting it this way means a crash at the third resolution does not lose
    the first two, the comparison can be re-run after installing LPIPS, and
    the analysis can happen somewhere the GPU is not.

The comparison needs a **divergence floor** to be readable at all: the same
resolution at two different seeds, which is two different videos at identical
quality. Without it a cross-resolution distance cannot be told apart from the
fact that the two clips simply show different things. ``generate`` emits the
floor runs by default; ``--floor-seeds 0`` opts out and the report then
refuses to present its numbers as quality.

Examples::

    # One run per resolution, plus a second seed at the reference for the floor.
    python -m benchmarks.ar_diffusion.run_quality_vs_resolution generate \
        --model <checkpoint> --prompt "..." --image first.png \
        --out runs/quality --chunks 8 \
        --resolution 832x480 --resolution 512x320 --resolution 384x288 \
        --note "hw=1xRTX-PRO-6000" --note "steps=4"

    # Anywhere, later, with or without a GPU.
    python -m benchmarks.ar_diffusion.run_quality_vs_resolution compare \
        --reference runs/quality/832x480_seed0 \
        --candidate runs/quality/512x320_seed0 \
        --candidate runs/quality/384x288_seed0 \
        --floor runs/quality/832x480_seed1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.ar_diffusion.quality_metrics import (
    ResolutionSample,
    build_report,
    format_report,
    frames_to_png_bytes,
    load_lpips,
)

MANIFEST = "manifest.json"


def parse_resolution(text: str) -> tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` and reject anything the pipeline would refuse."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"resolution must be WIDTHxHEIGHT, got {text!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"resolution must be WIDTHxHEIGHT, got {text!r}") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"resolution must be positive, got {text!r}")
    # The pipeline requires alignment with the VAE and DiT patch sizes; failing
    # here beats failing after a 30-second model load.
    if width % 16 or height % 16:
        raise argparse.ArgumentTypeError(f"width and height must be multiples of 16, got {text!r}")
    return width, height


def run_label(width: int, height: int, seed: int) -> str:
    return f"{width}x{height}_seed{seed}"


# ── generate ───────────────────────────────────────────────────────────────


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_vae(model: str, *, device, dtype):
    """Load the checkpoint's VAE on its own, independent of any engine.

    Deliberately not the ``Distributed`` subclass: that one calls
    ``init_distributed()`` and asserts on a world group this process does not
    have, because the engine's own distributed state lives in its worker. One
    VAE on one device is all a single-session decode needs.
    """
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        OmniAutoencoderKLWan,
    )

    vae = OmniAutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=dtype)
    return vae.to(device).eval()


def _latent_stats(vae):
    """Per-channel latent mean/std, shaped to broadcast over [B, C, T, H, W]."""
    import torch

    shape = (1, -1, 1, 1, 1)
    mean = torch.as_tensor(vae.config.latents_mean, device=vae.device, dtype=torch.float32).view(*shape)
    std = torch.as_tensor(vae.config.latents_std, device=vae.device, dtype=torch.float32).view(*shape)
    return mean, std


def _tick_latents(output: Any) -> Any:
    """Extract one tick's latent tensor, saying what was wrong if it is absent."""
    payload = getattr(output, "multimodal_output", None)
    if not isinstance(payload, dict):
        raise TypeError(
            f"a tick returned {type(output).__name__} with no multimodal_output mapping; "
            "the quality tool reads latents from there"
        )
    body = payload.get("payload")
    latents = body.get("latents") if isinstance(body, dict) else None
    if latents is None:
        raise KeyError(
            f"tick output carried no payload.latents (keys: {sorted(payload)}). "
            "The session must run with output_type='latent'."
        )
    return latents


def _frames_to_clip(frames: Any) -> Any:
    """Normalise a decoder's ``[B, 3, T, H, W]`` output to ``(T, 3, H, W)``."""
    import torch

    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"decoder returned {type(frames).__name__}, expected a tensor")
    if frames.dim() == 5:
        if frames.shape[0] != 1:
            raise ValueError(f"expected one session per decode, got batch {frames.shape[0]}")
        frames = frames[0]
    if frames.dim() != 4 or frames.shape[0] != 3:
        raise ValueError(f"expected (3, frames, height, width), got {tuple(frames.shape)}")
    clip = frames.permute(1, 0, 2, 3).float()
    # Wan decoders emit [-1, 1]; map to [0, 1] without clipping legitimate range.
    if float(clip.min()) < 0.0:
        clip = (clip + 1.0) / 2.0
    return clip.clamp(0.0, 1.0).cpu()


async def generate_one(args: argparse.Namespace, *, width: int, height: int, seed: int) -> Path:
    """Run one session at one resolution and write its frames to disk."""
    import torch

    from benchmarks.ar_diffusion.decoding_session import DecodingSession
    from benchmarks.ar_diffusion.engine_binding import build_realtime_backend
    from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

    label = run_label(width, height, seed)
    directory = args.out / label
    directory.mkdir(parents=True, exist_ok=True)

    run_args = argparse.Namespace(**vars(args))
    run_args.seed = seed
    run_args.model_config_overrides = {"ar_diffusion_height": height, "ar_diffusion_width": width}

    backend = await build_realtime_backend(run_args)

    # Load a VAE here rather than reaching into the engine for one. With
    # process isolation the pipeline lives in a worker, so ``engine.pipeline``
    # is None in this process; and a decoder that a benchmark owns outright is
    # the honest shape anyway, since decode is what this measures.
    vae = _load_vae(args.model, device=_device(), dtype=torch.bfloat16)
    latent_mean, latent_std = _latent_stats(vae)

    def to_model_space(output):
        """Pull the tick's latents out and undo the checkpoint's normalisation.

        A tick returns an ``OmniRequestOutput``; what the pipeline built is
        under ``multimodal_output``. The latents are in the model's normalised
        space, so they need the same inverse the pipeline applies before its
        own decode, or the decoder sees values it was never trained on.
        """
        latents = _tick_latents(output)
        return (latents.to(vae.device) * latent_std + latent_mean).to(dtype=vae.dtype)

    decoder = WanStreamingDecoder(vae)
    inner = await backend.manager.create_session(label)
    session = DecodingSession(
        inner=inner,
        decoder=decoder,
        session_id=label,
        latent_of=to_model_space,
    )

    clips = []
    try:
        for _ in range(args.chunks):
            clips.append(_frames_to_clip(await session.next_chunk()))
    finally:
        await session.close()

    clip = torch.cat(clips, dim=0)
    for index, blob in enumerate(frames_to_png_bytes(clip)):
        (directory / f"frame_{index:05d}.png").write_bytes(blob)
    (directory / MANIFEST).write_text(
        json.dumps(
            {
                "label": label,
                "width": width,
                "height": height,
                "seed": seed,
                "frames": int(clip.shape[0]),
                "chunks": args.chunks,
                "model": args.model,
                "prompt": args.prompt,
                "image": None if args.image is None else str(args.image),
                "notes": list(args.note),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory


async def _generate(args: argparse.Namespace) -> int:
    runs: list[Path] = []
    reference_width, reference_height = args.resolution[0]
    seeds = [args.seed, *[args.seed + offset for offset in range(1, args.floor_seeds + 1)]]

    for width, height in args.resolution:
        # Extra seeds run only at the reference resolution: the floor is
        # "different content, same quality", which needs the size held fixed.
        is_reference = (width, height) == (reference_width, reference_height)
        for seed in seeds if is_reference else seeds[:1]:
            print(f"--- generating {run_label(width, height, seed)}")
            runs.append(await generate_one(args, width=width, height=height, seed=seed))

    print("\nwrote:")
    for path in runs:
        print(f"  {path}")
    if args.floor_seeds < 1:
        print("\nNo floor runs were generated (--floor-seeds 0).")
        print("compare will refuse to present its distances as quality numbers.")
    return 0


# ── compare ────────────────────────────────────────────────────────────────


def load_sample(directory: Path) -> ResolutionSample:
    """Read one generated run back as a clip."""
    import torch
    from PIL import Image

    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"{directory} has no {MANIFEST}; it is not a generated run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"{directory} contains no frames")

    def read(path: Path):
        array = _as_array(Image.open(path).convert("RGB"))
        # copy(): PIL hands back a read-only buffer, which torch warns about.
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0

    frames = torch.stack([read(path) for path in paths])
    return ResolutionSample(
        label=manifest["label"],
        height=int(manifest["height"]),
        width=int(manifest["width"]),
        seed=int(manifest["seed"]),
        frames=frames,
    )


def _as_array(image: Any):
    import numpy as np

    return np.asarray(image)


def _compare(args: argparse.Namespace) -> int:
    reference = load_sample(args.reference)
    candidates = [load_sample(path) for path in args.candidate]
    floors = [load_sample(path) for path in args.floor]

    for sample in floors:
        if (sample.height, sample.width) != (reference.height, reference.width):
            raise ValueError(
                f"floor run {sample.label} is {sample.width}x{sample.height} but the reference is "
                f"{reference.width}x{reference.height}; the floor must hold resolution fixed."
            )
        if sample.seed == reference.seed:
            raise ValueError(
                f"floor run {sample.label} shares the reference seed; it would measure nothing. "
                "Generate the floor with a different --seed."
            )

    lpips = None if args.no_lpips else load_lpips()
    report = build_report(reference, candidates, floor_samples=floors, lpips=lpips)
    text = format_report(report)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generation quality across serving resolutions.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run one session per resolution and save frames. Needs a device.")
    gen.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint path.")
    gen.add_argument("--prompt", required=True, help="Scene prompt, identical across resolutions.")
    gen.add_argument("--image", default=None, help="Initial RGB image, if the pipeline needs one.")
    gen.add_argument("--out", type=Path, required=True, help="Directory to write runs into.")
    gen.add_argument(
        "--resolution",
        type=parse_resolution,
        action="append",
        required=True,
        help="WIDTHxHEIGHT, repeatable. The first is the reference every other is compared to.",
    )
    gen.add_argument("--chunks", type=int, default=8, help="Chunks to generate per session.")
    gen.add_argument("--seed", type=int, default=0, help="Seed for the primary run at every resolution.")
    gen.add_argument(
        "--floor-seeds",
        type=int,
        default=1,
        help=(
            "Extra seeds to run at the reference resolution, forming the divergence floor. "
            "0 opts out, and compare then refuses to call its distances quality."
        ),
    )
    gen.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    gen.add_argument("--max-pending-events", type=int, default=64)
    gen.add_argument(
        "--note",
        action="append",
        default=[],
        required=True,
        help="Conditions to record verbatim in every manifest. Repeatable, at least one.",
    )

    cmp_ = sub.add_parser("compare", help="Report on saved runs. Needs no device.")
    cmp_.add_argument("--reference", type=Path, required=True, help="The run every candidate is compared to.")
    cmp_.add_argument("--candidate", type=Path, action="append", default=[], help="A run to score. Repeatable.")
    cmp_.add_argument(
        "--floor",
        type=Path,
        action="append",
        default=[],
        help="A reference-resolution run at a different seed. Without one, nothing is readable.",
    )
    cmp_.add_argument("--no-lpips", action="store_true", help="Skip LPIPS even if it is installed.")
    cmp_.add_argument("--output", type=Path, default=None, help="Also write the report here.")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        return asyncio.run(_generate(args))
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
