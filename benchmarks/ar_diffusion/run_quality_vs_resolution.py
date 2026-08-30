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

    # Judging one candidate resolution properly: several scenes, several seeds,
    # and the extra seeds run at the candidate too rather than only at the
    # reference. One resolution condemned on one sample is not a finding.
    python -m benchmarks.ar_diffusion.run_quality_vs_resolution generate \
        --model <checkpoint> --out runs/quality-p0 --chunks 8 --all-seeds \
        --prompt "a forest path" --prompt "a city street" --prompt "a beach" \
        --resolution 768x480 --resolution 512x320 \
        --note "hw=1xRTX-PRO-6000"

Because a distance between two different prompts measures the prompts,
``compare`` refuses to mix them: every reference, candidate and floor it is
given must come from the same prompt. Run it once per prompt.
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

# LingBot World refuses a tick that carries no camera control, so every tick
# has to carry one. The script is FIXED and identical at every resolution:
# drawing keys at random would send each resolution down a different path, and
# the comparison would then be measuring the path rather than the size.
_CAMERA_SCHEMA = "lingbot.camera_actions.v1"
_CAMERA_SCRIPT = "WWAWWD"
_FRAMES_PER_BLOCK = 3

# Sampling settings the deployed path uses. Left unset they take library
# defaults, which silently produces frames from a different sampler than the
# one being characterised.
_NUM_FRAMES = 9
_NUM_INFERENCE_STEPS = 4
_MAX_SEQUENCE_LENGTH = 512
_FLOW_SHIFT = 5.0


def _camera_event(chunk_index: int):
    """One tick's worth of camera keys, taken from the fixed script."""
    from vllm_omni.experimental.ar_diffusion.session import ARDiffusionSessionEvent
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput

    start = chunk_index * _FRAMES_PER_BLOCK
    frames = [[_CAMERA_SCRIPT[(start + i) % len(_CAMERA_SCRIPT)]] for i in range(_FRAMES_PER_BLOCK)]
    return ARDiffusionSessionEvent(
        event_id=chunk_index,
        prompt=None,  # prompt_provider falls back to the run's single prompt
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema=_CAMERA_SCHEMA,
                data={"mode": "script", "frames": frames},
            ),
        ),
    )


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
    # A frame also had to be a legal KV block once, which additionally required
    # tokens_per_frame % 16 == 0 and ruled out the 832x480 this checkpoint
    # ships as its default -- 30 x 52 = 1560 tokens, and 1560 % 16 == 8. That
    # is no longer a constraint: the paging unit is chosen separately from the
    # frame, so 832x480 is measurable here like any other size.
    return width, height


def run_label(width: int, height: int, seed: int, prompt_index: int = 0, *, prompt_count: int = 1) -> str:
    """Name a run. A single prompt keeps the original two-part name so runs
    generated before prompts were repeatable stay addressable by the path they
    already have."""
    suffix = "" if prompt_count == 1 else f"_p{prompt_index}"
    return f"{width}x{height}{suffix}_seed{seed}"


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
    """Extract one tick's latent tensor, saying what was wrong if it is absent.

    With ``output_type="latent"`` the tick hands the latent back as
    ``images[0]`` -- that is what the shipped realtime example reads, and it
    is where this actually arrives. The two ``multimodal_output`` shapes are
    kept as fallbacks because other pipelines put it there; taking them in
    this order means the common case never depends on the fallbacks.
    """
    images = getattr(output, "images", None)
    if images:
        return images[0]

    payload = getattr(output, "multimodal_output", None)
    if isinstance(payload, dict):
        latents = payload.get("latents")
        if latents is not None:
            return latents
        body = payload.get("payload")
        if isinstance(body, dict) and body.get("latents") is not None:
            return body["latents"]
        raise KeyError(
            f"a tick carried neither images nor latents (multimodal_output keys: {sorted(payload)}). "
            "The session must run with output_type='latent'."
        )
    raise TypeError(
        f"a tick returned {type(output).__name__} with no images and no multimodal_output mapping; "
        "the quality tool reads latents from one of those"
    )


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


async def generate_one(
    args: argparse.Namespace,
    *,
    width: int,
    height: int,
    seed: int,
    prompt_index: int = 0,
    prompt_count: int = 1,
) -> Path:
    """Run one session at one resolution and prompt, and write its frames to disk."""
    import torch

    from benchmarks.ar_diffusion.decoding_session import DecodingSession
    from benchmarks.ar_diffusion.engine_binding import build_realtime_backend
    from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

    prompt = args.prompt[prompt_index]
    label = run_label(width, height, seed, prompt_index, prompt_count=prompt_count)
    directory = args.out / label
    directory.mkdir(parents=True, exist_ok=True)

    run_args = argparse.Namespace(**vars(args))
    run_args.seed = seed
    # args.prompt is the whole list; the backend builder wants this run's one.
    run_args.prompt = prompt
    run_args.model_config_overrides = {"ar_diffusion_height": height, "ar_diffusion_width": width}
    run_args.height, run_args.width = height, width
    run_args.num_frames = _NUM_FRAMES
    run_args.num_inference_steps = _NUM_INFERENCE_STEPS
    run_args.max_sequence_length = _MAX_SEQUENCE_LENGTH
    run_args.sampling_extra_args = {"flow_shift": _FLOW_SHIFT}

    from vllm_omni.diffusion.models.lingbot_world.actions import LingBotCameraControlReducer

    backend = await build_realtime_backend(run_args, control_reducer_factory=LingBotCameraControlReducer)

    # Generate everything first, keeping only latents, and let the engine go
    # before a VAE is loaded. The worker holds the model plus a whole session's
    # KV -- at 832x480 that is 45 GiB of weights and 28 GiB of cache -- so a
    # decoder sharing the device with it runs out of memory. Latents are tiny
    # by comparison, so parking them on the host costs nothing.
    latents: list[Any] = []
    inner = await backend.manager.create_session(label)
    try:
        # Written as they arrive, not just accumulated. A long session can die
        # on a ceiling nobody knew was there -- the paged block table is indexed
        # by absolute position and runs out around tick 224 at this resolution
        # -- and frames are only produced after the loop, so a crash at tick 225
        # used to throw away 17 minutes of generation. Latents are ~600 KB each.
        latent_dir = directory / "latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        for chunk_index in range(args.chunks):
            await inner.accept_event(_camera_event(chunk_index))
            output = await inner.next_chunk()
            tick_latent = _tick_latents(output).detach().cpu()
            torch.save(tick_latent, latent_dir / f"tick_{chunk_index:05d}.pt")
            latents.append(tick_latent)
    finally:
        await inner.close()

    shutdown = getattr(backend.engine, "shutdown", None)
    if callable(shutdown):
        shutdown()

    # Load a VAE here rather than reaching into the engine for one. With
    # process isolation the pipeline lives in a worker, so ``engine.pipeline``
    # is None in this process; and a decoder that a benchmark owns outright is
    # the honest shape anyway, since decode is what this measures.
    vae = _load_vae(args.model, device=_device(), dtype=torch.bfloat16)
    latent_mean, latent_std = _latent_stats(vae)
    decoder = WanStreamingDecoder(vae)
    state = decoder.new_decode_state(label)

    clips = []
    try:
        # inference_mode, not just eval(): eval() does not stop autograd, and
        # the retained graph across six chunks is what turns a 2 GiB decode
        # into an out-of-memory one.
        with torch.inference_mode():
            for chunk in latents:
                # Latents are in the model's normalised space; undo it with the
                # same inverse the pipeline applies before its own decode, or
                # the decoder sees values it was never trained on.
                scaled = (chunk.to(vae.device) * latent_std + latent_mean).to(dtype=vae.dtype)
                clips.append(_frames_to_clip(decoder.decode_chunk(scaled, state)))
    finally:
        decoder.release(state)

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
                "prompt": prompt,
                "prompt_index": prompt_index,
                "image": None if args.image is None else str(args.image),
                "camera_script": _CAMERA_SCRIPT,
                "num_inference_steps": _NUM_INFERENCE_STEPS,
                "flow_shift": _FLOW_SHIFT,
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

    prompt_count = len(args.prompt)

    for prompt_index in range(prompt_count):
        for width, height in args.resolution:
            # By default extra seeds run only at the reference resolution: the
            # floor is "different content, same quality", which needs the size
            # held fixed. --all-seeds runs them everywhere instead, which is
            # what turns a candidate resolution's verdict from a single sample
            # into several, and gives each resolution its own floor.
            is_reference = (width, height) == (reference_width, reference_height)
            run_seeds = seeds if (is_reference or args.all_seeds) else seeds[:1]
            for seed in run_seeds:
                label = run_label(width, height, seed, prompt_index, prompt_count=prompt_count)
                print(f"--- generating {label}")
                runs.append(
                    await generate_one(
                        args,
                        width=width,
                        height=height,
                        seed=seed,
                        prompt_index=prompt_index,
                        prompt_count=prompt_count,
                    )
                )

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


def _prompt_of(directory: Path) -> str:
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"{directory} has no {MANIFEST}; it is not a generated run")
    return json.loads(manifest_path.read_text(encoding="utf-8")).get("prompt", "")


def _reject_mixed_prompts(reference: Path, others: Sequence[Path]) -> None:
    """A distance between clips of two different prompts measures the prompts.

    Nothing downstream can separate that from a resolution effect, so refuse
    the comparison here rather than report a number that cannot mean what the
    report says it means.
    """
    expected = _prompt_of(reference)
    for path in others:
        found = _prompt_of(path)
        if found != expected:
            raise ValueError(
                f"{path.name} was generated from a different prompt than the reference "
                f"({found!r} vs {expected!r}); comparing them measures the prompts, "
                "not the resolution."
            )


def _as_array(image: Any):
    import numpy as np

    return np.asarray(image)


def _compare(args: argparse.Namespace) -> int:
    _reject_mixed_prompts(args.reference, [*args.candidate, *args.floor])
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
    gen.add_argument(
        "--prompt",
        action="append",
        required=True,
        help=(
            "Scene prompt, identical across resolutions. Repeatable: every prompt runs at "
            "every resolution, so a verdict rests on more than one scene."
        ),
    )
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
    gen.add_argument(
        "--all-seeds",
        action="store_true",
        help=(
            "Run every seed at every resolution rather than only at the reference. "
            "Costs one session per extra seed per resolution and buys a repeat of each "
            "candidate, plus a per-resolution divergence floor."
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
