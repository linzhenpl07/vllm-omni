#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure what coalescing sessions could actually recover.

Cross-session batching is usually justified two ways, and only one of them has
been measured here. The measured one is amortising per-tick switching cost:
19.7 ms, which is 0.95% of a 2.07 s tick and 3.1% of the 0.63 s tick that
320x256 now runs at. Three percent does not justify changing a public
protocol.

The unmeasured one is device utilisation. One chunk at 320x256 is 960 tokens
against an 18.5B model. If that GEMM is small enough to leave the device
weight-bandwidth-bound rather than compute-bound, then N sessions in one
forward read each weight once instead of N times, and the saving is a
different order of magnitude than 3%. If instead latency already grows
linearly with tokens, the device is compute-bound at this size, there is no
utilisation headroom, and coalescing is worth only the switching cost.

This settles that, and deliberately does not settle it with the resolution
sweep already in hand: those points vary the spatial dimensions, so attention's
quadratic term and the KV footprint move with the token count. Here everything
is fixed except sequence length, which is exactly what coalescing changes.

Weights are random. Latency does not depend on their values, and one
real-sized block is 200M parameters instead of 18.5B, so this runs in seconds
rather than needing the checkpoint. Self-attention and the feed-forward are
timed separately, because they answer differently: the FFN is a plain GEMM and
must benefit, while attention is varlen and each sequence attends only over
itself, so its work is linear in tokens no matter what. Reporting the split
says where any headroom actually lives.

Usage::

    python -m benchmarks.ar_diffusion.batch_scaling_probe \\
        --tokens-per-chunk 960 --sessions 1 2 3 4 \\
        --note "hw=1xRTX-PRO-6000" --note "res=320x256"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported lazily everywhere else so --help works without torch installed.
    import torch


@dataclass
class Timing:
    """Latency for one component at one sequence length."""

    sessions: int
    tokens: int
    median_s: float
    samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sessions": self.sessions,
            "tokens": self.tokens,
            "median_s": self.median_s,
            "min_s": min(self.samples) if self.samples else None,
            "max_s": max(self.samples) if self.samples else None,
        }


def coalescing_gain(timings: Sequence[Timing]) -> list[dict]:
    """How much one coalesced forward beats running the sessions separately.

    ``gain`` is ``N * latency(1) / latency(N)``. Above 1.0 the device had
    headroom that batching recovered; at 1.0 it was already saturated and
    coalescing buys nothing beyond switching cost.
    """
    if not timings:
        return []
    base = next((t for t in timings if t.sessions == 1), None)
    if base is None:
        raise ValueError("a single-session timing is needed as the baseline")
    rows = []
    for timing in timings:
        serial = base.median_s * timing.sessions
        rows.append(
            {
                "sessions": timing.sessions,
                "tokens": timing.tokens,
                "coalesced_s": timing.median_s,
                "serial_s": serial,
                "gain": serial / timing.median_s if timing.median_s else float("nan"),
            }
        )
    return rows


def format_report(results: dict) -> str:
    lines: list[str] = []
    for note in results.get("notes", ()):
        lines.append(f"# {note}")
    lines.append(
        f"# block dim={results['dim']} heads={results['num_heads']} ffn={results['ffn_dim']} "
        f"dtype={results['dtype']} device={results['device']}"
    )
    lines.append("")
    for component, timings in results["components"].items():
        lines.append(f"{component}")
        lines.append(f"  {'N':>2} {'tokens':>7} {'coalesced':>11} {'N x single':>11} {'gain':>7}")
        for row in timings:
            lines.append(
                f"  {row['sessions']:>2} {row['tokens']:>7} {row['coalesced_s'] * 1e3:>10.3f}ms "
                f"{row['serial_s'] * 1e3:>10.3f}ms {row['gain']:>7.3f}"
            )
        lines.append("")
    lines.append(
        "gain > 1 means the device had headroom that one coalesced forward recovered.\n"
        "gain ~ 1 means it was already compute-bound, and coalescing is worth only the\n"
        "19.7 ms per-tick switching cost -- 3.1% of a 0.63 s tick."
    )
    return "\n".join(lines)


def _time_calls(fn, *, warmup: int, repeats: int, sync) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - start)
    return samples


@contextlib.contextmanager
def single_process_parallel(device: torch.device):
    """Stand up a world of one so the model's TP-aware layers can be built.

    ``LingBotSelfAttention`` divides its heads by the tensor-parallel world size
    and reads the current vLLM config, both of which a served engine provides
    and a script does not. A probe is not a server, so it makes the smallest
    world that satisfies the layer: one rank, dividing by one, changing nothing
    it measures.

    Idempotent, and a no-op inside a real engine, so the probe can also be
    imported by something that already has a group.
    """
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    if model_parallel_is_initialized():
        yield
        return
    with set_current_vllm_config(VllmConfig()):
        if not dist.is_initialized():
            # A file rendezvous rather than a port: two probes on one box must
            # not race for the same address, and there is nothing to connect to.
            with tempfile.NamedTemporaryFile(prefix="ar-diffusion-probe-", delete=False) as handle:
                rendezvous = f"file://{handle.name}"
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=rendezvous,
                local_rank=0,
                backend="gloo" if device.type == "cpu" else "nccl",
            )
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield


def probe(args: argparse.Namespace) -> dict:
    import torch

    from vllm_omni.diffusion.models.lingbot_world.transformer import (
        LingBotCrossAttention,
        LingBotSelfAttention,
    )

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    def sync():
        # Every timing here brackets a device call, so the clock has to wait
        # for it; without this the measurement times the launch, not the work.
        if device.type != "cpu":
            torch.accelerator.synchronize(device)

    torch.manual_seed(0)
    torch.set_default_dtype(dtype)

    self_attn = LingBotSelfAttention(args.dim, args.num_heads, prefix="probe.self_attn").to(device).eval()
    cross_attn = LingBotCrossAttention(args.dim, args.num_heads, prefix="probe.cross_attn").to(device).eval()
    ffn = (
        torch.nn.Sequential(
            torch.nn.Linear(args.dim, args.ffn_dim),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(args.ffn_dim, args.dim),
        )
        .to(device)
        .eval()
    )

    text = torch.randn(1, args.text_tokens, args.dim, device=device, dtype=dtype)
    # Project the text K/V once and reuse it, which is what the model does: a
    # session's text K/V is constant, so re-projecting it per call would time
    # work that never happens and would swamp the query-side scaling this is
    # trying to observe.
    with torch.inference_mode():
        _, text_cache = cross_attn(text[:, :1], text, cache=None)
    components: dict[str, list[Timing]] = {"self_attention (varlen)": [], "cross_attention": [], "feed_forward": []}

    with torch.inference_mode():
        for sessions in args.sessions:
            tokens = args.tokens_per_chunk * sessions
            hidden = torch.randn(1, tokens, args.dim, device=device, dtype=dtype)

            # Self-attention over a request-local cache sized for this length.
            cache = _contiguous_cache(self_attn, tokens, device=device, dtype=dtype)

            def run_self():
                self_attn(hidden, cache=cache, current_start=0, sink_tokens=0, update_cache=False)

            def run_cross():
                cross_attn(hidden, None, cache=text_cache)

            def run_ffn():
                ffn(hidden)

            for name, fn in (
                ("self_attention (varlen)", run_self),
                ("cross_attention", run_cross),
                ("feed_forward", run_ffn),
            ):
                samples = _time_calls(fn, warmup=args.warmup, repeats=args.repeats, sync=sync)
                components[name].append(
                    Timing(sessions=sessions, tokens=tokens, median_s=statistics.median(samples), samples=samples)
                )

    return {
        "dim": args.dim,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "dtype": args.dtype,
        "device": str(device),
        "tokens_per_chunk": args.tokens_per_chunk,
        "notes": list(args.note),
        "components": {name: coalescing_gain(timings) for name, timings in components.items()},
        "raw": {name: [t.to_dict() for t in timings] for name, timings in components.items()},
    }


def _contiguous_cache(attention, tokens: int, *, device, dtype):
    """A request-local K/V cache wide enough to hold one forward's tokens."""
    import torch

    from vllm_omni.diffusion.models.lingbot_world.transformer import LingBotAttentionCache

    shape = (1, tokens, attention.num_local_heads, attention.head_dim)
    return LingBotAttentionCache(
        key=torch.zeros(shape, device=device, dtype=dtype),
        value=torch.zeros(shape, device=device, dtype=dtype),
        end=0,
        absolute_end=0,
        last_start=None,
        sink_end=0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe how block latency scales with coalesced tokens.")
    parser.add_argument("--dim", type=int, default=5120, help="Hidden size. LingBot World v2 uses 5120.")
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--ffn-dim", type=int, default=13824)
    parser.add_argument("--text-tokens", type=int, default=512)
    parser.add_argument(
        "--tokens-per-chunk",
        type=int,
        default=960,
        help="Tokens one session contributes. 320x256 is 320/frame x 3 frames.",
    )
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default=None)
    parser.add_argument("--note", action="append", default=[], required=True, help="Conditions, recorded verbatim.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    with single_process_parallel(torch.device(args.device)):
        results = probe(args)
    print(format_report(results))
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
