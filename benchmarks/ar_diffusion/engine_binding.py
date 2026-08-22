# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire the benchmark harness to a real AR-Diffusion engine.

This is the only module here that needs a device, a checkpoint and a CUDA
build, and therefore the only one the CPU tests cannot cover. Everything the
benchmark actually measures -- the load model, the deadline model and every
metric -- lives in ``realtime_harness`` and ``realtime_metrics`` and is tested
without it.

The wiring mirrors ``examples/offline_inference/diffusion/lingbot_world_v2_realtime.py``
but drops everything model-specific: no control reducer, no action schema, no
fixed frame count. A pipeline needing per-tick controls should extend
``prompt_provider``/``control_reducer_factory`` rather than change the harness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RealtimeBackend:
    """The session manager under test, plus the capability it declared."""

    manager: Any
    spec: Any
    engine: Any


async def build_realtime_backend(args: argparse.Namespace) -> RealtimeBackend:
    """Build an AsyncOmni engine and a session manager from CLI arguments."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import (
        ARDiffusionSessionManager,
        ARDiffusionWorkerLifecycle,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Resolution and any other pipeline-level knob arrive through model_config,
    # which is an engine construction parameter -- changing one means a new
    # engine, not a new request.
    model_config: dict[str, Any] = {
        "ar_diffusion_kv_config": {
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "warmup_cudagraph": not args.enforce_eager,
        },
    }
    model_config.update(getattr(args, "model_config_overrides", None) or {})

    engine = AsyncOmni(
        model=args.model,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        model_config=model_config,
    )

    sampling = OmniDiffusionSamplingParams(seed=args.seed, output_type="latent")

    def prompt_provider(tick: Any) -> dict[str, Any]:
        prompt: dict[str, Any] = {"prompt": tick.prompt or args.prompt}
        if args.image is not None:
            prompt["multi_modal_data"] = {"image": str(args.image)}
        return prompt

    consumer = ARDiffusionOmniTickConsumer(
        engine,
        prompt_provider=prompt_provider,
        sampling_params_list=[sampling],
        diffusion_stage_id=0,
    )
    manager = ARDiffusionSessionManager(
        tick_consumer=consumer,
        lifecycle=ARDiffusionWorkerLifecycle(engine, stage_ids=[0], timeout=180.0),
        max_pending_events=args.max_pending_events,
    )
    return RealtimeBackend(manager=manager, spec=_declared_spec(engine), engine=engine)


def _declared_spec(engine: Any) -> Any:
    """Fetch the pipeline's declared AR-Diffusion KV spec, or None.

    Returns None rather than raising, because the pipeline lives in the worker
    process and an engine built with process isolation cannot reach it from
    here. Only callers that need the chunk shape -- the load benchmark, to
    build its playout grid -- have to care; driving sessions does not.
    """
    pipeline = getattr(getattr(engine, "engine", None), "pipeline", None)
    spec_fn = getattr(pipeline, "ar_diffusion_kv_cache_spec", None)
    return None if spec_fn is None else spec_fn()
