# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-hint cache backend (RFC #4710, P1).

The backend handles the complete acceleration lifecycle: request reset,
denoising-step/CFG-branch bookkeeping, retained hint history, and the selected
reuse strategy.  Models expose only the acceleration-neutral
``ModelRegion.REFERENCE_HINTS`` execution seam and contain no cache state or
policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.ref_hint_cache.state import (
    SUPPORTED_REF_HINT_STRATEGIES,
    RefHintCacheState,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.model_region import (
    ModelRegion,
    ModelRegionHandler,
)

logger = init_logger(__name__)

T = TypeVar("T")
HintValue = list[torch.Tensor]

_TRANSFORMER_ATTRS = ("transformer", "transformer_2")
_FORECAST_GAIN = 0.5
_FORECAST_ALPHA_MAX = 1.5
_FORECAST_CORRECTION_MAX = 0.25


@dataclass
class _OffloadedHints:
    tensors: HintValue
    ready: torch.cuda.Event


CachedHintValue = HintValue | _OffloadedHints


class _PinnedHintBufferPool:
    """Shape-aware pinned CPU pool with a strict allocation budget."""

    def __init__(self, max_bytes: int, *, pin_memory: bool = True):
        self.max_bytes = max_bytes
        self.pin_memory = pin_memory
        self.allocated_bytes = 0
        self._free: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}

    @staticmethod
    def _key(tensor: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype]:
        return tuple(tensor.shape), tensor.dtype

    @staticmethod
    def _nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def acquire_like(self, tensor: torch.Tensor) -> torch.Tensor:
        key = self._key(tensor)
        available = self._free.get(key)
        if available:
            return available.pop()

        nbytes = self._nbytes(tensor)
        if self.allocated_bytes + nbytes > self.max_bytes:
            requested_mib = nbytes / (1024**2)
            allocated_mib = self.allocated_bytes / (1024**2)
            limit_mib = self.max_bytes / (1024**2)
            raise MemoryError(
                "ref_hint pinned CPU buffer limit exceeded: "
                f"requested={requested_mib:.1f} MiB "
                f"allocated={allocated_mib:.1f} MiB limit={limit_mib:.1f} MiB. "
                "Increase ref_hint_cpu_memory_limit_mb or disable "
                "ref_hint_cpu_offload."
            )
        buffer = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.allocated_bytes += nbytes
        return buffer

    def acquire_set(self, hints: HintValue) -> HintValue:
        acquired: HintValue = []
        try:
            for hint in hints:
                acquired.append(self.acquire_like(hint))
        except Exception:
            self.release(acquired)
            raise
        return acquired

    def release(self, hints: HintValue) -> None:
        for hint in hints:
            self._free.setdefault(self._key(hint), []).append(hint)


class RefHintCacheBackend(CacheBackend):
    """Framework-side reference-hint reuse and forecasting.

    ``ref_hint_strategy="reuse"`` returns the latest fresh hints on skipped
    steps.  ``"forecast50"`` retains two fresh observations and applies a
    damped first-order prediction with nominal gain 0.5 and a 0.25
    trust-region cap.  Both strategies remain approximate and require
    ``ref_hint_acknowledge_lossy=True`` when
    ``ref_hint_refresh_interval >= 2``.
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._states: dict[int, RefHintCacheState[CachedHintValue]] = {}
        limit_mb = int(getattr(config, "ref_hint_cpu_memory_limit_mb", 4096))
        self._pinned_pool = _PinnedHintBufferPool(max(limit_mb, 0) * 1024**2)
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}

    def _get_transformers(self, pipeline: object) -> list[nn.Module]:
        """Return every present transformer for multi-expert/expert-only pipelines."""
        transformers: list[nn.Module] = []
        for attr in _TRANSFORMER_ATTRS:
            transformer = cast(nn.Module | None, getattr(pipeline, attr, None))
            if transformer is not None:
                transformers.append(transformer)
        if not transformers:
            raise ValueError("ref_hint cache backend requires pipeline.transformer or pipeline.transformer_2")
        return transformers

    def _strategy(self) -> str:
        strategy = str(getattr(self.config, "ref_hint_strategy", "forecast50"))
        if strategy not in SUPPORTED_REF_HINT_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_REF_HINT_STRATEGIES))
            raise ValueError(f"Unsupported ref_hint_strategy={strategy!r}; expected one of: {supported}")
        return strategy

    def _refresh_interval(self) -> int:
        return max(1, int(getattr(self.config, "ref_hint_refresh_interval", 2)))

    def _new_state(self) -> RefHintCacheState[CachedHintValue]:
        return RefHintCacheState(
            refresh_interval=self._refresh_interval(),
            strategy=self._strategy(),
        )

    def _state_for(self, owner: nn.Module) -> RefHintCacheState[CachedHintValue]:
        owner_id = id(owner)
        state = self._states.get(owner_id)
        if state is None:
            state = self._new_state()
            self._states[owner_id] = state
        return state

    def _check_lossy_ack(self) -> None:
        refresh_interval = self._refresh_interval()
        if refresh_interval >= 2 and not getattr(self.config, "ref_hint_acknowledge_lossy", False):
            raise ValueError(
                "The 'ref_hint' cache is approximate: reusing or forecasting "
                f"reference hints (strategy={self._strategy()!r}, "
                f"ref_hint_refresh_interval={refresh_interval}) can change output quality. "
                "Set DiffusionCacheConfig.ref_hint_acknowledge_lossy=True to opt in, "
                "or use ref_hint_refresh_interval=1 for recompute-every-step."
            )

    def _cpu_offload_enabled(self) -> bool:
        return bool(getattr(self.config, "ref_hint_cpu_offload", False))

    def _check_cpu_offload_config(self) -> None:
        if not self._cpu_offload_enabled():
            return
        if self._strategy() != "forecast50" or self._refresh_interval() != 2:
            raise ValueError(
                "ref_hint_cpu_offload currently requires ref_hint_strategy='forecast50' and ref_hint_refresh_interval=2"
            )
        if int(getattr(self.config, "ref_hint_cpu_memory_limit_mb", 4096)) <= 0:
            raise ValueError("ref_hint_cpu_memory_limit_mb must be greater than zero")

    @staticmethod
    def _as_hints(value: T) -> HintValue:
        if not isinstance(value, list) or not all(torch.is_tensor(item) for item in value):
            raise TypeError(f"ModelRegion.REFERENCE_HINTS must return list[torch.Tensor], got {type(value).__name__}")
        return cast(HintValue, value)

    @staticmethod
    def _forecast(
        history: tuple[tuple[int, HintValue], ...],
        step: int,
    ) -> HintValue:
        if len(history) != 2:
            raise RuntimeError("forecast50 requires exactly two retained fresh hint observations")
        (previous_step, previous), (current_step, current) = history
        if len(previous) != len(current):
            raise RuntimeError("reference-hint history changed shape between refreshes")
        step_distance = max(current_step - previous_step, 1)
        alpha = min(max((step - current_step) / step_distance, 0.0), _FORECAST_ALPHA_MAX)
        # The nominal gain is 0.5, but the first skipped step can otherwise
        # extrapolate by half of the entire calibration delta.  A small trust
        # region prevents that one-step overshoot while retaining forecasting.
        correction = min(_FORECAST_GAIN * alpha, _FORECAST_CORRECTION_MAX)
        return [
            current_hint + (current_hint - previous_hint) * correction
            for previous_hint, current_hint in zip(previous, current)
        ]

    @staticmethod
    def _forecast_inplace_k2(
        history: tuple[tuple[int, HintValue], ...],
        step: int,
    ) -> HintValue:
        """Forecast into the oldest buffers for the K=2 schedule.

        With K=2 there is exactly one forecast between refreshes. The oldest
        fresh observation is dead immediately after that forecast, so its
        storage can safely become the output. This removes both the full
        forecast allocation and expression temporaries without adding another
        approximation.
        """
        if len(history) != 2:
            raise RuntimeError("forecast50 requires exactly two retained fresh hint observations")
        (previous_step, previous), (current_step, current) = history
        if len(previous) != len(current):
            raise RuntimeError("reference-hint history changed shape between refreshes")
        step_distance = max(current_step - previous_step, 1)
        alpha = min(max((step - current_step) / step_distance, 0.0), _FORECAST_ALPHA_MAX)
        correction = min(_FORECAST_GAIN * alpha, _FORECAST_CORRECTION_MAX)
        for output_hint, current_hint in zip(previous, current):
            # output = current + correction * (current - previous)
            output_hint.mul_(-correction).add_(current_hint, alpha=1.0 + correction)
        return previous

    def _copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._copy_streams[device] = stream
        return stream

    def _to_pinned_cpu_async(self, hints: HintValue) -> _OffloadedHints:
        """Start an asynchronous D2H copy into reusable pinned history."""
        device = hints[0].device
        current_stream = torch.cuda.current_stream(device)
        copy_stream = self._copy_stream(device)
        cpu_hints = self._pinned_pool.acquire_set(hints)

        producer_done = torch.cuda.Event()
        producer_done.record(current_stream)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(producer_done)
            for cpu_hint, hint in zip(cpu_hints, hints):
                cpu_hint.copy_(hint, non_blocking=True)
                hint.record_stream(copy_stream)
            ready = torch.cuda.Event()
            ready.record(copy_stream)
        return _OffloadedHints(cpu_hints, ready)

    def _forecast_from_cpu_k2(
        self,
        history: tuple[tuple[int, _OffloadedHints], ...],
        step: int,
        device: torch.device,
    ) -> HintValue:
        """Forecast in pooled pinned memory and enqueue H2D on a copy stream."""
        if len(history) != 2:
            raise RuntimeError("forecast50 requires exactly two retained fresh hint observations")
        (previous_step, previous_value), (current_step, current_value) = history
        if not isinstance(previous_value, _OffloadedHints) or not isinstance(current_value, _OffloadedHints):
            raise RuntimeError("CPU forecast requires two offloaded hint observations")
        previous_value.ready.synchronize()
        current_value.ready.synchronize()
        previous = previous_value.tensors
        current = current_value.tensors
        if len(previous) != len(current):
            raise RuntimeError("reference-hint history changed shape between refreshes")
        step_distance = max(current_step - previous_step, 1)
        alpha = min(max((step - current_step) / step_distance, 0.0), _FORECAST_ALPHA_MAX)
        correction = min(_FORECAST_GAIN * alpha, _FORECAST_CORRECTION_MAX)

        for previous_hint, current_hint in zip(previous, current):
            # The oldest observation is dead after this one K=2 forecast, so
            # it doubles as the pooled CPU output without a third allocation.
            previous_hint.mul_(-correction).add_(current_hint, alpha=1.0 + correction)

        current_stream = torch.cuda.current_stream(device)
        copy_stream = self._copy_stream(device)
        # Allocate on the consumer stream so the caching allocator can reuse
        # its existing blocks. Allocating on the copy stream creates a second
        # stream-local pool and can raise process-level peak VRAM.
        output = [torch.empty_like(cpu_hint, device=device) for cpu_hint in previous]
        allocation_ready = torch.cuda.Event()
        allocation_ready.record(current_stream)
        with torch.cuda.stream(copy_stream):
            # The allocator may recycle storage whose preceding use is still
            # queued on the consumer stream. Do not write it from the copy
            # stream until that stream reaches the allocation point.
            copy_stream.wait_event(allocation_ready)
            for output_hint, cpu_hint in zip(output, previous):
                output_hint.copy_(cpu_hint, non_blocking=True)
                output_hint.record_stream(copy_stream)
            ready = torch.cuda.Event()
            ready.record(copy_stream)
        current_stream.wait_event(ready)
        for output_hint in output:
            output_hint.record_stream(current_stream)
        return output

    def _release_cached(self, values: tuple[CachedHintValue, ...]) -> None:
        for value in values:
            if isinstance(value, _OffloadedHints):
                self._pinned_pool.release(value.tensors)

    def enable(self, pipeline: object) -> None:
        self._check_lossy_ack()
        self._check_cpu_offload_config()
        transformers = self._get_transformers(pipeline)
        for transformer in transformers:
            if getattr(transformer, "vace_blocks", None) is None:
                raise ValueError(
                    f"{transformer.__class__.__name__} does not expose a reference-hint "
                    "model region. The 'ref_hint' backend currently supports "
                    "reference-conditioned Wan-VACE transformers."
                )

        self._states = {id(transformer): self._new_state() for transformer in transformers}
        self.enabled = True
        logger.info(
            "Reference-hint cache enabled: strategy=%s refresh_interval=%d owners=%d cpu_offload=%s cpu_limit_mib=%d",
            self._strategy(),
            self._refresh_interval(),
            len(transformers),
            self._cpu_offload_enabled(),
            int(getattr(self.config, "ref_hint_cpu_memory_limit_mb", 4096)),
        )

    def refresh(self, pipeline: object, num_inference_steps: int, verbose: bool = True) -> None:
        for transformer in self._get_transformers(pipeline):
            self._release_cached(self._state_for(transformer).reset())
        if verbose:
            logger.debug(
                "Reference-hint cache reset for new %d-step generation",
                num_inference_steps,
            )

    def finish_request(self, pipeline: object) -> None:
        """Return retained CPU buffers to the pool when a request finishes."""
        for transformer in self._get_transformers(pipeline):
            self._release_cached(self._state_for(transformer).reset())

    def get_model_region_handler(self) -> ModelRegionHandler:
        """Install this backend only in the active request's ForwardContext."""
        return self

    def execute(
        self,
        region: ModelRegion,
        owner: nn.Module,
        compute: Callable[[], T],
    ) -> T:
        """Handle a reference-hint region; pass unrelated regions through."""
        if not self.enabled or region is not ModelRegion.REFERENCE_HINTS:
            return compute()
        if self._refresh_interval() == 1:
            # K=1 never reuses hints. Avoid retaining full hint tensors for a
            # mode that is intentionally equivalent to direct computation.
            return compute()

        step = get_forward_context().denoise_step_idx if is_forward_context_available() else None
        state = self._state_for(owner)
        branch, should_refresh = state.begin_call(step)
        if should_refresh:
            self._release_cached(state.prepare_refresh(branch))
            value = compute()
            hints = self._as_hints(value)
            stored_hints: CachedHintValue = hints
            if (
                self._cpu_offload_enabled()
                and hints
                and hints[0].is_cuda
                and self._strategy() == "forecast50"
                and self._refresh_interval() == 2
            ):
                stored_hints = self._to_pinned_cpu_async(hints)
            state.store(branch, step, stored_hints)
            return value

        assert branch is not None and step is not None
        history = state.history(branch)
        strategy = self._strategy()
        if strategy == "reuse":
            return cast(T, cast(HintValue, history[-1][1]))
        if self._refresh_interval() == 2:
            parameter = next(owner.parameters(), None)
            current = history[-1][1]
            if parameter is not None and parameter.is_cuda and isinstance(current, _OffloadedHints):
                return cast(
                    T,
                    self._forecast_from_cpu_k2(
                        cast(tuple[tuple[int, _OffloadedHints], ...], history),
                        step,
                        parameter.device,
                    ),
                )
            return cast(
                T,
                self._forecast_inplace_k2(cast(tuple[tuple[int, HintValue], ...], history), step),
            )
        return cast(
            T,
            self._forecast(cast(tuple[tuple[int, HintValue], ...], history), step),
        )
