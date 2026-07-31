# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the framework-side RefHintCacheBackend."""

from typing import cast

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")

from vllm_omni.diffusion.cache.ref_hint_cache import RefHintCacheBackend  # noqa: E402
from vllm_omni.diffusion.cache.ref_hint_cache.backend import (  # noqa: E402
    _PinnedHintBufferPool,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig  # noqa: E402
from vllm_omni.diffusion.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
)
from vllm_omni.diffusion.model_region import ModelRegion  # noqa: E402


class _VaceLikeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vace_blocks = nn.ModuleList([nn.Identity()])


class _PlainTransformer(nn.Module):
    pass


class _FakePipeline:
    def __init__(self, transformer=None, transformer_2=None):
        if transformer is not None:
            self.transformer = transformer
        if transformer_2 is not None:
            self.transformer_2 = transformer_2


def _cfg(**kw):
    return DiffusionCacheConfig(**kw)


def test_quality_validated_strategy_is_default():
    backend = RefHintCacheBackend(_cfg())
    assert backend._strategy() == "forecast50"


def test_lossy_interval_requires_acknowledgement():
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=2))
    with pytest.raises(ValueError, match="acknowledge_lossy"):
        backend.enable(_FakePipeline(_VaceLikeTransformer()))


def test_lossless_interval_is_exempt_and_exposes_handler():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(_FakePipeline(owner))
    assert backend.enabled
    assert backend.get_model_region_handler() is backend

    sentinel = [torch.tensor([1.0])]
    context = ForwardContext()
    with override_forward_context(context):
        context.denoise_step_idx = 0
        assert backend.execute(ModelRegion.REFERENCE_HINTS, owner, lambda: sentinel) is sentinel

    state = backend._states[id(owner)]
    assert state._history == {}
    assert state.misses == 0


def test_both_experts_get_isolated_state_and_reset():
    first, second = _VaceLikeTransformer(), _VaceLikeTransformer()
    pipeline = _FakePipeline(first, second)
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(pipeline)
    assert set(backend._states) == {id(first), id(second)}

    for state in backend._states.values():
        branch, _ = state.begin_call(0)
        state.store(branch, 0, [torch.tensor([1.0])])
    backend.refresh(pipeline, num_inference_steps=30)
    assert all(state.misses == 0 and state._history == {} for state in backend._states.values())


def test_second_expert_only_config():
    second = _VaceLikeTransformer()
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(_FakePipeline(transformer_2=second))
    assert set(backend._states) == {id(second)}


def test_no_transformer_raises():
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    with pytest.raises(ValueError, match="transformer_2"):
        backend.enable(_FakePipeline())


def test_unsupported_model_raises():
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    with pytest.raises(ValueError, match="does not expose"):
        backend.enable(_FakePipeline(_PlainTransformer()))


def test_reuse_strategy_skips_compute_on_second_step():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="reuse",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(_FakePipeline(owner))
    context = ForwardContext()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return [torch.tensor([float(calls)])]

    with override_forward_context(context):
        context.denoise_step_idx = 0
        first = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 1
        second = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)

    assert calls == 1
    assert torch.equal(first[0], second[0])


def test_forecast50_uses_two_fresh_values_and_damped_prediction():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="forecast50",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(_FakePipeline(owner))
    context = ForwardContext()
    values = iter((0.0, 2.0))

    def compute():
        return [torch.tensor([next(values)])]

    with override_forward_context(context):
        context.denoise_step_idx = 0
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 1
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 2
        forecast = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)

    # Nominal gain 0.5 is limited by the 0.25 trust region:
    # 2 + 0.25 * (2 - 0) = 2.5.
    assert torch.equal(forecast[0], torch.tensor([2.5]))


def test_forecast50_k2_reuses_oldest_storage_and_evicts_before_refresh():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="forecast50",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(_FakePipeline(owner))
    context = ForwardContext()
    created = []

    def compute():
        value = [torch.tensor([float(len(created) * 2)])]
        created.append(value)
        return value

    with override_forward_context(context):
        context.denoise_step_idx = 0
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 1
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        oldest_ptr = created[0][0].data_ptr()

        context.denoise_step_idx = 2
        forecast = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        assert forecast[0].data_ptr() == oldest_ptr
        assert torch.equal(forecast[0], torch.tensor([2.5]))

        context.denoise_step_idx = 3
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)

    history = backend._states[id(owner)].history(0)
    assert [step for step, _ in history] == [1, 3]
    assert len(created) == 3


def test_cpu_offload_rejects_unsupported_strategy_or_interval():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="reuse",
            ref_hint_acknowledge_lossy=True,
            ref_hint_cpu_offload=True,
        )
    )
    with pytest.raises(ValueError, match="forecast50"):
        backend.enable(_FakePipeline(owner))

    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=3,
            ref_hint_strategy="forecast50",
            ref_hint_acknowledge_lossy=True,
            ref_hint_cpu_offload=True,
        )
    )
    with pytest.raises(ValueError, match="refresh_interval=2"):
        backend.enable(_FakePipeline(owner))


def test_pinned_pool_reuses_matching_buffers_and_enforces_limit():
    source = torch.empty(4, dtype=torch.float32)
    pool = _PinnedHintBufferPool(source.numel() * source.element_size(), pin_memory=False)
    first = pool.acquire_like(source)
    first_ptr = first.data_ptr()
    pool.release([first])
    second = pool.acquire_like(source)
    assert second.data_ptr() == first_ptr
    assert pool.allocated_bytes == source.numel() * source.element_size()

    with pytest.raises(MemoryError, match="pinned CPU buffer limit exceeded"):
        pool.acquire_like(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_offload_reuses_pool_and_copies_on_dedicated_stream():
    owner = _VaceLikeTransformer().cuda()
    pipeline = _FakePipeline(owner)
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="forecast50",
            ref_hint_acknowledge_lossy=True,
            ref_hint_cpu_offload=True,
            ref_hint_cpu_memory_limit_mb=1,
        )
    )
    backend.enable(pipeline)
    context = ForwardContext()

    def run_request():
        with override_forward_context(context):
            context.denoise_step_idx = 0
            backend.execute(
                ModelRegion.REFERENCE_HINTS,
                owner,
                lambda: [torch.tensor([0.0], device="cuda")],
            )
            context.denoise_step_idx = 1
            backend.execute(
                ModelRegion.REFERENCE_HINTS,
                owner,
                lambda: [torch.tensor([2.0], device="cuda")],
            )
            context.denoise_step_idx = 2
            result = backend.execute(
                ModelRegion.REFERENCE_HINTS,
                owner,
                lambda: [torch.tensor([99.0], device="cuda")],
            )
        torch.accelerator.synchronize()
        return result

    result = run_request()
    assert torch.equal(result[0].cpu(), torch.tensor([2.5]))
    assert all(value.tensors[0].is_pinned() for _, value in backend._states[id(owner)].history(0))
    allocated = backend._pinned_pool.allocated_bytes
    assert allocated == 2 * torch.tensor([0.0]).nbytes
    assert len(backend._copy_streams) == 1

    backend.finish_request(pipeline)
    run_request()
    assert backend._pinned_pool.allocated_bytes == allocated


def test_finish_request_releases_retained_hints():
    owner = _VaceLikeTransformer()
    pipeline = _FakePipeline(owner)
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="reuse",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(pipeline)
    context = ForwardContext()

    with override_forward_context(context):
        context.denoise_step_idx = 0
        backend.execute(
            ModelRegion.REFERENCE_HINTS,
            owner,
            lambda: [torch.tensor([1.0])],
        )

    state = backend._states[id(owner)]
    assert state._history
    backend.finish_request(pipeline)
    assert state._history == {}
    assert state.misses == 0


def test_unrelated_region_behavior_is_direct_compute():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(_FakePipeline(owner))
    sentinel = [torch.tensor([7.0])]
    assert backend.execute(cast(ModelRegion, "other"), owner, lambda: sentinel) is sentinel
