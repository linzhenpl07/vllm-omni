# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for RefHintCacheBackend (RFC #4710, P1): the lossy opt-in gate and
multi-expert (transformer / transformer_2) handling. Uses fake transformers/pipelines --
no model or GPU. Skipped headlessly since the backend imports ``vllm.logger``."""

import pytest

pytest.importorskip("vllm")

from vllm_omni.diffusion.cache.ref_hint_cache import RefHintCacheBackend  # noqa: E402
from vllm_omni.diffusion.data import DiffusionCacheConfig  # noqa: E402


class _VaceLikeTransformer:
    """Implements the ref_hint contract; records how it was driven."""

    def __init__(self):
        self.enabled_with = None
        self.reset_count = 0

    def enable_ref_hint_cache(self, refresh_interval):
        self.enabled_with = refresh_interval

    def reset_ref_hint_cache(self):
        self.reset_count += 1


class _PlainTransformer:
    """A non-reference-conditioned model: no ref_hint hooks."""


class _FakePipeline:
    def __init__(self, transformer=None, transformer_2=None):
        if transformer is not None:
            self.transformer = transformer
        if transformer_2 is not None:
            self.transformer_2 = transformer_2


def _cfg(**kw):
    return DiffusionCacheConfig(**kw)


def test_lossy_interval_requires_acknowledgement():
    """K>=2 (reuse -> lossy) without acknowledgement is refused (RFC #4710 <=8% gate)."""
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=2))
    with pytest.raises(ValueError, match="acknowledge_lossy"):
        be.enable(_FakePipeline(_VaceLikeTransformer()))


def test_acknowledged_lossy_interval_enables():
    t = _VaceLikeTransformer()
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=2, ref_hint_acknowledge_lossy=True))
    be.enable(_FakePipeline(t))
    assert be.enabled and t.enabled_with == 2


def test_lossless_interval_is_exempt_from_gate():
    """K=1 recomputes every step (lossless) -> no acknowledgement needed."""
    t = _VaceLikeTransformer()
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    be.enable(_FakePipeline(t))
    assert be.enabled and t.enabled_with == 1


def test_both_experts_are_cached_and_reset():
    """Multi-expert (Wan2.2 VACE): both transformer and transformer_2 are driven."""
    t1, t2 = _VaceLikeTransformer(), _VaceLikeTransformer()
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    be.enable(_FakePipeline(t1, t2))
    assert t1.enabled_with == 1 and t2.enabled_with == 1
    be.refresh(_FakePipeline(t1, t2), num_inference_steps=30)
    assert t1.reset_count == 1 and t2.reset_count == 1


def test_second_expert_only_config():
    """A transformer_2-only config must work, not crash on the missing first expert."""
    t2 = _VaceLikeTransformer()
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    be.enable(_FakePipeline(transformer_2=t2))
    assert t2.enabled_with == 1


def test_no_transformer_raises():
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    with pytest.raises(ValueError, match="transformer_2"):
        be.enable(_FakePipeline())


def test_unsupported_model_raises():
    be = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    with pytest.raises(ValueError, match="does not support"):
        be.enable(_FakePipeline(_PlainTransformer()))
