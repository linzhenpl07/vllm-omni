# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""VACE reference-hint cache (RFC #4710, P1): lossy, opt-in reuse of the vace_blocks
hints across denoising steps. See ``state.py`` for the mechanism and ``backend.py`` for
the cache-backend integration (``cache_backend="vace_hint"``)."""

from vllm_omni.diffusion.cache.vace_hint_cache.state import VaceHintCacheState

__all__ = ["VaceHintCacheState", "VaceHintCacheBackend"]


def __getattr__(name):
    # Lazy import so ``VaceHintCacheState`` stays importable without vllm/torch deps.
    if name == "VaceHintCacheBackend":
        from vllm_omni.diffusion.cache.vace_hint_cache.backend import VaceHintCacheBackend

        return VaceHintCacheBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
