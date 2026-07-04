# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-hint cache (RFC #4710, P1): lossy, opt-in reuse of a reference-conditioned
model's side-branch hints across denoising steps. See ``state.py`` for the mechanism and
``backend.py`` for the cache-backend integration (``cache_backend="ref_hint"``). Currently
wired to Wan-VACE; the machinery is model-agnostic (any model implementing the
``enable_ref_hint_cache`` / ``reset_ref_hint_cache`` contract can use it)."""

from vllm_omni.diffusion.cache.ref_hint_cache.state import RefHintCacheState

__all__ = ["RefHintCacheState", "RefHintCacheBackend"]


def __getattr__(name):
    # Lazy import so ``RefHintCacheState`` stays importable without vllm/torch deps.
    if name == "RefHintCacheBackend":
        from vllm_omni.diffusion.cache.ref_hint_cache.backend import RefHintCacheBackend

        return RefHintCacheBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
