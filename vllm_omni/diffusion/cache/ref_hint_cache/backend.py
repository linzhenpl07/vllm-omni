# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-hint cache backend (RFC #4710, P1).

Plugs the P1 reference-hint cache into the unified cache-backend framework, alongside
TeaCache / MagCache / cache-dit. Enable via ``cache_backend="ref_hint"``.

NOTE — this is a *model-level integration*, not a generic backend. Unlike the generic
block caches (which act on any transformer via hooks/extractors and know nothing about the
model), this backend deliberately reaches into a model-provided contract: the transformer
must expose ``enable_ref_hint_cache`` / ``reset_ref_hint_cache``, and the reuse/skip logic
+ CFG branch-keying live *inside the model's forward*. The backend itself is only a
lifecycle adapter (enable once, reset per generation). It applies only to
reference-conditioned models that implement that contract — currently Wan-VACE — and
errors out on any other model, pointing them at the generic backends. This is by design:
P1 caches the *reference* contribution, which only exists in reference-conditioned models.
It is a lossy, opt-in fast path. Treat the abstraction boundary as model-level, not as a
clean generic backend.
"""

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend

logger = init_logger(__name__)

# Model-provided contract this backend drives (see module docstring). A model opts in to
# P1 by implementing these; the backend errors if they are absent.
_ENABLE_HOOK = "enable_ref_hint_cache"
_RESET_HOOK = "reset_ref_hint_cache"


class RefHintCacheBackend(CacheBackend):
    """P1 reference-hint cache backend (model-level integration; see module docstring).

    ``enable`` turns on the transformer's built-in hint cache with the configured refresh
    interval; ``refresh`` clears it at the start of each generation. The caching logic and
    CFG branch-keying live in the model (``RefHintCacheState`` + the transformer forward),
    so this backend is a thin lifecycle adapter, not a generic transformer-agnostic cache.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> backend = RefHintCacheBackend(DiffusionCacheConfig(ref_hint_refresh_interval=2))
        >>> backend.enable(pipeline)          # once, after the pipeline is built
        >>> backend.refresh(pipeline, num_inference_steps=30)  # before each generation
    """

    def _get_transformer(self, pipeline: Any) -> Any:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("ref_hint cache backend requires pipeline.transformer")
        return transformer

    def enable(self, pipeline: Any) -> None:
        transformer = self._get_transformer(pipeline)
        if not hasattr(transformer, _ENABLE_HOOK):
            raise ValueError(
                f"{transformer.__class__.__name__} does not support the 'ref_hint' cache (P1). "
                "It applies only to reference-conditioned models that implement the "
                f"'{_ENABLE_HOOK}' / '{_RESET_HOOK}' contract (e.g. WanVACETransformer3DModel). "
                "Use a generic backend ('tea_cache' / 'mag_cache' / 'cache_dit') for other models."
            )
        refresh_interval = getattr(self.config, "ref_hint_refresh_interval", 2)
        getattr(transformer, _ENABLE_HOOK)(refresh_interval=refresh_interval)
        self.enabled = True
        logger.info(
            "Reference-hint cache (P1, lossy) enabled with refresh_interval=%d on %s",
            refresh_interval,
            transformer.__class__.__name__,
        )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        transformer = self._get_transformer(pipeline)
        if hasattr(transformer, _RESET_HOOK):
            getattr(transformer, _RESET_HOOK)()
            if verbose:
                logger.debug("Reference-hint cache reset for new generation")
