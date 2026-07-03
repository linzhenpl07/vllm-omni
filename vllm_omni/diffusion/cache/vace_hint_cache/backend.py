# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""VACE reference-hint cache backend (RFC #4710, P1).

Plugs the P1 reference-hint cache into the unified cache-backend framework, alongside
TeaCache / MagCache / cache-dit. Enable via ``cache_backend="vace_hint"``.

Unlike the generic block caches (which act on any transformer via hooks/extractors), this
backend is *model-facing*: it only applies to models that expose a reference-hint branch
(``enable_vace_hint_cache`` / ``reset_vace_hint_cache``) — currently Wan-VACE. That is by
design: P1 caches the *reference* contribution, which only exists in reference-conditioned
models. It is a lossy, opt-in fast path.
"""

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend

logger = init_logger(__name__)


class VaceHintCacheBackend(CacheBackend):
    """P1 reference-hint cache backend.

    ``enable`` turns on the transformer's built-in hint cache with the configured refresh
    interval; ``refresh`` clears it at the start of each generation. The caching logic and
    CFG branch-keying live in the model (``VaceHintCacheState`` + the transformer forward),
    so this backend is a thin lifecycle adapter.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> backend = VaceHintCacheBackend(DiffusionCacheConfig(vace_hint_refresh_interval=2))
        >>> backend.enable(pipeline)          # once, after the pipeline is built
        >>> backend.refresh(pipeline, num_inference_steps=30)  # before each generation
    """

    def _get_transformer(self, pipeline: Any) -> Any:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("vace_hint cache backend requires pipeline.transformer")
        return transformer

    def enable(self, pipeline: Any) -> None:
        transformer = self._get_transformer(pipeline)
        if not hasattr(transformer, "enable_vace_hint_cache"):
            raise ValueError(
                f"{transformer.__class__.__name__} does not support the 'vace_hint' cache (P1). "
                "It applies only to reference-conditioned models with a hint branch "
                "(e.g. WanVACETransformer3DModel). Use a generic backend "
                "('tea_cache' / 'mag_cache' / 'cache_dit') for other models."
            )
        refresh_interval = getattr(self.config, "vace_hint_refresh_interval", 2)
        transformer.enable_vace_hint_cache(refresh_interval=refresh_interval)
        self.enabled = True
        logger.info(
            "VACE hint cache (P1, lossy) enabled with refresh_interval=%d on %s",
            refresh_interval,
            transformer.__class__.__name__,
        )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        transformer = self._get_transformer(pipeline)
        if hasattr(transformer, "reset_vace_hint_cache"):
            transformer.reset_vace_hint_cache()
            if verbose:
                logger.debug("VACE hint cache reset for new generation")
