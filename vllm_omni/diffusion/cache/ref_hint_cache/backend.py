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

# Transformers whose reference branch must be cached. Multi-expert models (e.g. Wan2.2
# VACE) run the reference branch in every expert, and some configs load only the second
# expert -- so cache each transformer that is present, not just ``pipeline.transformer``.
_TRANSFORMER_ATTRS = ("transformer", "transformer_2")


class RefHintCacheBackend(CacheBackend):
    """P1 reference-hint cache backend (model-level integration; see module docstring).

    ``enable`` turns on the transformer's built-in hint cache with the configured refresh
    interval; ``refresh`` clears it at the start of each generation. The caching logic and
    CFG branch-keying live in the model (``RefHintCacheState`` + the transformer forward),
    so this backend is a thin lifecycle adapter, not a generic transformer-agnostic cache.

    Reuse is lossy, so a reusing interval (>= 2) must be explicitly acknowledged via
    ``ref_hint_acknowledge_lossy=True`` (RFC #4710); ``enable`` raises otherwise.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> cfg = DiffusionCacheConfig(ref_hint_refresh_interval=2, ref_hint_acknowledge_lossy=True)
        >>> backend = RefHintCacheBackend(cfg)
        >>> backend.enable(pipeline)          # once, after the pipeline is built
        >>> backend.refresh(pipeline, num_inference_steps=30)  # before each generation
    """

    def _get_transformers(self, pipeline: Any) -> list[Any]:
        """Every transformer present on the pipeline (handles multi-expert / expert-only configs)."""
        transformers = [t for attr in _TRANSFORMER_ATTRS if (t := getattr(pipeline, attr, None)) is not None]
        if not transformers:
            raise ValueError("ref_hint cache backend requires pipeline.transformer or pipeline.transformer_2")
        return transformers

    def _check_lossy_ack(self, refresh_interval: int) -> None:
        """Refuse a reusing (lossy) interval unless the user explicitly acknowledged the loss."""
        if refresh_interval >= 2 and not getattr(self.config, "ref_hint_acknowledge_lossy", False):
            raise ValueError(
                "The 'ref_hint' cache is lossy: reusing reference hints "
                f"(ref_hint_refresh_interval={refresh_interval}) can degrade output well beyond "
                "RFC #4710's <=8% mean-DINOv2 guidance (measured ~20% DINOv2 drop at K=2). "
                "Set DiffusionCacheConfig.ref_hint_acknowledge_lossy=True to opt in, or use "
                "ref_hint_refresh_interval=1 for the lossless (recompute-every-step) path."
            )

    def enable(self, pipeline: Any) -> None:
        refresh_interval = getattr(self.config, "ref_hint_refresh_interval", 2)
        self._check_lossy_ack(refresh_interval)
        transformers = self._get_transformers(pipeline)
        for transformer in transformers:
            if not hasattr(transformer, _ENABLE_HOOK):
                raise ValueError(
                    f"{transformer.__class__.__name__} does not support the 'ref_hint' cache (P1). "
                    "It applies only to reference-conditioned models that implement the "
                    f"'{_ENABLE_HOOK}' / '{_RESET_HOOK}' contract (e.g. WanVACETransformer3DModel). "
                    "Use a generic backend ('tea_cache' / 'mag_cache' / 'cache_dit') for other models."
                )
        for transformer in transformers:
            getattr(transformer, _ENABLE_HOOK)(refresh_interval=refresh_interval)
            logger.info(
                "Reference-hint cache (P1, lossy) enabled with refresh_interval=%d on %s",
                refresh_interval,
                transformer.__class__.__name__,
            )
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        for transformer in self._get_transformers(pipeline):
            if hasattr(transformer, _RESET_HOOK):
                getattr(transformer, _RESET_HOOK)()
                if verbose:
                    logger.debug("Reference-hint cache reset for new generation on %s", transformer.__class__.__name__)
