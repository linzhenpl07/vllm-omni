# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter routing Cosmos3's per-CFG-branch UND text K/V through the session manager.

Cosmos3's cross-step cache is the UND (reasoner) text K/V, currently kept as
instance fields on ``Cosmos3VFMTransformer`` (``cached_kv``) with no session
keying -- so concurrent requests sharing the transformer instance can clobber
each other. ``Cosmos3StateAdapter`` keys it per session through
``SessionMemoryManager`` (RFC #4480), behind the ``enable_session_memory_manager``
opt-in flag.

Storage mirrors ``DreamZeroStateAdapter``'s cross-attention pattern: one
``EncodeOnceKV`` per (layer, CFG branch), each holding a
``{"is_init", "k", "v"}`` dict -- the class's native shape -- with the layer
count recorded in ``session.attrs`` so reads rebuild the per-layer
``list[(K, V)]``.

``freqs_gen`` (M-RoPE cos/sin for the GEN pathway) is intentionally *not* stored:
it depends only on per-request shape/fps, not on generated content or the denoise
step, so the transformer recomputes it each forward. This adapter handles only
``cached_kv``.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_omni.diffusion.memory.manager import SessionMemoryManager
from vllm_omni.diffusion.memory.objects import EncodeOnceKV

logger = logging.getLogger(__name__)


def _layer_key(layer_index: int, is_negative: bool) -> str:
    """Session key for one (layer, CFG branch) UND K/V object (cf. DreamZero ``_xattn_key``)."""
    return f"und_kv_neg/{layer_index}" if is_negative else f"und_kv/{layer_index}"


class Cosmos3StateAdapter:
    """Session-keyed view over Cosmos3's per-branch UND text K/V.

    A fresh adapter is built per forward; the durable state lives in the manager
    (mirroring ``DreamZeroStateAdapter``).
    """

    def __init__(self, session_id: str | None, manager: SessionMemoryManager) -> None:
        self._session_id = session_id
        # Pin the session: the manager may evict it from its lookup table under
        # LRU pressure, but an adapter holding a reference keeps its state alive.
        self._session = manager.get_or_create_session(session_id)

    @property
    def _num_layers(self) -> int | None:
        return self._session.attrs.get("_num_layers")

    def is_branch_initialized(self, is_negative: bool) -> bool:
        """Whether this CFG branch's UND K/V has been stored (encode-once)."""
        if self._num_layers is None:
            return False
        obj = self._session.get(_layer_key(0, is_negative))
        return obj is not None and obj.resident

    def get_branch_kv(self, is_negative: bool) -> list[tuple[Any, Any]] | None:
        """Return the branch's ``cached_kv`` as ``list[(K, V)]`` by layer, or ``None``."""
        if not self.is_branch_initialized(is_negative):
            return None
        out: list[tuple[Any, Any]] = []
        for i in range(int(self._num_layers)):  # type: ignore[arg-type]
            obj = self._session.get(_layer_key(i, is_negative))
            if obj is None or not obj.resident:
                raise RuntimeError("Cosmos3 UND K/V is partially initialized.")
            cache = obj.view()
            out.append((cache["k"], cache["v"]))
        return out

    def set_branch_kv(self, is_negative: bool, cached_kv: list[tuple[Any, Any]]) -> None:
        """Store this branch's freshly computed ``cached_kv`` (encode-once, once per generation).

        Lazily creates one ``EncodeOnceKV`` per layer (layer count from
        ``len(cached_kv)``) and commits a ``{"is_init", "k", "v"}`` dict each.
        """
        self._session.attrs["_num_layers"] = len(cached_kv)
        for i, (k, v) in enumerate(cached_kv):
            obj = self._session.get(_layer_key(i, is_negative))
            if obj is None:
                obj = EncodeOnceKV()
                self._session.put(_layer_key(i, is_negative), obj)
            # No clone: UND K/V is fixed after the encode-once pass, matching the
            # bespoke path which stored the tensors directly.
            obj.commit({"is_init": True, "k": k, "v": v})

    def load_into_transformer(self, transformer: Any, is_negative: bool) -> None:
        """Assign this branch's ``cached_kv`` onto the transformer (freqs_gen is recomputed)."""
        transformer.cached_kv = self.get_branch_kv(is_negative)

    def capture_from_transformer(self, transformer: Any, is_negative: bool) -> None:
        """Capture the freshly computed ``cached_kv`` into the session if not already stored."""
        if not self.is_branch_initialized(is_negative):
            self.set_branch_kv(is_negative, transformer.cached_kv)

    def reset(self) -> None:
        """Clear all UND K/V objects + metadata for this session (replaces ``reset_cache()``)."""
        self._session.reset()
