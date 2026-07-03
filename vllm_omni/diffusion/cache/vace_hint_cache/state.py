# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""State for the VACE reference-hint cache (RFC #4710, P1).

P1 is a *lossy, opt-in* fast path: on Wan-VACE the reference is injected via the
``vace_blocks`` (a ControlNet-style side branch computed from the reference/control
input). Those per-layer hints drift only slowly across denoising steps, so we can compute
them at a refresh step and *reuse* them on the steps in between, skipping the vace_blocks
recompute. This is lossy (the hint does drift) and therefore default-off and quality-gated.

This module holds only the pure-Python bookkeeping (no torch, no model deps) so it is
unit-testable in isolation. The actual reuse/skip happens in the transformer forward.

Cache keying
------------
Keyed by ``branch`` = the call index within a single denoising step, NOT by step. This is
robust to how CFG is run:
- batched CFG (one forward per step, batch=[cond, uncond]) -> branch is always 0, the
  cached tensor carries both branches;
- sequential CFG (two forwards per step: cond then uncond) -> branch 0 / 1 keep them apart
  so cond never reuses uncond's hint (the "CFG branch-keying is mandatory" corner case);
- no CFG (g=1) -> branch always 0.

``step`` comes from the forward context's ``denoise_step_idx`` (set once per denoising
step by the pipeline). If it is ``None`` (e.g. warmup / no forward context) we always
recompute and never reuse, so the cache is a safe no-op.
"""

from __future__ import annotations

from typing import Any


class VaceHintCacheState:
    """Per-request bookkeeping for reusing VACE hints across denoising steps.

    Args:
        refresh_interval: recompute (refresh) the hints every ``K`` denoising steps; a very
            large value means "compute once at the first step and reuse for the rest".
    """

    def __init__(self, refresh_interval: int = 2):
        self.refresh_interval = max(1, int(refresh_interval))
        self._cache: dict[int, Any] = {}       # branch -> cached vace_hints
        self._last_step: int | None = None
        self._call_idx: int = 0
        self.hits: int = 0
        self.misses: int = 0

    def reset(self) -> None:
        """Clear all cached hints and counters. Call at the start of each generation."""
        self._cache.clear()
        self._last_step = None
        self._call_idx = 0
        self.hits = 0
        self.misses = 0

    def begin_call(self, step: int | None) -> tuple[int | None, bool]:
        """Advance the per-step branch counter and decide whether to recompute.

        Returns ``(branch, should_refresh)``. ``should_refresh=True`` means the caller must
        recompute the hints (and then call :meth:`store`); ``False`` means it may reuse
        :meth:`get`. ``step is None`` always forces a refresh with ``branch=None``.
        """
        if step is None:
            return None, True
        if step != self._last_step:
            self._last_step = step
            self._call_idx = 0
        else:
            self._call_idx += 1
        branch = self._call_idx
        should_refresh = (step % self.refresh_interval == 0) or (branch not in self._cache)
        return branch, should_refresh

    def get(self, branch: int) -> Any:
        """Return the cached hints for ``branch`` (only valid when refresh was False)."""
        self.hits += 1
        return self._cache[branch]

    def store(self, branch: int | None, hints: Any) -> None:
        """Store freshly-computed hints for ``branch`` (no-op if branch is None)."""
        self.misses += 1
        if branch is not None:
            self._cache[branch] = hints
