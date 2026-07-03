# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the VACE reference-hint cache bookkeeping (RFC #4710, P1).

These exercise VaceHintCacheState in isolation (pure Python, no model / GPU) — the
refresh schedule, CFG branch-keying, full-reuse, safe no-op when the step is unknown,
and reset.
"""

from vllm_omni.diffusion.cache.vace_hint_cache import VaceHintCacheState


def test_refresh_schedule_k2():
    """K=2: refresh on even steps, reuse on odd steps."""
    st = VaceHintCacheState(refresh_interval=2)
    b, r = st.begin_call(0)
    assert r is True  # step 0: 0 % 2 == 0 -> refresh
    st.store(b, "h0")
    b, r = st.begin_call(1)
    assert r is False and st.get(b) == "h0"  # step 1: reuse
    b, r = st.begin_call(2)
    assert r is True  # step 2: refresh
    st.store(b, "h2")
    b, r = st.begin_call(3)
    assert r is False and st.get(b) == "h2"  # step 3: reuse the fresher h2
    assert st.hits == 2 and st.misses == 2


def test_branch_keying_two_forwards_per_step():
    """Two sequential forwards per step (cond then uncond) must not alias."""
    st = VaceHintCacheState(refresh_interval=2)
    b0, r0 = st.begin_call(0)
    b1, r1 = st.begin_call(0)
    assert (b0, r0) == (0, True) and (b1, r1) == (1, True)
    st.store(b0, "cond")
    st.store(b1, "uncond")
    # next step reuses per branch, cond never gets uncond's hint
    b0, r0 = st.begin_call(1)
    b1, r1 = st.begin_call(1)
    assert (b0, r0) == (0, False) and (b1, r1) == (1, False)
    assert st.get(b0) == "cond" and st.get(b1) == "uncond"


def test_full_reuse_large_k():
    """Large K = compute once at step 0, reuse for all later steps."""
    st = VaceHintCacheState(refresh_interval=10_000)
    b, r = st.begin_call(0)
    assert r is True
    st.store(b, "once")
    for s in range(1, 12):
        b, r = st.begin_call(s)
        assert r is False and st.get(b) == "once"
    assert st.misses == 1 and st.hits == 11


def test_first_use_of_a_branch_always_refreshes():
    """Even if step % K != 0, a branch never seen before must recompute (nothing to reuse)."""
    st = VaceHintCacheState(refresh_interval=2)
    _, r = st.begin_call(1)  # step 1 is not a refresh step, but branch 0 not cached yet
    assert r is True


def test_step_none_is_safe_noop():
    """Unknown step (no forward context / warmup) -> always refresh, never cache."""
    st = VaceHintCacheState(refresh_interval=2)
    b, r = st.begin_call(None)
    assert b is None and r is True
    st.store(b, "x")  # no-op store for branch None
    assert st.hits == 0
    b, r = st.begin_call(None)
    assert b is None and r is True  # still no reuse


def test_reset_clears_state():
    st = VaceHintCacheState(refresh_interval=2)
    b, _ = st.begin_call(0)
    st.store(b, "h")
    st.begin_call(1)
    st.reset()
    assert st._cache == {} and st._last_step is None and st._call_idx == 0
    assert st.hits == 0 and st.misses == 0


def test_refresh_interval_clamped_to_at_least_one():
    st = VaceHintCacheState(refresh_interval=0)
    assert st.refresh_interval == 1
    # K=1 -> every step is a refresh step, never reuse
    st.store(*(lambda br: (br, "h"))(st.begin_call(0)[0]))
    _, r = st.begin_call(1)
    assert r is True


if __name__ == "__main__":  # allow running standalone without pytest
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
    sys.exit(0)
