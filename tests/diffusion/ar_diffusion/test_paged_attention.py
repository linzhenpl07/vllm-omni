# SPDX-License-Identifier: Apache-2.0
"""Tests for AR-Diffusion paged self-attention contexts."""

from __future__ import annotations

import subprocess
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
    ar_diffusion_paged_attention,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    PagedSequenceMetadata,
    batch_paged_metadata,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
POS = "positive"
NEG = "negative"


def make_state(*, num_layers=1, window_chunks=2, dtype=torch.float32, device=torch.device("cpu")):
    cfg = ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window_chunks)
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=num_layers,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=dtype,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 26,
        kv_branches=(ARDiffusionKVBranchSpec(POS, 0), ARDiffusionKVBranchSpec(NEG, 1)),
        session_capacity=2,
        frames_per_block=2,
        max_scratch_tokens_per_branch=BLOCK,
        device=device,
    )
    pos = kv.begin_request("r-pos")
    neg = kv.begin_request("r-neg")
    return kv, ARDiffusionKVState(kv, "s1", {POS: pos, NEG: neg}, num_layers=num_layers)


def _dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * (HEAD_DIM**-0.5)
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value)


def _gpu_flash_attn_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is not None:
        for module in ("aiter", "flash_attn"):
            try:
                imported = __import__(module, fromlist=["flash_attn_varlen_func"])
                if getattr(imported, "flash_attn_varlen_func", None) is not None:
                    return True
            except ImportError:
                pass
        return False
    try:
        spec = find_spec("vllm.vllm_flash_attn")
        if spec is None or spec.origin is None:
            return True
        fa2_so = Path(spec.origin).parent / "_vllm_fa2_C.abi3.so"
        linked = subprocess.check_output(["ldd", str(fa2_so)], text=True, timeout=5)
    except Exception:
        return True
    if "libcudart.so.13" not in linked:
        return True
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        driver_major = int(out.splitlines()[0].split(".")[0])
    except Exception:
        return True
    return driver_major >= 580


def _commit_video_span(
    kv: ARDiffusionKVCache,
    st: ARDiffusionKVState,
    *,
    kv_branch: str,
    n_chunks: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = st.get_kv_caches(kv_branch, seq_len=n_chunks * BLOCK, commit_current=True)[0].forward_ctx
    ctx.ensure_video_slots(device)
    k = torch.randn(1, n_chunks * BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(1, n_chunks * BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][ctx.current_video_slot_mapping] = k[0]
    kv._v_pools[0][ctx.current_video_slot_mapping] = v[0]
    st.commit_paged_context(kv_branch)
    return k, v


def test_paged_context_allocates_lazily_and_commits_after_forward():
    _, st = make_state()

    contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)
    ctx = contexts[0].forward_ctx
    assert isinstance(contexts[0], ARDiffusionPagedLayerContext)
    assert st.adapter(POS).completed_chunks == 0
    assert ctx.current_video_slot_mapping is None

    ctx.ensure_video_slots(torch.device("cpu"))
    assert st.adapter(POS).completed_chunks == 0
    assert len(ctx.current_video_block_ids) == 1

    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == 1
    assert st._committed[POS] == BLOCK


def test_scratch_video_and_action_blocks_do_not_commit():
    kv, st = make_state()

    ctx = st.get_kv_caches(POS, seq_len=2 * BLOCK, commit_current=False)[0].forward_ctx
    ctx.ensure_video_slots(torch.device("cpu"))
    ctx.ensure_action_slots(3, torch.device("cpu"))

    assert ctx.current_video_block_ids == kv.scratch_block_ids(POS, 0, 2)
    assert ctx.action_scratch_block_ids == kv.scratch_block_ids(POS, 2, 1)
    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == 0
    assert st._committed[POS] == 0


def test_pipeline_kv_get_paged_path_has_no_gather_backend():
    kv, st = make_state()
    assert not hasattr(kv, "gather_window_all_layers")

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline._ar_diffusion_kv_state = st
    contexts = pipeline._kv_get(MagicMock(), False, seq_len=BLOCK, update_kv_cache=False)

    assert len(contexts) == 1
    assert isinstance(contexts[0], ARDiffusionPagedLayerContext)


@pytest.mark.parametrize("history_chunks", [0, 1, 3])
@pytest.mark.parametrize("action_len", [0, 3])
@pytest.mark.parametrize("commit_current", [False, True])
def test_paged_attention_matches_dense_reference_cpu(history_chunks, action_len, commit_current):
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2)

    history_k_parts: list[torch.Tensor] = []
    history_v_parts: list[torch.Tensor] = []
    if history_chunks:
        k, v = _commit_video_span(
            kv,
            st,
            kv_branch=POS,
            n_chunks=history_chunks,
            dtype=dtype,
            device=device,
        )
        history_k_parts.append(k)
        history_v_parts.append(v)

    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit_current)[0].forward_ctx
    ctx.ensure_video_slots(device)
    current_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv._k_pools[0][ctx.current_video_slot_mapping] = current_k[0]
    kv._v_pools[0][ctx.current_video_slot_mapping] = current_v[0]

    action_k = action_v = None
    if action_len:
        ctx.ensure_action_slots(action_len, device)
        action_k = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        action_v = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        kv._k_pools[0][ctx.action_slot_mapping] = action_k[0]
        kv._v_pools[0][ctx.action_slot_mapping] = action_v[0]

    query = torch.randn(1, BLOCK + action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    block_table, query_start_loc, seq_lens, max_query_len, max_seq_len = ctx.build_block_table(
        action_len=action_len,
        query_len=query.shape[1],
        device=device,
    )
    paged = ar_diffusion_paged_attention(
        query,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )

    if history_k_parts:
        history_k = torch.cat(history_k_parts, dim=1)
        history_v = torch.cat(history_v_parts, dim=1)
    else:
        history_k = torch.empty(1, 0, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        history_v = torch.empty(1, 0, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    new_k = torch.cat([history_k, current_k], dim=1)[:, -kv.spec.sliding_window :]
    new_v = torch.cat([history_v, current_v], dim=1)[:, -kv.spec.sliding_window :]
    if action_len:
        new_k = torch.cat([new_k, action_k], dim=1)
        new_v = torch.cat([new_v, action_v], dim=1)
    ref = _dense_attention(query, new_k, new_v)

    torch.testing.assert_close(paged, ref, rtol=1e-5, atol=1e-5)

    before = st.adapter(POS).completed_chunks
    st.commit_paged_context(POS)
    assert st.adapter(POS).completed_chunks == before + (1 if commit_current else 0)


@pytest.mark.skipif(not _gpu_flash_attn_usable(), reason="usable GPU FlashAttention is required")
@pytest.mark.parametrize("history_chunks", [1, 3])
@pytest.mark.parametrize("action_len", [0, 3])
@pytest.mark.parametrize("commit_current", [False, True])
def test_paged_attention_matches_dense_reference_gpu(history_chunks, action_len, commit_current):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2)

    history_k, history_v = _commit_video_span(
        kv,
        st,
        kv_branch=POS,
        n_chunks=history_chunks,
        dtype=dtype,
        device=device,
    )

    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit_current)[0]
    ctx = layer_ctx.forward_ctx
    current_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    current_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    action_k = action_v = None
    if action_len:
        action_k = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
        action_v = torch.randn(1, action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    query = torch.randn(1, BLOCK + action_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    # The production path: once-per-forward host prep, then the fused
    # write+attend custom op consuming the NamedTuple payload.
    ctx.prepare(device=device, action_len=action_len, query_len=query.shape[1])
    inputs = layer_ctx.to_layer_inputs()
    assert isinstance(inputs, ARDiffusionPagedLayerInputs)
    paged = paged_write_attn(
        inputs,
        query[0],
        current_k[0],
        current_v[0],
        action_k[0] if action_len else None,
        action_v[0] if action_len else None,
        HEAD_DIM**-0.5,
    ).unsqueeze(0)

    # Direct python-fn call on the same (already written) pools must be
    # bit-exact: identical kernel, identical inputs.
    direct = ar_diffusion_paged_attention(
        query,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=ctx.block_table,
        query_start_loc=ctx.query_start_loc,
        seq_lens=ctx.seq_lens,
        max_query_len=ctx.max_query_len,
        max_seq_len=ctx.max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )
    assert torch.equal(paged, direct)

    new_k = torch.cat([history_k, current_k], dim=1)[:, -kv.spec.sliding_window :]
    new_v = torch.cat([history_v, current_v], dim=1)[:, -kv.spec.sliding_window :]
    if action_len:
        new_k = torch.cat([new_k, action_k], dim=1)
        new_v = torch.cat([new_v, action_v], dim=1)
    ref = _dense_attention(query, new_k, new_v)

    torch.testing.assert_close(paged, ref, rtol=2e-2, atol=2e-2)


def test_block_table_padded_to_fixed_width():
    """Shapes must be constant across window growth: only values change."""
    device = torch.device("cpu")
    kv, st = make_state(window_chunks=2)

    ctx1 = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    ctx1.prepare(device=device, action_len=0, query_len=BLOCK)
    st.commit_paged_context(POS)

    ctx2 = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=True)[0].forward_ctx
    ctx2.prepare(device=device, action_len=0, query_len=BLOCK)
    st.commit_paged_context(POS)

    # 1-block vs 2-block visible history: same table width, same max_seq_len.
    assert ctx1.block_table.shape == ctx2.block_table.shape
    assert ctx1.max_seq_len == ctx2.max_seq_len
    expected_width = kv.spec.sliding_window // kv.block_size + 1
    assert ctx1.block_table.shape == (1, expected_width)
    # Real lengths live in seq_lens, not the padded table.
    assert int(ctx1.seq_lens[0]) == BLOCK
    assert int(ctx2.seq_lens[0]) == 2 * BLOCK


def test_prepare_is_idempotent_and_layers_share_metadata():
    device = torch.device("cpu")
    kv, st = make_state(num_layers=2)
    contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)
    fctx = contexts[0].forward_ctx
    fctx.prepare(device=device, action_len=0, query_len=BLOCK)
    table = fctx.block_table
    fctx.prepare(device=device, action_len=0, query_len=BLOCK)
    assert fctx.block_table is table  # memoized, not rebuilt

    i0, i1 = contexts[0].to_layer_inputs(), contexts[1].to_layer_inputs()
    # 0-dim tensors (NOT python ints) so dynamo doesn't install per-layer
    # value guards on the shared block code object.
    assert isinstance(i0.layer_idx, torch.Tensor) and int(i0.layer_idx) == 0
    assert isinstance(i1.layer_idx, torch.Tensor) and int(i1.layer_idx) == 1
    # All layers share the same metadata tensor objects.
    assert i0.block_table is i1.block_table
    assert i0.seq_lens is i1.seq_lens
    assert i0.video_slots is i1.video_slots
    assert i0.key_pool is kv._k_pools[0]
    assert i0.value_pool is kv._v_pools[0]
    assert i1.key_pool is kv._k_pools[1]
    assert i1.value_pool is kv._v_pools[1]


def test_layer_inputs_before_prepare_raises():
    _, st = make_state()
    layer_ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0]
    with pytest.raises(RuntimeError, match="before prepare"):
        layer_ctx.to_layer_inputs()


def test_custom_op_registration_idempotent():
    import importlib
    import sys

    assert hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn")
    mod = "vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention"
    saved = sys.modules.pop(mod)
    try:
        importlib.import_module(mod)  # re-registration must not raise
    finally:
        sys.modules[mod] = saved
    assert hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn")


def test_custom_op_compiles_fullgraph_without_recompile_on_value_change():
    """The op must trace as one opaque node: fullgraph OK, and changed tensor
    VALUES (new slots / block ids) must not trigger recompilation."""
    import torch._dynamo

    device = torch.device("cpu")
    kv, st = make_state(num_layers=2, window_chunks=2)

    def run_one_forward(commit):
        contexts = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=commit)
        fctx = contexts[0].forward_ctx
        fctx.prepare(device=device, action_len=0, query_len=BLOCK)
        q = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        k = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        v = torch.randn(BLOCK, N_HEADS, HEAD_DIM)
        # Both layers through ONE compiled fn: layer_idx is a tensor, so a
        # different layer must NOT recompile (all 40 DiT blocks share the
        # block-forward code object in production).
        for layer_ctx in contexts:
            out = compiled(layer_ctx.to_layer_inputs(), q, k, v)
        st.commit_paged_context(POS)
        return out

    torch._dynamo.reset()
    try:
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def fn(inputs, q, k, v):
            return paged_write_attn(inputs, q, k, v, None, None, HEAD_DIM**-0.5) * 1.0

        compiled = torch.compile(fn, backend=counter, fullgraph=True)

        run_one_forward(commit=True)  # history grows between calls ->
        run_one_forward(commit=True)  # block-table VALUES change, shapes don't
        run_one_forward(commit=False)

        assert counter.frame_count == 1, f"recompiled: frame_count={counter.frame_count}"
    finally:
        # Leave a clean dynamo state for later suites in the same pytest process
        # (e.g. model_executor transformers models).
        torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Cross-session batching: is the attention path already varlen-capable?
#
# Coalescing several sessions' chunks into one forward is only a plumbing
# change if the attention path already handles a batch of sequences with
# different KV lengths. These tests establish that against the same dense
# reference the single-sequence tests use, so the claim rests on a measurement
# rather than on reading the code.
# ---------------------------------------------------------------------------


def _write_sequence(kv, *, blocks, kv_len, dtype, device):
    """Fill ``kv_len`` tokens across ``blocks``, returning the dense K/V written."""
    k = torch.randn(1, kv_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(1, kv_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(kv_len, device=device)
    logical = torch.div(positions, BLOCK, rounding_mode="floor")
    offsets = positions % BLOCK
    physical = torch.tensor(blocks, device=device)[logical]
    slots = physical * BLOCK + offsets
    kv._k_pools[0][slots] = k[0]
    kv._v_pools[0][slots] = v[0]
    return k, v


def _batch_metadata(sequences, *, width, device):
    """Build batch metadata through the primitive the runtime would use."""
    table, starts, lens, _, _ = batch_paged_metadata(
        [PagedSequenceMetadata(block_ids=tuple(b), kv_len=k, query_len=q) for b, k, q in sequences],
        block_size=BLOCK,
        device=device,
        width=width,
    )
    return table, starts, lens


def _run_paged(kv, query_flat, table, starts, lens, *, max_seq_len):
    return ar_diffusion_paged_attention(
        query_flat,
        kv.key_cache(0),
        kv.value_cache(0),
        block_table=table,
        query_start_loc=starts,
        seq_lens=lens,
        max_query_len=int((starts[1:] - starts[:-1]).max().item()),
        max_seq_len=max_seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
    )


@pytest.mark.parametrize(
    "spec",
    [
        # (blocks, kv_len, query_len) per sequence. Unequal KV lengths are the
        # point: equal lengths would pass even for a batch-oblivious path.
        [((1, 2), 32, 16), ((3, 4, 5), 40, 8)],
        [((1,), 5, 3), ((2, 3), 27, 11), ((4, 5, 6), 48, 16)],
    ],
)
def test_batched_sequences_match_running_them_one_at_a_time(spec):
    """A coalesced batch must equal the same sequences run separately.

    This is the load-bearing question for cross-session chunk batching: if it
    holds, coalescing needs new metadata construction and nothing else in the
    attention path.
    """
    torch.manual_seed(0)
    device, dtype = torch.device("cpu"), torch.float32
    kv, _ = make_state(dtype=dtype, device=device, window_chunks=8)

    queries = []
    for blocks, kv_len, query_len in spec:
        _write_sequence(kv, blocks=blocks, kv_len=kv_len, dtype=dtype, device=device)
        queries.append(torch.randn(query_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device))

    width = max(len(blocks) for blocks, _, _ in spec)
    max_seq_len = width * BLOCK
    table, starts, lens = _batch_metadata(spec, width=width, device=device)
    batched = _run_paged(kv, torch.cat(queries, dim=0), table, starts, lens, max_seq_len=max_seq_len)

    alone = []
    for index, (blocks, kv_len, _) in enumerate(spec):
        one = _batch_metadata([(blocks, kv_len, queries[index].shape[0])], width=width, device=device)
        alone.append(_run_paged(kv, queries[index], *one, max_seq_len=max_seq_len))

    expected = torch.cat(alone, dim=0)
    assert batched.shape == expected.shape
    torch.testing.assert_close(batched, expected, rtol=1e-5, atol=1e-5)


def test_each_batched_sequence_matches_the_dense_reference():
    """Anchor the batch against dense attention, not only against itself.

    Comparing a batch to per-sequence paged calls would pass even if both
    shared a bug, so each member is also checked against plain attention over
    the exact K/V it should see.
    """
    torch.manual_seed(1)
    device, dtype = torch.device("cpu"), torch.float32
    kv, _ = make_state(dtype=dtype, device=device, window_chunks=8)

    spec = [((1, 2), 32, 16), ((3, 4, 5), 40, 8)]
    dense_kv, queries = [], []
    for blocks, kv_len, query_len in spec:
        dense_kv.append(_write_sequence(kv, blocks=blocks, kv_len=kv_len, dtype=dtype, device=device))
        queries.append(torch.randn(query_len, N_HEADS, HEAD_DIM, dtype=dtype, device=device))

    width = max(len(blocks) for blocks, _, _ in spec)
    table, starts, lens = _batch_metadata(spec, width=width, device=device)
    batched = _run_paged(kv, torch.cat(queries, dim=0), table, starts, lens, max_seq_len=width * BLOCK)

    offset = 0
    for index, (_, _, query_len) in enumerate(spec):
        k, v = dense_kv[index]
        ref = _dense_attention(queries[index].unsqueeze(0), k, v)[0]
        torch.testing.assert_close(batched[offset : offset + query_len], ref, rtol=1e-5, atol=1e-5)
        offset += query_len


def test_batching_is_not_vacuous_because_the_sequences_differ():
    """Negative control for the two tests above.

    If both members produced the same output, an implementation that ignored
    the per-sequence block table entirely would still pass.
    """
    torch.manual_seed(2)
    device, dtype = torch.device("cpu"), torch.float32
    kv, _ = make_state(dtype=dtype, device=device, window_chunks=8)

    spec = [((1, 2), 32, 8), ((3, 4), 32, 8)]
    for blocks, kv_len, _ in spec:
        _write_sequence(kv, blocks=blocks, kv_len=kv_len, dtype=dtype, device=device)
    query = torch.randn(8, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    table, starts, lens = _batch_metadata(spec, width=2, device=device)
    batched = _run_paged(kv, torch.cat([query, query], dim=0), table, starts, lens, max_seq_len=2 * BLOCK)

    # Same query, different KV: the two halves must differ.
    assert not torch.allclose(batched[:8], batched[8:], rtol=1e-3, atol=1e-3)


def test_padding_beyond_seq_len_is_never_read():
    """A short sequence in a wide batch must ignore its padded block slots.

    Coalescing pads every session's table to the widest member, so a session
    reading past its own KV would silently mix in another session's blocks --
    a plausible but wrong result that no timing metric can detect.
    """
    torch.manual_seed(3)
    device, dtype = torch.device("cpu"), torch.float32
    kv, _ = make_state(dtype=dtype, device=device, window_chunks=8)

    short_len = 12
    k, v = _write_sequence(kv, blocks=(1,), kv_len=short_len, dtype=dtype, device=device)
    _write_sequence(kv, blocks=(3, 4, 5), kv_len=40, dtype=dtype, device=device)
    query = torch.randn(6, N_HEADS, HEAD_DIM, dtype=dtype, device=device)

    # Pad the short sequence's table with blocks holding real, different data.
    table = torch.tensor([[1, 3, 4]], dtype=torch.int32, device=device)
    starts = torch.tensor([0, 6], dtype=torch.int32, device=device)
    lens = torch.tensor([short_len], dtype=torch.int32, device=device)
    out = _run_paged(kv, query, table, starts, lens, max_seq_len=3 * BLOCK)

    ref = _dense_attention(query.unsqueeze(0), k, v)[0]
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_batch_paged_metadata_rejects_tables_too_small_for_their_kv():
    """Silently truncating a session's KV is the failure to avoid here."""
    with pytest.raises(ValueError, match="needs 3 blocks"):
        batch_paged_metadata(
            [PagedSequenceMetadata(block_ids=(1, 2), kv_len=40, query_len=4)],
            block_size=BLOCK,
            device=torch.device("cpu"),
        )


def test_batch_paged_metadata_pads_every_row_to_the_widest_member():
    table, starts, lens, max_query_len, max_seq_len = batch_paged_metadata(
        [
            PagedSequenceMetadata(block_ids=(1,), kv_len=10, query_len=4),
            PagedSequenceMetadata(block_ids=(2, 3, 4), kv_len=40, query_len=9),
        ],
        block_size=BLOCK,
        device=torch.device("cpu"),
    )
    assert table.shape == (2, 3)
    assert table[0].tolist() == [1, 0, 0]
    assert starts.tolist() == [0, 4, 13]
    assert lens.tolist() == [10, 40]
    assert (max_query_len, max_seq_len) == (9, 3 * BLOCK)


def test_batch_paged_metadata_matches_the_single_sequence_builder():
    """One sequence through the batch primitive equals the existing path."""
    device, dtype = torch.device("cpu"), torch.float32
    kv, st = make_state(dtype=dtype, device=device, window_chunks=2)
    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
    ctx.ensure_video_slots(device)
    single = ctx.build_block_table(action_len=0, query_len=BLOCK, device=device)

    blocks, _ = ctx.video_block_table(device)
    batched = batch_paged_metadata(
        [PagedSequenceMetadata(block_ids=tuple(blocks), kv_len=ctx.kv_len, query_len=BLOCK)],
        block_size=BLOCK,
        device=device,
        width=single[0].shape[1],
    )
    torch.testing.assert_close(batched[0], single[0])
    torch.testing.assert_close(batched[1], single[1])
    torch.testing.assert_close(batched[2], single[2])
    assert batched[3] == single[3]
