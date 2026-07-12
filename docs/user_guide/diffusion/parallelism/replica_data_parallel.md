# Replica Data Parallelism Guide


## Table of Content

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Configuration Parameters](#configuration-parameters)
- [Expected Performance](#expected-performance)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

Replica Data Parallelism (replica-DP) scales **request throughput** by running `N`
independent diffusion engine replicas, one per GPU (or per GPU group), and routing
each incoming request to a single replica. It is the serving-level, model-agnostic
data-parallel axis for diffusion — the analogue of vLLM's dense-model DP, where the
serving layer holds multiple engine replicas and load-balances requests across them.

This is different from the other parallelism methods in this guide
([Tensor](tensor_parallel.md), [Sequence](sequence_parallel.md),
[CFG](cfg_parallel.md), [Pipeline](pipeline_parallel.md), [VAE](vae_parallelism.md)),
which split a **single request** across GPUs to reduce its latency. Replica-DP instead
replicates the **whole engine** to raise aggregate throughput:

| | Intra-request axes (TP / SP / CFG / PP / VAE) | **Replica-DP** |
|---|---|---|
| Splits | one request across GPUs | requests across replicas |
| Improves | single-request **latency** | aggregate **throughput** |
| Per-request latency | goes down | **unchanged** |
| Cross-GPU collectives | yes (per step) | **none** (replicas are isolated) |

The two are composable: use an intra-request axis **inside** each replica (via
`tensor_parallel_size`, etc.) and replica-DP **across** replicas to scale both latency
and throughput at once.

---

## How It Works

Setting `runtime.num_replicas: N` on a diffusion stage builds a pool of `N` replicas,
each an independent `DiffusionEngine` on its own device(s) as assigned by
`runtime.devices`. Replicas do not share torch collectives; a request runs end-to-end
on exactly one replica, so outputs are bit-identical to single-replica execution for a
fixed seed. The serving layer distributes requests across the live replicas.

```text
                         Requests
                   A       B       C       D
                   |       |       |       |
                   +----- StagePool / router -----+
                   |            |            |
                   v            v            v
             Replica 0     Replica 1     Replica 2 ...
          DiffusionEngine DiffusionEngine DiffusionEngine
           (GPU 0)         (GPU 1)         (GPU 2)
```

---

## Quick Start

Add `num_replicas` and a matching `devices` list to the diffusion stage's `runtime`
block in your stage config, then serve with `--omni`. Example for a 2-replica,
single-GPU-per-replica deployment of `Wan-AI/Wan2.2-TI2V-5B`:

```yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      num_replicas: 2       # fan out 2 independent engine replicas
      devices: "0,1"        # replica 0 -> GPU 0, replica 1 -> GPU 1
    engine_args:
      model_stage: dit
```

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B --omni \
    --port 8098 \
    --stage-configs-path /path/to/your_stage_config.yaml
```

Requests to the video endpoint (see [Videos API](../../../serving/videos_api.md)) are
now spread across both replicas; send concurrent requests to see throughput scale.

---

## Configuration Parameters

Set on the diffusion stage's `runtime` block (see
[Stage Configuration](../../../configuration/stage_configs.md#runtime)):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_replicas` | int | `1` | Number of independent engine replicas for this stage. Requests are load-balanced across them. |
| `devices` | str | `"0"` | Logical device list for the replica pool. Two accepted shapes (see below). |

The length of `devices` must match one of two shapes when `num_replicas > 1`:

- **Pool mode** — `len(devices) == num_replicas * tensor_parallel_size`: enumerate the
  full pool; each replica takes `tensor_parallel_size` consecutive entries.
  e.g. `num_replicas: 4`, `tensor_parallel_size: 1`, `devices: "0,1,2,3"`.
- **Template mode** — `len(devices) == tensor_parallel_size`: declare one per-replica
  template; replica `r` is offset by `r * tensor_parallel_size`.
  e.g. `num_replicas: 4`, `tensor_parallel_size: 2`, `devices: "0,1"` → GPUs
  `{0,1}, {2,3}, {4,5}, {6,7}`.

!!! info
    In single-runtime `vllm serve`, replica fan-out is driven **only** by the config's
    `runtime.num_replicas` — there is no CLI replica flag for this path. (The headless /
    multi-runtime launch path uses `--omni-dp-size-local`, which is process-local and
    requires `--stage-id`.) If only one replica loads, check `runtime.num_replicas` in
    the stage config first.

---

## Expected Performance

Throughput scales near-linearly with replica count. Measured on
`Wan-AI/Wan2.2-TI2V-5B` (832×480, 33 frames, 30 steps), 4× A800-80GB (NVLink),
one GPU per replica:

| Replicas | Throughput (videos/min) | Scaling | Efficiency |
|----------|-------------------------|---------|------------|
| 1 | 4.71 | 1.00× | — |
| 2 | 9.20 | 1.95× | 98% |
| 4 | 18.02 | 3.83× | 96% |

Per-request latency stayed flat (~13–14 s) across all replica counts, and seed-fixed
outputs were bit-identical regardless of `N` — replicas scale throughput, not
single-request latency, with no cross-replica interference.

---

## Best Practices

### When to Use

**Good for:**

- Throughput-bound serving: many concurrent generation requests
- The model (plus its VAE) fits on a single replica's device(s)
- Combining with an intra-request axis inside each replica (e.g.
  `tensor_parallel_size` for a large backbone) to scale latency **and** throughput

**Not for:**

- Reducing single-request latency — use an intra-request axis instead
- Models too large to fit one replica per device — shard the model first with
  [Tensor Parallelism](tensor_parallel.md) or [HSDP](hsdp.md), then optionally add
  replicas on top
- Single-GPU setups

### Sizing

Pick `num_replicas` so that `num_replicas * tensor_parallel_size` equals the number of
GPUs you want the stage to occupy. Give each replica the smallest device group that
fits the model to maximize the replica count (and thus throughput).

---

## Troubleshooting

### Only one replica loads / no throughput gain

**Symptom**: throughput does not scale; only one GPU is busy.

**Solutions**:

1. Set `runtime.num_replicas` in the **stage config** — in single-runtime `serve` this
   is the only knob that fans out replicas (there is no CLI flag for it; the
   `--omni-dp-size-local` flag applies to the headless / multi-runtime path and requires
   `--stage-id`).
2. Ensure `runtime.devices` enumerates the full pool (pool mode) or a valid per-replica
   template (template mode) — see [Configuration Parameters](#configuration-parameters).
3. Replicas warm up sequentially: replica 0 is ready first and the rest a few seconds
   later. Send load after all replicas report ready.

### `ValueError` on device length

**Symptom**: startup fails complaining about the `devices` length.

**Solution**: `len(devices)` must equal `num_replicas * tensor_parallel_size` (pool
mode) or `tensor_parallel_size` (template mode). Any other length is rejected.

---

## Summary

1. ✅ **Enable replica-DP** — set `runtime.num_replicas: N` on the diffusion stage to
   scale request throughput near-linearly.
2. ✅ **Size `devices`** — `num_replicas * tensor_parallel_size` entries (pool mode) or
   `tensor_parallel_size` (template mode).
3. ✅ **Use the config** — in single-runtime `serve`, fan-out comes from
   `runtime.num_replicas` (the headless / multi-runtime path uses `--omni-dp-size-local`).
4. ✅ **Compose for both axes** — replica-DP across replicas + an intra-request axis
   inside each replica scales throughput and latency together.
