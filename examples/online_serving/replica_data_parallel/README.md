# Replica Data Parallelism for Video Generation

A runnable recipe + benchmark for **replica data parallelism** (replica-DP) on a
video DiT. Replica-DP runs `N` independent diffusion engine replicas, one per GPU,
and routes each request to a single replica. It scales **request throughput**
near-linearly; it does **not** change single-request latency (for that, use an
intra-request axis such as tensor/sequence/CFG parallelism).

This example uses `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with one GPU per replica.

## Files

| File | Purpose |
|------|---------|
| `wan2_2_ti2v_dp.yaml` | Diffusion stage config; `runtime.num_replicas` + `runtime.devices` drive the fan-out. |
| `run_server.sh` | Substitutes `NUM_REPLICAS` / `DEVICES` into the config and serves. |
| `bench_replica_dp.py` | Replica-agnostic load driver; reports throughput, latency, and an md5 isolation check. |

## Run

Start the server with the replica count you want (one GPU per replica):

```bash
# baseline
NUM_REPLICAS=1 DEVICES=0        ./run_server.sh
# 2 replicas
NUM_REPLICAS=2 DEVICES=0,1      ./run_server.sh
# 4 replicas
NUM_REPLICAS=4 DEVICES=0,1,2,3  ./run_server.sh
```

In another shell, drive a fixed workload and read the throughput:

```bash
python bench_replica_dp.py --url http://127.0.0.1:8098 \
    --num-requests 28 --concurrency 8 --label "replicas=4"
```

Re-run the server at `NUM_REPLICAS=1,2,4` (same client workload each time) and
compare `videos/min`. Pass `--baseline-tpm <N=1 value>` to print a scaling factor.

### Isolation check (optional)

With a fixed `--seed`, outputs are deterministic, so a byte-identical result
across replica counts confirms the replicas do not cross-talk:

```bash
python bench_replica_dp.py --seed 42 --save-dir out_n1 ...   # against NUM_REPLICAS=1
python bench_replica_dp.py --seed 42 --save-dir out_n4 ...   # against NUM_REPLICAS=4
# md5 dedup should report "all identical" in each run, and the two runs should match
```

## Measured scaling

Wan2.2-TI2V-5B (832×480, 33 frames, 30 steps), 4× A800-80GB (NVLink), one GPU per replica:

| Replicas | Throughput (videos/min) | Scaling | Efficiency |
|----------|-------------------------|---------|------------|
| 1 | 4.71 | 1.00× | — |
| 2 | 9.20 | 1.95× | 98% |
| 4 | 18.02 | 3.83× | 96% |

Per-request latency stayed flat (~13–14 s) across all replica counts, and
seed-fixed outputs were byte-identical regardless of `N`.

## Notes on the config surface

- `runtime.num_replicas: N` fans out `N` replicas; `runtime.devices` assigns their
  GPUs. With `tensor_parallel_size = 1`, list one GPU per replica
  (`num_replicas * tensor_parallel_size` entries, pool mode).
- Replica fan-out here comes from the stage config, not a CLI flag. (The headless /
  multi-runtime launch path uses the process-local `--omni-dp-size-local`, which
  requires `--stage-id`.)
- This recipe serves via `--stage-configs-path`. Expressing replica-DP under the
  newer `--deploy-config` schema is part of the ongoing serving-config alignment
  (see the diffusion parallelism guide and the `#4707` discussion); `num_replicas`
  is accepted by both paths.
