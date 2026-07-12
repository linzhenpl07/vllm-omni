# Parallelism Acceleration Guide

This guide covers the parallelism methods in vLLM-Omni for speeding up diffusion model inference and reducing per-device memory requirements.

## Supported Methods

| Method                                             | Description                                                                                                         |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **[Tensor Parallelism](tensor_parallel.md)**       | Shards DiT weights across GPUs to reduce per-GPU memory                                                             |
| **[Sequence Parallelism](sequence_parallel.md)**   | Splits sequence dimension across GPUs (Ulysses-SP, Ring-Attention, or hybrid) for high-resolution images and videos |
| **[CFG-Parallel](cfg_parallel.md)**                | Runs CFG positive/negative branches on separate GPUs for ~1.8x speedup on guided generation                         |
| **[Pipeline Parallelism](pipeline_parallel.md)**   | Splits the denoising transformer block-wise across sequential GPU stages to reduce per-GPU model memory             |
| **[VAE Parallelism](vae_parallelism.md)** | Distributes VAE decode spatially across GPUs to reduce peak VAE memory                                              |
| **[HSDP](hsdp.md)**                                | Shards full model weights via PyTorch FSDP2 to enable large-model inference on memory-constrained GPUs              |
| **[Expert Parallelism](expert_parallel.md)**       | Shards MoE expert blocks across GPUs for MoE models (e.g. HunyuanImage3.0)                                          |
| **[Replica Data Parallelism](replica_data_parallel.md)** | Runs N independent engine replicas and routes requests across them to scale request throughput near-linearly |

The methods above split a single request across GPUs to reduce its latency; Replica
Data Parallelism instead replicates the whole engine to scale throughput, and composes
with any of them.

See [Supported Models](../../diffusion_features.md#supported-models) for per-model compatibility.
