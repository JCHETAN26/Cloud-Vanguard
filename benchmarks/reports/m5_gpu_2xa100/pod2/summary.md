# GPU benchmark: run2

GPU: NVIDIA A100-SXM4-80GB. Task: AG News topic classification (4 labels), exact match on held-out test rows.

## Stages

| stage | result | seconds |
|---|---|---|
| setup | ok | 290 |
| data | ok | 7 |
| download | ok | 13 |
| lora_bf16 | ok | 433 |
| ddp_qlora_nf4 | ok | 482 |
| vllm_eval | ok | 192 |

## Training (same data, steps, global batch, sequence cap, gradient checkpointing)

| variant | GPUs | params (B) | trainable % | weights on GPU (GB) | peak allocated (GB) | peak nvidia-smi (GB) | tok/s | step p50 (ms) | train (min) | loss first→last | exact match | ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora_bf16 | 1 | 14.77 | 0.46 | 29.8 | 39.2 | - | 1099 | 1067 | 4.8 | 0.345→0.109 | 0.924 (500) | 1.125 |
| ddp_qlora_nf4 | 2 | 14.77 | 0.84 | 10.3 | 15.5 | 29.6 | 1698 | 691 | 3.1 | 0.488→0.108 | 0.920 (500) | 1.124 |

## Evaluation

- base model, no adapter (hf): -
- LoRA bf16 adapter via vLLM: exact match 0.924 on 500 rows, ppl 1.215, latency p50/p95/p99 79.7/132.3/132.7 ms

## GPU telemetry from Prometheus (gpu_exporter, 1 s scrapes), per stage

| stage | GPU | peak memory used (GB) | mean utilization % | samples |
|---|---|---|---|---|
| setup | - | - | | |
| data | 0 | 0.8 | 0.0 | 3 |
| data | 1 | 0.8 | 0.0 | 3 |
| download | 0 | 0.8 | 0.0 | 13 |
| download | 1 | 0.8 | 0.0 | 13 |
| lora_bf16 | 0 | 77.4 | 83.0 | 433 |
| lora_bf16 | 1 | 0.8 | 0.0 | 433 |
| ddp_qlora_nf4 | 0 | 77.4 | 57.8 | 482 |
| ddp_qlora_nf4 | 1 | 29.6 | 78.1 | 482 |
| vllm_eval | 0 | 72.2 | 40.7 | 192 |
| vllm_eval | 1 | 1.2 | 0.5 | 192 |
