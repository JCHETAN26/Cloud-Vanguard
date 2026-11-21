# Milestone 5: GPU benchmark record (Qwen2.5-14B on 2× A100-SXM4-80GB)

Produced by `benchmarks/gpu/run_gpu_bench.sh` on RunPod Secure Cloud, torch 2.15.0+cu132, bitsandbytes NF4, NCCL, vLLM.
Task: AG News topic classification, 4000 train / 500 held-out rows, 250 steps, global batch 16, max length 256,
LoRA r=16 on all attention + MLP projections (68.8M trainable parameters, 336 modules).

| Directory | Pod | Stages |
|---|---|---|
| `pod1/` | `lq7ebrwaed7b16`, 2026-11-20 | setup, gpu_tests, smoke, data, download, base_eval, lora_bf16, qlora_nf4, ddp_lora_bf16 (all ok). `ddp_qlora_nf4.log` is a run interrupted when the pod was stopped; it has no result. |
| `pod2/` | `as6pg8vl2b2l8w`, 2026-11-21 | setup, data, download, lora_bf16 (rerun: vLLM needs its adapter), ddp_qlora_nf4, vllm_eval (all ok); `summary.md` / `summary.json` |

The pod-1 run was split across two pods because the first pod was stopped mid-run and its host GPUs were taken. The
`lora_bf16` rerun on pod 2 reproduced pod 1 exactly (peak allocated 37,411.7 MiB, loss 0.3452 → 0.1093, exact match
92.4%; throughput within 0.5%), so cross-pod comparisons of memory and accuracy are sound. Eval latency varies by host.

Per-run records: `runs/*/train_summary.json` (memory four ways, throughput, step times, loss, eval) and
`runs/*.json` (evaluations). Not kept here: adapters, the MLflow database and the Prometheus TSDB (reproducible).
Headline comparisons and caveats: [`STATUS.md`](../../../STATUS.md) §4.2.
