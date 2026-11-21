# CloudVanguard — Project Status

_Last updated: 2026-11-21 06:40 UTC · branch `m5-gpu-validation` · maintained alongside the work; numbers here are
copied from real run outputs only (nothing estimated is reported as a result)._

## 1. Summary

| | |
|---|---|
| **Overall** | Phases 1–4 and Milestones 1–5 merged to `main`. M5 GPU validation complete on 2× A100-80GB; record committed in `benchmarks/reports/m5_gpu_2xa100/`; GPU pods terminated (0 running). |
| **Current step** | ✅ GPU run complete (pod 1 2026-11-20, pod 2 2026-11-21). Results: `benchmarks/results/gpu/run1-partial/` and `benchmarks/results/gpu/run2/` (`summary.md`, `summary.json`, Prometheus TSDB). No pods running. |
| **Blockers** | None. |
| **Next** | Interview prep. (Resume bullets are owned by the user — not edited here.) |

## 2. Plan and progress

| # | Milestone | Status | Branch / merge |
|---|---|---|---|
| P1 | C++20 epoll proxy, exact cache, consistent-hash ring | ✅ Done | merged to `main` |
| P2 | Semantic cache, circuit breaker + failover | ✅ Done | merged to `main` |
| P3 | Go operator (CRDs), control protocol, multi-node routing | ✅ Done | `e202031` |
| P4 | CSI driver on QEMU images, warm-cache snapshots, resilience suite | ✅ Done | `6014651` |
| M1 | OpenTelemetry tracing, observability, monorepo layout | ✅ Done | `d24ce55` |
| M2 | Operator tracing (`spec.tracing`, `TracingReady`), CSI/QMP hardening | ✅ Done | `488413a` |
| M3 | PyTorch LoRA/QLoRA, quantization, Arrow data, DDP, MLflow, eval harness (CPU) | ✅ Done | `074c1cc` |
| M4 | Images build, full stack on a real kind cluster, deploy bundle under RBAC | ✅ Done | `8040921` |
| **M5** | **GPU validation: real 14B model through LoRA/QLoRA/DDP, GPU telemetry, vLLM eval** | ✅ Done | `m5-gpu-validation` → `main` |

## 3. What is implemented

- **Gateway (`proxy-cpp/`)** — C++20 epoll proxy in front of OpenAI-compatible upstreams: exact + semantic cache
  (neural mode, threshold 0.96 + salient-token guard), streamed SSE caching, per-tenant rate limits and token budgets,
  circuit breaker + failover, consistent-hash ring (1000 vnodes), hot-reloadable control plane, crash-safe A/B
  warm-cache snapshots, OpenTelemetry tracing.
- **Control plane (`operator-go/`)** — Kubernetes operator (`CacheCluster`, `TenantCachePolicy`), churn-free
  reconciles, `TracingReady`; CSI driver `csi.cloudvanguard.io` on qemu-img images with loop/QMP attach, snapshots,
  expansion.
- **Observability (`observability/`)** — OTel Collector pipeline, Prometheus scrape config + rules,
  `gpu_exporter.py` (NVML → Prometheus) for GPU telemetry.
- **Training (`training/`)** — Arrow data pipeline, LoRA/QLoRA (own int8/NF4 + bitsandbytes), bf16/fp16, DDP,
  PEFT-format adapters, MLflow tracking, eval harness (torch / HF / vLLM / through the gateway);
  `training/hf` runs real Hugging Face models (Qwen2.5-14B) through the same pipeline.
- **Deployment (`deploy/`)** — Dockerfiles for 3 images, kustomize bundle, kind config, `e2e-kind.sh`.
- **GPU benchmark (`benchmarks/gpu/`)** — `run_gpu_bench.sh` (10 stages), `summarize.py`, Prometheus config.

## 4. Results

### 4.1 Verified on CPU / kind (Codespace, 4 vCPU, 2026-11-20)

| Area | Result |
|---|---|
| Proxy unit + integration | 139/139 pass; 139/139 under ASan+UBSan, 0 sanitizer reports |
| Operator + CSI | 39 Go tests pass; csi-sanity 64/64 per volume mode; real QEMU hot-plug |
| Training (CPU) | 52 passed, 4 skipped (CUDA-only) |
| Observability | Collector received 4 spans, 0 export errors; promtool: 8 rules valid |
| Real cluster (kind) | Warm restart 50/50 cache hits via re-attached CSI volume, 0 upstream calls after restart, `TracingReady=True`, 100 spans at Collector |

Details: [`README.md`](README.md#verified-vs-not-verified), `benchmarks/reports/`.

### 4.2 GPU run (M5) — RunPod, 2× NVIDIA A100-SXM4-80GB, torch 2.15.0+cu132

Model **Qwen/Qwen2.5-14B** (14.77B params). Task: AG News topic classification, 4000 train / 500 held-out rows,
250 steps, global batch 16, max length 256, LoRA r=16 on all attention + MLP projections.

| Stage | Status | Duration | Notes |
|---|---|---|---|
| setup | ✅ ok | 3587 s | venvs, vLLM, Prometheus |
| gpu_tests | ✅ ok | 851 s | `training/tests`: 59 passed, 2 skipped on CUDA (after test-expectation fixes, see §5) |
| smoke (Qwen2.5-0.5B, every GPU path) | ✅ ok | 1005 s | bnb NF4, NCCL DDP, HF + vLLM eval |
| data | ✅ ok | 55 s | |
| download | ✅ ok | 105 s | ~30 GB checkpoint |
| base_eval | ✅ ok | 141 s | |
| lora_bf16 | ✅ ok | 525 s | |
| qlora_nf4 | ✅ ok | 637 s | bitsandbytes NF4 backend |
| ddp_lora_bf16 | ✅ ok | 605 s | 2 GPUs, NCCL; per-rank batch 8 |
| ddp_qlora_nf4 | ✅ ok (pod 2) | 482 s | 2 GPUs, NCCL, bnb NF4 |
| vllm_eval | ✅ ok (pod 2) | 192 s | LoRA bf16 adapter via vLLM `LoRARequest` |

**Measured so far**

| Metric | Base (untuned) | LoRA bf16 (1× A100) | QLoRA NF4 (1× A100) | DDP LoRA bf16 (2× A100) | DDP QLoRA NF4 (2× A100) | vLLM |
|---|---|---|---|---|---|---|
| Exact match (500 rows) | 82.4% | **92.4%** | **92.0%** | 91.8% | 92.0% | **92.4%** |
| Perplexity | 1.494 | 1.125 | 1.118 | 1.125 | 1.124 | 1.215 † |
| Eval latency p50 / p95 | 60 / 112 ms | 98 / 187 ms | 148 / 271 ms | 106 / 202 ms | 212 / 390 ms | **80 / 132 ms** |
| Trainable params | — | 68.8M (0.46%), 336 modules | 68.8M, 336 modules | 68.8M, 336 modules | 68.8M, 336 modules | n/a |
| Weights on GPU | — | 28.4 GB | 9.8 GB | 28.4 GB per rank | 9.8 GB per rank | n/a |
| Peak allocated VRAM | — | 37.4 GB | **18.8 GB** | 33.4 GB per rank (batch 8/rank) | **14.8 GB per rank** | n/a |
| Peak reserved VRAM | — | 72.5 GB | 54.0 GB | 44.2 GB | 25.7 GB | n/a |
| Training throughput | — | 1,105 tokens/s | 1,020 tokens/s | **1,955 tokens/s** | 1,698 tokens/s | n/a |
| Training loop time (250 steps) | — | 290.6 s | 314.8 s | 163.8 s | 188.0 s | n/a |
| Step time p50 / p95 | — | 1.06 / 1.73 s | 1.16 / 1.82 s | 0.60 / 0.92 s | 0.69 / 1.02 s | n/a |
| Loss first → last-10 mean | — | 0.345 → 0.109 | 0.468 → 0.108 | 0.355 → 0.098 | 0.488 → 0.108 | n/a |

**QLoRA NF4 vs LoRA bf16 (same data, steps, batch, 1× A100):**
peak allocated VRAM **−49.8%** (37.4 → 18.8 GB); weights on GPU −65.5% (28.4 → 9.8 GB); peak reserved −25.6%
(72.5 → 54.0 GB); accuracy −0.4 pts (92.4% → 92.0%); training throughput −7.7%; eval latency p50 +51%.

**DDP LoRA bf16 on 2 GPUs vs 1 GPU (same global batch 16, 250 steps):** training throughput **1.77×**
(1,105 → 1,955 tokens/s, 88.5% scaling efficiency); training loop time **−43.6%** (290.6 → 163.8 s); accuracy
91.8% vs 92.4% (−0.6 pts). NVML: peak 46.8 GB used, mean GPU utilization 94.4% (rank 0, 645 samples).
Caveat: model loading took 210 s vs 36 s (two processes reading the checkpoint at once), so end-to-end stage time
was 605 s vs 525 s — the speedup is in the training loop, not the whole job.

**Reproducibility (pod 2, different host, 2026-11-21):** `lora_bf16` rerun gave identical peak allocated VRAM
(37,411.7 MB), identical loss curve (0.3452 → 0.1093, seeded), identical exact match (92.4%), throughput 1,099 vs
1,105 tokens/s (−0.5%). Eval latency p50 138 ms vs 98 ms on pod 1 — eval latency varies by host; treat it as noisy.
Pod 2 stage times: setup 290 s (vs 3,587 s on pod 1, venvs on local disk), data 7 s, download 13 s, lora_bf16 433 s.

**DDP QLoRA NF4 (2× A100, pod 2):** vs 1-GPU QLoRA: throughput **1.66×** (1,020 → 1,698 tokens/s), training loop
**−40.3%** (314.8 → 188.0 s); vs DDP LoRA bf16: peak allocated per rank **−55.7%** (33.4 → 14.8 GB), NVML peak used
−39.7% (46.8 → 28.2 GB), accuracy 92.0% vs 91.8%. Mean GPU utilization 94.8% (rank 0, 746 samples). Load time
88 s for two processes (pod 2's local disk; pod 1's DDP bf16 load was 210 s). Cross-pod caveat: DDP LoRA ran on
pod 1, DDP QLoRA on pod 2 — memory figures are deterministic (the lora_bf16 rerun matched to the MB), throughput
on the two pods agreed within 0.5%.

**vLLM vs HF generate (same LoRA bf16 adapter, same 500 rows, same pod 2):** identical exact match (92.4%);
latency p50 **−42.5%** (138.7 → 79.7 ms), p95 −50.0% (264.4 → 132.3 ms); generated tokens/s 1.84× (14.8 → 27.3).
† vLLM perplexity is scored over 770 tokens vs 1,270 for HF (different scoring path) — not comparable, don't quote it.

**GPU telemetry (Prometheus scraping `gpu_exporter`, 1 s, pod 2):** per-stage peak memory used / mean utilization on
GPU 0: lora_bf16 77.4 GB / 83.0%; ddp_qlora_nf4 GPU 1 29.6 GB / 78.1%; vllm_eval 72.2 GB / 40.7% (vLLM pre-allocates
its KV cache). Caveat: stage windows are wall-clock, so a stage's first seconds can include the previous stage's
memory (GPU 0 shows 77.4 GB in ddp_qlora_nf4 while that run's own NVML peak was 27.6 GB) — quote per-run NVML peaks
from `train_summary.json`, use Prometheus for the time series.

**Units:** this file's GB = GiB (MiB / 1024, from `train_summary.json`); `summary.md` divides bytes by 10⁹, so its
numbers are ~7% larger (37.4 GiB = 39.2 GB). Percentages are the same either way.

Observations: (1) eval with an adapter is slower than the base — likely because the adapter is evaluated unmerged
(and dequantized per layer for NF4); not yet investigated. (2) Peak *reserved* falls far less than peak *allocated*
(PyTorch caching allocator); quote "peak allocated" and say so. (3) Per-process NVML peak was not recorded in
`train_summary.json` (`nvml: {}`); the Prometheus GPU-exporter series will provide the `nvidia-smi`-level peak.

## 5. Issues found and fixed during M5

| Issue | Fix | Commit |
|---|---|---|
| Benchmark driver died after the first failing stage: setup's `set -e` leaked into the driver shell | Stages run in a subshell, failures are recorded and the run continues | `a9fbaae` |
| `test_lora_run_outputs_summary_adapter_and_mlflow` assumed fp32 (CPU default); GPU auto-precision is bf16 | Test expects the device's auto precision | `a9fbaae` |
| `test_quantize_model_freezes_and_skips_lm_head` expected `Int8Linear`; with bitsandbytes it is `Linear8bitLt` | Test accepts the backend's layer type | `a9fbaae` |
| `want()` used before definition when resetting `stages.tsv` | Moved after the definition | `be7b132` |

## 6. Resume claims vs evidence

| Claim | Status |
|---|---|
| PEFT/LoRA fine-tuning of a 10B+ model | ✅ Measured: Qwen2.5-14B LoRA, +10.0 pts exact match (82.4% → 92.4%) |
| ~45% VRAM reduction with QLoRA | ✅ Measured: **49.8% lower peak allocated VRAM** (37.4 → 18.8 GB) at −0.4 pts accuracy; reserved only −25.6% — word the claim as "peak allocated" |
| GPU telemetry (Prometheus) | ✅ Measured: NVML exporter scraped by Prometheus at 1 s through every stage; `promtool` validated config, rules and exporter output; per-stage memory/utilization in `summary.md` |
| Multi-GPU DDP training | ✅ Measured: 2× A100 DDP, 1.77× training throughput (88.5% scaling), −43.6% training-loop time; QLoRA DDP: 1.66× throughput, −40.3% training-loop time |
| vLLM evaluation | ✅ Measured: LoRA adapter served by vLLM, same 92.4% accuracy, p50 latency −42.5% vs HF generate |
| 55% / 40% | ✅ Now backed, **only with this exact wording**: 2-GPU QLoRA cut peak allocated memory per GPU **55.7%** vs 2-GPU LoRA bf16; 2-GPU DDP cut training-loop time **40.3%** (QLoRA) / 43.6% (LoRA) vs 1 GPU. Also available: vLLM −42.5% p50 latency. Don't present them as end-to-end job speedups |

## 7. Remaining work

| # | Item | Status |
|---|---|---|
| 1 | Commit GPU result records (`benchmarks/reports/m5_gpu_2xa100/`), `STATUS.md`, README GPU row | ✅ Done |
| 2 | Merge `m5-gpu-validation` → `main` | ✅ Done |
| 3 | Resume bullets | User-owned — Claude does not edit them; §6 lists what the measurements support |
| 4 | Optional: merge the LoRA adapter before HF eval to remove its latency overhead, re-measure | Open |
| 5 | Optional: fp16 mixed-precision training on a GPU (only bf16 was run) | Open |
| 6 | Interview prep from the measured results and design trade-offs | Open |

## 8. Cost

| Pod | GPUs | Rate | Ran | Approx. cost |
|---|---|---|---|---|
| `lq7ebrwaed7b16` (terminated) | 2× A100-SXM4-80GB | $4.38/hr | 2026-11-20 19:38–22:33 UTC | ≈ $12.80 + stopped-disk time |
| `as6pg8vl2b2l8w` (terminated) | 2× A100-SXM4-80GB | $4.38/hr | 2026-11-21 05:52–06:21 UTC | ≈ $2.10 |
| **Total** | | | | **≈ $15** (check RunPod billing for the exact figure) |

No pods are running (verified with `list-pods` after termination).
