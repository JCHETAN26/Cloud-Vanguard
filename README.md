<div align="center">

# CloudVanguard

**A self-hosted LLM gateway that answers repeated prompts from cache, so the model never runs twice for the same question.**
<br>Multi-tenant · Kubernetes-native · cache survives restarts · with a fine-tuning pipeline for the models behind it

[![CI](https://github.com/JCHETAN26/Cloud-Vanguard/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JCHETAN26/Cloud-Vanguard/actions/workflows/ci.yml)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![Go](https://img.shields.io/badge/Go-operator%20%2B%20CSI-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-PyTorch-3776AB?logo=python&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-kind%20verified-326CE5?logo=kubernetes&logoColor=white)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-traces-425CC7?logo=opentelemetry&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-2%C3%97%20A100%20verified-76B900?logo=nvidia&logoColor=white)

[The problem](#the-problem) · [How it works](#how-it-works) · [Results](#results) · [Architecture](#architecture) ·
[Quick start](#quick-start) · [Verification](#verification) · [Limitations](#known-limitations)

</div>

---

## What is CloudVanguard?

CloudVanguard sits between your applications and your LLM servers (vLLM, OpenAI or any OpenAI-compatible API).
Applications keep sending ordinary OpenAI-format requests; they only change the URL. The gateway:

- **answers repeated and near-identical prompts from memory**, in under a millisecond instead of a full model call;
- **keeps each customer (tenant) within its rate limit and token budget**, so one tenant can't starve the others;
- **fails over when a model server is down**, without any retry code in the applications;
- **keeps its cache across restarts and deploys**, so a rollout doesn't send a burst of traffic to the GPUs.

It ships as four parts that work together:

| Component | Language | What it does |
|---|---|---|
| [`proxy-cpp/`](proxy-cpp/) **Gateway** | C++20 | The request path: caching, rate limits, budgets, failover, multi-node routing, snapshots, tracing |
| [`operator-go/`](operator-go/) **Control plane** | Go | Kubernetes operator that deploys and configures the gateways, plus a CSI storage driver for the cache disks |
| [`observability/`](observability/) **Telemetry** | YAML / Python | OpenTelemetry Collector pipeline, Prometheus rules, NVML GPU exporter |
| [`training/`](training/) **Model pipeline** | Python | LoRA / QLoRA fine-tuning on one or more GPUs (DDP), evaluation (PyTorch, Hugging Face, vLLM, or through the gateway), MLflow tracking |

---

## The problem

Teams running LLMs in production hit the same three problems:

| Problem | Why it hurts |
|---|---|
| 💸 **Repeated work** | Many users ask the same thing in slightly different words ("How do I reset my password?"). Every repeat is a full GPU call or a paid API call. |
| 🚦 **Noisy neighbours** | On a shared platform, one tenant's traffic spike can exhaust the GPUs or the budget for everyone. |
| 🔥 **Fragile upstreams** | Model servers fail, stall and get redeployed. Every application ends up writing its own retries and fallbacks. |

**Example:** a support-chatbot platform serving 50 client companies. Thousands of end users ask the same handful of
questions. Without a cache, each one costs a model call; with CloudVanguard, the first answer is reused for the rest,
each client stays inside its quota, and the bots keep answering when one model backend goes down.

---

## How it works

```
 app ──► Gateway ──► exact-match cache hit?  ────────── yes ──► answer from memory (X-Cache: HIT)
            │        similar-prompt cache hit + guard?  yes ──► answer from memory (X-Cache: SEMANTIC-HIT)
            │ miss
            ├─ tenant within rate limit and token budget?   no ──► 429 Too Many Requests
            ├─ primary upstream healthy? else fail over to the fallback (circuit breaker)
            └─► vLLM / OpenAI ──► answer ──► stored in cache ──► app (X-Cache: MISS)
```

1. **Identify the tenant** from the `Authorization: Bearer` API key. Tenants never share cache entries.
2. **Exact cache.** The key is a SHA-256 of tenant, model, messages and generation parameters (temperature,
   `max_tokens`, tools…). A hit is copied straight into the socket; streamed (SSE) answers are replayed too.
3. **Semantic cache, with a safety guard.** The final user message is embedded and compared with earlier prompts in
   the same scope. A match needs high similarity (0.96 with a neural embedder) **and** must pass a salient-token guard:
   numbers, negations and key words have to match. So *"refund order 123"* never gets *"refund order 456"*'s answer,
   and *"is X safe"* never matches *"is X not safe"*.
4. **Limits before spend.** Requests-per-minute is charged on every request; tokens-per-minute only on misses, since
   cache hits cost the upstream nothing.
5. **Failover.** A per-upstream circuit breaker opens after repeated failures, and a failed or stalled request is
   retried once on the fallback before any byte reaches the client.
6. **Scale out.** N gateways form one logical cache: a 1,000-virtual-node consistent-hash ring routes each prompt to
   its owner node, so entries aren't duplicated and any node can take any request.
7. **Survive restarts.** Each node snapshots its cache to a persistent disk (crash-safe, checksummed, two alternating
   slots). After a restart the disk is re-attached and the cache is restored before the first request is served.

The model side: [`training/`](training/) fine-tunes a model for the domain (LoRA or the memory-saving QLoRA, on one or
more GPUs), evaluates it, and vLLM serves the result behind the gateway.

---

## Results

All numbers are measured, not estimated. Raw records live in [`benchmarks/reports/`](benchmarks/reports/).

### Gateway

GitHub Codespace, 4 vCPU (AMD EPYC), 2 gateway workers, mock upstream answering in 20–40 ms
([`milestone1_codespace_4vcpu.json`](benchmarks/reports/milestone1_codespace_4vcpu.json)).

| Metric | Result |
|---|---|
| Latency, cache miss (goes to the model) | p50 **31.2 ms** |
| Latency, cache hit | p50 **0.45 ms**, p99 1.0 ms |
| Throughput, cache hits (`hey`, 32 connections) | **49,397 req/s**, p99 2.8 ms |
| Throughput with 100% trace sampling | 36,417 req/s (10% sampling: 48,721 req/s) |
| Semantic-cache safety (hand-written paraphrase set) | **0 false hits**; 3 of 12 paraphrases hit (safety chosen over recall) |

### Reliability on a real Kubernetes cluster (kind)

| Check | Result |
|---|---|
| Warm restart through the CSI volume | **50/50 cache hits** after the pod restarted, **0** upstream calls |
| Tracing end to end (gateway → Collector) | 50/50 upstream calls traced, 100 spans received, `TracingReady=True` |

### Fine-tuning on GPUs (Qwen2.5-14B, 2× NVIDIA A100-80GB)

Topic classification (AG News), 4,000 training rows, 500 held-out rows, 250 steps, LoRA rank 16
([`benchmarks/reports/m5_gpu_2xa100/`](benchmarks/reports/m5_gpu_2xa100/)).

| Result | Value |
|---|---|
| Accuracy after LoRA fine-tuning | **82.4% → 92.4%** exact match (+10.0 points) |
| QLoRA (4-bit NF4) vs LoRA (bf16) | **−49.8%** peak allocated GPU memory (37.4 → 18.8 GB) at 92.0% accuracy |
| 2-GPU DDP vs 1 GPU (LoRA) | **1.77×** training throughput (88.5% scaling), −43.6% training-loop time |
| 2-GPU QLoRA vs 2-GPU LoRA | **−55.7%** peak allocated memory per GPU (33.4 → 14.8 GB) |
| vLLM vs Hugging Face `generate` (same adapter) | p50 latency **−42.5%** (138.7 → 79.7 ms), same 92.4% accuracy |
| Reproducibility | LoRA rerun on a second host: identical memory, loss curve and accuracy |

Memory is peak *allocated* by PyTorch, in GiB. Speedups are for the training loop, not the whole job (two processes
loading a 30 GB checkpoint add start-up time). Full tables and caveats: [`STATUS.md`](STATUS.md) §4.2.

---

## Architecture

```
                     tenants (OpenAI-compatible HTTP, Bearer key)
                                   │
         ┌─────────────────────────▼─────────────────────────┐      OTLP/HTTP      ┌───────────────────┐
         │ vanguard-proxy ×N  (proxy-cpp, C++20 epoll)        │ ──────────────────► │ OTel Collector    │
         │  exact + semantic cache · rate limits · budgets    │      /metrics       │ Prometheus rules  │
         │  breaker + failover · consistent-hash ring         │ ──────────────────► │ (observability/)  │
         │  A/B block snapshots of the cache                  │                     └───────────────────┘
         └──────┬──────────────────▲──────────────┬───────────┘
   misses only  │     admin API    │              │ /dev/vanguard-cache (raw block PVC)
                ▼     (control.v1) │              ▼
      vLLM / OpenAI upstreams   ┌──┴─────────────────────┐   ┌────────────────────────────────────┐
      (and the eval harness     │ operator (operator-go) │   │ csi.cloudvanguard.io (operator-go) │
       measures through the     │ CacheCluster →         │   │ qemu-img images · loop/QMP attach  │
       gateway: training/)      │ StatefulSet, config    │   │ snapshots · expansion              │
                                │ push, TracingReady     │   └────────────────────────────────────┘
                                └────────────────────────┘
      training/: Arrow data → LoRA/QLoRA (PyTorch, DDP) → PEFT adapters → eval (torch / vLLM / via gateway) → MLflow
```

### Components

<details open>
<summary><b>Gateway — <code>proxy-cpp/</code></b></summary>

- One edge-triggered `epoll` event loop per core (`SO_REUSEPORT`), no framework; the hot path takes only cache-shard
  and limiter-shard locks.
- Sharded, byte-bounded, TTL-aware LRU cache; streamed responses cached already chunk-framed and only after a complete
  `[DONE]`.
- Semantic index with a lexical encoder (~20 µs, in-process) or a neural one (any `/v1/embeddings` service, async,
  with its own breaker: if it's down, the gateway falls back to exact matching).
- Hot-reloadable control plane: ring and tenant-policy updates are versioned and swapped atomically.
- Response headers explain every decision: `X-Cache`, `X-Vanguard-Similarity`, `X-Vanguard-Upstream`,
  `X-Vanguard-Failover`, `X-Vanguard-Owner`.

Details: [`proxy-cpp/README.md`](proxy-cpp/README.md)
</details>

<details>
<summary><b>Control plane — <code>operator-go/</code></b></summary>

- CRDs `CacheCluster` and `TenantCachePolicy`: describe a cache cluster in YAML, the operator renders a StatefulSet
  and pushes configuration to running gateways over the `vanguard.control.v1` protocol ([`proto/`](proto/)).
- Reconciles are churn-free (0 writes in steady state); `spec.tracing` becomes pod `OTEL_*` settings and a
  `TracingReady` condition.
- CSI driver `csi.cloudvanguard.io`: volumes are QEMU disk images attached through loop devices or QMP hot-plug, with
  snapshots and expansion.

Details: [`operator-go/README.md`](operator-go/README.md)
</details>

<details>
<summary><b>Observability — <code>observability/</code></b></summary>

- OpenTelemetry traces over OTLP/HTTP with W3C `traceparent`: a request forwarded to its owner node is one trace.
- Prometheus metrics for cache, rate limits, upstreams and latency; alert rules checked with `promtool`.
- `gpu_exporter.py`: NVML → Prometheus GPU memory and utilization.

Details: [`observability/README.md`](observability/README.md)
</details>

<details>
<summary><b>Model pipeline — <code>training/</code></b></summary>

- Memory-mapped Apache Arrow datasets; LoRA and QLoRA (own int8/NF4 quantization, or bitsandbytes); bf16/fp16.
- Multi-GPU training with PyTorch DDP over NCCL (`torchrun`); PEFT-format adapters.
- Evaluation with PyTorch, Hugging Face, vLLM, or through the gateway; experiments tracked in MLflow.
- `training/hf` runs real Hugging Face models (Qwen2.5-14B) through the same pipeline.

Details: [`training/README.md`](training/README.md)
</details>

### Repository layout

| Path | Contents |
|---|---|
| `proxy-cpp/` | Gateway source, unit tests (GoogleTest) |
| `operator-go/` | Operator, CSI driver, `csictl` client, envtest suites |
| `proto/` | Control-plane protocol `vanguard.control.v1` |
| `observability/` | Collector pipeline and manifests, Prometheus config and rules, GPU exporter |
| `training/` | Fine-tuning and evaluation pipeline |
| `deploy/` | Dockerfiles, CSI and StorageClass manifests, kustomize bundle, kind config, `e2e-kind.sh` |
| `tests/` | Cross-component integration tests and the index of every end-to-end check ([`tests/README.md`](tests/README.md)) |
| `benchmarks/` | Mock LLM with fault injection, load generators, multi-node, CSI chaos and GPU benchmarks, reports |

---

## Quick start

> **Requirements:** Linux (the gateway uses `epoll`; on macOS `make test` builds only the portable core).
> The easiest setup is a **GitHub Codespace**: [`.devcontainer/`](.devcontainer/) installs every toolchain plus
> Docker-in-Docker, kubectl and kind.

**1. Run the gateway against a mock LLM**

```bash
make build                         # CMake + Ninja → proxy-cpp/build/vanguard-proxy
python3 benchmarks/mock_upstream.py --port 9000 &
UPSTREAM_URL=http://127.0.0.1:9000 ./proxy-cpp/build/vanguard-proxy &

# send the same request twice: the first is a MISS, the second a HIT
for i in 1 2; do
  curl -s localhost:8080/v1/chat/completions -H 'Authorization: Bearer sk-demo' \
    -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}' -D - -o /dev/null | grep X-Cache
done
```

**2. Deploy the full stack on a local Kubernetes cluster**

```bash
make images && make e2e-kind        # build the 3 images, run gateway + operator + CSI + Collector on kind
```

**3. Fine-tune and evaluate a model** (CPU-sized example; no network needed)

```bash
python -m training info                                            # what this machine supports
python -m training prepare-data --out runs/data --task reverse
python -m training train --config training/configs/pretrain_tiny.json \
    --train-data runs/data/pretrain.arrow --eval-data runs/data/pretrain_eval.arrow --output-dir runs/base
python -m training train --config training/configs/lora_tiny.json --base-model runs/base/model \
    --train-data runs/data/reverse_train.arrow --eval-data runs/data/reverse_eval.arrow --output-dir runs/lora
python -m training eval --base-model runs/base/model --adapter runs/lora/adapter \
    --data runs/data/reverse_eval.arrow --output runs/lora/eval.json
```

Multi-GPU: `torchrun --nproc_per_node=2 -m training train ...`. The 14B GPU benchmark: [`benchmarks/gpu/`](benchmarks/gpu/).

### Entry points

| Command | What |
|---|---|
| `vanguard-proxy` | The gateway. Configured by env vars or flags (`UPSTREAM_URL`, `LISTEN_PORT`, `OTEL_*`…); see [`proxy-cpp/README.md`](proxy-cpp/README.md#configuration) |
| `operator-go/cmd` (`manager`) | The operator; `kubectl apply -k deploy/` installs it with the CRDs, CSI driver and Collector |
| `csi-driver --mode=controller\|node`, `csictl` | The CSI driver, and a CSI gRPC client for driving it by hand |
| `python -m training {info,prepare-data,train,eval}` | The training and evaluation pipeline |
| `make help` | Every build, test, verification and benchmark target |

---

## Verification

```bash
make test                # C++ unit tests + integration (3-node ring, tracing, robustness, eval via gateway) — ctest
make test-asan           # the same under AddressSanitizer + UBSan
make test-go             # operator + CSI: envtest (incl. deploy/ bundle under its RBAC), csi-sanity, QEMU hot-plug
make test-training PYTHON=.venv/bin/python   # training/ pytest (torch, pyarrow, mlflow)
make verify-observability                    # real OpenTelemetry Collector + promtool
make verify-deploy                           # render every kustomization, bash -n + shellcheck all scripts
make images && make e2e-kind                 # build the 3 images; full stack on a kind cluster
make bench | bench-m1 | bench-m3 | bench-phase2 | bench-phase4
```

**Continuous integration** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `make test`, `make test-asan`,
`make test-go`, `make test-training` (CPU) and `make verify-deploy` on every push to `main` and every pull request.
The GPU benchmark and `make e2e-kind` are run by hand (no GPU on hosted runners; kind is slow there).

Last full run: 2026-11-20, GitHub Codespace (4 vCPU AMD EPYC, Ubuntu 24.04, Docker 29.9, kind v0.24 with Kubernetes
1.31 nodes, no GPU, no KVM), at commit `7ee394d` plus the HostMounter probe fix. GPU run: 2026-11-20/21 on RunPod.

| Area | Status | Result | Verified how |
|---|---|---|---|
| Gateway unit + integration | ✅ | **139/139** (`make test`), **139/139** under ASan+UBSan, 0 sanitizer reports | ctest: 135 unit, 3-node ring, tracing, robustness, eval-via-gateway |
| Operator + CSI | ✅ | **39 Go tests pass**; csi-sanity **64/64** specs per volume mode; real QEMU (TCG) hot-plug; generated CRDs/RBAC/deepcopy match the committed files | `make test-go` (envtest = real kube-apiserver + etcd 1.32) |
| CSI node data path | ✅ | loop devices and mount, 2/2 as root | `sudo` test binary (skipped in `make test-go`) |
| Training (CPU) | ✅ | **52 passed, 4 skipped** (skips need CUDA) | `make test-training` |
| Training (GPU) | ✅ | **59 passed, 2 skipped** on 2× A100; full Qwen2.5-14B benchmark (see [Results](#results)) | `benchmarks/gpu/run_gpu_bench.sh`; record in [`m5_gpu_2xa100/`](benchmarks/reports/m5_gpu_2xa100/) |
| Observability | ✅ | Collector 0.111 received 4 spans, 0 export errors; promtool: config valid, 8 rules, live `/metrics` lints clean | `make verify-observability` |
| Manifests | ✅ | `deploy/` 23 objects, operator 9, Collector 4, all applied to kube-apiserver; operator runs under its shipped RBAC; shellcheck clean | `make verify-deploy`, `deploy_test.go` |
| Container images | ✅ | all 3 build and run (gateway as uid 10001, healthz 200, clean exit on SIGTERM) | `make images` |
| Real cluster (kind) | ✅ | warm restart 50/50 hits, 0 upstream calls after restart, `TracingReady=True`, 100 spans at the Collector | `make e2e-kind` |

---

## Known limitations

Stated plainly so nothing above is read as more than it is.

**Not yet verified**
- fp16 mixed-precision training on a GPU (the A100 runs use bf16; tests only check the gradient-scaler setup)
- `--attacher=qmp` inside a cluster (QMP is tested against a real VM outside Kubernetes)
- Storage across multiple Kubernetes nodes, and VolumeSnapshot through the Kubernetes API
- Images pushed to a registry, and production-scale load on a cluster

**Not built yet** (from [`proxy-cpp/README.md`](proxy-cpp/README.md#design-decisions-and-trade-offs))
- TLS to upstreams (plain HTTP only) and an IPv6 listener
- Request coalescing for concurrent identical misses (cache stampede)
- An approximate vector index (HNSW) for very large semantic scopes; the exact scan costs ~1 ms at the per-scope cap
- Reconciling the estimated token count with the upstream's reported `usage`

---

## Project status

All planned work is complete: Phases 1–4 and Milestones 1–5 are merged to `main`. Progress, every measured number,
issues found along the way and remaining items are tracked in [`STATUS.md`](STATUS.md).
