# CloudVanguard

A multi-tenant LLM response-caching and cost-control gateway, its Kubernetes control plane and warm-cache storage, and
the fine-tuning/evaluation pipeline for the models behind it. The gateway sits in front of OpenAI-compatible endpoints
(vLLM, OpenAI, Anthropic via a compatible shim) and serves repeated completions, including streamed SSE responses,
from memory, with per-tenant rate limits and token budgets protecting the upstream.

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

| Path | What | Milestone |
|---|---|---|
| `proxy-cpp/` | `vanguard-proxy`: exact + semantic cache, rate limits, breaker/failover, multi-node ring forwarding, hot-reloadable control plane, crash-safe warm-cache snapshots, OpenTelemetry tracing | Phases 1–4, M1, M2 |
| `operator-go/` | Kubernetes operator (`CacheCluster`, `TenantCachePolicy`; `spec.tracing`, `TracingReady`) and the `csi.cloudvanguard.io` CSI driver on QEMU disk images | Phases 3–4, M2, M4 |
| `proto/` | Control-plane protocol `vanguard.control.v1` (config push, `ApplyResult` with tracing status) | Phase 3, M2 |
| `observability/` | OpenTelemetry Collector pipeline (+ in-cluster manifests), Prometheus scrape config + rules, verification scripts | M1, M2 |
| `training/` | PyTorch LoRA/QLoRA: memory-mapped Arrow data, int8/NF4 quantization, bf16/fp16, DDP, PEFT-format adapters, MLflow, eval harness | M3 |
| `deploy/` | Dockerfiles, CSI + StorageClass manifests, kustomize bundle, kind config, `env-check.sh`, `e2e-kind.sh` | Phase 4, M4 |
| `tests/` | Cross-subsystem integration tests and the index of every end-to-end check ([`tests/README.md`](tests/README.md)) | M1, M2, M4 |
| `benchmarks/` | Fault-injecting mock LLM, load generators, 3-node test, CSI/QEMU persistence + chaos, M1 and M3 benchmarks | Phases 1–4, M1, M3 |

### The four milestones

1. **Core proxy + telemetry** — OpenTelemetry tracing (W3C `traceparent`, parent-based sampling, OTLP/HTTP export),
   Prometheus metrics, monorepo layout. Checked against a real Collector 0.111 and promtool 2.54.
2. **Operator + CSI/QEMU** — `spec.tracing` rendered into pod `OTEL_*` env with churn-free reconciles (0 writes in
   steady state), `TracingReady` from pod reports, CSI/QMP failure-path hardening; fixed a remote crash on invalid UTF-8.
3. **Training** — LoRA/QLoRA fine-tuning, quantization, Arrow data loading, DDP, MLflow, evaluation harness; CPU-verified.
4. **End-to-end + deployment** — the three images build and run; the full stack runs on a real kind cluster (warm restart
   through CSI, tracing through the operator to the Collector); deploy bundle checked under its own RBAC; training eval
   measured through the gateway. Found and fixed three deployment bugs that only real builds/clusters expose
   ([`deploy/README.md`](deploy/README.md#verification)).

## Entry points

| Command | What |
|---|---|
| `vanguard-proxy` | the gateway (env/flags: `UPSTREAM_URL`, `LISTEN_PORT`, `OTEL_*`, ...; see `proxy-cpp/README.md`) |
| `operator-go/cmd` (`manager`) | the operator; `kubectl apply -k deploy/` installs it with the CRDs, CSI driver and Collector |
| `csi-driver --mode=controller\|node`, `csictl` | the CSI driver and a CSI gRPC client for driving it by hand |
| `python -m training {info,prepare-data,train,eval}` | the training/evaluation pipeline (`training/README.md`); `torchrun ... -m training train` for DDP |
| `make help` | every build, test, verification and benchmark target |

## Tests and verification

Linux is required for the epoll proxy (on macOS `make test` builds the portable core only); open the repo in a GitHub
Codespace — `.devcontainer/` installs every toolchain plus Docker-in-Docker, kubectl and kind.

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

### Verified vs not verified

Last full run: 2026-11-20, GitHub Codespace (4 vCPU AMD EPYC, Ubuntu 24.04, Docker 29.9, kind v0.24 with Kubernetes
1.31 nodes, no GPU, no KVM), at commit `7ee394d` plus the HostMounter probe fix.

| Area | Result | Verified how |
|---|---|---|
| Proxy unit + integration | **139/139** (`make test`), **139/139** under ASan+UBSan, 0 sanitizer reports | ctest: 135 unit, 3-node ring, tracing, robustness, eval-via-gateway |
| Operator + CSI | **39 Go tests pass, 0 fail**; csi-sanity **64/64** specs per volume mode; real QEMU (TCG) hot-plug; generated CRDs/RBAC/deepcopy match the committed files | `make test-go` (envtest = real kube-apiserver + etcd 1.32) |
| CSI node data path (loop devices, mount) | 2/2 as root | `sudo` test binary; skipped in `make test-go` (not root) |
| Training | **52 passed, 4 skipped** (skips need CUDA: bitsandbytes ×2, fp16, vLLM) | `make test-training` |
| Observability | Collector 0.111 received 4 spans, 0 export errors; promtool: config valid, 8 rules, live `/metrics` lints clean | `make verify-observability` |
| Manifests | `deploy/` 23 objects, `operator-go/config/default` 9, collector 4; all applied to kube-apiserver; operator runs under its shipped RBAC; shellcheck clean | `make verify-deploy`, `deploy_test.go` |
| Container images | all 3 build and run (proxy as uid 10001, healthz 200, exit 0 on SIGTERM) | `make images` |
| **Real cluster** | **PASS**: warm restart 50/50 hits through a re-attached CSI block volume, 0 upstream calls after restart, 50/50 upstream calls traced, `TracingReady=True`, 100 spans at the in-cluster Collector | `make e2e-kind` |
| **Not verified** | GPU paths (CUDA/bitsandbytes/fp16/NCCL/vLLM); `--attacher=qmp` inside a cluster (QMP is tested against a real VM outside Kubernetes); multi-node storage; VolumeSnapshot through the Kubernetes API; images pushed to a registry; production-scale load on a cluster | — |

## Run it by hand

```bash
python3 benchmarks/mock_upstream.py --port 9000 &
UPSTREAM_URL=http://127.0.0.1:9000 ./proxy-cpp/build/vanguard-proxy &
curl -s localhost:8080/v1/chat/completions -H 'Authorization: Bearer sk-demo' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}' -D - -o /dev/null | grep X-Cache
```

See [`proxy-cpp/README.md`](proxy-cpp/README.md) for the gateway's architecture, configuration and design trade-offs,
[`operator-go/README.md`](operator-go/README.md), [`training/README.md`](training/README.md) and
[`deploy/README.md`](deploy/README.md) for the other components.
