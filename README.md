# CloudVanguard

A multi-tenant LLM response-caching and cost-control gateway. It sits in front of
OpenAI-compatible endpoints (vLLM, OpenAI, Anthropic via a compatible shim) and
serves repeated completions, including streamed SSE responses, from memory.
Per-tenant rate limits and token budgets protect the upstream.

| Path | What | Status |
|---|---|---|
| `proxy-cpp/` | C++20 epoll data plane (`vanguard-proxy`): exact + semantic cache, rate limits, breaker/failover, multi-node ring forwarding, hot-reloadable control plane, crash-safe warm-cache snapshots, OpenTelemetry tracing | **Phases 1–4, Milestone 1** |
| `operator-go/` | Kubernetes operator (`CacheCluster`, `TenantCachePolicy`) and the `csi.cloudvanguard.io` CSI driver on QEMU disk images | **Phases 3–4** |
| `proto/` | Control-plane protocol (`vanguard.control.v1`) | **Phase 3** |
| `observability/` | OpenTelemetry Collector pipeline, Prometheus scrape config + rules, OTLP test receiver, real-tool verification scripts | **Milestone 1** |
| `tests/` | Cross-component integration tests (tracing across a 2-node ring) | **Milestone 1** |
| `benchmarks/` | Fault-injecting mock LLM, embedding sidecar, load generators, 3-node integration test, CSI/QEMU persistence + chaos suite, Milestone 1 latency/tracing-overhead benchmark | Phases 1–4, Milestone 1 |
| `deploy/` | Dockerfiles, CSI driver + StorageClass manifests, kustomize, kind config, env check and kind e2e scripts | **Phase 4** |
| `training/` | PEFT/LoRA + quantization pipeline | **Milestone 3: scaffold only, no code yet** |

### Roadmap

1. **Milestone 1** — core proxy + telemetry baseline: existing proxy, ring, snapshots and Prometheus metrics, plus OpenTelemetry tracing and the monorepo layout. *(this branch)*
2. **Milestone 2** — operator (`CacheCluster` reconciliation) and CSI/QEMU storage orchestration (built in Phases 3–4; tracing settings are not yet exposed through the CRD).
3. **Milestone 3** — Python/PyTorch distributed PEFT/LoRA + quantization, Arrow data loading, MLflow/vLLM evaluation.
4. **Milestone 4** — reproducible end-to-end benchmarks, Docker/cluster deployment checks, fault injection.

## Quick start (Linux / Codespaces)

The proxy uses Linux `epoll`. On macOS, `make test` builds and tests the
portable core only. Open the repo in a GitHub Codespace (`.devcontainer/`) for
the full build.

```bash
make help                      # all entry points
make test     # build + unit tests + integration tests: 3-node forwarding, tracing (ctest)
make test-asan                 # the same under ASan + UBSan
make test-go                   # operator + CSI driver tests
make bench    # Phase 1: 10k-request MISS/HIT benchmark
make bench-m1                  # Milestone 1: latency/throughput baseline + tracing overhead
make verify-observability      # real OpenTelemetry Collector + promtool checks
./benchmarks/bench_phase2.sh   # Phase 2: semantic cache, failover, neural sidecar
./benchmarks/bench_phase4.sh   # Phase 4: CSI/QEMU warm-cache persistence, chaos, load
```

```bash
# run it by hand
python3 benchmarks/mock_upstream.py --port 9000 &
UPSTREAM_URL=http://127.0.0.1:9000 ./proxy-cpp/build/vanguard-proxy &
curl -s localhost:8080/v1/chat/completions -H 'Authorization: Bearer sk-demo' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}' -D - -o /dev/null | grep X-Cache
```

See [`proxy-cpp/README.md`](proxy-cpp/README.md) for the architecture, configuration and design trade-offs.
