# CloudVanguard

A multi-tenant LLM response-caching and cost-control gateway. It sits in front of
OpenAI-compatible endpoints (vLLM, OpenAI, Anthropic via a compatible shim) and
serves repeated completions, including streamed SSE responses, from memory.
Per-tenant rate limits and token budgets protect the upstream.

| Path | What | Status |
|---|---|---|
| `proxy-cpp/` | C++20 epoll data plane (`vanguard-proxy`): exact + semantic cache, rate limits, breaker/failover | **Phases 1–2 done** |
| `benchmarks/` | Fault-injecting mock LLM, embedding sidecar, load generators, bench scripts | Phases 1–2 done |
| `operator-go/` | Kubernetes operator (`TenantCachePolicy` CRD) | Phase 4 |
| `proto/` | gRPC definitions (peer cache protocol, CSI) | Phase 3/6 |
| `deploy/` | kind config, manifests | Phase 4 |

## Quick start (Linux / Codespaces)

The proxy uses Linux `epoll`. On macOS, `make test` builds and tests the
portable core only. Open the repo in a GitHub Codespace (`.devcontainer/`) for
the full build.

```bash
make test     # build + all unit tests (ctest)
make bench    # Phase 1: 10k-request MISS/HIT benchmark
./benchmarks/bench_phase2.sh   # Phase 2: semantic cache, failover, neural sidecar
```

```bash
# run it by hand
python3 benchmarks/mock_upstream.py --port 9000 &
UPSTREAM_URL=http://127.0.0.1:9000 ./proxy-cpp/build/vanguard-proxy &
curl -s localhost:8080/v1/chat/completions -H 'Authorization: Bearer sk-demo' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}' -D - -o /dev/null | grep X-Cache
```

See [`proxy-cpp/README.md`](proxy-cpp/README.md) for the architecture, configuration and design trade-offs.
