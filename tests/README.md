# tests/ — cross-subsystem and end-to-end checks

Unit tests live next to their code (`proxy-cpp/tests`, `operator-go/**/*_test.go`, `training/tests`).
This page indexes every check that crosses a subsystem boundary, what it proves, and how to run it.
Linux is required for anything that starts `vanguard-proxy` (epoll); the devcontainer (`.devcontainer/`)
provides every tool listed, including Docker and kind.

## In this directory (run by `ctest`, i.e. `make test` / `make test-asan`)

| Test | Subsystems | What it proves |
|---|---|---|
| `integration/test_tracing_e2e.py` | proxy x2 (ring) + mock LLM + OTLP receiver | W3C trace context flows caller → gateway → peer → upstream; sampling; span status; exporter metrics; tracing off |
| `integration/test_http_robustness.py` | proxy data + admin ports | 38 hostile requests (invalid UTF-8, wrongly typed JSON, malformed control documents) never take the process down |
| `integration/test_coalescing.py` | proxy x2 (4 workers each) + capacity-limited mock LLM | A 100-client cold burst makes 1 upstream call with coalescing (100 without); an upstream 500 is shared, not retried per waiter; a leader whose client resets hands the call to a waiter; streams are not coalesced |
| `integration/test_eval_via_gateway.py` | `training.eval` + proxy + mock LLM | The evaluation harness measured through the gateway: cold pass reaches the upstream, warm pass is served from cache with zero upstream calls and byte-identical answers; `/metrics` agrees |

`benchmarks/multinode_test.py` (3-node ring routing, loop prevention, control plane) is registered in the same
`ctest` run as `multinode_forwarding`.

## Elsewhere in the monorepo

| Check | Subsystems | Run | Needs |
|---|---|---|---|
| Cross-language goldens | proxy-cpp ↔ operator-go ↔ proto | `make test` + `make test-go` | — |
| ↳ the operator's rendered pod env and control document load in the proxy's config loader; the proxy's `ApplyResult` parses strictly as `vanguard.control.v1` in Go | | | |
| `operator-go/internal/controller/deploy_test.go` | `deploy/` bundle + operator RBAC + kube-apiserver | `make test-go` | envtest binaries (`make -C operator-go envtest`) |
| ↳ `kubectl kustomize deploy/` server-side-applied to a real API server; both reconcilers run on the published sample *as the operator's ServiceAccount*, so a missing permission fails as Forbidden | | | |
| csi-sanity (64 specs, block + mount) | CSI driver gRPC + `qemu-img` | `make test-go` | `qemu-img` |
| QMP hot-plug into a real VM | CSI attacher + QEMU | `make test-go` | `qemu-system-x86_64` (TCG ok) |
| `internal/csi/hostmounter_test.go` | CSI node data path + loop devices + mount | `sudo` test binary (see the file) | root, loop devices |
| `make verify-observability` | proxy + real OpenTelemetry Collector + promtool | `make verify-observability` | network (downloads pinned binaries) |
| `benchmarks/bench_phase4.sh` | CSI gRPC + `qemu-img` + proxy block snapshots + chaos | `make bench-phase4` | `qemu-img` |
| `training/tests/test_cli_e2e.py` | training CLI: Arrow data → train → adapter → eval, torchrun DDP, MLflow | `make test-training PYTHON=...` | torch, pyarrow, mlflow |
| `make verify-deploy` | every kustomization + all shell scripts | `make verify-deploy` | kubectl (shellcheck optional) |
| **`deploy/scripts/e2e-kind.sh`** | **everything on a real cluster** | `make e2e-kind` | docker, kind, kubectl |
| ↳ builds the 3 images, creates kind, installs operator + CSI driver + snapshot controller + OTel Collector, deploys a persistent traced CacheCluster, serves 50 prompts, deletes the pod, and requires 50/50 cache hits from the replacement pod (restored from its re-attached CSI block volume), zero upstream calls after the restart, a traceparent on every upstream call, `TracingReady=True` via the operator's push to the pod, and spans at the in-cluster Collector | | | |

## Running everything

```bash
make test-all                         # ctest (unit + integration), ASan/UBSan, Go (envtest, csi-sanity, QEMU), training
make verify-observability verify-deploy
make e2e-kind                         # needs docker + kind (the devcontainer has both)
```
