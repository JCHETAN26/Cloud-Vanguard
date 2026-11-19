# tests/

Cross-component tests that exercise more than one part of the monorepo. Unit tests stay next to their code
(`proxy-cpp/tests`, `operator-go/**/_test.go`).

| Path | What | How it runs |
|---|---|---|
| `integration/test_tracing_e2e.py` | OpenTelemetry trace propagation across a 2-node ring: caller → gateway → peer → LLM upstream, sampling, status mapping, exporter metrics, tracing-off | `ctest` (`tracing_e2e`, Linux), or by hand with `--binary` |

Related suites that predate this directory and stay where the build already references them:

| Path | What |
|---|---|
| `benchmarks/multinode_test.py` | 3-node ring routing, loop prevention, control plane (`ctest`: `multinode_forwarding`) |
| `benchmarks/phase4_csi_e2e.py`, `benchmarks/phase4_resilience.py` | CSI/QEMU persistence and chaos (`benchmarks/bench_phase4.sh`) |

Run everything with `make test-all` from the repo root (Linux for the epoll proxy; see the root README).
