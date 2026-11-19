#!/usr/bin/env bash
# Milestone 1 benchmark: Release build of vanguard-proxy, then
# bench_milestone1.py (miss/hit latency + throughput with tracing off, on at
# 100% and on at 10% sampling, and off again). Linux only (epoll).
#
# Env knobs: UNIQUE (2000), HITS (20000), WORKERS (2), HEY_SECONDS (10),
# BUILD_DIR (proxy-cpp/build-release).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/proxy-cpp/build-release}"
[[ "$(uname -s)" == "Linux" ]] || { echo "needs Linux (epoll); use a Codespace" >&2; exit 1; }

echo "==> building vanguard-proxy (Release)"
cmake -S "$ROOT/proxy-cpp" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVANGUARD_BUILD_TESTS=OFF >/dev/null
cmake --build "$BUILD_DIR" --target vanguard-proxy >/dev/null
command -v hey >/dev/null || [[ -x "$HOME/go/bin/hey" ]] || echo "note: 'hey' not found; max-throughput column skipped (go install github.com/rakyll/hey@latest)"

python3 "$ROOT/benchmarks/bench_milestone1.py" --binary "$BUILD_DIR/vanguard-proxy" \
  --unique "${UNIQUE:-2000}" --hits "${HITS:-20000}" --workers "${WORKERS:-2}" \
  --hey-seconds "${HEY_SECONDS:-10}" --out "$ROOT/benchmarks/results/milestone1_bench.json"
