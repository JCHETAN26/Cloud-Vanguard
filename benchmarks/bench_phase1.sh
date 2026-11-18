#!/usr/bin/env bash
# Phase 1 benchmark: build vanguard-proxy, start the mock upstream and the
# proxy, then run the 10k-request MISS vs HIT comparison plus the SSE replay,
# rate-limit, auth and invariant checks. Linux only (the proxy uses epoll).
#
# Env knobs: REQUESTS (10000), CONCURRENCY (64), HIT_CONCURRENCY (4),
# STREAM_REQUESTS (200), WORKERS (2), PROXY_PORT (8180), UPSTREAM_PORT (9100),
# BUILD_DIR (proxy-cpp/build-release).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/proxy-cpp/build-release}"
RESULTS="$ROOT/benchmarks/results"
PROXY_PORT="${PROXY_PORT:-8180}"
UPSTREAM_PORT="${UPSTREAM_PORT:-9100}"
WORKERS="${WORKERS:-2}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bench_phase1.sh needs Linux (vanguard-proxy is built on epoll). Use a Codespace / Linux VM." >&2
  exit 1
fi

mkdir -p "$RESULTS"
echo "==> building vanguard-proxy (Release)"
cmake -S "$ROOT/proxy-cpp" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVANGUARD_BUILD_TESTS=OFF >/dev/null
cmake --build "$BUILD_DIR" --target vanguard-proxy

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for() {  # url name
  for _ in $(seq 1 100); do
    if curl -fsS "$1" >/dev/null 2>&1; then return 0; fi
    sleep 0.1
  done
  echo "$2 did not become healthy at $1" >&2
  exit 1
}

echo "==> starting mock upstream on :$UPSTREAM_PORT"
python3 "$ROOT/benchmarks/mock_upstream.py" --port "$UPSTREAM_PORT" \
  --min-latency-ms 50 --max-latency-ms 200 --token-interval-ms 5 --tokens 24 \
  >"$RESULTS/mock_upstream.log" 2>&1 &
PIDS+=($!)
wait_for "http://127.0.0.1:$UPSTREAM_PORT/healthz" "mock upstream"

echo "==> starting vanguard-proxy on :$PROXY_PORT ($WORKERS workers)"
LISTEN_PORT="$PROXY_PORT" WORKERS="$WORKERS" \
UPSTREAM_URL="http://127.0.0.1:$UPSTREAM_PORT" \
TENANTS_FILE="$ROOT/benchmarks/tenants.bench.json" \
CACHE_MAX_MB=512 \
  "$BUILD_DIR/vanguard-proxy" >"$RESULTS/proxy.log" 2>&1 &
PIDS+=($!)
wait_for "http://127.0.0.1:$PROXY_PORT/healthz" "vanguard-proxy"

echo "==> running load generator"
set +e
python3 "$ROOT/benchmarks/loadgen.py" \
  --proxy "http://127.0.0.1:$PROXY_PORT" \
  --upstream "http://127.0.0.1:$UPSTREAM_PORT" \
  --requests "${REQUESTS:-10000}" \
  --concurrency "${CONCURRENCY:-64}" \
  --hit-concurrency "${HIT_CONCURRENCY:-4}" \
  --stream-requests "${STREAM_REQUESTS:-200}" \
  --out "$RESULTS/phase1.json" | tee "$RESULTS/phase1.txt"
status=${PIPESTATUS[0]}
set -e

# Optional: an independent, non-Python client for the hit path.
if command -v hey >/dev/null 2>&1; then
  echo
  echo "==> hey (Go) cache-hit throughput: 20000 identical requests, 32 connections"
  body='{"model":"mock-llm-1","messages":[{"role":"user","content":"hey benchmark prompt"}]}'
  hey -n 20000 -c 32 -m POST -T application/json -H "Authorization: Bearer sk-bench-0001" \
    -d "$body" "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" | tee "$RESULTS/hey_hit.txt" |
    sed -n '/Summary/,/Requests\/sec/p;/Latency distribution/,/99%/p;/Status code distribution/,$p'
fi

echo
echo "results written to $RESULTS/"
exit "$status"
