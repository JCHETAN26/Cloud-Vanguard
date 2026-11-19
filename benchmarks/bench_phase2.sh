#!/usr/bin/env bash
# Phase 2 benchmark: semantic cache + circuit breaker / failover.
#
#   1. Phase 1 regression (exact cache, SSE replay, rate limits) on proxy A
#   2. Semantic cache with the in-process lexical encoder (proxy A)
#   3. Failover: primary 503s / slow (>300 ms TTFB) / dropped connections,
#      exponential backoff, recovery, both upstreams down (proxy A)
#   4. Semantic cache with a real neural encoder sidecar (proxy B,
#      bge-small via fastembed), then the sidecar killed (degraded mode)
#
# Linux only (epoll). Env knobs: REQUESTS (2000 semantic prompts),
# PHASE1_REQUESTS (10000), SKIP_NEURAL=1, SKIP_PHASE1=1, WORKERS (2).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/proxy-cpp/build-release}"
RESULTS="$ROOT/benchmarks/results"
B="$ROOT/benchmarks"
WORKERS="${WORKERS:-2}"
PRIMARY=9100 FALLBACK=9101 EMBEDDER=9200 PROXY_A=8180 PROXY_B=8181

[[ "$(uname -s)" == "Linux" ]] || { echo "needs Linux (epoll); use a Codespace" >&2; exit 1; }
mkdir -p "$RESULTS"

echo "==> building vanguard-proxy (Release)"
cmake -S "$ROOT/proxy-cpp" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVANGUARD_BUILD_TESTS=OFF >/dev/null || exit 1
cmake --build "$BUILD_DIR" --target vanguard-proxy >/dev/null || exit 1

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT
wait_for() {  # url name seconds
  for _ in $(seq 1 $((${3:-10} * 10))); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 0.1; done
  echo "$2 did not become healthy at $1" >&2; exit 1
}
FAIL=0
# run_tee <logfile> <cmd...>: run, tee output, remember failures (pipeline-safe).
run_tee() { local out=$1; shift; "$@" | tee "$out"; [[ ${PIPESTATUS[0]} -eq 0 ]] || FAIL=1; }

echo "==> mock upstreams: primary :$PRIMARY, fallback :$FALLBACK"
python3 "$B/mock_upstream.py" --name primary --port $PRIMARY >"$RESULTS/primary.log" 2>&1 & PIDS+=($!)
python3 "$B/mock_upstream.py" --name fallback --port $FALLBACK --seed 7 >"$RESULTS/fallback.log" 2>&1 & PIDS+=($!)
wait_for "http://127.0.0.1:$PRIMARY/healthz" primary
wait_for "http://127.0.0.1:$FALLBACK/healthz" fallback

common_env=(
  WORKERS="$WORKERS" TENANTS_FILE="$B/tenants.bench.json" CACHE_MAX_MB=512
  UPSTREAM_URL="http://127.0.0.1:$PRIMARY" FALLBACK_UPSTREAM_URL="http://127.0.0.1:$FALLBACK"
  UPSTREAM_TTFB_TIMEOUT_MS=300 BREAKER_FAILURE_THRESHOLD=5 BREAKER_OPEN_MS=1000 BREAKER_MAX_OPEN_MS=8000
)

echo "==> proxy A :$PROXY_A (lexical semantic cache, primary + fallback)"
env "${common_env[@]}" LISTEN_PORT=$PROXY_A SEMANTIC_MODE=lexical \
  "$BUILD_DIR/vanguard-proxy" >"$RESULTS/proxy_a.log" 2>&1 & PIDS+=($!)
wait_for "http://127.0.0.1:$PROXY_A/healthz" "proxy A"

if [[ "${SKIP_PHASE1:-0}" != 1 ]]; then
  echo; echo "==> [1/4] Phase 1 regression"
  run_tee "$RESULTS/phase2_regression.txt" python3 "$B/loadgen.py" --proxy "http://127.0.0.1:$PROXY_A" --upstream "http://127.0.0.1:$PRIMARY" \
    --requests "${PHASE1_REQUESTS:-10000}" --out "$RESULTS/phase2_regression.json"
fi

echo; echo "==> [2/4] semantic cache (lexical)"
run_tee "$RESULTS/phase2_semantic.txt" python3 "$B/loadgen_phase2.py" --section semantic --proxy "http://127.0.0.1:$PROXY_A" \
  --upstream "http://127.0.0.1:$PRIMARY" --requests "${REQUESTS:-2000}" --out "$RESULTS/phase2_semantic.json" \
 

echo; echo "==> [3/4] circuit breaker + failover"
run_tee "$RESULTS/phase2_failover.txt" python3 "$B/loadgen_phase2.py" --section failover --proxy "http://127.0.0.1:$PROXY_A" \
  --upstream "http://127.0.0.1:$PRIMARY" --fallback "http://127.0.0.1:$FALLBACK" --max-open-s 8 \
  --out "$RESULTS/phase2_failover.json"

if [[ "${SKIP_NEURAL:-0}" != 1 ]]; then
  echo; echo "==> [4/4] semantic cache (neural sidecar)"
  VENV="$B/.venv"
  if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import fastembed" 2>/dev/null; then
    echo "    installing fastembed into $VENV (one-time)"
    python3 -m venv "$VENV" && "$VENV/bin/pip" install -q fastembed >"$RESULTS/pip.log" 2>&1 \
      || { echo "    fastembed install failed (see $RESULTS/pip.log); skipping neural section"; SKIP_NEURAL=1; }
  fi
fi
if [[ "${SKIP_NEURAL:-0}" != 1 ]]; then
  "$VENV/bin/python" "$B/embedder_sidecar.py" --port $EMBEDDER >"$RESULTS/embedder.log" 2>&1 & EMB_PID=$!
  PIDS+=($EMB_PID)
  wait_for "http://127.0.0.1:$EMBEDDER/healthz" "embedder sidecar" 300
  head -1 "$RESULTS/embedder.log"
  env "${common_env[@]}" LISTEN_PORT=$PROXY_B SEMANTIC_MODE=neural EMBEDDER_URL="http://127.0.0.1:$EMBEDDER" \
    EMBEDDER_TIMEOUT_MS=200 "$BUILD_DIR/vanguard-proxy" >"$RESULTS/proxy_b.log" 2>&1 & PIDS+=($!)
  wait_for "http://127.0.0.1:$PROXY_B/healthz" "proxy B"
  run_tee "$RESULTS/phase2_neural.txt" python3 "$B/loadgen_phase2.py" --section neural --proxy "http://127.0.0.1:$PROXY_B" \
    --embedder "http://127.0.0.1:$EMBEDDER" --threshold 0.96 --out "$RESULTS/phase2_neural.json"
  kill "$EMB_PID" 2>/dev/null; wait "$EMB_PID" 2>/dev/null
  run_tee "$RESULTS/phase2_degraded.txt" python3 "$B/loadgen_phase2.py" --section degraded --proxy "http://127.0.0.1:$PROXY_B" \
    --out "$RESULTS/phase2_degraded.json"
fi

echo; echo "results written to $RESULTS/"
[[ $FAIL -eq 0 ]] && echo "ALL SECTIONS PASSED" || echo "SOME CHECKS FAILED"
exit $FAIL
