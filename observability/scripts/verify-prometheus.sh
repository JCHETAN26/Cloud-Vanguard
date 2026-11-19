#!/usr/bin/env bash
# Validates the Prometheus side with the real promtool:
#   promtool check config  observability/prometheus/prometheus.yml (+ rules.yml)
#   promtool check rules   observability/prometheus/rules.yml
#   promtool check metrics on a live vanguard-proxy /metrics scrape
# Usage: verify-prometheus.sh <path/to/vanguard-proxy>     (Linux)
set -euo pipefail
BINARY="${1:?usage: verify-prometheus.sh <vanguard-proxy binary>}"
OBS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$OBS")"
PROMTOOL="$(command -v promtool || true)"
if [[ -z "$PROMTOOL" ]]; then "$OBS/scripts/fetch-tools.sh" >/dev/null; PROMTOOL="$OBS/.bin/promtool"; fi
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
FAIL=0
echo "== $("$PROMTOOL" --version | head -1)"
"$PROMTOOL" check config "$OBS/prometheus/prometheus.yml" || FAIL=1
"$PROMTOOL" check rules "$OBS/prometheus/rules.yml" || FAIL=1

UP_PORT=19490; PX_PORT=18490
python3 "$ROOT/benchmarks/mock_upstream.py" --port $UP_PORT --min-latency-ms 1 --max-latency-ms 2 >/dev/null 2>&1 & PIDS+=($!)
LISTEN_PORT=$PX_PORT ADMIN_PORT=0 WORKERS=1 UPSTREAM_URL="http://127.0.0.1:$UP_PORT" \
FALLBACK_UPSTREAM_URL="http://127.0.0.1:$UP_PORT" "$BINARY" >/dev/null 2>&1 & PIDS+=($!)
for _ in $(seq 100); do curl -fsS "http://127.0.0.1:$PX_PORT/healthz" >/dev/null 2>&1 && break; sleep 0.1; done
curl -fsS -o /dev/null "http://127.0.0.1:$PX_PORT/v1/chat/completions" -H "Authorization: Bearer sk" \
  -d '{"model":"m","messages":[{"role":"user","content":"metrics check"}]}'
echo "== promtool check metrics (live /metrics, fallback configured)"
if curl -fsS "http://127.0.0.1:$PX_PORT/metrics" | "$PROMTOOL" check metrics; then
  echo "  [PASS] /metrics parses and lints clean"
else
  echo "  [FAIL] promtool check metrics reported problems (above)"; FAIL=1
fi
exit $FAIL
