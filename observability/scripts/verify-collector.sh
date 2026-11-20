#!/usr/bin/env bash
# End-to-end check against a REAL OpenTelemetry Collector (not the test sink):
#   1. `otelcol validate` on observability/otel-collector/config.yaml
#   2. run the collector (same pipeline, debug exporter at detailed verbosity),
#      point vanguard-proxy at it, send requests with a known traceparent
#   3. require: the collector logged our trace id and span names, and the
#      proxy's exporter reports spans exported with zero errors (=> 2xx).
# Usage: verify-collector.sh <path/to/vanguard-proxy>      (Linux)
set -euo pipefail
BINARY="${1:?usage: verify-collector.sh <vanguard-proxy binary>}"
OBS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$OBS")"
OTELCOL="$(command -v otelcol || command -v otelcol-contrib || true)"
if [[ -z "$OTELCOL" ]]; then "$OBS/scripts/fetch-tools.sh" >/dev/null; OTELCOL="$OBS/.bin/otelcol"; fi
WORK="$(mktemp -d)"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
OTLP_PORT=14318; UP_PORT=19480; PX_PORT=18480; HC_PORT=13134
FAIL=0
check() { if eval "$2"; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; FAIL=1; fi; }

echo "== otelcol: $("$OTELCOL" --version)"
check "otelcol validate observability/otel-collector/config.yaml" "'$OTELCOL' validate --config '$OBS/otel-collector/config.yaml' >'$WORK/validate.log' 2>&1"

# Same pipeline on test ports, debug exporter at detailed verbosity.
sed -e "s/0.0.0.0:4318/127.0.0.1:$OTLP_PORT/" -e "s/0.0.0.0:4317/127.0.0.1:14317/" \
    -e "s/0.0.0.0:13133/127.0.0.1:$HC_PORT/" -e "s/verbosity: basic/verbosity: detailed/" \
    -e "s/timeout: 2s/timeout: 200ms/" "$OBS/otel-collector/config.yaml" > "$WORK/collector.yaml"
"$OTELCOL" --config "$WORK/collector.yaml" >"$WORK/collector.log" 2>&1 & PIDS+=($!)
python3 "$ROOT/benchmarks/mock_upstream.py" --port $UP_PORT --min-latency-ms 2 --max-latency-ms 5 --token-interval-ms 0 \
  >/dev/null 2>&1 & PIDS+=($!)
for _ in $(seq 100); do curl -fsS "http://127.0.0.1:$UP_PORT/healthz" >/dev/null 2>&1 && break; sleep 0.1; done
for _ in $(seq 100); do curl -fsS "http://127.0.0.1:$HC_PORT/" >/dev/null 2>&1 && break; sleep 0.1; done
LISTEN_PORT=$PX_PORT ADMIN_PORT=0 WORKERS=2 NODE_ID=collector-check UPSTREAM_URL="http://127.0.0.1:$UP_PORT" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:$OTLP_PORT" OTEL_BSP_SCHEDULE_DELAY=100 \
  "$BINARY" >"$WORK/proxy.log" 2>&1 & PIDS+=($!)
for _ in $(seq 100); do curl -fsS "http://127.0.0.1:$PX_PORT/healthz" >/dev/null 2>&1 && break; sleep 0.1; done

TRACE="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
for _ in 1 2 3; do  # miss, hit, hit
  curl -fsS -o /dev/null "http://127.0.0.1:$PX_PORT/v1/chat/completions" -H "Authorization: Bearer sk-check" \
    -H "traceparent: 00-$TRACE-00f067aa0ba902b7-01" -d '{"model":"m","messages":[{"role":"user","content":"collector check"}]}'
done
sleep 2
METRICS="$(curl -fsS "http://127.0.0.1:$PX_PORT/metrics")"
val() { echo "$METRICS" | awk -v n="$1" '$1==n {print $2}'; }
echo "   proxy exporter: exported=$(val vanguard_trace_spans_exported_total) errors=$(val vanguard_trace_export_errors_total) dropped=$(val vanguard_trace_spans_dropped_total)"
check "collector received the trace (trace id in debug output)" "grep -q '$TRACE' '$WORK/collector.log'"
check "collector logged SERVER + llm.upstream spans" "grep -q 'POST /v1/chat/completions' '$WORK/collector.log' && grep -q 'llm.upstream' '$WORK/collector.log'"
check "collector logged resource service.name=vanguard-proxy" "grep -q 'service.name: Str(vanguard-proxy)' '$WORK/collector.log'"
check "4 spans exported (miss: 2, hit: 1, hit: 1), 0 export errors" "[[ '$(val vanguard_trace_spans_exported_total)' == 4 && '$(val vanguard_trace_export_errors_total)' == 0 ]]"
echo "   collector log: $WORK/collector.log"
grep -E "Trace ID|^ +-> Name|Name +:" "$WORK/collector.log" | head -12 || true
exit $FAIL
