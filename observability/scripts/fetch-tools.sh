#!/usr/bin/env bash
# Downloads pinned OpenTelemetry Collector (core) and Prometheus release
# binaries into observability/.bin (gitignored), Linux amd64/arm64 only.
# Used by verify-collector.sh and verify-prometheus.sh when the tools are not
# already on PATH.
set -euo pipefail
OTELCOL_VERSION="${OTELCOL_VERSION:-0.111.0}"
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-2.54.1}"
BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.bin"
mkdir -p "$BIN"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ARCH=amd64 ;;
  Linux-aarch64|Linux-arm64) ARCH=arm64 ;;
  *) echo "fetch-tools.sh supports Linux amd64/arm64 only" >&2; exit 1 ;;
esac
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

if [[ ! -x "$BIN/otelcol" ]]; then
  echo "==> otelcol $OTELCOL_VERSION"
  curl -fsSL -o "$TMP/otelcol.tgz" \
    "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/otelcol_${OTELCOL_VERSION}_linux_${ARCH}.tar.gz"
  tar -xzf "$TMP/otelcol.tgz" -C "$TMP" otelcol
  mv "$TMP/otelcol" "$BIN/otelcol"
fi
if [[ ! -x "$BIN/promtool" ]]; then
  echo "==> promtool $PROMETHEUS_VERSION"
  curl -fsSL -o "$TMP/prom.tgz" \
    "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}.tar.gz"
  tar -xzf "$TMP/prom.tgz" -C "$TMP" "prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/promtool"
  mv "$TMP/prometheus-${PROMETHEUS_VERSION}.linux-${ARCH}/promtool" "$BIN/promtool"
fi
"$BIN/otelcol" --version
"$BIN/promtool" --version | head -1
