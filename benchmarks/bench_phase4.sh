#!/usr/bin/env bash
# Phase 4: CSI/QEMU warm-cache persistence + resilience/chaos/load.
#
#   0. environment report (which layers can be real here)
#   1. build vanguard-proxy (Release), csi-driver, csictl
#   2. phase4_csi_e2e.py     CSI gRPC lifecycle on QEMU images, warm restore
#                            across crash / re-attach / snapshot-restore,
#                            torn-slot fallback, QMP hot-plug (if QEMU present)
#   3. phase4_resilience.py  3-node ring: load, chaos (node kill/restore, LLM
#                            outage), 429 burst, 508 loop guard, shutdown
#   4. deploy/scripts/e2e-kind.sh  only if docker + kind are present
#
# Linux only. Env: SKIP_KIND=1, WARM, STEADY.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/benchmarks/results"
BUILD_DIR="${BUILD_DIR:-$ROOT/proxy-cpp/build-release}"
mkdir -p "$RESULTS"
[[ "$(uname -s)" == "Linux" ]] || { echo "needs Linux (epoll); use a Codespace" >&2; exit 1; }

FAIL=0
run_tee() { local out=$1; shift; "$@" 2>&1 | tee "$out"; [[ ${PIPESTATUS[0]} -eq 0 ]] || FAIL=1; }

echo "==> [0] environment"
"$ROOT/deploy/scripts/env-check.sh" | tee "$RESULTS/phase4_env.txt"
"$ROOT/deploy/scripts/env-check.sh" --json > "$RESULTS/phase4_env.json"

echo; echo "==> [1] build"
cmake -S "$ROOT/proxy-cpp" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVANGUARD_BUILD_TESTS=OFF >/dev/null || exit 1
cmake --build "$BUILD_DIR" --target vanguard-proxy >/dev/null || exit 1
(cd "$ROOT/operator-go" && go build -o bin/csi-driver ./cmd/csi-driver && go build -o bin/csictl ./cmd/csictl) || exit 1

echo; echo "==> [2] CSI + QEMU warm-cache persistence"
run_tee "$RESULTS/phase4_csi.txt" python3 "$ROOT/benchmarks/phase4_csi_e2e.py" \
  --proxy-binary "$BUILD_DIR/vanguard-proxy" --csi-driver "$ROOT/operator-go/bin/csi-driver" \
  --csictl "$ROOT/operator-go/bin/csictl" --out "$RESULTS/phase4_csi.json"

echo; echo "==> [3] resilience, chaos and load (3-node ring)"
run_tee "$RESULTS/phase4_resilience.txt" python3 "$ROOT/benchmarks/phase4_resilience.py" \
  --binary "$BUILD_DIR/vanguard-proxy" --warm "${WARM:-3000}" --steady "${STEADY:-15000}" \
  --out "$RESULTS/phase4_resilience.json"

echo; echo "==> [4] real cluster (kind)"
if [[ "${SKIP_KIND:-0}" == 1 ]]; then
  echo "skipped (SKIP_KIND=1)"
else
  "$ROOT/deploy/scripts/e2e-kind.sh" 2>&1 | tee "$RESULTS/phase4_kind.txt"
  case ${PIPESTATUS[0]} in
    0) echo "kind e2e passed" ;;
    3) echo "kind e2e not run: required tools missing (see environment report)" ;;
    *) echo "kind e2e FAILED"; FAIL=1 ;;
  esac
fi

echo; echo "results in $RESULTS/"
[[ $FAIL -eq 0 ]] && echo "PHASE 4 SUITE PASSED" || echo "PHASE 4 SUITE: FAILURES"
exit $FAIL
