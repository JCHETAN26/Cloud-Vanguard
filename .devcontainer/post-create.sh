#!/usr/bin/env bash
# Toolchain for every phase: C++20 data plane, Go operator, Python harness.
set -euo pipefail
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build libgtest-dev nlohmann-json3-dev curl jq
# Independent Go load generator used by benchmarks/bench_phase1.sh when present.
go install github.com/rakyll/hey@v0.1.4 || echo "hey install failed (optional)"

# post-create addition (Phase 4): real QEMU tools for CSI tests
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends qemu-utils qemu-system-x86
