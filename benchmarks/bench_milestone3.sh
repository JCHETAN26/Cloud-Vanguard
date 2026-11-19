#!/usr/bin/env bash
# Milestone 3 benchmark (CPU or GPU). Needs torch, pyarrow and mlflow:
#   pip install -r training/requirements.txt
# PYTHON selects the interpreter (default python3).
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" benchmarks/bench_milestone3.py "$@"
