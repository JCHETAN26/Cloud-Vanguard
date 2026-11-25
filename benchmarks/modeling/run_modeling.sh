#!/usr/bin/env bash
# Modeling experiment: does fine-tuning beat a classical baseline and an untuned LLM on AG News?
#
#   tfidf_small  TF-IDF + logistic regression on the same 4,000 training rows   (CPU, tfidf_baseline.py)
#   tfidf_full   TF-IDF + logistic regression on all 120,000 training rows      (CPU)
#   base         Qwen2.5-14B, NF4, zero-shot, same prompt                        (GPU)
#   lora         the same NF4 base + a QLoRA adapter trained on the 4,000 rows   (GPU)
#
# All four are scored on the same EVAL_ROWS held-out test rows, with every prediction kept, then compared by
# analyze.py (accuracy, macro-F1 with bootstrap CI, McNemar, confusions, error examples). Base and LoRA use the same
# NF4 weights so the adapter is the only difference. Run from the repository root on a Linux box with one GPU:
#
#   ./benchmarks/modeling/run_modeling.sh
#   STAGES="base_eval" ./benchmarks/modeling/run_modeling.sh                   # a subset
#   MODEL=Qwen/Qwen2.5-0.5B QUANT_BITS=0 DEVICE=cpu STEPS=4 TRAIN_ROWS=64 EVAL_ROWS=16 \
#     STAGES="data tfidf base_eval lora tuned_eval analyze" PY=python3 ./benchmarks/modeling/run_modeling.sh   # dry run
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-Qwen/Qwen2.5-14B}"
QUANT_BITS="${QUANT_BITS:-4}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-$ROOT/benchmarks/results/modeling/$(date -u +%Y%m%dT%H%M%SZ)}"
VENV="${VENV:-$ROOT/.venv-gpu}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ROWS="${TRAIN_ROWS:-4000}"
EVAL_ROWS="${EVAL_ROWS:-2000}"
STEPS="${STEPS:-250}"
BATCH="${BATCH:-16}"
STAGES="${STAGES:-setup data download tfidf base_eval lora tuned_eval analyze}"
export MLFLOW_DISABLE_AGENT_HINT=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$OUT/logs" "$OUT/runs"
DATA="$OUT/data"
STATUS="$OUT/stages.tsv"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/logs/driver.log"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
stage() {  # stage <name> <command...>; a failing stage is recorded and the rest still run
  local name="$1"; shift
  want "$name" || return 0
  log "== $name"
  local t0 t1 rc; t0=$(date +%s)
  ( "$@" ) >"$OUT/logs/$name.log" 2>&1; rc=$?
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$t0" "$t1" >>"$STATUS"
  if [[ $rc -eq 0 ]]; then log "   ok ($((t1 - t0)) s)"; else log "   FAILED rc=$rc ($((t1 - t0)) s)"; tail -15 "$OUT/logs/$name.log" | sed 's/^/      /'; fi
}

do_setup() {
  set -e
  nvidia-smi
  python3 -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q torch
  "$PY" -m pip install -q "transformers==5.23.0" "peft==0.22.0" bitsandbytes nvidia-ml-py "pyarrow==26.0.1" \
    "mlflow==3.18.0" safetensors accelerate huggingface_hub hf_transfer scikit-learn
  "$PY" -m training info
  "$PY" -m pip freeze >"$OUT/pip-freeze.txt"
}

do_data() {
  "$PY" -m training prepare-hf-data --out "$DATA" --tokenizer "$MODEL" --train-rows "$TRAIN_ROWS" --eval-rows "$EVAL_ROWS"
}

do_download() {
  "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt']))"
}

do_tfidf() {
  "$PY" benchmarks/modeling/tfidf_baseline.py --out-dir "$OUT/runs" --train-rows "$TRAIN_ROWS" --eval-rows "$EVAL_ROWS"
}

evaluate() {  # evaluate <name> [adapter]
  local args=(eval --backend hf --hf-model "$MODEL" --quant-bits "$QUANT_BITS" --data "$DATA/agnews_eval.arrow"
              --limit "$EVAL_ROWS" --precision bf16 --device "$DEVICE" --max-new-tokens 6 --keep-samples -1
              --output "$OUT/runs/$1.json")
  [[ -n "${2:-}" ]] && args+=(--adapter "$2")
  "$PY" -m training "${args[@]}"
}

do_lora() {
  "$PY" -m training train-hf --model "$MODEL" --train-data "$DATA/agnews_train.arrow" --output-dir "$OUT/runs/lora" \
    --quant-bits "$QUANT_BITS" --dtype bf16 --device "$DEVICE" --steps "$STEPS" --batch-size "$BATCH" \
    --eval-examples 0 --run-name lora --tracking-uri "sqlite:///$OUT/mlflow.db"
}

do_analyze() {
  "$PY" benchmarks/modeling/analyze.py --out "$OUT/analysis.json" \
    tfidf_small="$OUT/runs/tfidf_lr_small.json" tfidf_full="$OUT/runs/tfidf_lr_full.json" \
    base="$OUT/runs/base.json" lora="$OUT/runs/lora_eval.json" | tee "$OUT/analysis.txt"
}

log "model=$MODEL quant_bits=$QUANT_BITS device=$DEVICE out=$OUT stages: $STAGES"
stage setup do_setup
stage data do_data
stage download do_download
stage tfidf do_tfidf
stage base_eval evaluate base
stage lora do_lora
stage tuned_eval evaluate lora_eval "$OUT/runs/lora/adapter"
stage analyze do_analyze
log "done: $OUT/analysis.txt"
