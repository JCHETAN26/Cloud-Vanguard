#!/usr/bin/env bash
# GPU benchmark: Qwen2.5-14B (14.77B params) LoRA vs QLoRA-NF4, DDP on 2 GPUs, vLLM evaluation, GPU telemetry
# through Prometheus. Run on a rented Linux box with NVIDIA GPU(s) from the repository root:
#
#   ./benchmarks/gpu/run_gpu_bench.sh                 # everything
#   STAGES="setup data lora_bf16" ./benchmarks/gpu/run_gpu_bench.sh   # a subset
#
# Hardware it was sized for: 2x 80 GB GPUs (A100/H100), >= 128 GB host RAM, >= 250 GB disk. With one GPU the
# DDP stages are skipped. Every stage logs to $OUT/logs/<stage>.log and records start/end times; a failing
# stage is recorded as failed and the others still run. Results: $OUT/summary.md and $OUT/summary.json.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-Qwen/Qwen2.5-14B}"
OUT="${OUT:-$ROOT/benchmarks/results/gpu/$(date -u +%Y%m%dT%H%M%SZ)}"
VENV="${VENV:-$ROOT/.venv-gpu}"
VLLM_VENV="${VLLM_VENV:-$ROOT/.venv-vllm}"
TRAIN_ROWS="${TRAIN_ROWS:-4000}"
EVAL_ROWS="${EVAL_ROWS:-500}"
STEPS="${STEPS:-250}"
BATCH="${BATCH:-16}"                  # global batch; DDP stages use BATCH/NGPU per rank
PROM_VERSION="${PROM_VERSION:-2.54.1}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B}"
STAGES="${STAGES:-setup gpu_tests smoke data download base_eval lora_bf16 qlora_nf4 ddp_lora_bf16 ddp_qlora_nf4 vllm_eval}"
# PCI_BUS_ID: CUDA device N == NVML device N, so per-GPU telemetry lines up with ranks.
export CUDA_DEVICE_ORDER=PCI_BUS_ID MLFLOW_DISABLE_AGENT_HINT=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$OUT/logs" "$OUT/runs"
DATA="$OUT/data"
PY="$VENV/bin/python"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
STATUS="$OUT/stages.tsv"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/logs/driver.log"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
want setup && : >"$STATUS"  # a rerun without setup appends to the same record
stage() {  # stage <name> <command...>
  local name="$1"; shift
  want "$name" || return 0
  log "== $name"
  local t0 t1 rc; t0=$(date +%s)
  # Subshell: a stage's `set -e` (setup, smoke) must not leak into the driver, or the first failing stage kills
  # the whole run instead of being recorded.
  ( "$@" ) >"$OUT/logs/$name.log" 2>&1; rc=$?
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$t0" "$t1" >>"$STATUS"
  if [[ $rc -eq 0 ]]; then log "   ok ($((t1 - t0)) s)"; else log "   FAILED rc=$rc ($((t1 - t0)) s): tail of $OUT/logs/$name.log"; tail -15 "$OUT/logs/$name.log" | sed 's/^/      /'; fi
  return 0
}

# ---------------------------------------------------------------- setup
do_setup() {
  set -e
  nvidia-smi
  nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
  python3 -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q torch                      # Linux wheels on PyPI bundle CUDA 12
  "$PY" -m pip install -q "transformers==5.23.0" "peft==0.22.0" bitsandbytes nvidia-ml-py \
    "pyarrow==26.0.1" "mlflow==3.18.0" safetensors accelerate huggingface_hub hf_transfer pytest
  if want vllm_eval; then
    python3 -m venv "$VLLM_VENV"
    "$VLLM_VENV/bin/python" -m pip install -q --upgrade pip
    "$VLLM_VENV/bin/python" -m pip install -q vllm "pyarrow==26.0.1" numpy   # vLLM pins its own torch
  fi
  "$PY" -m training info
  "$PY" -m pip freeze >"$OUT/pip-freeze.txt"
  [[ -x "$VLLM_VENV/bin/python" ]] && "$VLLM_VENV/bin/python" -m pip freeze >"$OUT/pip-freeze-vllm.txt" || true
}

do_gpu_tests() {  # the tests that skip on CPU (bitsandbytes, fp16 GradScaler, NCCL) run here
  "$PY" -m pytest training/tests -q -rs
}

# ---------------------------------------------------------------- telemetry: exporter + Prometheus
start_telemetry() {
  local bin="$ROOT/observability/.bin"
  mkdir -p "$bin"
  if [[ ! -x "$bin/prometheus" ]]; then
    local tmp; tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/p.tgz" "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
    tar -xzf "$tmp/p.tgz" -C "$tmp"
    mv "$tmp/prometheus-${PROM_VERSION}.linux-amd64/prometheus" "$tmp/prometheus-${PROM_VERSION}.linux-amd64/promtool" "$bin/"
  fi
  "$PY" "$ROOT/observability/gpu_exporter.py" --port 9400 --host 127.0.0.1 >"$OUT/logs/gpu_exporter.log" 2>&1 & PIDS+=($!)
  "$bin/prometheus" --config.file="$ROOT/benchmarks/gpu/prometheus-gpu.yml" --storage.tsdb.path="$OUT/prometheus" \
    --web.listen-address=127.0.0.1:9490 >"$OUT/logs/prometheus.log" 2>&1 & PIDS+=($!)
  for _ in $(seq 60); do curl -fsS http://127.0.0.1:9490/-/ready >/dev/null 2>&1 && break; sleep 1; done
  "$bin/promtool" check config "$ROOT/benchmarks/gpu/prometheus-gpu.yml" >"$OUT/logs/promtool_config.log" 2>&1 \
    && log "   promtool check config (GPU scrape + rules.yml): ok" || log "   promtool check config: FAILED (see logs)"
  curl -fsS http://127.0.0.1:9400/metrics >"$OUT/gpu_metrics_sample.txt"
  "$bin/promtool" check metrics <"$OUT/gpu_metrics_sample.txt" >"$OUT/logs/promtool_gpu_metrics.log" 2>&1 \
    && log "   gpu_exporter /metrics: promtool check metrics clean" || log "   gpu_exporter /metrics: promtool reported issues"
}

# ---------------------------------------------------------------- stages
do_data() {
  "$PY" -m training prepare-hf-data --out "$DATA" --tokenizer "$MODEL" --train-rows "$TRAIN_ROWS" --eval-rows "$EVAL_ROWS"
}

do_download() {
  "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt']))"
}

do_base_eval() {  # the untuned base, same prompts, same scoring: the "before" number
  "$PY" -m training eval --backend hf --hf-model "$MODEL" --data "$DATA/agnews_eval.arrow" --limit "$EVAL_ROWS" \
    --precision bf16 --device cuda --max-new-tokens 6 --output "$OUT/runs/base_eval.json"
}

train() {  # train <name> <quant_bits> <nproc>
  local name="$1" bits="$2" nproc="$3"
  local per_rank=$((BATCH / nproc))
  local args=(train-hf --model "$MODEL" --train-data "$DATA/agnews_train.arrow" --eval-data "$DATA/agnews_eval.arrow"
              --output-dir "$OUT/runs/$name" --quant-bits "$bits" --dtype bf16 --steps "$STEPS" --batch-size "$per_rank"
              --eval-examples "$EVAL_ROWS" --run-name "$name" --tracking-uri "sqlite:///$OUT/mlflow.db")
  if [[ "$nproc" -gt 1 ]]; then
    "$PY" -m torch.distributed.run --nproc_per_node="$nproc" --master_addr=127.0.0.1 --master_port=29511 -m training "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES=0 "$PY" -m training "${args[@]}"
  fi
}

do_smoke() {  # every GPU code path on a 0.5B model in a few minutes, before paying for 14B runs
  set -e
  local d="$OUT/smoke"
  "$PY" -m training prepare-hf-data --out "$d/data" --tokenizer "$SMOKE_MODEL" --train-rows 256 --eval-rows 16
  local common=(train-hf --model "$SMOKE_MODEL" --train-data "$d/data/agnews_train.arrow" --eval-data "$d/data/agnews_eval.arrow"
                --dtype bf16 --steps 10 --eval-examples 8 --no-track)
  CUDA_VISIBLE_DEVICES=0 "$PY" -m training "${common[@]}" --output-dir "$d/bf16" --batch-size 8
  CUDA_VISIBLE_DEVICES=0 "$PY" -m training "${common[@]}" --output-dir "$d/nf4" --batch-size 8 --quant-bits 4
  grep -q '"quant_backend": "bnb"' "$d/nf4/train_summary.json"   # bitsandbytes, not the portable fallback
  if [[ "$NGPU" -ge 2 ]]; then
    "$PY" -m torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29512 -m training \
      "${common[@]}" --output-dir "$d/ddp_nf4" --batch-size 4 --quant-bits 4
  fi
  "$PY" -m training eval --backend hf --hf-model "$SMOKE_MODEL" --adapter "$d/nf4/adapter" --quant-bits 4 \
    --data "$d/data/agnews_eval.arrow" --limit 4 --precision bf16 --device cuda --output "$d/eval_hf_nf4.json"
  if want vllm_eval; then
    "$VLLM_VENV/bin/python" -m training eval --backend vllm --hf-model "$SMOKE_MODEL" --adapter "$d/bf16/adapter" \
      --data "$d/data/agnews_eval.arrow" --limit 4 --output "$d/eval_vllm.json"
  fi
  echo "smoke: all GPU paths ran"
}

do_vllm_eval() {
  "$VLLM_VENV/bin/python" -m training eval --backend vllm --hf-model "$MODEL" --adapter "$OUT/runs/lora_bf16/adapter" \
    --data "$DATA/agnews_eval.arrow" --limit "$EVAL_ROWS" --max-new-tokens 6 --output "$OUT/runs/vllm_eval_lora_bf16.json"
}

# ---------------------------------------------------------------- run
log "model=$MODEL gpus=$NGPU out=$OUT stages: $STAGES"
stage setup do_setup
[[ -x "$PY" ]] || { log "no venv at $VENV: run the setup stage first"; exit 1; }
start_telemetry
stage gpu_tests do_gpu_tests
stage smoke do_smoke
if want smoke && ! grep -q $'^smoke\t0\t' "$STATUS"; then
  log "smoke failed: stopping before the $MODEL runs (fix, then rerun with STAGES without setup)"; exit 1
fi
stage data do_data
stage download do_download
stage base_eval do_base_eval
stage lora_bf16 train lora_bf16 0 1
stage qlora_nf4 train qlora_nf4 4 1
if [[ "$NGPU" -ge 2 ]]; then
  stage ddp_lora_bf16 train ddp_lora_bf16 0 2
  stage ddp_qlora_nf4 train ddp_qlora_nf4 4 2
else
  log "== ddp stages skipped: $NGPU GPU(s)"
fi
stage vllm_eval do_vllm_eval
sleep 3  # one more scrape after the last stage
"$PY" "$ROOT/benchmarks/gpu/summarize.py" "$OUT" --prometheus http://127.0.0.1:9490 | tee "$OUT/summary.md"
log "done: $OUT/summary.md"
