# training/ — Milestone 3: LoRA / QLoRA fine-tuning, quantization, data loading, evaluation

A PyTorch pipeline that runs fully offline on a CPU (tests, laptops, Codespaces) and uses GPU paths
(bitsandbytes, NCCL, vLLM, bf16/fp16) when they exist. Everything is a package, `training`, run from the
repository root.

```
training/
  data/      arrow.py      memory-mapped Arrow IPC datasets (file + Hugging Face stream format), zero-copy rows
             loader.py     DataLoader: padding + prompt masking, pin_memory, worker prefetch, DistributedSampler
             synthetic.py  deterministic offline corpora: pretrain text + 4 graded tasks (reverse/upper/add/sort)
             tokenizer.py  fixed char vocabulary (no tokenizer download)
  models/    tiny_lm.py    small Llama-style causal LM (q/k/v/o_proj, gate/up/down_proj naming)
             lora.py       LoRA: frozen W + (alpha/r)·B·A; inject / freeze / merge / save / load (PEFT layout)
             quant.py      int8 per-channel + NF4 4-bit weight quantization; bitsandbytes backend on CUDA
             precision.py  device + fp32/bf16/fp16 resolution, autocast, GradScaler (fp16 only)
             distributed.py torchrun / DDP hooks (Gloo on CPU, NCCL on CUDA)
             checkpoint.py base-model checkpoints (safetensors)
  eval/      tracking.py   MLflow (local sqlite or file store by default), rank-0 only, jsonl fallback
             backends.py   inference backends: torch (local), vLLM (offline engine + LoRARequest), OpenAI-HTTP
             harness.py    exact match, perplexity, per-request latency p50/p95/p99, tokens/s
  train.py   training loop (full / LoRA / QLoRA, grad accumulation, clipping, cosine LR, DDP, metrics)
  cli.py     python -m training {info,prepare-data,train,eval}
  configs/   pretrain_tiny.json, lora_tiny.json, qlora_tiny.json
  tests/     pytest suite (unit + CLI end-to-end, CPU)
```

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or a CUDA build
pip install -r training/requirements.txt                              # pyarrow, mlflow, pytest, (transformers, peft)
python -m training info                                               # what this machine supports
```

## Run it

```bash
python -m training prepare-data --out runs/data --task reverse          # 4 Arrow files, no network
python -m training train --config training/configs/pretrain_tiny.json \
    --train-data runs/data/pretrain.arrow --eval-data runs/data/pretrain_eval.arrow --output-dir runs/base
python -m training train --config training/configs/lora_tiny.json --base-model runs/base/model \
    --train-data runs/data/reverse_train.arrow --eval-data runs/data/reverse_eval.arrow --output-dir runs/lora
python -m training train --config training/configs/qlora_tiny.json ...   # 4-bit frozen base + LoRA
torchrun --nproc_per_node=2 -m training train --config training/configs/lora_tiny.json ...   # DDP
python -m training eval --base-model runs/base/model --adapter runs/lora/adapter \
    --data runs/data/reverse_eval.arrow --output runs/lora/eval.json
mlflow ui --backend-store-uri sqlite:///runs/lora/mlflow.db             # browse the run
```

Every flag overrides the JSON config (`python -m training train --help`). A run writes
`train_summary.json` (losses, throughput, step latency, peak memory, eval metrics), the adapter
(`adapter/`) or full model (`model/`), and an MLflow run with params, per-step metrics and artifacts.

## Design notes

**Data.** Datasets are Arrow IPC files opened with `pyarrow.memory_map`; `input_ids` offsets and values
are numpy views over the mapping, so opening and iterating a dataset allocates nothing in Arrow's heap
(tested) and DataLoader workers share page-cache pages instead of pickling data — the dataset drops its
mapping when pickled and re-maps in each worker. Hugging Face `datasets.save_to_disk` files (IPC
*stream* format) load too; a `text`-only table is tokenized on the fly. Prompt tokens are masked out of the
loss via `prompt_len`. `pin_memory="auto"` pins only when CUDA exists.

**LoRA.** `inject_lora` wraps each targeted linear layer (suffix list or one regex) in `LoRALinear`
computing `W x + (alpha/r)·B(A(dropout(x)))`, with A Kaiming-initialised and B zero, so the injected model is
bit-identical to the base until the first step (tested). Everything except `lora_A`/`lora_B` is frozen;
a target list that matches nothing is an error. `merge_lora` folds `ΔW = (alpha/r)·B·A` into plain
`nn.Linear` layers (dequantizing quantized bases). Adapters are written in the PEFT layout: an adapter
trained here on a Hugging Face Llama loads with `peft.PeftModel.from_pretrained` and gives the same logits,
and a PEFT-trained adapter loads here (both directions tested with peft 0.22), which is also the format
vLLM's `LoRARequest` takes.

**Quantization (QLoRA).** `quantize_model(model, bits)` runs before `inject_lora`, so adapters wrap frozen
quantized layers. With bitsandbytes and CUDA it uses `Linear8bitLt` / `Linear4bit(nf4)`. Otherwise a
portable implementation: int8 symmetric per-output-channel, and NF4 — blocks of 64 scaled by their absmax,
each weight mapped to the nearest of the 16 QLoRA NormalFloat levels, two codes per byte. It saves weight
memory, not compute: weights are dequantized in forward, which costs time on CPU (see the benchmark).

**Precision.** `auto` = bf16 on GPUs that support it, else fp16 with a `GradScaler`, else fp32 on CPU.
bf16 on CPU is allowed when requested (CPU autocast); fp16 on CPU falls back to fp32 with a warning. The
loss is always computed in fp32.

**DDP.** Launch with torchrun; `init_distributed` reads `RANK/WORLD_SIZE/LOCAL_RANK`, picks NCCL or Gloo,
and the loader shards with `DistributedSampler`. Only adapter gradients are all-reduced under LoRA, and
`broadcast_buffers=False` stops DDP re-sending the frozen quantized weights before every forward. Rank 0
alone logs to MLflow, saves, and evaluates. A 2-process Gloo run is tested to produce the same adapter
weights as single-process training on the full batch.

**Tracking.** The tracking URI is the explicit one, else `MLFLOW_TRACKING_URI`, else
`sqlite:///<output_dir>/mlflow.db` — no server needed. MLflow 3.x refuses `file:` stores unless
`MLFLOW_ALLOW_FILE_STORE=true`; the tracker sets that for file URIs since a local file store is a supported
choice here. Without MLflow installed, metrics go to `metrics.jsonl`.

**Evaluation.** One harness, three backends: `torch` (in-process; base + adapter, optionally quantized),
`vllm` (offline engine for a Hugging Face model + PEFT adapter; needs a GPU), and `openai` (any
OpenAI-compatible endpoint — `vllm serve`, or the CloudVanguard gateway in front of it — measuring the
latency clients see; it cannot score perplexity). Perplexity is `exp(mean NLL)` of the completion given the
prompt, the same quantity the trainer minimises (tested to match the training loss).

## Verification status

| Path | Status |
|---|---|
| Arrow mmap data, DataLoader workers/prefetch, DistributedSampler | tested on CPU |
| LoRA inject/freeze/forward/backward/merge/save/load, PEFT interop | tested on CPU |
| int8 / NF4 torch quantizer, QLoRA training | tested on CPU |
| bf16 autocast | tested on CPU |
| DDP (Gloo, 2 processes, `mp.spawn` and torchrun) | tested on CPU |
| MLflow sqlite + file stores, jsonl fallback | tested |
| Eval harness: torch backend, OpenAI-HTTP backend (against `benchmarks/mock_upstream.py`) | tested |
| bitsandbytes backend, fp16 + GradScaler, NCCL, pinned-memory transfer, vLLM backend | **implemented, not run**: needs a CUDA GPU; the tests for them skip with a reason |

## Benchmark

`make bench-m3 PYTHON=...` (`benchmarks/bench_milestone3.py`) measures on the machine it runs on and writes
`benchmarks/results/milestone3_bench.json`. The committed record is
`benchmarks/reports/milestone3_codespace_4vcpu.json`.

Measured 2026-11-19 on the Codespace: AMD EPYC 7763 (Zen 3), 4 vCPU (torch uses 2 threads), no GPU,
torch 2.15.0+cpu. Tiny model: 833k parameters, 4 layers, d_model 128. Base pre-trained 300 steps (loss
4.61 → 0.28), then each variant fine-tuned 1200 steps on the reverse task, global batch 32, seed 0.
Untuned base: exact match 0.000, perplexity 5074.8.

| variant | ranks | trainable | weights in memory (incl. adapter) | tok/s | step p50 / p95 ms | exact match | ppl | gen p50 ms |
|---|---|---|---|---|---|---|---|---|
| full fine-tune fp32 | 1 | 833,152 (100%) | 3.18 MiB | 16,507 | 36.4 / 38.9 | 1.000 | 1.000 | 8.77 |
| LoRA r=8 fp32 | 1 | 78,080 (8.6%) | 3.48 MiB | 13,967 | 42.9 / 46.9 | 1.000 | 1.094 | 17.32 |
| LoRA r=8 bf16 autocast | 1 | 78,080 (8.6%) | 3.48 MiB | 946 | 639.0 / 673.1 | 0.740 | 1.254 | 40.35 |
| QLoRA int8 (torch) | 1 | 78,080 (8.6%) | 1.23 MiB | 14,004 | 42.8 / 46.8 | 1.000 | 1.093 | 22.02 |
| QLoRA NF4 (torch) | 1 | 78,080 (8.6%) | 0.89 MiB | 12,496 | 47.8 / 52.3 | 0.550 | 1.545 | 48.64 |
| LoRA r=8, DDP Gloo | 2 | 78,080 (8.6%) | 3.48 MiB | 12,465 | 47.6 / 57.8 | 0.455 | 1.667 | 18.06 |

How to read it:

* **Exact match from one seed is noisy at 1200 steps.** Re-running with seeds 1 and 2, LoRA DDP, QLoRA
  int8 and QLoRA NF4 all reached 1.000 (single-process LoRA: 1.000 and 0.955). The seed-0 lows for NF4
  and DDP are data-order luck near the convergence edge, not a defect; DDP is separately tested to give
  the same weights as single-process training. Throughput, latency and size columns are not seed-sensitive.
* **Weights:** excluding the 305 KiB of fp32 adapter parameters, the int8 base is 3.4x and the NF4 base
  5.4x smaller than the 3.18 MiB fp32 base (embeddings, norms and lm_head stay fp32; on large models the
  ratios approach 4x and ~7x, since NF4 costs 4.5 bits/weight with its fp32 block scales). The adapter
  file on disk is 312 KiB.
* **Speed:** the torch quantizer dequantizes in forward, so it trades time for memory: int8 is at parity,
  NF4 costs 11% throughput and 2.8x generation latency. bf16 autocast is 15x *slower* here: Zen 3 has no
  native bf16 instructions, so CPU bf16 is emulated. Use bf16 on GPUs (or CPUs with AVX512-BF16/AMX).
* **DDP on 4 shared vCPUs** (2 ranks x 2 threads) is slower than one process: the model is too small for
  data parallelism to pay for Gloo all-reduce. It is here to exercise the multi-process path.
* **Peak RSS** is 535-567 MB per variant, of which ~350 MB is the interpreter + torch + MLflow baseline;
  with a 3 MB model the memory columns measure overhead, so the table reports weight bytes instead
  (both are in the JSON).
* **Data:** opening the 9.6 MB, 20k-row Arrow file takes 251 ms and allocates 80 KB of Arrow heap (the
  `prompt_len` column; token buffers stay in the mapping), unchanged after full epochs. The DataLoader
  delivers 47.9k samples/s in-process vs 16.0k with 2 workers: rows are tiny, so worker IPC costs more
  than it saves. Data wait is 0.7-0.8 ms of a ~43 ms step, so loading is not the bottleneck.
