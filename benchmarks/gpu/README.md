# GPU benchmark: 14B LoRA vs QLoRA, 2-GPU DDP, vLLM evaluation, GPU telemetry

`run_gpu_bench.sh` runs the `training/` pipeline on a real 14.77B-parameter model (Qwen2.5-14B, Apache-2.0,
ungated) and measures, with the same data/steps/batch/sequence cap for every variant:

| stage | what it measures |
|---|---|
| `setup` | venvs (`.venv-gpu`: torch + transformers/peft/bitsandbytes; `.venv-vllm`: vLLM), `nvidia-smi -q`, pip freeze |
| `gpu_tests` | `training/tests` on CUDA: the 4 tests that skip on CPU (bitsandbytes int8/NF4, fp16 GradScaler, NCCL) run here |
| `smoke` | every GPU code path on Qwen2.5-0.5B in minutes (bnb NF4, NCCL DDP, hf + vLLM eval); a failure stops the run before the 14B stages |
| `data` | AG News topic classification (4 labels), 4000 train rows, 500 held-out test rows, Qwen tokenizer |
| `download` | the 14B checkpoint (~30 GB) |
| `base_eval` | the untuned base model on the same 500 rows (the "before" number) |
| `lora_bf16` | LoRA r=16 on all attention + MLP projections, bf16 base, 1 GPU |
| `qlora_nf4` | the same with the frozen base in bitsandbytes NF4 (QLoRA), 1 GPU |
| `ddp_lora_bf16`, `ddp_qlora_nf4` | both again with DDP over 2 GPUs (NCCL), same global batch |
| `vllm_eval` | the bf16 adapter served by vLLM (`LoRARequest`) on the same 500 rows |

GPU memory is recorded four ways per run (`train_summary.json`): weights on the GPU after loading, peak
`torch.cuda.max_memory_allocated` during training, peak reserved, and peak NVML "used" (what `nvidia-smi`
shows). Throughout the run, `observability/gpu_exporter.py` is scraped every second by a real Prometheus, which
loads `observability/prometheus/rules.yml`; `summarize.py` queries it for the per-stage peak memory and utilization.

## Hardware

Sized for **2x 80 GB GPUs (A100 or H100), >= 128 GB RAM, >= 250 GB disk**, Ubuntu with an NVIDIA driver. With one
GPU the DDP stages are skipped and everything else runs. 40/48 GB GPUs fit QLoRA but not the bf16 14B baseline
(~30 GB of weights plus activations and logits); use `MODEL=Qwen/Qwen2.5-7B` there and say so in any write-up.

The QLoRA path loads the bf16 checkpoint into host RAM and quantizes layer by layer onto the GPU, so each
process needs ~30 GB of RAM while loading (DDP: two processes).

## Run it

```bash
git clone https://github.com/JCHETAN26/Cloud-Vanguard.git && cd Cloud-Vanguard && git checkout m5-gpu-validation
./benchmarks/gpu/run_gpu_bench.sh 2>&1 | tee bench.log          # or: make bench-gpu
```

Partial runs reuse the venvs; pass the same `OUT` to keep results in one directory:

```bash
OUT=benchmarks/results/gpu/<run> STAGES="qlora_nf4 vllm_eval" ./benchmarks/gpu/run_gpu_bench.sh
```

Knobs (environment): `MODEL`, `SMOKE_MODEL`, `STEPS` (250), `BATCH` (global, 16), `TRAIN_ROWS` (4000), `EVAL_ROWS` (500).

## Bring the results back

Everything is under `benchmarks/results/gpu/<run>/` (gitignored): `summary.md`, `summary.json`, `stages.tsv`,
`logs/`, `runs/*/train_summary.json`, `runs/*.json` evaluations, `mlflow.db`, `nvidia-smi-q.txt`,
`pip-freeze*.txt`, and the Prometheus TSDB. Copy it off the box before releasing the instance:

```bash
scp -r <box>:Cloud-Vanguard/benchmarks/results/gpu/<run> benchmarks/results/gpu/
```

Commit `summary.md`, `summary.json` and the environment files as the record (as `benchmarks/reports/` does for the
CPU benchmarks); the adapters and TSDB are reproducible and stay out of git.

## Reading the numbers

* The memory comparison that matters is bf16 LoRA vs QLoRA NF4 at the same batch and sequence cap. Weights
  shrink ~3.5x (NF4 is ~4.5 bits/weight with block scales; embeddings, norms and lm_head stay bf16); the
  peak-during-training reduction is smaller, because activations, logits and the CUDA context do not shrink.
* Exact match on 500 rows has a binomial standard error of about 1-2 points near 90%; differences smaller than
  that are noise, not a quantization effect.
* DDP speedup is reported as 2-GPU tokens/s over 1-GPU tokens/s at the same global batch; it includes the
  NCCL all-reduce of the adapter gradients (~70M parameters at r=16 on a 14B model).
* The base model is evaluated zero-shot with the same strict exact-match scoring; a low number there mostly
  means it does not answer in the one-word format, which is what fine-tuning teaches.
