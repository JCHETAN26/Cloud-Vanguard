# Modeling experiment: classical baseline vs untuned LLM vs QLoRA

**Question:** on a simple classification task, does fine-tuning a 14B model earn its cost? Compared against an
untuned LLM and a classical TF-IDF model, with every prediction kept so the errors can be read, not just counted.

**Answer:**
- **QLoRA beats both the untuned model and TF-IDF given the same data.** Trained on 4,000 rows, it scores
  92.0% accuracy and 0.921 macro-F1. The untuned model scores 82.1% (0.819) and TF-IDF on those 4,000 rows scores
  86.4% (0.864). Both gaps are significant (McNemar p < 1e-16).
- **It only ties TF-IDF trained on all 120,000 rows:** 91.65% and 0.917 macro-F1, p = 0.56. So fine-tuning buys
  the same accuracy from 30× less labelled data, not more accuracy.
- **It costs far more to run:** about 231 ms per answer on a GPU, against well under a millisecond on a CPU for
  TF-IDF.

## Setup

| | |
|---|---|
| Task | AG News topic classification: World / Sports / Business / Sci/Tech |
| Data | 4,000 training rows; 2,000 held-out test rows (496–524 per class), the same rows for every model |
| LLM | Qwen2.5-14B, 4-bit NF4 (bitsandbytes), zero-shot prompt from [`training/hf/data.py`](../../training/hf/data.py) |
| Fine-tuning | QLoRA r=16 on all attention + MLP projections (68.8M trainable), 250 steps, batch 16, 1 epoch |
| Baseline | TF-IDF (1–2-grams, sublinear tf) + logistic regression, on the same 600-character article text |
| Hardware | 1× NVIDIA A40 48 GB, RunPod Secure Cloud; 47 min wall time, $0.48 |

The untuned and tuned models load the same NF4 weights, so the adapter is the only difference between them.

**How it was scored ([`analyze.py`](analyze.py)):**
- **Parsed:** each output's first word is mapped to a label.
- **Strict:** the exact-match number the harness reports.
- **Macro-F1:** a 95% CI from 1,000 bootstrap resamples of the eval rows.
- **McNemar:** an exact paired test on the rows where exactly one of two models is right.

## Results ([`modeling_agnews_a40/`](../reports/modeling_agnews_a40/))

| Model | Training rows | Accuracy (strict) | Accuracy (parsed) | Macro-F1 [95% CI] | Latency p50 |
|---|---|---|---|---|---|
| TF-IDF + LR | 4,000 | 86.4% | 86.4% | 0.864 [0.849, 0.878] | < 1 ms (CPU) |
| Qwen2.5-14B NF4, untuned | 0 | 77.1% | 82.1% | 0.819 [0.804, 0.834] | 167 ms (A40) |
| TF-IDF + LR | 120,000 | 91.65% | 91.65% | 0.917 [0.906, 0.928] | < 1 ms (CPU) |
| **Qwen2.5-14B NF4 + QLoRA** | 4,000 | **92.0%** | **92.0%** | **0.921 [0.909, 0.932]** | 231 ms (A40) |

**Paired tests** (rows only one model of the pair gets right):

| Comparison | Only the first right | Only the second right | McNemar p |
|---|---|---|---|
| QLoRA vs untuned | 235 | 36 | 6e-37 |
| QLoRA vs TF-IDF (4k) | 150 | 38 | 6e-17 |
| QLoRA vs TF-IDF (120k) | 57 | 50 | 0.56: no detectable difference |

**Per-class F1:**

| Model | World | Sports | Business | Sci/Tech |
|---|---|---|---|---|
| TF-IDF (4k) | 0.879 | 0.936 | 0.813 | 0.828 |
| Untuned | 0.896 | 0.974 | 0.751 | 0.655 |
| TF-IDF (120k) | 0.926 | 0.967 | 0.885 | 0.890 |
| QLoRA | 0.923 | 0.980 | 0.884 | 0.895 |

**Training ([`lora_train_summary.json`](../reports/modeling_agnews_a40/runs/lora_train_summary.json)):**
- Loss fell from 0.46 to 0.114 (mean of the last 10 steps).
- The training loop took 725 s at 439 tokens/s.
- Peak allocated memory was 18,791 MiB, the same as the A100 QLoRA run in M5.

## Error analysis

**What fine-tuning fixed.** The untuned model has two separate problems:
- **Format:** 110 outputs ran on past the label ("Sports\n\nThe news article discusses…"). That accounts for the
  5-point gap between strict and parsed accuracy. After fine-tuning, every output is a bare label.
- **A bias toward "Business":** it answered Business 732 times for 496 Business articles. 233 of 524 Sci/Tech
  articles were called Business (Sci/Tech recall 0.52). Fine-tuning learns the dataset's
  convention that company tech news is Sci/Tech.

**What is left.**
- **Business ↔ Sci/Tech is the main remaining error:** 82 of QLoRA's 160 errors, and the same pair is TF-IDF's top
  confusion.
- **Many of the remaining "errors" read as debatable labels rather than model mistakes.** Examples where QLoRA
  disagrees with the gold label:
  - "Telescope snaps distant 'planet'": labelled World, predicted Sci/Tech.
  - "Dow Jones Agrees to Buy MarketWatch in $519 Million Deal": labelled World, predicted Business.
  - "Google puts desktop search privacy up front": labelled Business, predicted Sci/Tech.
- **86 rows (4.3%) are wrong for all four models,** which points to a ceiling set by label noise and genuinely
  ambiguous articles. I did not hand-audit them, so this is an observation, not a measured noise rate.

**Precision matters for the untuned number.**
- The first 500 eval rows here are exactly M5's 500 eval rows (same sampling seed).
- On those rows, the untuned model scores 75.8% strict in NF4, against 82.4% strict in bf16 in M5.
- QLoRA scores 91.8% here, against 92.0% in M5.
- So 4-bit quantization costs the zero-shot model about 6.6 points but makes no difference once tuned. The 10-point
  "before → after" gain in M5 was measured in bf16; the NF4 comparison here shows a larger gap.

## When is fine-tuning worth it here?

- **With only a few thousand labelled rows,** QLoRA is clearly better than TF-IDF: +5.6 points and 0.057 macro-F1.
- **With 120k labelled rows,** a CPU model that trains in under a minute matches it, at a fraction of a millisecond
  per prediction.
- **The untuned 14B model is the weakest option** for this task unless the output is parsed and its Business bias
  is corrected.

The LLM's advantages, such as no feature engineering and taking new labels from a prompt, matter more on harder,
data-poor tasks than on a 4-class news benchmark that 2004-era methods already handle well.

## Reproduce

```bash
./benchmarks/modeling/run_modeling.sh   # Linux, one ≥ 24 GB NVIDIA GPU; stages and env knobs in the script header
python3 benchmarks/modeling/analyze.py --out analysis.json \
    tfidf_small=benchmarks/reports/modeling_agnews_a40/runs/tfidf_lr_small.json \
    tfidf_full=benchmarks/reports/modeling_agnews_a40/runs/tfidf_lr_full.json \
    base=benchmarks/reports/modeling_agnews_a40/runs/base.json \
    lora=benchmarks/reports/modeling_agnews_a40/runs/lora_eval.json   # reproduces analysis.txt exactly
```

The committed prediction files hold labels and model outputs, not the article text. Rows are identified by index
into the seeded sample of the AG News test split.

## Limits

- **One task, one seed, one training run.** There is no variance across seeds; the CIs cover eval-set sampling
  only.
- **Zero-shot only for the untuned model.** A few-shot prompt or constrained decoding would likely narrow its gap.
- **TF-IDF was not tuned** beyond sensible defaults (C=10, 1–2-grams). The full-data score was 91.70% on a laptop
  and 91.65% on the pod, a scikit-learn version difference.
- **Latency is single-request Hugging Face `generate`.** vLLM was 42.5% faster in M5 and was not rerun here.
