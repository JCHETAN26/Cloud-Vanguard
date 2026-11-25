#!/usr/bin/env python3
"""TF-IDF + logistic regression baseline on the same AG News rows the LLM runs use.

Rows come from training.hf.data's own sampling (same seed, same shuffle, same 600-character article truncation as
the LLM prompt), so the eval rows here are exactly the rows in agnews_eval.arrow. Two training sets:

  small  the TRAIN_ROWS rows the LoRA adapter is trained on (same data budget)
  full   all 120,000 AG News training rows (what a classical model would actually get)

Writes one JSON per training set in the eval report shape ({"samples": [{"i", "expected", "got", "ok"}]}) so
analyze.py treats it like any other model. CPU only, under a minute.

    python3 benchmarks/modeling/tfidf_baseline.py --out-dir runs/modeling --train-rows 4000 --eval-rows 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

# training/hf/data.py by path: the training.hf package imports torch, which this baseline doesn't need.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("agnews_data", os.path.join(_ROOT, "training", "hf", "data.py"))
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
LABELS, MAX_ARTICLE_CHARS, load_split, sample = _data.LABELS, _data.MAX_ARTICLE_CHARS, _data.load_split, _data.sample


def article(text: str) -> str:
    return " ".join(text.split())[:MAX_ARTICLE_CHARS]


def run(train: list[tuple[str, int]], test: list[tuple[str, int]], name: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x = vec.fit_transform([article(a) for a, _ in train])
    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(x, [y for _, y in train])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(vec.transform([article(a) for a, _ in test]))
    pred_ms = (time.perf_counter() - t0) * 1000 / len(test)
    samples = [{"i": i, "expected": LABELS[y], "got": LABELS[int(p)], "ok": int(p) == y}
               for i, ((_, y), p) in enumerate(zip(test, pred))]
    return {"backend": name, "examples": len(test), "train_rows": len(train), "features": len(vec.vocabulary_),
            "exact_match": sum(s["ok"] for s in samples) / len(samples), "fit_s": fit_s,
            "predict_ms_per_example": pred_ms, "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-rows", type=int, default=4000)
    ap.add_argument("--eval-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parquet-dir", default=None, help="local AG News parquet files (default: the Hub)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    full = load_split("train", a.parquet_dir)
    test = sample(load_split("test", a.parquet_dir), a.eval_rows, a.seed)
    for name, train in (("tfidf_lr_small", sample(full, a.train_rows, a.seed)), ("tfidf_lr_full", full)):
        rep = run(train, test, name)
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"{name}: train {rep['train_rows']} rows, accuracy {rep['exact_match']:.4f}, fit {rep['fit_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
