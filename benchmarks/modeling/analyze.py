#!/usr/bin/env python3
"""Compare classifiers on the same AG News eval rows: accuracy, macro-F1, confusions, paired tests, error analysis.

Every input is an eval report holding one prediction per eval row ({"samples": [{"i", "expected", "got", ...}]}):
`training eval --keep-samples -1` for the LLMs, tfidf_baseline.py for the classical baseline. Rows are matched by
index and the script fails if the expected labels disagree, so every model is scored on identical rows.

Scoring:
  strict    the generated text, stripped, equals the label exactly (what the harness reports as exact_match)
  parsed    the first word of the output is mapped to a label ("Sci/Tech", "Science", "Technology" -> Sci/Tech);
            outputs that map to no label count as wrong. Accuracy, macro-F1 and confusions use this, so a model is
            not marked down only for formatting; the gap between the two is reported.
  macro-F1  mean of the four per-class F1 scores; 95% CI by a seeded bootstrap over eval rows.
  McNemar   for each pair of models, rows only A gets right (b) and only B gets right (c); exact two-sided p.

    python3 benchmarks/modeling/analyze.py --out runs/modeling/analysis.json \\
        tfidf_small=runs/modeling/tfidf_lr_small.json base=runs/modeling/base_nf4.json lora=runs/modeling/lora_nf4.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
INVALID = "(none)"


def parse(text: str) -> str:
    t = text.strip().lower()
    m = re.match(r"[a-z/\-]+", t)
    w = m.group(0) if m else ""
    if w.startswith("world") or w in ("international", "politics"):
        return "World"
    if w.startswith("sport"):
        return "Sports"
    if w.startswith("business") or w in ("finance", "economy"):
        return "Business"
    if w.startswith(("sci", "tech")):
        return "Sci/Tech"
    return INVALID


def f1_scores(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for c in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "support": tp + fn}
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    return sum(v["f1"] for v in f1_scores(gold, pred).values()) / len(LABELS)


def bootstrap_ci(gold: list[str], pred: list[str], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    idx = range(len(gold))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(macro_f1([gold[i] for i in s], [pred[i] for i in s]))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial(b + c, 0.5) tail."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(path: str) -> dict:
    with open(path) as f:
        rep = json.load(f)
    rows = sorted(rep["samples"], key=lambda s: s["i"])
    if [s["i"] for s in rows] != list(range(len(rows))):
        raise SystemExit(f"{path}: predictions are not one per row 0..n-1 (was it run with --keep-samples -1?)")
    return {"report": rep, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="name=eval_report.json, in the order to print")
    ap.add_argument("--out", required=True, help="analysis JSON")
    ap.add_argument("--examples", type=int, default=8, help="error examples kept per confusion")
    a = ap.parse_args()
    models = {}
    for spec in a.models:
        name, _, path = spec.partition("=")
        models[name] = load(path)
    names = list(models)
    gold = [s["expected"] for s in models[names[0]]["rows"]]
    for n in names:
        if [s["expected"] for s in models[n]["rows"]] != gold:
            raise SystemExit(f"{n}: expected labels differ from {names[0]}: not the same eval rows")
    prompts = next(([s.get("prompt", "") for s in m["rows"]] for m in models.values() if "prompt" in m["rows"][0]),
                   [""] * len(gold))

    out: dict = {"rows": len(gold), "class_counts": dict(Counter(gold)), "models": {}, "pairs": [], "errors": {}}
    preds: dict[str, list[str]] = {}
    for n in names:
        rows = models[n]["rows"]
        pred = [parse(s["got"]) for s in rows]
        preds[n] = pred
        lo, hi = bootstrap_ci(gold, pred)
        conf = Counter((g, p) for g, p in zip(gold, pred) if g != p)
        out["models"][n] = {
            "strict_accuracy": sum(s["got"].strip() == s["expected"] for s in rows) / len(rows),
            "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
            "macro_f1": macro_f1(gold, pred), "macro_f1_ci95": [lo, hi],
            "unparseable": sum(p == INVALID for p in pred),
            "per_class": f1_scores(gold, pred),
            "confusions": [{"gold": g, "pred": p, "count": c} for (g, p), c in conf.most_common()],
            "raw_outputs_top": Counter(s["got"].strip() for s in rows).most_common(8),
            "latency_ms": models[n]["report"].get("latency_ms"),
        }
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            b = sum(1 for g, p, q in zip(gold, preds[x], preds[y]) if p == g and q != g)
            c = sum(1 for g, p, q in zip(gold, preds[x], preds[y]) if p != g and q == g)
            out["pairs"].append({"a": x, "b": y, "only_a_right": b, "only_b_right": c, "mcnemar_p": mcnemar_p(b, c)})
    wrong_all = [i for i in range(len(gold)) if all(preds[n][i] != gold[i] for n in names)]
    out["wrong_for_all"] = len(wrong_all)
    for n in names:
        errs: dict[str, list] = {}
        for i, (g, p) in enumerate(zip(gold, preds[n])):
            if g != p:
                bucket = errs.setdefault(f"{g} -> {p}", [])
                if len(bucket) < a.examples:
                    bucket.append({"i": i, "got": models[n]["rows"][i]["got"], "others": {m: preds[m][i] for m in names},
                                   "article": prompts[i].split("Article: ", 1)[-1].rsplit("\nTopic:", 1)[0][:300]})
        out["errors"][n] = errs
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{len(gold)} eval rows, classes {dict(Counter(gold))}")
    print(f"{'model':<16} {'strict':>7} {'acc':>7} {'macroF1':>8}  {'95% CI':<15} {'unparsed':>8}")
    for n in names:
        m = out["models"][n]
        print(f"{n:<16} {m['strict_accuracy']:>7.4f} {m['accuracy']:>7.4f} {m['macro_f1']:>8.4f}  "
              f"[{m['macro_f1_ci95'][0]:.4f}, {m['macro_f1_ci95'][1]:.4f}] {m['unparseable']:>8}")
    for p in out["pairs"]:
        print(f"{p['a']} vs {p['b']}: only {p['a']} right {p['only_a_right']}, only {p['b']} right "
              f"{p['only_b_right']}, McNemar p={p['mcnemar_p']:.3g}")
    for n in names:
        top = ", ".join(f"{c['gold']}->{c['pred']} {c['count']}" for c in out["models"][n]["confusions"][:4])
        print(f"{n} top confusions: {top}")
    print(f"wrong for every model: {len(wrong_all)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
