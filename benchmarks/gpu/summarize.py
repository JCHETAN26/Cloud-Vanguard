#!/usr/bin/env python3
"""Summarize a run_gpu_bench.sh output directory: train summaries, evaluations, and per-stage GPU telemetry
queried from Prometheus. Writes <out>/summary.json and prints Markdown.

    python3 benchmarks/gpu/summarize.py benchmarks/results/gpu/<run> [--prometheus http://127.0.0.1:9490]

Every number comes from a file or a Prometheus query; a missing input shows as "-", never as an estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

TRAIN_STAGES = ["lora_bf16", "qlora_nf4", "ddp_lora_bf16", "ddp_qlora_nf4"]


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def stages(out: str) -> dict[str, dict]:
    res = {}
    try:
        with open(os.path.join(out, "stages.tsv")) as f:
            for line in f:
                name, rc, t0, t1 = line.rstrip("\n").split("\t")
                res[name] = {"rc": int(rc), "start": int(t0), "end": int(t1), "seconds": int(t1) - int(t0)}
    except OSError:
        pass
    return res


def prom_query(base: str, expr: str, at: int) -> list[dict]:
    url = f"{base}/api/v1/query?" + urllib.parse.urlencode({"query": expr, "time": at})
    with urllib.request.urlopen(url, timeout=10) as r:
        doc = json.load(r)
    return doc["data"]["result"] if doc.get("status") == "success" else []


def stage_telemetry(base: str, st: dict) -> dict | None:
    """Peak memory and mean utilization per GPU over the stage's wall-clock window."""
    window = max(st["seconds"], 2)
    try:
        mem = prom_query(base, f"max_over_time(cloudvanguard_gpu_memory_used_bytes[{window}s])", st["end"])
        util = prom_query(base, f"avg_over_time(cloudvanguard_gpu_utilization_ratio[{window}s])", st["end"])
        n = prom_query(base, f"count_over_time(cloudvanguard_gpu_memory_used_bytes[{window}s])", st["end"])
    except Exception as e:  # Prometheus not running (summarizing an old run offline)
        return {"error": str(e)}
    out: dict = {}
    for r in mem:
        out.setdefault(r["metric"]["gpu"], {})["peak_memory_used_gb"] = float(r["value"][1]) / 1e9
    for r in util:
        out.setdefault(r["metric"]["gpu"], {})["mean_utilization_pct"] = 100 * float(r["value"][1])
    for r in n:
        out.setdefault(r["metric"]["gpu"], {})["samples"] = int(float(r["value"][1]))
    return out


def fmt(v, spec: str = ".1f") -> str:
    return "-" if v is None else format(v, spec)


def gb(mb) -> float | None:
    return None if mb is None else mb * 2**20 / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prometheus", default="")
    a = ap.parse_args()
    st = stages(a.out)
    runs = {s: load(os.path.join(a.out, "runs", s, "train_summary.json")) for s in TRAIN_STAGES}
    base_eval = load(os.path.join(a.out, "runs", "base_eval.json"))
    vllm_eval = load(os.path.join(a.out, "runs", "vllm_eval_lora_bf16.json"))
    telemetry = {name: stage_telemetry(a.prometheus, s) for name, s in st.items()} if a.prometheus else {}

    rows = []
    for name, s in runs.items():
        if not s:
            continue
        ev = s.get("eval") or {}
        nvml_peak = max((v["peak_memory_used_mb"] for v in (s.get("nvml") or {}).values()), default=None)
        rows.append({
            "variant": name, "gpus": s["world_size"], "gpu": s.get("gpu_name"), "quant_bits": s["quant_bits"],
            "params_b": s.get("total_params_base", 0) / 1e9, "trainable_pct": s["trainable_pct"],
            "weights_gpu_gb": gb(s.get("weights_gpu_mb")), "peak_allocated_gb": gb(s.get("peak_allocated_mb_max_rank")),
            "peak_reserved_gb": gb(s.get("peak_reserved_mb")),
            "nvml_peak_gb": gb(s.get("nvml_peak_memory_used_mb_max_rank", nvml_peak)),
            "tokens_per_s": s.get("throughput_tokens_per_s"), "step_ms_p50": (s.get("step_ms") or {}).get("p50"),
            "train_min": s["train_s"] / 60, "exact_match": ev.get("exact_match"), "perplexity": ev.get("perplexity"),
            "eval_examples": ev.get("examples"), "first_loss": s["first_loss"],
            "final_loss": s["final_loss_mean_last10"], "adapter_mb": s.get("adapter_bytes", 0) / 1e6,
            "load_s": s.get("load_s"),
        })

    by = {r["variant"]: r for r in rows}
    comparisons = {}
    if "lora_bf16" in by and "qlora_nf4" in by:
        b, q = by["lora_bf16"], by["qlora_nf4"]
        for k in ("weights_gpu_gb", "peak_allocated_gb", "nvml_peak_gb"):
            if b[k] and q[k]:
                comparisons[f"nf4_vs_bf16_{k}_reduction_pct"] = 100 * (1 - q[k] / b[k])
        if b["exact_match"] is not None and q["exact_match"] is not None:
            comparisons["nf4_minus_bf16_exact_match_pts"] = 100 * (q["exact_match"] - b["exact_match"])
        if b["tokens_per_s"] and q["tokens_per_s"]:
            comparisons["nf4_vs_bf16_throughput_ratio"] = q["tokens_per_s"] / b["tokens_per_s"]
    for single, ddp in (("lora_bf16", "ddp_lora_bf16"), ("qlora_nf4", "ddp_qlora_nf4")):
        if single in by and ddp in by and by[single]["tokens_per_s"] and by[ddp]["tokens_per_s"]:
            comparisons[f"{ddp}_speedup_vs_1gpu"] = by[ddp]["tokens_per_s"] / by[single]["tokens_per_s"]

    summary = {"stages": st, "train": rows, "comparisons": comparisons, "telemetry": telemetry,
               "base_eval": {k: v for k, v in (base_eval or {}).items() if k != "samples"} or None,
               "vllm_eval": {k: v for k, v in (vllm_eval or {}).items() if k != "samples"} or None}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    p = print
    gpu = next((r["gpu"] for r in rows if r["gpu"]), "?")
    p(f"# GPU benchmark: {os.path.basename(os.path.abspath(a.out))}\n")
    p(f"GPU: {gpu}. Task: AG News topic classification (4 labels), exact match on held-out test rows.\n")
    p("## Stages\n")
    p("| stage | result | seconds |\n|---|---|---|")
    for name, s in st.items():
        p(f"| {name} | {'ok' if s['rc'] == 0 else 'FAILED rc=' + str(s['rc'])} | {s['seconds']} |")
    p("\n## Training (same data, steps, global batch, sequence cap, gradient checkpointing)\n")
    p("| variant | GPUs | params (B) | trainable % | weights on GPU (GB) | peak allocated (GB) | peak nvidia-smi (GB) "
      "| tok/s | step p50 (ms) | train (min) | loss first→last | exact match | ppl |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        p(f"| {r['variant']} | {r['gpus']} | {fmt(r['params_b'], '.2f')} | {fmt(r['trainable_pct'], '.2f')} "
          f"| {fmt(r['weights_gpu_gb'])} | {fmt(r['peak_allocated_gb'])} | {fmt(r['nvml_peak_gb'])} "
          f"| {fmt(r['tokens_per_s'], '.0f')} | {fmt(r['step_ms_p50'], '.0f')} | {fmt(r['train_min'])} "
          f"| {fmt(r['first_loss'], '.3f')}→{fmt(r['final_loss'], '.3f')} "
          f"| {fmt(r['exact_match'], '.3f')} ({r['eval_examples']}) | {fmt(r['perplexity'], '.3f')} |")
    if comparisons:
        p("\n## Comparisons\n")
        for k, v in comparisons.items():
            p(f"- {k}: {v:.2f}")
    p("\n## Evaluation\n")
    for label, e in (("base model, no adapter (hf)", base_eval), ("LoRA bf16 adapter via vLLM", vllm_eval)):
        if e:
            lat = e.get("latency_ms") or {}
            p(f"- {label}: exact match {fmt(e.get('exact_match'), '.3f')} on {e.get('examples')} rows, "
              f"ppl {fmt(e.get('perplexity'), '.3f')}, latency p50/p95/p99 {fmt(lat.get('p50'))}/"
              f"{fmt(lat.get('p95'))}/{fmt(lat.get('p99'))} ms")
        else:
            p(f"- {label}: -")
    if telemetry:
        p("\n## GPU telemetry from Prometheus (gpu_exporter, 1 s scrapes), per stage\n")
        p("| stage | GPU | peak memory used (GB) | mean utilization % | samples |\n|---|---|---|---|---|")
        for name, t in telemetry.items():
            if not t or "error" in t:
                p(f"| {name} | - | {(t or {}).get('error', '-')} | | |")
                continue
            for g, v in sorted(t.items()):
                p(f"| {name} | {g} | {fmt(v.get('peak_memory_used_gb'))} | {fmt(v.get('mean_utilization_pct'))} "
                  f"| {v.get('samples', '-')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
