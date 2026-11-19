#!/usr/bin/env python3
"""Milestone 3 benchmark: data loading, LoRA / QLoRA / full fine-tuning, DDP, evaluation.

Everything is measured on the machine it runs on; nothing is hard-coded. Each
training variant runs in its own process (the real CLI) so peak RSS is per
variant. Steps:

  1. generate the synthetic Arrow datasets (pretrain + reverse task);
  2. data pipeline: memory-mapped open time, heap allocated by Arrow, samples/s
     through the DataLoader with 0 and 2 workers;
  3. pre-train a base model (full training);
  4. fine-tune the base on the reverse task, same steps and global batch for
     every variant: full fine-tuning, LoRA fp32, LoRA bf16 autocast, QLoRA int8,
     QLoRA nf4, and LoRA on 2 DDP ranks (torchrun, Gloo on CPU);
  5. evaluate the untuned base and every result: exact match, perplexity,
     per-request generation latency.

  python3 benchmarks/bench_milestone3.py [--steps 1200] [--out benchmarks/results/milestone3_bench.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def sh(cmd: list[str], env: dict | None = None) -> str:
    e = dict(os.environ, PYTHONPATH=ROOT, **(env or {}))
    p = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def cli(*args: str, env: dict | None = None) -> str:
    return sh([sys.executable, "-m", "training", *args], env)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def bench_data(data: str) -> dict:
    import pyarrow as pa
    import torch

    from training.data import ArrowTokenDataset, LoaderConfig, build_dataloader, dataset_tokens

    path = os.path.join(data, "pretrain.arrow")
    before = pa.total_allocated_bytes()
    t0 = time.perf_counter()
    ds = ArrowTokenDataset(path, max_len=128)
    open_ms = (time.perf_counter() - t0) * 1000
    out = {"file": "pretrain.arrow", "rows": len(ds), "file_bytes": os.path.getsize(path),
           "tokens": dataset_tokens(ds), "memory_mapped": ds.is_memory_mapped, "open_ms": open_ms,
           "arrow_heap_bytes_after_open": pa.total_allocated_bytes() - before, "loader": []}
    for workers in (0, 2):
        loader = build_dataloader(ds, LoaderConfig(batch_size=32, num_workers=workers, prefetch_factor=4, seed=0))
        it = iter(loader)
        next(it)  # worker start-up is not throughput
        t0 = time.perf_counter()
        n = 32
        for b in it:
            n += b["input_ids"].shape[0]
        dt = time.perf_counter() - t0
        out["loader"].append({"num_workers": workers, "prefetch_factor": 4 if workers else None,
                              "pin_memory": loader.pin_memory, "samples": n, "samples_per_s": (n - 32) / dt})
        del loader, it
    out["arrow_heap_bytes_after_epochs"] = pa.total_allocated_bytes() - before
    return out


def evaluate(base: str, data: str, adapter: str = "", quant: int = 0, limit: int = 200) -> dict:
    args = ["eval", "--backend", "torch", "--base-model", base, "--data", os.path.join(data, "reverse_eval.arrow"),
            "--limit", str(limit)]
    if adapter:
        args += ["--adapter", adapter]
    if quant:
        args += ["--quant-bits", str(quant)]
    r = json.loads(cli(*args))
    r.pop("samples", None)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1200, help="fine-tuning steps per variant")
    ap.add_argument("--pretrain-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32, help="global batch size")
    ap.add_argument("--work", default=os.path.join(ROOT, "benchmarks", "results", "m3"))
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results", "milestone3_bench.json"))
    ap.add_argument("--skip", default="", help="comma-separated variant names to skip")
    a = ap.parse_args()

    import torch

    data, work = os.path.join(a.work, "data"), a.work
    env = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
           "platform": platform.platform(), "cpu_model": cpu_model(), "cpus": os.cpu_count(),
           "python": platform.python_version(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
           "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    info = json.loads(cli("info"))
    env.update({k: info[k] for k in ("bitsandbytes_usable", "vllm_usable", "pyarrow", "mlflow")})
    print(json.dumps(env, indent=2), flush=True)

    t0 = time.perf_counter()
    cli("prepare-data", "--out", data, "--task", "reverse")
    prep_s = time.perf_counter() - t0
    print(f"data prepared in {prep_s:.1f}s", flush=True)
    data_bench = bench_data(data)
    data_bench["prepare_s"] = prep_s
    print(json.dumps(data_bench, indent=2), flush=True)

    common = ["--train-data", os.path.join(data, "reverse_train.arrow"), "--eval-data",
              os.path.join(data, "reverse_eval.arrow"), "--steps", str(a.steps), "--warmup-steps", "50",
              "--log-every", "200", "--eval-examples", "200", "--tracking-uri", f"sqlite:///{work}/mlflow.db",
              "--experiment", "milestone3-bench"]
    base_dir = os.path.join(work, "base")
    cli("train", "--config", "training/configs/pretrain_tiny.json", "--train-data", os.path.join(data, "pretrain.arrow"),
        "--eval-data", os.path.join(data, "pretrain_eval.arrow"), "--output-dir", base_dir,
        "--steps", str(a.pretrain_steps), "--tracking-uri", f"sqlite:///{work}/mlflow.db",
        "--experiment", "milestone3-bench", "--run-name", "pretrain")
    base = json.load(open(os.path.join(base_dir, "train_summary.json")))
    base_model = os.path.join(base_dir, "model")
    print(f"base pre-trained: loss {base['first_loss']:.3f} -> {base['final_loss']:.3f}", flush=True)

    lora = ["--mode", "lora", "--lora-r", "8", "--lora-alpha", "16", "--lora-targets", TARGETS, "--lr", "5e-3",
            "--base-model", base_model]
    variants = [
        ("full_ft_fp32", ["--mode", "full", "--lr", "1e-3", "--base-model", base_model], {}, 1),
        ("lora_r8_fp32", lora, {}, 1),
        ("lora_r8_bf16", lora + ["--precision", "bf16"], {}, 1),
        ("qlora_int8", lora + ["--quant-bits", "8"], {}, 1),
        ("qlora_nf4", lora + ["--quant-bits", "4"], {}, 1),
        ("lora_r8_ddp2", lora, {"OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 2) // 2))}, 2),
    ]
    skip = set(filter(None, a.skip.split(",")))
    results = []
    reference = evaluate(base_model, data)
    print(f"untuned base: exact match {reference['exact_match']:.3f}, perplexity {reference['perplexity']:.1f}",
          flush=True)
    for name, args, venv, world in variants:
        if name in skip:
            continue
        out = os.path.join(work, name)
        flags = ["train", *common, *args, "--output-dir", out, "--run-name", name,
                 "--batch-size", str(a.batch_size // world)]
        t0 = time.perf_counter()
        if world > 1:
            sh([sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world}",
                "--master_addr=127.0.0.1", f"--master_port={free_port()}", "-m", "training", *flags], venv)
        else:
            cli(*flags, env=venv)
        wall = time.perf_counter() - t0
        s = json.load(open(os.path.join(out, "train_summary.json")))
        row = {"variant": name, "world_size": s["world_size"], "precision": s["precision"],
               "quant_bits": s["quant_bits"], "quant_backend": s.get("quant_backend"),
               "trainable_params": s["trainable_params"], "total_params": s["total_params"],
               "trainable_pct": s["trainable_pct"], "weight_bytes": s["weight_bytes"],
               "adapter_bytes": s.get("adapter_bytes"), "tokens_per_s": s["throughput_tokens_per_s"],
               "step_ms_p50": s["step_ms"]["p50"], "step_ms_p95": s["step_ms"]["p95"],
               "data_wait_ms_mean": s["data_wait_ms_mean"], "peak_rss_mb": s["peak_rss_mb_max_rank"],
               "baseline_rss_mb": s["baseline_rss_mb"], "rss_over_baseline_mb": s["peak_rss_over_baseline_mb"],
               "first_loss": s["first_loss"], "final_loss_last10": s["final_loss_mean_last10"],
               "eval_exact_match": s["eval"]["exact_match"], "eval_perplexity": s["eval"]["perplexity"],
               "gen_latency_ms_p50": s["eval"]["latency_ms"]["p50"],
               "gen_latency_ms_p95": s["eval"]["latency_ms"]["p95"], "wall_s": wall,
               "mlflow_run_id": s["mlflow_run_id"]}
        results.append(row)
        print(f"{name:14s} {row['tokens_per_s']:9.0f} tok/s  step p50 {row['step_ms_p50']:6.1f} ms  "
              f"p95 {row['step_ms_p95']:6.1f} ms  rss {row['peak_rss_mb']:6.0f} MB  "
              f"EM {row['eval_exact_match']:.3f}  ppl {row['eval_perplexity']:.3f}", flush=True)

    report = {"env": env, "params": vars(a), "data": data_bench,
              "pretrain": {k: base[k] for k in ("steps", "first_loss", "final_loss", "throughput_tokens_per_s",
                                                "step_ms", "peak_rss_mb", "total_params", "eval_loss")},
              "untuned_base_eval": reference, "variants": results}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out}")
    print("\n| variant | ranks | precision | trainable | weights | tok/s | step p50 / p95 ms | peak RSS MB | "
          "exact match | ppl | gen p50 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['variant']} | {r['world_size']} | {r['precision']} | {r['trainable_params']:,} "
              f"({r['trainable_pct']:.1f}%) | {r['weight_bytes'] / 2**20:.2f} MiB | {r['tokens_per_s']:,.0f} | "
              f"{r['step_ms_p50']:.1f} / {r['step_ms_p95']:.1f} | {r['peak_rss_mb']:.0f} (+{r['rss_over_baseline_mb']:.0f}) | "
              f"{r['eval_exact_match']:.3f} | {r['eval_perplexity']:.3f} | {r['gen_latency_ms_p50']:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
