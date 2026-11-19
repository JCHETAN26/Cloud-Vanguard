#!/usr/bin/env python3
"""Milestone 1 benchmark: latency/throughput baseline and tracing overhead.

For each configuration it starts a fresh vanguard-proxy (same binary, same
mock upstream) and measures, over real local sockets:

  miss   N unique prompts (every request goes to the mock LLM upstream)
  hit    H requests cycling over the same prompts (all served from cache),
         at low concurrency for latency (asyncio client)
  hey    optional: `hey` (Go HTTP load generator) on one cached prompt for
         maximum hit throughput, if `hey` is on PATH

Configurations (run in this order; `off` is repeated last to show run-to-run
noise on the same machine):

  off        tracing disabled (no OTLP endpoint)
  on-100     OTLP export, every request sampled (parentbased_always_on)
  on-100-unreachable  every request sampled, collector unreachable (spans built,
             serialized and failed): the proxy-side cost without the receiver
  on-10      OTLP export, 10% sampled (parentbased_traceidratio 0.1)
  off-again  tracing disabled again

Spans go to observability/tools/otlp_sink.py. Everything reported is measured
in this run; nothing is estimated. Results: --out JSON plus a markdown table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import loadgen  # noqa: E402

SINK = os.path.join(ROOT, "observability", "tools", "otlp_sink.py")
KEY = "sk-bench-0001"

CONFIGS = [
    ("off", {}),
    ("on-100", {"OTEL_TRACES_SAMPLER": "parentbased_always_on"}),
    # Same spans and serialization, but the collector is unreachable (connect
    # refused): separates the proxy's own tracing cost from the CPU the OTLP
    # receiver uses on this shared machine.
    ("on-100-unreachable", {"OTEL_TRACES_SAMPLER": "parentbased_always_on", "_endpoint": "http://127.0.0.1:1"}),
    ("on-10", {"OTEL_TRACES_SAMPLER": "parentbased_traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0.1"}),
    ("off-again", {}),
]


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def wait_healthy(url: str, seconds: float = 20) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            get(url)
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"{url} did not become healthy")


def metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    return 0.0


def run_hey(port: int, seconds: int, concurrency: int) -> dict | None:
    hey = shutil.which("hey") or (os.path.expanduser("~/go/bin/hey") if os.path.exists(os.path.expanduser("~/go/bin/hey")) else None)
    if not hey:
        return None
    body = loadgen.chat_body("benchmark question 0 about distributed caches")
    out = subprocess.run(
        [hey, "-z", f"{seconds}s", "-c", str(concurrency), "-m", "POST", "-T", "application/json",
         "-H", f"Authorization: Bearer {KEY}", "-d", body.decode(), f"http://127.0.0.1:{port}/v1/chat/completions"],
        capture_output=True, text=True, check=False).stdout
    res = {"tool": "hey", "seconds": seconds, "concurrency": concurrency}
    m = re.search(r"Requests/sec:\s+([\d.]+)", out)
    res["rps"] = float(m.group(1)) if m else None
    for q in ("50", "95", "99"):
        m = re.search(rf"\s{q}% in ([\d.]+) secs", out)
        res[f"p{q}_ms"] = float(m.group(1)) * 1000 if m else None
    # Only the "Status code distribution" section lists status codes; the
    # latency histogram above it also uses [n] for bucket counts.
    codes = dict(re.findall(r"^\s+\[(\d{3})\]\s+(\d+) responses", out.split("Status code distribution:")[-1], re.M))
    res["status_codes"] = {k: int(v) for k, v in codes.items()}
    res["ok_responses"] = res["status_codes"].get("200", 0)
    res["non_200"] = sorted(k for k in codes if k != "200")
    res["errors"] = "Error distribution:" in out
    return res


async def measure(port: int, prompts: list[str], hits: int, miss_conc: int, hit_conc: int) -> dict:
    jobs = [loadgen.Job(KEY, loadgen.chat_body(p)) for p in prompts]
    res, err, el = await loadgen.run_jobs("127.0.0.1", port, jobs, miss_conc)
    miss = loadgen.summarize("miss", res, el)
    miss["errors"] = len(err)
    jobs = [loadgen.Job(KEY, loadgen.chat_body(prompts[i % len(prompts)])) for i in range(hits)]
    res, err, el = await loadgen.run_jobs("127.0.0.1", port, jobs, hit_conc)
    hit = loadgen.summarize("hit", res, el)
    hit["errors"] = len(err)
    return {"miss": miss, "hit": hit}


def env_info(binary: str) -> dict:
    info = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "kernel": platform.release(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "binary": binary,
    }
    try:
        info["git_sha"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout
        info["git_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return info


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--port", type=int, default=8380)
    ap.add_argument("--upstream-port", type=int, default=9380)
    ap.add_argument("--sink-port", type=int, default=4380)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--unique", type=int, default=2000, help="unique prompts (miss phase)")
    ap.add_argument("--hits", type=int, default=20000, help="requests in the hit phase")
    ap.add_argument("--miss-concurrency", type=int, default=32)
    ap.add_argument("--hit-concurrency", type=int, default=8)
    ap.add_argument("--hey-seconds", type=int, default=10)
    ap.add_argument("--hey-concurrency", type=int, default=32)
    ap.add_argument("--min-latency-ms", type=float, default=20)
    ap.add_argument("--max-latency-ms", type=float, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results", "milestone1_bench.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logdir = os.path.dirname(args.out)

    procs = []
    report = {"env": env_info(args.binary), "params": vars(args), "runs": []}
    try:
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_upstream.py"), "--port", str(args.upstream_port),
             "--min-latency-ms", str(args.min_latency_ms), "--max-latency-ms", str(args.max_latency_ms),
             "--token-interval-ms", "0", "--tokens", "24"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([sys.executable, SINK, "--port", str(args.sink_port)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        wait_healthy(f"http://127.0.0.1:{args.upstream_port}/healthz")
        wait_healthy(f"http://127.0.0.1:{args.sink_port}/healthz")

        for idx, (name, otel) in enumerate(CONFIGS):
            env = dict(os.environ, LISTEN_PORT=str(args.port), ADMIN_PORT="0", WORKERS=str(args.workers),
                       UPSTREAM_URL=f"http://127.0.0.1:{args.upstream_port}", DEFAULT_RPM="0", DEFAULT_TPM="0",
                       CACHE_MAX_MB="512", NODE_ID="bench")
            if otel:
                endpoint = otel.get("_endpoint", f"http://127.0.0.1:{args.sink_port}")
                env.update(OTEL_EXPORTER_OTLP_ENDPOINT=endpoint, OTEL_BSP_SCHEDULE_DELAY="200",
                           OTEL_BSP_MAX_QUEUE_SIZE="65536", OTEL_BSP_MAX_EXPORT_BATCH_SIZE="2048",
                           **{k: v for k, v in otel.items() if not k.startswith("_")})
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{args.sink_port}/spans", method="DELETE"))
            log = open(os.path.join(logdir, f"milestone1_proxy_{name}.log"), "w")
            proxy = subprocess.Popen([args.binary], env=env, stdout=log, stderr=subprocess.STDOUT)
            run = {"config": name, "otel": otel}
            try:
                wait_healthy(f"http://127.0.0.1:{args.port}/healthz")
                # Distinct prompt set per run so every run starts with a cold cache
                # and the miss phase really reaches the upstream.
                prompts = [f"benchmark question {i} run {idx} about distributed caches" for i in range(args.unique)]
                prompts[0] = "benchmark question 0 about distributed caches"  # the hey prompt
                print(f"\n== {name}: miss x{args.unique} (c={args.miss_concurrency}), "
                      f"hit x{args.hits} (c={args.hit_concurrency})", flush=True)
                run.update(asyncio.run(measure(args.port, prompts, args.hits, args.miss_concurrency,
                                               args.hit_concurrency)))
                run["hey"] = run_hey(args.port, args.hey_seconds, args.hey_concurrency)
                m = get(f"http://127.0.0.1:{args.port}/metrics")
                run["server_hit_histogram"] = loadgen.server_histogram(m, 'cache="hit",stream="false"')
                time.sleep(0.6)  # let the exporter drain (schedule delay 200 ms)
                m = get(f"http://127.0.0.1:{args.port}/metrics")
                run["tracing"] = {k: metric(m, f"vanguard_{k}") for k in (
                    "tracing_enabled", "trace_spans_started_total", "trace_spans_exported_total",
                    "trace_spans_dropped_total", "trace_spans_failed_total", "trace_export_batches_total")}
                run["tracing"]["sink"] = json.loads(get(f"http://127.0.0.1:{args.sink_port}/stats"))
                run["cache_hits_total"] = metric(m, "vanguard_cache_hits_total")
                run["cache_misses_total"] = metric(m, "vanguard_cache_misses_total")
            finally:
                proxy.terminate()
                run["exit_code"] = proxy.wait(timeout=15)
                log.close()
            report["runs"].append(run)
            h, mi, hy = run["hit"], run["miss"], run["hey"] or {}
            print(f"   miss p50/p95/p99 {fmt(mi['p50_ms'])}/{fmt(mi['p95_ms'])}/{fmt(mi['p99_ms'])} ms  "
                  f"{fmt(mi['rps'], 0)} req/s  statuses={mi['statuses']} x_cache={mi['x_cache']}")
            print(f"   hit  p50/p95/p99 {fmt(h['p50_ms'], 3)}/{fmt(h['p95_ms'], 3)}/{fmt(h['p99_ms'], 3)} ms  "
                  f"{fmt(h['rps'], 0)} req/s  statuses={h['statuses']} x_cache={h['x_cache']}")
            if hy:
                print(f"   hey  p50/p95/p99 {fmt(hy.get('p50_ms'), 3)}/{fmt(hy.get('p95_ms'), 3)}/"
                      f"{fmt(hy.get('p99_ms'), 3)} ms  {fmt(hy.get('rps'), 0)} req/s  codes={hy.get('status_codes')}")
            t = run["tracing"]
            print(f"   spans started={t['trace_spans_started_total']:.0f} exported={t['trace_spans_exported_total']:.0f} "
                  f"dropped={t['trace_spans_dropped_total']:.0f} failed={t['trace_spans_failed_total']:.0f} "
                  f"sink={t['sink']['spans']}  exit={run['exit_code']}")
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)

    # Sanity: every miss/hit request succeeded with the expected cache outcome.
    problems = []
    for r in report["runs"]:
        if r["miss"]["statuses"] != {200: args.unique} or r["miss"]["x_cache"].get("MISS") != args.unique:
            problems.append(f"{r['config']}: miss phase {r['miss']['statuses']} {r['miss']['x_cache']}")
        if r["hit"]["statuses"] != {200: args.hits} or r["hit"]["x_cache"].get("HIT") != args.hits:
            problems.append(f"{r['config']}: hit phase {r['hit']['statuses']} {r['hit']['x_cache']}")
        if r["hey"] and (r["hey"]["non_200"] or r["hey"]["errors"] or not r["hey"]["ok_responses"]):
            problems.append(f"{r['config']}: hey status codes {r['hey']['status_codes']} errors={r['hey']['errors']}")
        if r["exit_code"] != 0:
            problems.append(f"{r['config']}: proxy exit code {r['exit_code']}")
    report["problems"] = problems
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n| config | miss P50 / P95 / P99 (ms) | hit P50 / P95 / P99 (ms) | hit req/s (asyncio) "
          "| hey P50 / P95 / P99 (ms) | hey req/s | spans started / exported / dropped / failed |")
    print("|---|---|---|---|---|---|---|")
    for r in report["runs"]:
        mi, h, hy, t = r["miss"], r["hit"], r["hey"] or {}, r["tracing"]
        print(f"| {r['config']} | {fmt(mi['p50_ms'], 1)} / {fmt(mi['p95_ms'], 1)} / {fmt(mi['p99_ms'], 1)} "
              f"| {fmt(h['p50_ms'], 3)} / {fmt(h['p95_ms'], 3)} / {fmt(h['p99_ms'], 3)} | {fmt(h['rps'], 0)} "
              f"| {fmt(hy.get('p50_ms'), 3)} / {fmt(hy.get('p95_ms'), 3)} / {fmt(hy.get('p99_ms'), 3)} "
              f"| {fmt(hy.get('rps'), 0)} | {t['trace_spans_started_total']:.0f} / {t['trace_spans_exported_total']:.0f} / "
              f"{t['trace_spans_dropped_total']:.0f} / {t['trace_spans_failed_total']:.0f} |")
    print(f"\nenv: {report['env']}")
    print(f"results: {args.out}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
