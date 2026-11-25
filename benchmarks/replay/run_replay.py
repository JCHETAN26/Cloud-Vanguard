#!/usr/bin/env python3
"""Replay a labelled support workload through the gateway: no cache vs exact vs semantic caching.

Pipeline (everything local; the upstream is the mock LLM, so latency is simulated):

1. Tune. Replay the *tune* stream at each semantic threshold in a grid, for the
   lexical encoder and (if the embedding sidecar runs) the neural one. A fresh
   gateway per run. The chosen threshold is the one with the most correct reuses
   among those with zero incorrect reuses on the tune stream.
2. Test. Replay the held-out *test* stream once per mode, fresh gateway each:
     off        caching disabled (Cache-Control: no-cache, no-store on every request)
     exact      exact-match cache only (SEMANTIC_MODE=off)
     lexical    exact + lexical semantic cache at the tuned threshold
     neural     exact + bge-small semantic cache at the tuned threshold
   plus lexical/neural at the shipped defaults (0.94 / 0.96) for reference.

A reuse (X-Cache HIT or SEMANTIC-HIT) is *correct* when the cached answer was
produced for a request with the same label (intent + slot values + numbers in
the text), and *incorrect* otherwise. The mock LLM's response id is a hash of
the prompt, so every served id maps back to the request that filled the cache.

Requests are replayed sequentially, in stream order, from one client.

    python3 benchmarks/replay/run_replay.py --binary proxy-cpp/build/vanguard-proxy \\
        --tune tune.jsonl --test test.jsonl --out benchmarks/reports/replay.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEM = "You are the customer-support assistant of an online store."
KEY = "sk-replay"


def wait_port(port: int, timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def mock_requests(port: int) -> int:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5) as r:
        return int(json.loads(r.read())["requests"])


class Mock:
    def __init__(self, port: int, min_ms: float, max_ms: float):
        self.port = port
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "benchmarks", "mock_upstream.py"), "--port", str(port),
             "--min-latency-ms", str(min_ms), "--max-latency-ms", str(max_ms), "--tokens", "24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_port(port):
            raise RuntimeError("mock did not start")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=5)


def start_gateway(binary: str, port: int, upstream: int, env_extra: dict, logdir: str) -> subprocess.Popen:
    env = dict(os.environ, LISTEN_PORT=str(port), WORKERS="2", NODE_ID="replay",
               UPSTREAM_URL=f"http://127.0.0.1:{upstream}", DEFAULT_RPM="0", DEFAULT_TPM="0",
               UPSTREAM_TTFB_TIMEOUT_MS="10000", **env_extra)
    log = open(os.path.join(logdir, f"gateway-{port}-{int(time.time() * 1000)}.log"), "w")
    p = subprocess.Popen([binary], env=env, stdout=log, stderr=subprocess.STDOUT)
    if not wait_port(port, 15):
        p.kill()
        raise RuntimeError("gateway did not start")
    return p


def replay(port: int, rows: list[dict], no_cache: bool) -> list[dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if no_cache:
        headers["Cache-Control"] = "no-cache, no-store"
    out = []
    for r in rows:
        body = json.dumps({"model": "support-bot", "temperature": 0,
                           "messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": r["text"]}]})
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        ms = (time.perf_counter() - t0) * 1000.0
        rid = None
        if resp.status == 200:
            rid = json.loads(data).get("id")
        out.append({"status": resp.status, "x_cache": resp.getheader("X-Cache"),
                    "similarity": resp.getheader("X-Vanguard-Similarity"), "id": rid, "ms": ms})
    conn.close()
    return out


def score(rows: list[dict], res: list[dict], upstream_calls: int) -> dict:
    source: dict[str, dict] = {}  # response id -> the request whose miss produced it
    correct = incorrect = 0
    by_cat: dict[str, dict[str, int]] = {}
    words_avoided = 0
    errors = []
    for r, x in zip(rows, res):
        cat = by_cat.setdefault(r["category"], {"requests": 0, "reused_correct": 0, "reused_incorrect": 0})
        cat["requests"] += 1
        if x["x_cache"] in ("HIT", "SEMANTIC-HIT"):
            src = source.get(x["id"])
            words_avoided += r["ref_words"]
            if src is not None and src["label"] == r["label"]:
                correct += 1
                cat["reused_correct"] += 1
            else:
                incorrect += 1
                cat["reused_incorrect"] += 1
                if len(errors) < 25:
                    errors.append({"request": r["text"], "label": r["label"],
                                   "served_from": src["text"] if src else None,
                                   "served_label": src["label"] if src else None,
                                   "x_cache": x["x_cache"], "similarity": x["similarity"]})
        elif x["id"] is not None and x["id"] not in source:
            source[x["id"]] = r
    ms = sorted(x["ms"] for x in res)
    statuses: dict[str, int] = {}
    for x in res:
        statuses[str(x["status"])] = statuses.get(str(x["status"]), 0) + 1
    kinds: dict[str, int] = {}
    for x in res:
        kinds[str(x["x_cache"])] = kinds.get(str(x["x_cache"]), 0) + 1
    return {"requests": len(rows), "statuses": statuses, "x_cache": kinds, "upstream_calls": upstream_calls,
            "reuses": correct + incorrect, "correct_reuses": correct, "incorrect_reuses": incorrect,
            "reference_answer_words_avoided": words_avoided,
            "latency_ms": {"p50": round(statistics.median(ms), 1), "p95": round(ms[int(0.95 * (len(ms) - 1))], 1),
                           "mean": round(statistics.fmean(ms), 1)},
            "by_category": by_cat, "incorrect_examples": errors}


def run_mode(a, rows, mock: Mock, env_extra: dict, no_cache: bool, logdir: str) -> dict:
    gw = start_gateway(a.binary, a.base_port + 1, mock.port, env_extra, logdir)
    try:
        before = mock_requests(mock.port)
        res = replay(a.base_port + 1, rows, no_cache)
        return score(rows, res, mock_requests(mock.port) - before)
    finally:
        gw.terminate()
        gw.wait(timeout=5)


def semantic_env(mode: str, threshold: float, embedder_port: int) -> dict:
    env = {"SEMANTIC_MODE": mode, "SEMANTIC_THRESHOLD": f"{threshold:.2f}"}
    if mode == "neural":
        env.update(EMBEDDER_URL=f"http://127.0.0.1:{embedder_port}", EMBEDDER_TIMEOUT_MS="5000")
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--tune", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-port", type=int, default=19400)
    ap.add_argument("--neural", action="store_true", help="also run the bge-small sidecar (needs fastembed)")
    ap.add_argument("--upstream-min-ms", type=float, default=200)
    ap.add_argument("--upstream-max-ms", type=float, default=600)
    a = ap.parse_args()
    tune = [json.loads(line) for line in open(a.tune)]
    test = [json.loads(line) for line in open(a.test)]
    logdir = tempfile.mkdtemp(prefix="vanguard-replay-")
    embedder_port = a.base_port + 2
    procs = []
    results: dict = {"method": {
        "tune_requests": len(tune), "test_requests": len(test), "client": "1 client, sequential, stream order",
        "upstream": f"mock LLM, uniform {a.upstream_min_ms:.0f}-{a.upstream_max_ms:.0f} ms (simulated)",
        "gateway": "2 workers, fresh process (empty cache) per run",
        "threshold_rule": "most correct reuses with zero incorrect reuses on the tune stream"}}
    modes = ["lexical"] + (["neural"] if a.neural else [])
    try:
        if a.neural:
            procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "benchmarks", "embedder_sidecar.py"),
                                           "--port", str(embedder_port)],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if not wait_port(embedder_port, 300):
                raise RuntimeError("embedding sidecar did not start")

        # 1. Tune (latency is irrelevant here: a fast mock).
        fast = Mock(a.base_port, 1, 1)
        grids = {"lexical": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
                 "neural": [0.70, 0.75, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]}
        results["tune"] = {}
        chosen = {}
        for mode in modes:
            sweep = []
            for t in grids[mode]:
                s = run_mode(a, tune, fast, semantic_env(mode, t, embedder_port), False, logdir)
                sweep.append({"threshold": t, "correct_reuses": s["correct_reuses"],
                              "incorrect_reuses": s["incorrect_reuses"], "upstream_calls": s["upstream_calls"]})
                print(f"tune {mode} {t:.2f}: correct {s['correct_reuses']} incorrect {s['incorrect_reuses']}",
                      flush=True)
            safe = [x for x in sweep if x["incorrect_reuses"] == 0]
            best = max(safe, key=lambda x: (x["correct_reuses"], -x["threshold"])) if safe else None
            chosen[mode] = best["threshold"] if best else None
            # A choice at the bottom of the grid means the tune stream never
            # produced an error: the threshold is not really constrained.
            results["tune"][mode] = {"sweep": sweep, "chosen_threshold": chosen[mode],
                                     "chosen_at_grid_edge": chosen[mode] == grids[mode][0]}
        exact_tune = run_mode(a, tune, fast, {"SEMANTIC_MODE": "off"}, False, logdir)
        results["tune"]["exact"] = {"correct_reuses": exact_tune["correct_reuses"],
                                    "incorrect_reuses": exact_tune["incorrect_reuses"]}
        fast.stop()

        # 2. Test on the held-out stream with a realistic (simulated) upstream delay.
        mock = Mock(a.base_port, a.upstream_min_ms, a.upstream_max_ms)
        results["test"] = {}
        plan = [("off", {"SEMANTIC_MODE": "off"}, True), ("exact", {"SEMANTIC_MODE": "off"}, False)]
        for mode in modes:
            if chosen[mode] is not None:
                plan.append((f"{mode}@{chosen[mode]:.2f} (tuned)", semantic_env(mode, chosen[mode], embedder_port),
                             False))
            default = 0.94 if mode == "lexical" else 0.96
            if chosen[mode] != default:
                plan.append((f"{mode}@{default:.2f} (default)", semantic_env(mode, default, embedder_port), False))
        for name, env, no_cache in plan:
            s = run_mode(a, test, mock, env, no_cache, logdir)
            results["test"][name] = s
            print(f"test {name}: upstream {s['upstream_calls']} correct {s['correct_reuses']} "
                  f"incorrect {s['incorrect_reuses']} p95 {s['latency_ms']['p95']} ms", flush=True)
        mock.stop()
    finally:
        for p in procs:
            p.terminate()
    off = results["test"]["off"]["upstream_calls"]
    for s in results["test"].values():
        s["upstream_calls_avoided_vs_off"] = off - s["upstream_calls"]
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}; gateway logs in {logdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
