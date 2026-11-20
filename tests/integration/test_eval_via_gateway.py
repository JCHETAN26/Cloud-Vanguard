#!/usr/bin/env python3
"""Cross-subsystem: the training/ evaluation harness measured *through* the CloudVanguard gateway.

training.eval's OpenAI-HTTP backend (stdlib only; no torch needed) evaluates a
task set against vanguard-proxy in front of the mock LLM, twice:

  pass 1 (cold): every request is a miss and reaches the upstream; per-request
                 latency includes the upstream's fixed 30 ms;
  pass 2 (warm): identical prompts are served from the exact cache: no upstream
                 calls, latency drops, and every answer is byte-identical, so
                 evaluation results through the gateway are deterministic.

The proxy's /metrics must agree (misses = N, hits = N). Exits 0 on success.

    python3 tests/integration/test_eval_via_gateway.py --binary proxy-cpp/build/vanguard-proxy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from training.eval import OpenAIHTTPBackend, evaluate  # noqa: E402

N = 40
UPSTREAM_MS = 30


def wait_port(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def metric(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        m = re.match(rf"^{name}(\{{[^}}]*\}})? ([0-9.e+]+)$", line)
        if m:
            total += float(m.group(2))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--base-port", type=int, default=18700)
    a = ap.parse_args()
    port, admin, up = a.base_port, a.base_port + 1, a.base_port + 2
    log = open(os.path.join(tempfile.mkdtemp(prefix="vanguard-eval-"), "proxy.log"), "w")
    mock = subprocess.Popen([sys.executable, os.path.join(ROOT, "benchmarks", "mock_upstream.py"), "--port", str(up),
                             "--min-latency-ms", str(UPSTREAM_MS), "--max-latency-ms", str(UPSTREAM_MS), "--tokens", "6"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = dict(os.environ, LISTEN_PORT=str(port), ADMIN_PORT=str(admin), WORKERS="2", NODE_ID="eval",
               UPSTREAM_URL=f"http://127.0.0.1:{up}", DEFAULT_RPM="0", DEFAULT_TPM="0")
    proxy = subprocess.Popen([a.binary], env=env, stdout=log, stderr=subprocess.STDOUT)
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures.append(what)

    try:
        deadline = time.time() + 15
        if not (wait_port(port, deadline) and wait_port(up, deadline)):
            print("FAIL proxy or mock did not start")
            return 1
        examples = [{"prompt": f"rev word{i:02d} =>", "answer": f"{i:02d}drow", "text": f"rev word{i:02d} => {i:02d}drow"}
                    for i in range(N)]
        backend = OpenAIHTTPBackend(f"http://127.0.0.1:{port}/v1", model="eval-model", api_key="sk-eval")
        cold = evaluate(backend, examples, max_new_tokens=6, warmup=0, keep_samples=N)
        upstream_after_cold = json.loads(get(f"http://127.0.0.1:{up}/stats"))["requests"]
        warm = evaluate(backend, examples, max_new_tokens=6, warmup=0, keep_samples=N)
        upstream_after_warm = json.loads(get(f"http://127.0.0.1:{up}/stats"))["requests"]
        backend.close()
        metrics = get(f"http://127.0.0.1:{port}/metrics")

        print(f"cold: p50 {cold.latency_ms['p50']:.2f} ms  p95 {cold.latency_ms['p95']:.2f} ms  "
              f"warm: p50 {warm.latency_ms['p50']:.2f} ms  p95 {warm.latency_ms['p95']:.2f} ms")
        check(cold.examples == warm.examples == N, f"{N} examples evaluated per pass")
        check(upstream_after_cold == N, f"cold pass reached the upstream {upstream_after_cold}/{N} times")
        check(upstream_after_warm == N, f"warm pass made no upstream calls (total still {upstream_after_warm})")
        check(cold.latency_ms["p50"] >= UPSTREAM_MS, f"cold p50 includes the {UPSTREAM_MS} ms upstream")
        check(warm.latency_ms["p50"] < cold.latency_ms["p50"] / 3, "warm p50 is < 1/3 of cold p50")
        check([s["got"] for s in warm.samples] == [s["got"] for s in cold.samples] and len(warm.samples) == N,
              "every cached answer is identical to the upstream answer")
        check(warm.exact_match == cold.exact_match, "exact match is the same through the cache")
        check(metric(metrics, "vanguard_cache_misses_total") == N, "proxy /metrics: misses == N")
        check(metric(metrics, "vanguard_cache_hits_total") == N, "proxy /metrics: hits == N")
        proxy.send_signal(signal.SIGTERM)
        check(proxy.wait(timeout=15) == 0, "proxy exits 0 on SIGTERM")
    finally:
        for p in (proxy, mock):
            if p.poll() is None:
                p.kill()
                p.wait()
        log.close()
    if failures:
        print("--- proxy.log (tail) ---")
        print("".join(open(log.name, errors="replace").readlines()[-20:]))
        return 1
    print("PASS: evaluation through the gateway is cached, deterministic and upstream-free on repeat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
