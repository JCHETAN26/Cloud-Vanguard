#!/usr/bin/env python3
"""Request coalescing: a cold-cache burst of identical requests reaches the upstream once.

Two gateways in front of one mock LLM that answers in UPSTREAM_MS and serves at
most CAPACITY completions at a time (the rest queue, like a model server with a
fixed batch size). Both gateways run 4 workers, so a burst spreads across event
loops and coalescing has to work across threads.

  off    COALESCE=0: N identical requests -> N upstream calls
  on     COALESCE=1: N identical requests -> 1 upstream call, N-1 COALESCED answers,
         all bodies identical
  error  the upstream returns 500: 1 upstream call, every client gets the 500
         (no per-waiter retry stampede)
  rst    the leader's client resets its connection mid-call: the upstream call is
         aborted, one waiter takes over (2 upstream calls), every waiter gets 200
  close  the leader's client closes cleanly: the non-streaming call still
         completes (1 upstream call), every waiter gets 200
  stream stream=true is not coalesced: N streams -> N upstream calls

Prints client latency (p50/p95/max) for off/on; --json writes every result.
Exits 0 on success.

    python3 tests/integration/test_coalescing.py --binary proxy-cpp/build/vanguard-proxy
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = 100
UPSTREAM_MS = 600
CAPACITY = 8
KEY = "sk-coalesce"


def wait_port(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^(vanguard_coalesce_[a-z_]+_total) ([0-9.e+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def body_for(prompt: str, stream: bool = False) -> bytes:
    return json.dumps({"model": "m", "stream": stream, "messages": [{"role": "user", "content": prompt}]}).encode()


def request_bytes(prompt: str) -> bytes:
    b = body_for(prompt)
    return (f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(b)}\r\n\r\n").encode() + b


def one(port: int, prompt: str, stream: bool, start: threading.Barrier | None, out: list, i: int) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        conn.connect()
        if start is not None:
            start.wait()
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body_for(prompt, stream),
                     headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        out[i] = {"status": r.status, "x_cache": r.getheader("X-Cache"), "body": data,
                  "ms": (time.perf_counter() - t0) * 1000.0}
    except Exception as e:  # noqa: BLE001 - recorded and checked
        out[i] = {"status": 0, "x_cache": None, "body": b"", "ms": 0.0, "error": repr(e)}
    finally:
        conn.close()


def burst(port: int, prompt: str, n: int, stream: bool = False) -> list[dict]:
    out: list = [None] * n
    start = threading.Barrier(n)
    threads = [threading.Thread(target=one, args=(port, prompt, stream, start, out, i)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def start_waiters(port: int, prompt: str, n: int) -> tuple[list, list[threading.Thread]]:
    out: list = [None] * n
    threads = [threading.Thread(target=one, args=(port, prompt, False, None, out, i)) for i in range(n)]
    for t in threads:
        t.start()
    return out, threads


def upstream_requests(up: int) -> int:
    """Chat requests the mock has received, once nothing is in flight or queued."""
    deadline = time.time() + 30
    while time.time() < deadline:
        s = get_json(f"http://127.0.0.1:{up}/stats")
        if s["in_flight"] == 0:
            return int(s["requests"])
        time.sleep(0.05)
    raise RuntimeError("mock upstream never went idle")


def latency(rows: list[dict]) -> dict:
    ms = sorted(r["ms"] for r in rows)
    return {"p50_ms": round(statistics.median(ms), 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1),
            "max_ms": round(ms[-1], 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--base-port", type=int, default=18900)
    ap.add_argument("--json", help="write the measured results here")
    a = ap.parse_args()
    up, off_port, on_port = a.base_port, a.base_port + 1, a.base_port + 2
    tmp = tempfile.mkdtemp(prefix="vanguard-coalesce-")
    procs = [subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "benchmarks", "mock_upstream.py"), "--port", str(up),
         "--min-latency-ms", str(UPSTREAM_MS), "--max-latency-ms", str(UPSTREAM_MS), "--tokens", "8",
         "--max-concurrency", str(CAPACITY)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    for port, coalesce in ((off_port, "0"), (on_port, "1")):
        env = dict(os.environ, LISTEN_PORT=str(port), WORKERS="4", NODE_ID=f"coalesce{coalesce}",
                   UPSTREAM_URL=f"http://127.0.0.1:{up}", DEFAULT_RPM="0", DEFAULT_TPM="0", COALESCE=coalesce,
                   UPSTREAM_TTFB_TIMEOUT_MS="60000", SEMANTIC_MODE="off")
        log = open(os.path.join(tmp, f"proxy-{port}.log"), "w")
        procs.append(subprocess.Popen([a.binary], env=env, stdout=log, stderr=subprocess.STDOUT))
    failures: list[str] = []
    results: dict = {"n": N, "upstream_ms": UPSTREAM_MS, "upstream_capacity": CAPACITY, "workers": 4}

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures.append(what)

    try:
        deadline = time.time() + 15
        if not all(wait_port(p, deadline) for p in (up, off_port, on_port)):
            print("FAIL proxy or mock did not start")
            return 1

        # off: every request goes upstream.
        before = upstream_requests(up)
        rows = burst(off_port, "coalesce burst off", N)
        calls = upstream_requests(up) - before
        results["off"] = {"upstream_calls": calls, **latency(rows)}
        check(all(r["status"] == 200 for r in rows), f"off: all {N} answered 200")
        check(calls == N, f"off: {calls} upstream calls (expected {N})")

        # on: one upstream call for the whole burst.
        before = upstream_requests(up)
        rows = burst(on_port, "coalesce burst on", N)
        calls = upstream_requests(up) - before
        kinds = {k: sum(1 for r in rows if r["x_cache"] == k) for k in ("MISS", "COALESCED", "HIT")}
        results["on"] = {"upstream_calls": calls, "x_cache": kinds, **latency(rows)}
        check(all(r["status"] == 200 for r in rows), f"on: all {N} answered 200")
        check(calls == 1, f"on: {calls} upstream call(s) (expected 1)")
        check(kinds["MISS"] == 1 and kinds["COALESCED"] + kinds["HIT"] == N - 1,
              f"on: 1 MISS + {N - 1} COALESCED/HIT (got {kinds})")
        check(len({r["body"] for r in rows}) == 1, "on: every client got the same body")
        print(f"     off p50/p95 {results['off']['p50_ms']}/{results['off']['p95_ms']} ms, "
              f"on p50/p95 {results['on']['p50_ms']}/{results['on']['p95_ms']} ms")

        # error: the leader's 500 is shared, not retried per waiter.
        before = upstream_requests(up)
        rows = burst(on_port, "coalesce __mock_error_500__", N)
        calls = upstream_requests(up) - before
        results["error"] = {"upstream_calls": calls, "statuses": sorted({r["status"] for r in rows})}
        check(calls == 1, f"error: {calls} upstream call(s) (expected 1)")
        check(all(r["status"] == 500 for r in rows), f"error: all {N} clients got the 500")
        check(sum(1 for r in rows if r["x_cache"] == "COALESCED") == N - 1, f"error: {N - 1} COALESCED")

        # rst: the leader's client resets mid-call; a waiter takes over.
        for mode in ("rst", "close"):
            prompt = f"coalesce leader {mode}"
            before = upstream_requests(up)
            leader = socket.create_connection(("127.0.0.1", on_port))
            leader.sendall(request_bytes(prompt))
            time.sleep(0.15)  # leader is registered and its upstream call is out
            waiters, threads = start_waiters(on_port, prompt, 20)
            time.sleep(0.15)  # waiters have joined
            if mode == "rst":
                leader.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            leader.close()
            for t in threads:
                t.join()
            calls = upstream_requests(up) - before
            results[mode] = {"upstream_calls": calls, "statuses": sorted({w["status"] for w in waiters}),
                             "x_cache": sorted({str(w["x_cache"]) for w in waiters})}
            expected = 2 if mode == "rst" else 1
            check(all(w["status"] == 200 for w in waiters), f"{mode}: all 20 waiters answered 200")
            check(calls == expected, f"{mode}: {calls} upstream calls (expected {expected})")

        # stream: not coalesced.
        before = upstream_requests(up)
        rows = burst(on_port, "coalesce stream", 10, stream=True)
        calls = upstream_requests(up) - before
        results["stream"] = {"upstream_calls": calls}
        check(all(r["status"] == 200 for r in rows), "stream: all 10 answered 200")
        check(calls == 10, f"stream: {calls} upstream calls (expected 10)")

        m = metrics(on_port)
        results["metrics_on"] = m
        check(m.get("vanguard_coalesce_waiters_total", 0) >= 2 * (N - 1), "metrics: waiters counted")
        check(m.get("vanguard_coalesce_retries_total", 0) >= 1, "metrics: rst waiters retried")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "metrics_on"}))
    if failures:
        print(f"{len(failures)} check(s) failed; proxy logs in {tmp}")
        return 1
    print("all coalescing checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
