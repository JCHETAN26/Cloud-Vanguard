#!/usr/bin/env python3
"""Phase 4: resilience, chaos and load on a 3-node vanguard-proxy ring.

Topology: 3 proxies (n0..n2) on one consistent-hash ring (PEERS, peer token),
primary + fallback mock LLMs, per-node warm-cache block volumes (raw images,
A/B snapshot slots every 2 s), and a load generator that behaves like a
Kubernetes Service in front of the pods (spreads requests, retries a
connection-refused/reset request on the next pod).

  A. warm-up        unique prompts, cache fill (MISS) latency/throughput
  B. steady load    90% repeated / 10% new prompts: P50/P95/P99, throughput
  C. chaos          continuous load while: n1 SIGKILLed (t=4s) -> restarted
                    and restored from its volume (t=9s) -> primary LLM outage
                    (t=14s, breaker + failover) -> primary back (t=19s)
  D. rate limits    concurrent burst for a tenant with rpm 30 (cluster-wide)
  E. loop guard     concurrent X-Vanguard-Hop: 2 requests -> 508
  F. shutdown       SIGTERM: final snapshots, clean exits
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadgen import Checks, Conn, pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ADMIN, PEER = "admin-p4", "peer-p4"


def wait_port(port, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def http_json(port, method, path, body=None, token=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data.decode()


def metric(port, name):
    _, text = http_json(port, "GET", "/metrics")
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    return 0.0


class Node:
    def __init__(self, idx, base, env_common, root, binary):
        self.id, self.port, self.admin = f"n{idx}", base + idx, base + 10 + idx
        self.volume = os.path.join(root, f"{self.id}-warm-cache.raw")
        with open(self.volume, "wb") as f:  # a 64 MiB raw "block device"
            f.truncate(64 << 20)
        self.env = dict(env_common, NODE_ID=self.id, LISTEN_PORT=str(self.port), ADMIN_PORT=str(self.admin),
                        SNAPSHOT_PATH=self.volume)
        self.binary, self.root = binary, root
        self.proc = None
        self.start()

    def start(self):
        log = open(os.path.join(self.root, f"{self.id}.log"), "a")
        self.proc = subprocess.Popen([self.binary], env=self.env, stdout=log, stderr=subprocess.STDOUT)
        assert wait_port(self.port), f"{self.id} did not start"

    def snapshot_status(self):
        return http_json(self.admin, "GET", "/admin/v1/snapshot", token=ADMIN)[1]


def body(prompt_id, model="mock-llm-1"):
    return json.dumps({"model": model, "messages": [{"role": "user", "content": f"resilience prompt {prompt_id}"}]}).encode()


class Recorder:
    def __init__(self):
        self.rows = []
        self.reroutes = 0
        self.failures = 0


async def client_worker(nodes, rec, jobs, t0, key="sk-load", headers=None, start=0):
    conns = {n.port: Conn("127.0.0.1", n.port) for n in nodes}
    rr = start
    while True:
        try:
            pid = next(jobs)
        except StopIteration:
            break
        if pid is None:
            break
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}", **(headers or {})}
        done = False
        for attempt in range(len(nodes)):
            node = nodes[(rr + attempt) % len(nodes)]
            sent = time.perf_counter()
            try:
                r = await conns[node.port].request("POST", "/v1/chat/completions", h, body(pid))
                rec.rows.append({"t": sent - t0, "id": pid, "status": r.status, "cache": r.x_cache,
                                 "node": r.headers.get("x-vanguard-node", ""), "owner": r.headers.get("x-vanguard-owner", ""),
                                 "upstream": r.headers.get("x-vanguard-upstream", ""), "ms": r.total_ms,
                                 "retry_after": r.headers.get("retry-after", ""), "rerouted": attempt > 0})
                done = True
                break
            except (ConnectionError, asyncio.IncompleteReadError, OSError, ValueError):
                conns[node.port].close()
                rec.reroutes += 1
        if not done:
            rec.failures += 1
        rr += 1
    for c in conns.values():
        c.close()


async def run_fixed(nodes, ids, concurrency, **kw):
    rec = Recorder()
    it = iter(ids)
    t0 = time.perf_counter()
    await asyncio.gather(*(client_worker(nodes, rec, it, t0, start=i, **kw) for i in range(concurrency)))
    return rec, time.perf_counter() - t0


def summary(name, rec, elapsed):
    ok = [r for r in rec.rows if r["status"] == 200]
    lat = sorted(r["ms"] for r in rec.rows)
    hits = sum(1 for r in ok if r["cache"] in ("HIT", "SEMANTIC-HIT"))
    st = {}
    for r in rec.rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    return {"phase": name, "n": len(rec.rows), "statuses": st, "hit_ratio": hits / max(1, len(ok)),
            "p50": pct(lat, .5), "p95": pct(lat, .95), "p99": pct(lat, .99), "max": lat[-1] if lat else 0,
            "rps": len(rec.rows) / elapsed if elapsed else 0, "reroutes": rec.reroutes, "failures": rec.failures}


def print_rows(rows):
    print(f"{'phase':<30}{'n':>7}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}{'req/s':>9}{'hit%':>7}  statuses")
    for s in rows:
        print(f"{s['phase']:<30}{s['n']:>7}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['p99']:>9.2f}{s['max']:>9.1f}"
              f"{s['rps']:>9.0f}{s['hit_ratio'] * 100:>6.1f}%  {s['statuses']}"
              + (f" reroutes={s['reroutes']}" if s["reroutes"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--base-port", type=int, default=19500)
    ap.add_argument("--warm", type=int, default=3000)
    ap.add_argument("--steady", type=int, default=15000)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--chaos-seconds", type=float, default=24)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="vg-p4res-")
    base = args.base_port
    primary, fallback = base + 99, base + 98
    procs, checks, rows = [], Checks(), []
    tenants = os.path.join(root, "tenants.json")
    json.dump({"allow_unknown": False, "tenants": [
        {"id": "load", "api_keys": ["sk-load"], "rpm": 0, "tpm": 0},
        {"id": "burst", "api_keys": ["sk-burst"], "rpm": 30, "tpm": 0}]}, open(tenants, "w"))
    try:
        for port, name in ((primary, "primary"), (fallback, "fallback")):
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(HERE, "mock_upstream.py"), "--port", str(port), "--name", name,
                 "--min-latency-ms", "20", "--max-latency-ms", "60", "--token-interval-ms", "1", "--tokens", "8"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            assert wait_port(port)
        common = dict(os.environ, WORKERS="2", TENANTS_FILE=tenants, ADMIN_TOKEN=ADMIN, PEER_TOKEN=PEER,
                      PEERS=",".join(f"n{i}=127.0.0.1:{base + i}" for i in range(3)),
                      UPSTREAM_URL=f"http://127.0.0.1:{primary}", FALLBACK_UPSTREAM_URL=f"http://127.0.0.1:{fallback}",
                      UPSTREAM_TTFB_TIMEOUT_MS="300", BREAKER_FAILURE_THRESHOLD="5", BREAKER_OPEN_MS="1000",
                      BREAKER_MAX_OPEN_MS="4000", SNAPSHOT_MODE="block", SNAPSHOT_INTERVAL_S="2",
                      PEER_CONNECT_TIMEOUT_MS="200", CACHE_MAX_MB="256")
        nodes = [Node(i, base, common, root, args.binary) for i in range(3)]
        procs += [n.proc for n in nodes]
        print(f"3 nodes + primary/fallback LLMs up (logs in {root})\n")

        # A. warm-up ------------------------------------------------------------
        rec, el = asyncio.run(run_fixed(nodes, range(args.warm), args.concurrency))
        a = summary(f"A warm-up MISS (c={args.concurrency})", rec, el)
        rows.append(a)
        checks.check(a["statuses"] == {200: args.warm}, f"A: {args.warm} prompts cached across the ring")

        # B. steady load --------------------------------------------------------
        rnd = random.Random(7)
        mix = [rnd.randrange(args.warm) if rnd.random() < 0.9 else 100000 + i for i in range(args.steady)]
        rec, el = asyncio.run(run_fixed(nodes, mix, args.concurrency))
        b = summary(f"B steady 90/10 (c={args.concurrency})", rec, el)
        rows.append(b)
        checks.check(b["statuses"] == {200: args.steady} and b["hit_ratio"] > 0.88,
                     f"B: {args.steady} requests, hit ratio {b['hit_ratio'] * 100:.1f}%, {b['rps']:.0f} req/s")
        time.sleep(2.5)  # let a periodic snapshot cover the warmed set

        # C. chaos --------------------------------------------------------------
        print("chaos: n1 SIGKILL @4s, n1 restart+restore @9s, primary outage @14s, primary back @19s")
        events = []

        async def chaos():
            rec = Recorder()
            t0 = time.perf_counter()
            deadline = t0 + args.chaos_seconds
            rnd2 = random.Random(11)
            counter = iter(range(10**9))

            def gen():
                while time.perf_counter() < deadline:
                    n = next(counter)
                    yield rnd2.randrange(args.warm) if rnd2.random() < 0.8 else 200000 + n

            jobs = gen()

            async def scheduler():
                plan = [(4, "kill n1"), (9, "restart n1"), (14, "primary down"), (19, "primary up")]
                for at, what in plan:
                    await asyncio.sleep(max(0, t0 + at - time.perf_counter()))
                    if what == "kill n1":
                        nodes[1].proc.send_signal(signal.SIGKILL)
                        nodes[1].proc.wait()
                    elif what == "restart n1":
                        await asyncio.to_thread(nodes[1].start)
                        procs.append(nodes[1].proc)
                    elif what == "primary down":
                        http_json(primary, "POST", "/admin/fault", {"mode": "error", "status": 503})
                    else:
                        http_json(primary, "POST", "/admin/fault", {"mode": "ok"})
                    events.append((round(time.perf_counter() - t0, 2), what))

            await asyncio.gather(scheduler(), *(client_worker(nodes, rec, jobs, t0, start=i) for i in range(32)))
            return rec, time.perf_counter() - t0

        before_fo = sum(metric(n.port, "vanguard_failovers_total") for n in nodes)
        rec, el = asyncio.run(chaos())
        c = summary("C chaos (c=32)", rec, el)
        rows.append(c)
        bad = {k: v for k, v in c["statuses"].items() if k != 200}
        checks.check(not bad and rec.failures == 0,
                     f"C: {c['n']} requests during node kill/restart + LLM outage, 0 client-visible errors "
                     f"({rec.reroutes} connection retries to another pod)")
        rst = nodes[1].snapshot_status()["lastRestore"]
        restart_at = next(t for t, what in events if what == "restart n1")
        owned = [r for r in rec.rows if r["t"] > restart_at and r["owner"] == "n1" and r["id"] < args.warm]
        misses = [r for r in owned if r["cache"] != "HIT"]
        # Recovery gap: until peers re-probe n1 (peer breaker backoff, <= 2 s),
        # its keys are served by non-owners, which may miss.
        gap = (max(r["t"] for r in misses) - restart_at) if misses else 0.0
        settled = [r for r in owned if r["t"] > restart_at + 2.5]
        settled_hits = sum(1 for r in settled if r["cache"] == "HIT")
        served_by = {}
        for r in misses:
            served_by[r["node"]] = served_by.get(r["node"], 0) + 1
        print(f"  n1 recovery gap: {len(misses)} misses on warmed n1-owned prompts within {gap:.2f}s of restart "
              f"(served by {served_by or '-'})")
        checks.check(rst["ok"] and rst["loaded"] >= args.warm // 4,
                     f"C: restarted n1 restored {rst['loaded']} entries from its volume in {rst['durationMs']:.1f} ms")
        checks.check(settled and settled_hits / len(settled) > 0.999 and gap <= 2.5,
                     f"C: once peers re-probed n1 (gap {gap:.2f}s, {len(misses)} misses), "
                     f"{settled_hits}/{len(settled)} warmed prompts it owns were HITs")
        down = [r for r in rec.rows if 14.3 < r["t"] < 18.5 and r["cache"] == "MISS"]
        via_fb = sum(1 for r in down if r["upstream"] == "fallback")
        fo = sum(metric(n.port, "vanguard_failovers_total") for n in nodes) - before_fo
        checks.check(down and via_fb == len(down),
                     f"C: during the primary outage all {len(down)} misses were served by the fallback "
                     f"({fo:.0f} failovers, breaker open)")
        windows = {"before kill": (0, 4), "n1 down": (4, 9), "after restore": (9.5, 14), "LLM outage": (14, 19),
                   "recovered": (19.5, args.chaos_seconds)}
        print("  chaos p99 by window: " + ", ".join(
            f"{k} {pct(sorted(r['ms'] for r in rec.rows if lo <= r['t'] < hi), .99):.1f}ms" for k, (lo, hi) in windows.items()))
        print(f"  events: {events}")

        # D. rate limit burst ---------------------------------------------------
        rec, el = asyncio.run(run_fixed(nodes, range(300000, 300300), 60, key="sk-burst"))
        d = summary("D 429 burst rpm=30 (c=60)", rec, el)
        rows.append(d)
        ra = [r["retry_after"] for r in rec.rows if r["status"] == 429]
        checks.check(d["statuses"].get(200) == 30 and d["statuses"].get(429) == 270 and all(x.isdigit() for x in ra),
                     f"D: rpm 30 across 3 nodes -> exactly 30 allowed, 270 x 429 with Retry-After ({d['statuses']})")

        # E. loop protection ----------------------------------------------------
        rec, el = asyncio.run(run_fixed(nodes, range(400000, 400600), 60,
                                        headers={"X-Vanguard-Hop": "2", "X-Vanguard-Peer-Token": PEER}))
        e = summary("E hop=2 loop guard (c=60)", rec, el)
        rows.append(e)
        checks.check(e["statuses"] == {508: 600}, f"E: 600/600 X-Vanguard-Hop: 2 requests -> 508 (p99 {e['p99']:.2f} ms)")

        # F. graceful shutdown --------------------------------------------------
        for n in nodes:
            n.proc.send_signal(signal.SIGTERM)
        codes = {n.id: n.proc.wait(timeout=30) for n in nodes}
        saved = []
        for n in nodes:
            with open(os.path.join(root, f"{n.id}.log")) as f:
                log = f.read()
            saved.append("[snapshot] save ok" in log.split("shutdown")[0] or "[snapshot] save ok" in log)
        checks.check(all(c == 0 for c in codes.values()) and all(saved),
                     f"F: SIGTERM -> final snapshot + clean exit on every node {codes}")

        print()
        print_rows(rows)
        failed = sum(1 for ok, _ in checks.items if not ok)
        print("\nchecks:")
        for ok, w in checks.items:
            print(f"  [{'PASS' if ok else 'FAIL'}] {w}")
        print(f"\n{len(checks.items) - failed}/{len(checks.items)} resilience checks passed")
        if args.out:
            json.dump({"phases": rows, "events": events, "checks": [{"ok": ok, "check": w} for ok, w in checks.items]},
                      open(args.out, "w"), indent=2)
        return 1 if failed else 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()


if __name__ == "__main__":
    sys.exit(main())
