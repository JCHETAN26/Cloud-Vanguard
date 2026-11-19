#!/usr/bin/env python3
"""Phase 1 load generator and end-to-end verifier for vanguard-proxy.

Stdlib-only asyncio HTTP/1.1 client with keep-alive connections. Runs these
phases against a running proxy (+ mock upstream) and exits non-zero if any
check fails:

  1. MISS   N unique non-streaming prompts (every request goes upstream)
  2. HIT    the same N prompts again (every request served from cache)
  3. STREAM M unique streaming prompts, then the same again; verifies the
            replayed SSE bytes are identical and end with data: [DONE]
  4. 429    RPM limit (hits count) and TPM limit (only misses count)
  5. Auth, error-not-cached, and an upstream request-count invariant
            (upstream requests == responses marked X-Cache: MISS)

Client-side latency includes Python asyncio overhead, which grows with
concurrency, so the HIT phase defaults to low concurrency. The proxy's own
server-side histogram (/metrics) is reported alongside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import socket
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class Resp:
    status: int
    headers: dict
    body: bytes
    ttfb_ms: float
    total_ms: float

    @property
    def x_cache(self) -> str:
        return self.headers.get("x-cache", "-")


class Conn:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def open(self) -> None:
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port, limit=1 << 22)
        sock = self.writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.reader = self.writer = None

    async def request(self, method: str, path: str, headers: dict, body: bytes = b"") -> Resp:
        if self.writer is None:
            await self.open()
        lines = [f"{method} {path} HTTP/1.1", f"Host: {self.host}:{self.port}", f"Content-Length: {len(body)}"]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        raw = ("\r\n".join(lines) + "\r\n\r\n").encode() + body

        t0 = time.perf_counter()
        self.writer.write(raw)
        head = await self.reader.readuntil(b"\r\n\r\n")
        t_head = time.perf_counter()
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        status = int(status_line.split(" ", 2)[1])
        hdrs = {}
        for line in header_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                hdrs[k.strip().lower()] = v.strip()

        if "chunked" in hdrs.get("transfer-encoding", "").lower():
            parts = []
            while True:
                size_line = await self.reader.readline()
                size = int(size_line.split(b";")[0].strip(), 16)
                if size == 0:
                    while (await self.reader.readline()) not in (b"\r\n", b""):
                        pass
                    break
                parts.append((await self.reader.readexactly(size + 2))[:-2])
            data = b"".join(parts)
        elif "content-length" in hdrs:
            data = await self.reader.readexactly(int(hdrs["content-length"]))
        else:
            data = await self.reader.read()
        t_end = time.perf_counter()
        if hdrs.get("connection", "").lower() == "close":
            self.close()
        return Resp(status, hdrs, data, (t_head - t0) * 1000, (t_end - t0) * 1000)


@dataclass
class Job:
    key: str
    body: bytes
    extra_headers: dict = field(default_factory=dict)


async def run_jobs(host: str, port: int, jobs: list[Job], concurrency: int, keep_body: bool = False):
    queue: asyncio.Queue = asyncio.Queue()
    for i, j in enumerate(jobs):
        queue.put_nowait((i, j))
    results: list[Resp | None] = [None] * len(jobs)
    errors: list[str] = []

    async def worker() -> None:
        conn = Conn(host, port)
        while True:
            try:
                i, job = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {job.key}", **job.extra_headers}
            for attempt in range(2):
                try:
                    r = await conn.request("POST", "/v1/chat/completions", headers, job.body)
                    if not keep_body and r.status == 200:
                        r.body = b""
                    results[i] = r
                    break
                except (ConnectionError, asyncio.IncompleteReadError, OSError) as e:
                    conn.close()
                    if attempt == 1:
                        errors.append(f"{type(e).__name__}: {e}")
        conn.close()

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(max(1, min(concurrency, len(jobs))))))
    return results, errors, time.perf_counter() - t0


def chat_body(prompt: str, stream: bool = False, max_tokens: int | None = None) -> bytes:
    body = {"model": "mock-llm-1", "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": prompt}]}
    if stream:
        body["stream"] = True
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return json.dumps(body).encode()


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def summarize(name: str, results: list, elapsed: float, metric: str = "total_ms") -> dict:
    ok = [r for r in results if r is not None]
    lat = sorted(getattr(r, metric) for r in ok)
    caches: dict[str, int] = {}
    statuses: dict[int, int] = {}
    for r in ok:
        caches[r.x_cache] = caches.get(r.x_cache, 0) + 1
        statuses[r.status] = statuses.get(r.status, 0) + 1
    return {
        "phase": name,
        "requests": len(results),
        "completed": len(ok),
        "statuses": statuses,
        "x_cache": caches,
        "p50_ms": pct(lat, 0.50),
        "p95_ms": pct(lat, 0.95),
        "p99_ms": pct(lat, 0.99),
        "max_ms": lat[-1] if lat else float("nan"),
        "mean_ms": sum(lat) / len(lat) if lat else float("nan"),
        "rps": len(ok) / elapsed if elapsed > 0 else 0,
    }


def http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read().decode()


def server_histogram(metrics_text: str, labels: str, name: str = "vanguard_request_duration_seconds") -> dict:
    """Server-side percentile upper bounds from a Prometheus histogram."""
    buckets = []
    for line in metrics_text.splitlines():
        if line.startswith(name + "_bucket{") and labels in line:
            le = line.split('le="')[1].split('"')[0]
            buckets.append((math.inf if le == "+Inf" else float(le), float(line.rsplit(" ", 1)[1])))
    buckets.sort()
    total = buckets[-1][1] if buckets else 0
    out = {"count": int(total)}
    for q in (0.5, 0.95, 0.99):
        bound = next((b for b, c in buckets if total and c >= q * total), math.inf)
        out[f"p{int(q * 100)}_le_ms"] = bound * 1000
    return out


class Checks:
    def __init__(self):
        self.items: list[tuple[bool, str]] = []

    def check(self, ok: bool, what: str) -> bool:
        self.items.append((bool(ok), what))
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for ok, _ in self.items if not ok)


def fmt_row(s: dict) -> str:
    xc = " ".join(f"{k}={v}" for k, v in sorted(s["x_cache"].items()))
    st = " ".join(f"{k}={v}" for k, v in sorted(s["statuses"].items()))
    return (
        f"{s['phase']:<26}{s['completed']:>7}  {s['p50_ms']:>8.2f} {s['p95_ms']:>8.2f} {s['p99_ms']:>8.2f} "
        f"{s['max_ms']:>8.2f} {s['rps']:>9.0f}   {xc}  [{st}]"
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy", default="http://127.0.0.1:8080")
    ap.add_argument("--upstream", default="http://127.0.0.1:9000")
    ap.add_argument("--requests", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=64, help="concurrency for the MISS phase")
    ap.add_argument("--hit-concurrency", type=int, default=4, help="concurrency for the latency-measured HIT phase")
    ap.add_argument("--stream-requests", type=int, default=200)
    ap.add_argument("--key", default="sk-bench-0001")
    ap.add_argument("--rpm-key", default="sk-rpm-limited")
    ap.add_argument("--tpm-key", default="sk-tpm-limited")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pu = urlparse(args.proxy)
    host, port = pu.hostname, pu.port or 80
    run = format(int(time.time() * 1000) % (1 << 40), "x")
    checks = Checks()
    summaries: list[dict] = []
    upstream_before = json.loads(http_get(args.upstream + "/stats"))
    all_results: list[Resp] = []

    def track(results):
        all_results.extend(r for r in results if r is not None)

    print(f"run id {run}: {args.requests} requests per phase\n")

    # 1. MISS ------------------------------------------------------------------
    jobs = [Job(args.key, chat_body(f"[{run}] question #{i}: explain consistent hashing")) for i in range(args.requests)]
    res, errs, el = await run_jobs(host, port, jobs, args.concurrency)
    track(res)
    s_miss = summarize(f"json MISS (c={args.concurrency})", res, el)
    summaries.append(s_miss)
    checks.check(not errs, f"MISS phase transport errors: {len(errs)} {errs[:2]}")
    checks.check(s_miss["x_cache"].get("MISS") == args.requests, "every first-time prompt is X-Cache: MISS")

    # 2. HIT -------------------------------------------------------------------
    res, errs, el = await run_jobs(host, port, jobs, args.hit_concurrency)
    track(res)
    s_hit = summarize(f"json HIT  (c={args.hit_concurrency})", res, el)
    summaries.append(s_hit)
    checks.check(not errs, f"HIT phase transport errors: {len(errs)}")
    checks.check(s_hit["x_cache"].get("HIT") == args.requests, "every repeated prompt is X-Cache: HIT")
    checks.check(s_hit["p99_ms"] < 5.0, f"client-observed HIT P99 < 5 ms ({s_hit['p99_ms']:.3f} ms)")

    res, errs, el = await run_jobs(host, port, jobs, args.concurrency)
    track(res)
    s_hit_hi = summarize(f"json HIT  (c={args.concurrency})", res, el)
    summaries.append(s_hit_hi)

    # 3. STREAM ----------------------------------------------------------------
    sjobs = [Job(args.key, chat_body(f"[{run}] stream #{i}: what is SSE?", stream=True)) for i in range(args.stream_requests)]
    sc = min(args.concurrency, 32)
    res_m, errs_m, el = await run_jobs(host, port, sjobs, sc, keep_body=True)
    track(res_m)
    summaries.append(summarize(f"SSE MISS total (c={sc})", res_m, el))
    summaries.append(summarize(f"SSE MISS ttfb  (c={sc})", res_m, el, "ttfb_ms"))
    res_h, errs_h, el = await run_jobs(host, port, sjobs, args.hit_concurrency, keep_body=True)
    track(res_h)
    summaries.append(summarize(f"SSE HIT  total (c={args.hit_concurrency})", res_h, el))
    identical = sum(1 for a, b in zip(res_m, res_h) if a and b and a.body == b.body and a.body)
    frames = res_h[0].body.count(b"\n\n") if res_h and res_h[0] else 0
    checks.check(not errs_m and not errs_h, "SSE phases had no transport errors")
    checks.check(all(r and r.x_cache == "MISS" for r in res_m), "first SSE requests are X-Cache: MISS")
    checks.check(all(r and r.x_cache == "HIT" for r in res_h), "repeated SSE requests are X-Cache: HIT")
    checks.check(
        all(r and r.headers.get("content-type", "").startswith("text/event-stream") for r in res_h),
        "SSE hits are served as text/event-stream",
    )
    checks.check(identical == len(sjobs), f"SSE replay is byte-identical to the live stream ({identical}/{len(sjobs)})")
    checks.check(
        all(r and r.body.endswith(b"data: [DONE]\n\n") for r in res_h), f"SSE replay ends with data: [DONE] ({frames} events/stream)"
    )
    sample_stream = res_h[0].body.decode(errors="replace") if res_h and res_h[0] else ""

    # 4. RATE LIMITS -----------------------------------------------------------
    conn = Conn(host, port)
    hdr = lambda k: {"Content-Type": "application/json", "Authorization": f"Bearer {k}"}  # noqa: E731
    rpm_seq = []
    for _ in range(7):
        r = await conn.request("POST", "/v1/chat/completions", hdr(args.rpm_key), chat_body(f"[{run}] rpm probe"))
        rpm_seq.append(r)
    track(rpm_seq)
    codes = [f"{r.status}/{r.x_cache}" for r in rpm_seq]
    checks.check([r.status for r in rpm_seq] == [200] * 5 + [429] * 2, f"RPM=5 tenant: 5 allowed then 429 ({codes})")
    ra = rpm_seq[-1].headers.get("retry-after", "")
    checks.check(ra.isdigit() and int(ra) >= 1, f"429 carries Retry-After ({ra}s, reason={rpm_seq[-1].headers.get('x-ratelimit-reason')})")
    checks.check(rpm_seq[1].x_cache == "HIT", "RPM counts cache hits too (requests 2-5 were hits)")

    a = await conn.request("POST", "/v1/chat/completions", hdr(args.tpm_key), chat_body(f"[{run}] tpm A", max_tokens=600))
    b = await conn.request("POST", "/v1/chat/completions", hdr(args.tpm_key), chat_body(f"[{run}] tpm B", max_tokens=600))
    a2 = await conn.request("POST", "/v1/chat/completions", hdr(args.tpm_key), chat_body(f"[{run}] tpm A", max_tokens=600))
    track([a, b, a2])
    checks.check(a.status == 200 and a.x_cache == "MISS", "TPM=1000 tenant: first 600-token miss allowed")
    checks.check(
        b.status == 429 and b.headers.get("x-ratelimit-reason") == "tpm",
        f"second 600-token miss -> 429 tpm (Retry-After {b.headers.get('retry-after')}s)",
    )
    checks.check(a2.status == 200 and a2.x_cache == "HIT", "cache hit still served while TPM is exhausted (hits cost 0 tokens)")

    # 5. AUTH / ERRORS / INVARIANTS --------------------------------------------
    r = await conn.request("POST", "/v1/chat/completions", {"Content-Type": "application/json"}, chat_body("x"))
    checks.check(r.status == 401, f"missing bearer token -> 401 ({r.status})")
    r = await conn.request("POST", "/v1/chat/completions", hdr("sk-not-a-tenant"), chat_body("x"))
    checks.check(r.status == 401, f"unknown key -> 401 ({r.status})")
    r = await conn.request("POST", "/v1/chat/completions", hdr(args.key), b'{"model": 1}')
    checks.check(r.status == 400, f"invalid body -> 400 ({r.status})")
    e1 = await conn.request("POST", "/v1/chat/completions", hdr(args.key), chat_body(f"[{run}] __mock_error_500__"))
    e2 = await conn.request("POST", "/v1/chat/completions", hdr(args.key), chat_body(f"[{run}] __mock_error_500__"))
    track([e1, e2])
    checks.check(
        e1.status == 500 and e2.status == 500 and e2.x_cache == "MISS", "upstream 500s are passed through and never cached"
    )
    conn.close()

    upstream_after = json.loads(http_get(args.upstream + "/stats"))
    upstream_calls = upstream_after["requests"] - upstream_before["requests"]
    observed_misses = sum(1 for r in all_results if r.x_cache == "MISS")
    checks.check(
        upstream_calls == observed_misses,
        f"upstream saw exactly the MISS responses ({upstream_calls} upstream calls == {observed_misses} MISS)",
    )

    metrics = http_get(args.proxy + "/metrics")
    srv_hit = server_histogram(metrics, 'cache="hit",stream="false"')
    srv_hit_s = server_histogram(metrics, 'cache="hit",stream="true"')
    srv_miss = server_histogram(metrics, 'cache="miss",stream="false"')
    checks.check(srv_hit["p99_le_ms"] <= 5.0, f"server-side HIT P99 <= {srv_hit['p99_le_ms']:g} ms (<5 ms target)")

    # REPORT -------------------------------------------------------------------
    print(f"{'phase':<26}{'n':>7}  {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'req/s':>9}   (latency ms, client-observed)")
    print("-" * 118)
    for s in summaries:
        print(fmt_row(s))
    print()
    print("server-side (proxy histogram, upper bucket bounds):")
    for name, h in (("json HIT", srv_hit), ("SSE HIT", srv_hit_s), ("json MISS", srv_miss)):
        print(f"  {name:<10} n={h['count']:<7} P50<={h['p50_le_ms']:g}ms  P95<={h['p95_le_ms']:g}ms  P99<={h['p99_le_ms']:g}ms")
    speedup = s_miss["p50_ms"] / s_hit["p50_ms"] if s_hit["p50_ms"] else float("nan")
    print(f"\nP50 speedup HIT vs MISS: {speedup:,.0f}x   upstream calls avoided: {s_hit['completed'] + s_hit_hi['completed']}")
    print("\nSSE replay sample (first 2 events):")
    for ev in sample_stream.split("\n\n")[:2]:
        print("  " + ev[:150])
    print("\nchecks:")
    for ok, what in checks.items:
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    print(f"\n{len(checks.items) - checks.failed}/{len(checks.items)} checks passed")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "run": run,
                    "phases": summaries,
                    "server_side": {"hit": srv_hit, "hit_stream": srv_hit_s, "miss": srv_miss},
                    "checks": [{"ok": ok, "check": w} for ok, w in checks.items],
                },
                f,
                indent=2,
            )
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
