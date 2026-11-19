#!/usr/bin/env python3
"""Phase 2 scenarios: semantic cache and upstream failover.

Sections (pick with --section):

  semantic   lexical encoder: seed N prompts, then re-ask each with different
             phrasing (expect X-Cache: SEMANTIC-HIT), repeat the variants
             (expect exact HIT via aliasing), then ask N different questions
             (expect MISS: zero false positives).
  failover   primary + fallback mocks: baseline, primary 5xx, primary slow
             (>300 ms time to first byte), primary dropping connections,
             exponential-backoff probe spacing, recovery time, and both down.
  neural     remote encoder (bge-small sidecar): calibration table, true
             paraphrase hit rate and false positives through the proxy.
  degraded   remote encoder with the sidecar stopped: requests must still be
             served (exact-match only) and the embedder circuit must open.

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from urllib.parse import urlparse

from loadgen import Checks, Conn, Job, fmt_row, http_get, pct, run_jobs, server_histogram, summarize

KEY = "sk-bench-0001"
ADJ = ("quantum classical distributed neural relational functional reactive embedded parallel probabilistic "
       "symbolic linear nonlinear stochastic adaptive genetic cellular modular generative evolutionary "
       "cryptographic statistical numerical semantic lexical spectral thermal optical acoustic magnetic "
       "electric chemical biological cognitive social economic political historical geometric topological").split()
NOUN = ("computing networks databases programming algorithms compilers storage caching scheduling encryption "
        "learning inference routing consensus replication sharding indexing compression optimization simulation "
        "modeling rendering parsing testing debugging monitoring logging tracing profiling deployment "
        "virtualization containers kernels filesystems protocols sensors robotics vision speech translation "
        "planning search ranking recommendation forecasting clustering classification regression segmentation "
        "detection").split()

VARIANTS = [  # same question, different phrasing
    lambda a, n: f"Can you please explain {a} {n}?",
    lambda a, n: f"What is {a} {n}?",
    lambda a, n: f"  EXPLAIN {a.upper()} {n.upper()}!!! ",
    lambda a, n: f"Tell me about {a} {n}.",
    lambda a, n: f"explain {a} {n}, please",
]
DISTINCT = [  # different questions about the same topic
    lambda a, n: f"What are the risks of {a} {n}?",
    lambda a, n: f"How does {a} {n} fail?",
    lambda a, n: f"Explain the history of {a} {n}",
    lambda a, n: f"Why is {a} {n} not used?",
    lambda a, n: f"Explain {a} {n} in 3 sentences",
]

PARAPHRASES = [
    ("Explain quantum computing", "What is quantum computing in simple terms?"),
    ("How do I reverse a list in Python?", "What's the way to reverse a Python list?"),
    ("What causes the seasons on Earth?", "Why does Earth have seasons?"),
    ("How do vaccines train the immune system?", "In what way do vaccines teach the immune system?"),
    ("What is the difference between TCP and UDP?", "How does TCP differ from UDP?"),
    ("How can I improve my sleep quality?", "What are ways to sleep better?"),
    ("Summarize the causes of World War I", "What were the main causes of the First World War?"),
    ("What is a black hole?", "Can you explain what black holes are?"),
    ("How does photosynthesis work?", "Explain the process of photosynthesis"),
    ("What are the health benefits of green tea?", "Is green tea good for your health, and why?"),
    ("How do I center a div in CSS?", "What's the CSS to center a div?"),
    ("What is the capital of Australia?", "Which city is Australia's capital?"),
]
NEAR_MISSES = [
    ("What is the capital of Australia?", "What is the population of Australia?"),
    ("How do I reverse a list in Python?", "How do I sort a list in Python?"),
    ("What is the difference between TCP and UDP?", "What is the difference between HTTP and HTTPS?"),
    ("How does photosynthesis work?", "How does cellular respiration work?"),
    ("Explain quantum computing", "Explain quantum entanglement"),
    ("What causes the seasons on Earth?", "What causes the tides on Earth?"),
    ("How do I center a div in CSS?", "How do I center a div vertically in CSS?"),
    ("What are the health benefits of green tea?", "What are the health risks of green tea?"),
    ("Is it safe to eat raw eggs?", "Is it safe to eat raw chicken?"),
    ("Translate 'thank you' to Spanish", "Translate 'thank you' to Japanese"),
    ("Write a poem about autumn", "Write a poem about spring"),
    ("How do I reverse a list in Python?", "How do I reverse a list in JavaScript?"),
]


def body(system: str, user: str, stream: bool = False) -> bytes:
    b = {"model": "mock-llm-1", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if stream:
        b["stream"] = True
    return json.dumps(b).encode()


def post_json(url: str, obj) -> dict:
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def stats(upstream: str) -> dict:
    return json.loads(http_get(upstream + "/stats"))


def set_fault(upstream: str, **fault) -> None:
    post_json(upstream + "/admin/fault", fault)


def hostport(url: str) -> tuple[str, int]:
    u = urlparse(url)
    return u.hostname, u.port or 80


def counts(results, header: str) -> dict:
    out: dict[str, int] = {}
    for r in results:
        if r is not None:
            k = r.headers.get(header, "-")
            out[k] = out.get(k, 0) + 1
    return out


def metric(text: str, name: str, labels: str = "") -> float:
    for line in text.splitlines():
        if line.startswith(name) and (not labels or labels in line) and not line.startswith("#"):
            head = line.rsplit(" ", 1)[0]
            if head == name or head.startswith(name + "{"):
                return float(line.rsplit(" ", 1)[1])
    return float("nan")


# ---------------------------------------------------------------------------
# semantic (lexical encoder)
# ---------------------------------------------------------------------------

async def section_semantic(args, checks: Checks, run: str) -> list:
    host, port = hostport(args.proxy)
    system = f"You are terse. [run {run}]"
    n = min(args.requests, len(ADJ) * len(NOUN))
    topics = [(ADJ[i % len(ADJ)], NOUN[i // len(ADJ)]) for i in range(n)]
    before = stats(args.upstream)["requests"]
    rows, all_res = [], []

    seed = [Job(KEY, body(system, f"Explain {a} {t}")) for a, t in topics]
    res, errs, el = await run_jobs(host, port, seed, args.concurrency)
    all_res += res
    rows.append(summarize(f"seed MISS (c={args.concurrency})", res, el))
    checks.check(not errs and counts(res, "x-cache").get("MISS") == n, f"{n} seed prompts are MISS")

    variants = [Job(KEY, body(system, VARIANTS[i % len(VARIANTS)](a, t))) for i, (a, t) in enumerate(topics)]
    res, errs, el = await run_jobs(host, port, variants, args.hit_concurrency)
    all_res += res
    s = summarize(f"rephrased SEMANTIC (c={args.hit_concurrency})", res, el)
    rows.append(s)
    sims = sorted(float(r.headers.get("x-vanguard-similarity", "nan")) for r in res if r)
    checks.check(counts(res, "x-cache").get("SEMANTIC-HIT") == n,
                 f"all {n} rephrasings are SEMANTIC-HIT ({counts(res, 'x-cache')})")
    checks.check(sims and sims[0] >= 0.94, f"every semantic hit has similarity >= 0.94 (min {sims[0] if sims else 'n/a'})")
    checks.check(s["p99_ms"] < 5.0, f"client-observed SEMANTIC-HIT P99 < 5 ms ({s['p99_ms']:.3f} ms)")

    res, errs, el = await run_jobs(host, port, variants, args.hit_concurrency)
    all_res += res
    rows.append(summarize(f"rephrased again HIT (c={args.hit_concurrency})", res, el))
    checks.check(counts(res, "x-cache").get("HIT") == n, "repeating a rephrasing is an exact HIT (aliased)")

    distinct = [Job(KEY, body(system, DISTINCT[i % len(DISTINCT)](a, t))) for i, (a, t) in enumerate(topics)]
    res, errs, el = await run_jobs(host, port, distinct, args.concurrency)
    all_res += res
    rows.append(summarize(f"different Qs MISS (c={args.concurrency})", res, el))
    fp = counts(res, "x-cache").get("SEMANTIC-HIT", 0) + counts(res, "x-cache").get("HIT", 0)
    checks.check(fp == 0, f"0 false positives across {n} different questions on the same topics ({fp})")

    calls = stats(args.upstream)["requests"] - before
    misses = sum(1 for r in all_res if r and r.x_cache == "MISS")
    checks.check(calls == misses, f"upstream saw exactly the MISS responses ({calls} == {misses})")
    m = http_get(args.proxy + "/metrics")
    srv = server_histogram(m, 'cache="semantic_hit"')
    print("\n== semantic (lexical encoder) ==")
    print_rows(rows)
    print(f"server-side SEMANTIC-HIT: n={srv['count']} P50<={srv['p50_le_ms']:g}ms "
          f"P99<={srv['p99_le_ms']:g}ms   index entries={metric(m, 'vanguard_semantic_entries'):.0f}")
    print(f"similarity of semantic hits: min={sims[0]:.4f} p50={pct(sims, 0.5):.4f}")
    return rows


# ---------------------------------------------------------------------------
# failover
# ---------------------------------------------------------------------------

async def burst(args, label: str, n: int, c: int, run: str, tag: str):
    host, port = hostport(args.proxy)
    jobs = [Job(KEY, body("failover bench", f"[{run}] {tag} unique question {i}")) for i in range(n)]
    res, errs, el = await run_jobs(host, port, jobs, c)
    return res, errs, summarize(label, res, el)


async def recover(args, run: str, tag: str, limit_s: float) -> float:
    """Single requests every 50 ms until one is served by the primary again."""
    host, port = hostport(args.proxy)
    conn = Conn(host, port)
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < limit_s:
        r = await conn.request("POST", "/v1/chat/completions",
                               {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
                               body("failover bench", f"[{run}] {tag} recovery probe {i}"))
        i += 1
        if r.headers.get("x-vanguard-upstream") == "primary":
            conn.close()
            return time.perf_counter() - t0
        await asyncio.sleep(0.05)
    conn.close()
    return float("inf")


async def section_failover(args, checks: Checks, run: str) -> list:
    P, F = args.upstream, args.fallback
    rows = []
    set_fault(P, mode="ok")
    set_fault(F, mode="ok")
    print("\n== failover ==")

    # A. baseline
    res, errs, s = await burst(args, "A healthy primary", 200, 16, run, "A")
    rows.append(s)
    checks.check(not errs and s["statuses"] == {200: 200} and counts(res, "x-vanguard-upstream") == {"primary": 200},
                 "A: healthy primary serves all traffic")

    # B. primary returns 503s
    set_fault(P, mode="error", status=503)
    p0 = stats(P)["requests"]
    res, errs, s = await burst(args, "B primary 503 -> fallback", 400, 8, run, "B")
    rows.append(s)
    primary_calls = stats(P)["requests"] - p0
    reasons = counts(res, "x-vanguard-failover")
    checks.check(s["statuses"] == {200: 400}, f"B: 400/400 succeeded while the primary returned 503 ({s['statuses']})")
    checks.check(counts(res, "x-vanguard-upstream").get("fallback") == 400, "B: all served by the fallback")
    checks.check(primary_calls <= 40,
                 f"B: breaker shielded the primary ({primary_calls} primary calls for 400 requests); reasons={reasons}")
    set_fault(P, mode="ok")
    t = await recover(args, run, "B", 20)
    checks.check(t < 10, f"B: traffic returned to the primary {t:.2f}s after it recovered")

    # C. primary slow: time-to-first-byte timeout (300 ms)
    set_fault(P, mode="slow", delay_ms=1500)
    res, errs, s = await burst(args, "C primary slow -> fallback", 200, 8, run, "C")
    rows.append(s)
    timeouts = sorted(r.total_ms for r in res if r and r.headers.get("x-vanguard-failover") == "upstream-timeout")
    direct = sorted(r.total_ms for r in res if r and r.headers.get("x-vanguard-failover") == "primary-circuit-open")
    checks.check(s["statuses"] == {200: 200}, f"C: 200/200 succeeded with a 1.5 s primary ({s['statuses']})")
    checks.check(bool(timeouts) and 300 <= timeouts[0] and timeouts[-1] < 1500,
                 f"C: timed-out attempts failed over after ~300 ms, not 1.5 s "
                 f"(n={len(timeouts)}, min {timeouts[0] if timeouts else 0:.0f} ms, max {timeouts[-1] if timeouts else 0:.0f} ms)")
    if direct:
        print(f"  C: timeout+failover p50 {pct(timeouts, .5):.0f} ms (n={len(timeouts)}) | "
              f"circuit-open direct-to-fallback p50 {pct(direct, .5):.0f} ms (n={len(direct)})")
    set_fault(P, mode="ok")
    t = await recover(args, run, "C", 20)
    checks.check(t < 10, f"C: traffic returned to the primary {t:.2f}s after it recovered")

    # D. primary drops connections without responding
    set_fault(P, mode="drop")
    res, errs, s = await burst(args, "D primary drops conns", 150, 8, run, "D")
    rows.append(s)
    checks.check(s["statuses"] == {200: 150}, f"D: 150/150 succeeded while the primary dropped connections "
                                              f"({counts(res, 'x-vanguard-failover')})")
    set_fault(P, mode="ok")
    await recover(args, run, "D", 20)

    # E. exponential backoff: probe spacing while the primary stays down
    set_fault(P, mode="error", status=500)
    host, port = hostport(args.proxy)
    conn = Conn(host, port)
    t_start = stats(P)["now"]
    ok = 0
    end = time.perf_counter() + args.backoff_seconds
    i = 0
    while time.perf_counter() < end:
        r = await conn.request("POST", "/v1/chat/completions",
                               {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
                               body("failover bench", f"[{run}] E trickle {i}"))
        ok += r.status == 200
        i += 1
        await asyncio.sleep(0.02)
    conn.close()
    arrivals = [t for t in stats(P)["chat_times"] if t >= t_start]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:]) if b - a > 0.5]
    checks.check(ok == i, f"E: {ok}/{i} trickle requests succeeded while the primary stayed down")
    doubling = all(g2 >= 1.6 * g1 or g2 >= args.max_open_s * 0.9 for g1, g2 in zip(gaps, gaps[1:]))
    checks.check(len(gaps) >= 3 and doubling,
                 "E: probe spacing backs off exponentially: " + ", ".join(f"{g:.2f}s" for g in gaps))
    set_fault(P, mode="ok")
    t = await recover(args, run, "E", args.max_open_s + 5)
    checks.check(t <= args.max_open_s + 1, f"E: recovered {t:.2f}s after restore (<= max backoff {args.max_open_s:.0f}s)")

    # F. both upstreams down
    set_fault(P, mode="error", status=500)
    set_fault(F, mode="error", status=500)
    res, errs, s = await burst(args, "F primary+fallback down", 60, 4, run, "F")
    rows.append(s)
    r503 = [r for r in res if r and r.status == 503]
    checks.check(bool(r503) and all(r.headers.get("retry-after", "").isdigit() for r in r503),
                 f"F: once both circuits open, clients get fast 503 + Retry-After ({s['statuses']})")
    set_fault(P, mode="ok")
    set_fault(F, mode="ok")
    await recover(args, run, "F", args.max_open_s + 5)

    m = http_get(args.proxy + "/metrics")
    print_rows(rows)
    primary = 'upstream="primary"'
    print(f"failovers={metric(m, 'vanguard_failovers_total'):.0f} "
          f"primary trips={metric(m, 'vanguard_circuit_trips_total', primary):.0f} "
          f"primary timeouts={metric(m, 'vanguard_upstream_timeouts_total', primary):.0f} "
          f"primary circuit state={metric(m, 'vanguard_circuit_state', primary):.0f}")
    return rows


# ---------------------------------------------------------------------------
# neural (remote encoder)
# ---------------------------------------------------------------------------

async def section_neural(args, checks: Checks, run: str) -> list:
    print("\n== semantic (neural sidecar) ==")
    cal = post_json(args.embedder + "/similarity", {"pairs": PARAPHRASES + NEAR_MISSES})
    sims = cal["cosine"]
    para, near = sims[: len(PARAPHRASES)], sims[len(PARAPHRASES):]
    print(f"calibration ({cal['model']}): cosine similarity, threshold {args.threshold}")
    for (a, b), s in zip(PARAPHRASES, para):
        print(f"  same  {s:.3f} {'HIT ' if s >= args.threshold else 'miss'}  {a} | {b}")
    for (a, b), s in zip(NEAR_MISSES, near):
        print(f"  diff  {s:.3f} {'HIT!' if s >= args.threshold else 'miss'}  {a} | {b}")
    print(f"  paraphrases min/median {min(para):.3f}/{sorted(para)[len(para) // 2]:.3f}; "
          f"different questions max {max(near):.3f}")

    host, port = hostport(args.proxy)
    system = f"You are terse. [neural run {run}]"
    bases = sorted({a for a, _ in PARAPHRASES + NEAR_MISSES})
    res, errs, el = await run_jobs(host, port, [Job(KEY, body(system, b)) for b in bases], 4)
    checks.check(all(r and r.x_cache == "MISS" for r in res), f"{len(bases)} seed questions are MISS")
    res_p, _, el_p = await run_jobs(host, port, [Job(KEY, body(system, b)) for _, b in PARAPHRASES], 1)
    res_n, _, _ = await run_jobs(host, port, [Job(KEY, body(system, b)) for _, b in NEAR_MISSES], 1)
    hits = [r for r in res_p if r and r.x_cache == "SEMANTIC-HIT"]
    fps = [(q, r.headers.get("x-vanguard-similarity")) for (_, q), r in zip(NEAR_MISSES, res_n)
           if r and r.x_cache != "MISS"]
    s = summarize("neural paraphrases (c=1)", res_p, el_p)
    print_rows([s])
    lat = sorted(r.total_ms for r in hits)
    print(f"  paraphrase hit rate {len(hits)}/{len(PARAPHRASES)} at threshold {args.threshold}; "
          f"SEMANTIC-HIT latency incl. embedding p50 {pct(lat, .5):.2f} ms max {lat[-1] if lat else 0:.2f} ms")
    m = http_get(args.proxy + "/metrics")
    emb = server_histogram(m, "le=", "vanguard_embedding_duration_seconds")
    print(f"  embedding call latency (server-side): P50<={emb['p50_le_ms']:g}ms P99<={emb['p99_le_ms']:g}ms")
    checks.check(not fps, f"neural: 0 false positives on {len(NEAR_MISSES)} near-miss questions {fps}")
    checks.check(len(hits) >= 1, f"neural: true paraphrases produce SEMANTIC-HITs ({len(hits)}/{len(PARAPHRASES)})")
    return [s]


async def section_degraded(args, checks: Checks, run: str) -> list:
    print("\n== semantic (sidecar down) ==")
    res, errs, s = await burst(args, "sidecar down", 50, 4, run, "degraded")
    print_rows([s])
    m = http_get(args.proxy + "/metrics")
    skipped = metric(m, "vanguard_semantic_skipped_total")
    state = metric(m, "vanguard_circuit_state", 'upstream="embedder"')
    checks.check(s["statuses"] == {200: 50}, f"sidecar down: 50/50 requests still served ({s['statuses']})")
    checks.check(skipped >= 5 and state in (1, 2),
                 f"sidecar down: semantic lookup skipped ({skipped:.0f}x), embedder circuit opened (state {state:.0f})")
    return [s]


def print_rows(rows: list) -> None:
    print(f"{'phase':<34}{'n':>6}  {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'req/s':>8}   (ms)")
    for r in rows:
        print(fmt_row(r).replace(f"{r['phase']:<26}", f"{r['phase']:<34}", 1))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", required=True, choices=["semantic", "failover", "neural", "degraded"])
    ap.add_argument("--proxy", default="http://127.0.0.1:8180")
    ap.add_argument("--upstream", default="http://127.0.0.1:9100")
    ap.add_argument("--fallback", default="http://127.0.0.1:9101")
    ap.add_argument("--embedder", default="http://127.0.0.1:9200")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--hit-concurrency", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.94)
    ap.add_argument("--backoff-seconds", type=float, default=20.0)
    ap.add_argument("--max-open-s", type=float, default=8.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = format(int(time.time() * 1000) % (1 << 40), "x")
    checks = Checks()
    section = {"semantic": section_semantic, "failover": section_failover,
               "neural": section_neural, "degraded": section_degraded}[args.section]
    rows = await section(args, checks, run)
    print("\nchecks:")
    for ok, what in checks.items:
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    print(f"{len(checks.items) - checks.failed}/{len(checks.items)} checks passed")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"section": args.section, "phases": rows,
                       "checks": [{"ok": ok, "check": w} for ok, w in checks.items]}, f, indent=2)
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
