#!/usr/bin/env python3
"""Milestone 1 integration test: OpenTelemetry tracing across a 2-node ring.

Starts the mock LLM upstream, an OTLP/JSON receiver (observability/tools/otlp_sink.py),
two tracing-enabled proxies (n1, n2) on one ring and one proxy with tracing off,
then verifies over real sockets:

  * a caller's W3C traceparent is continued: the gateway's SERVER span is its
    child and keeps its trace id;
  * every outbound call carries a fresh traceparent: the LLM upstream sees the
    llm.upstream CLIENT span as its parent;
  * a request forwarded to its ring owner is ONE trace across both nodes:
    n1 SERVER -> n1 peer.forward -> n2 SERVER -> n2 llm.upstream -> upstream;
  * span attributes record the cache outcome (miss / hit / forwarded), tenant,
    status codes, and 5xx are span errors while 4xx are not;
  * an unsampled caller (flags 00) gets no recorded spans but its decision is
    propagated; no caller header starts a new root trace;
  * streamed responses are traced; /metrics exposes the exporter counters;
  * with tracing off the proxy adds no traceparent anywhere.

Used by ctest (Linux) and runnable by hand:
    python3 tests/integration/test_tracing_e2e.py --binary proxy-cpp/build/vanguard-proxy
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOCK = os.path.join(ROOT, "benchmarks", "mock_upstream.py")
SINK = os.path.join(ROOT, "observability", "tools", "otlp_sink.py")
SERVER, CLIENT = 2, 3


class Checks:
    def __init__(self):
        self.results = []

    def check(self, ok, what):
        self.results.append((bool(ok), what))
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}", flush=True)
        return ok

    @property
    def failed(self):
        return sum(1 for ok, _ in self.results if not ok)


def req(port, method, path, body=None, headers=None, timeout=10):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    hdrs = {k.lower(): v for k, v in r.getheaders()}
    c.close()
    return r.status, hdrs, data


def wait_healthy(port, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if req(port, "GET", "/healthz", timeout=1)[0] == 200:
                return True
        except OSError:
            time.sleep(0.05)
    return False


def chat(port, prompt, traceparent=None, key="sk-trace", stream=False, no_store=False):
    body = json.dumps({"model": "mock-model", "stream": stream,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    headers = {"Content-Type": "application/json"}
    if no_store:
        headers["Cache-Control"] = "no-store"
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if traceparent:
        headers["traceparent"] = traceparent
    return req(port, "POST", "/v1/chat/completions", body, headers)


def new_traceparent(sampled=True):
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-{'01' if sampled else '00'}"


def spans_of(sink_port, trace_id, expect, seconds=5.0):
    """Polls the sink until `expect` spans of this trace arrived (or timeout)."""
    deadline = time.time() + seconds
    found = []
    while time.time() < deadline:
        found = [s for s in json.loads(req(sink_port, "GET", "/spans")[2]) if s["traceId"] == trace_id]
        if len(found) >= expect:
            time.sleep(0.2)  # catch any unexpected extras
            return [s for s in json.loads(req(sink_port, "GET", "/spans")[2]) if s["traceId"] == trace_id]
        time.sleep(0.05)
    return found


def upstream_traceparents(port):
    return json.loads(req(port, "GET", "/stats")[2]).get("traceparents", [])


def one(spans, **match):
    out = [s for s in spans if all(s.get(k) == v for k, v in match.items())]
    return out[0] if len(out) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--base-port", type=int, default=18300)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="vanguard-tracing-")
    up_port, sink_port = args.base_port + 90, args.base_port + 91
    n1, n2, n3 = args.base_port + 1, args.base_port + 2, args.base_port + 3
    procs = []
    checks = Checks()
    try:
        procs.append(subprocess.Popen(
            [sys.executable, MOCK, "--port", str(up_port), "--min-latency-ms", "2", "--max-latency-ms", "5",
             "--token-interval-ms", "0", "--tokens", "8"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([sys.executable, SINK, "--port", str(sink_port)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        assert wait_healthy(up_port) and wait_healthy(sink_port), "mock upstream / OTLP sink did not start"

        peers = f"n1=127.0.0.1:{n1},n2=127.0.0.1:{n2}"
        base = dict(os.environ, UPSTREAM_URL=f"http://127.0.0.1:{up_port}", WORKERS="2", DEFAULT_RPM="0",
                    DEFAULT_TPM="0", PEER_TOKEN="peer-secret")
        configs = [
            ("n1", n1, dict(PEERS=peers, OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{sink_port}",
                            OTEL_BSP_SCHEDULE_DELAY="100")),
            ("n2", n2, dict(PEERS=peers, OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{sink_port}",
                            OTEL_BSP_SCHEDULE_DELAY="100")),
            ("n3", n3, {}),  # tracing off, standalone
        ]
        logs = {}
        for i, (nid, port, extra) in enumerate(configs):
            env = dict(base, NODE_ID=nid, LISTEN_PORT=str(port), ADMIN_PORT=str(args.base_port + 10 + i), **extra)
            logs[nid] = open(os.path.join(tmp, f"{nid}.log"), "w")
            procs.append(subprocess.Popen([args.binary], env=env, stdout=logs[nid], stderr=subprocess.STDOUT))
        for nid, port, _ in configs:
            assert wait_healthy(port), f"{nid} did not start (see {tmp}/{nid}.log)"
        print(f"upstream, OTLP sink and 3 proxies up (logs in {tmp})")

        # Ring ownership depends on the cache key, so discover prompts owned by
        # each node with Cache-Control: no-store (the cache stays cold; the
        # discovery requests are separate traces that the checks ignore).
        counter = iter(range(10**6))

        def find_prompt(owner, stream=False, marker=""):
            for _ in range(400):
                p = f"tracing question {next(counter)}{marker}"
                _, h, _ = chat(n1, p, stream=stream, no_store=True)
                if h.get("x-vanguard-owner") == owner:
                    return p
            raise AssertionError(f"no prompt owned by {owner}")

        local_prompt, remote_prompt = find_prompt("n1"), find_prompt("n2")
        local_tp, remote_tp = new_traceparent(), new_traceparent()
        assert chat(n1, local_prompt, local_tp)[0] == 200
        assert chat(n1, remote_prompt, remote_tp)[0] == 200

        # 1. Local miss: caller -> n1 SERVER -> n1 llm.upstream -> upstream ----
        print("\n== local miss continues the caller's trace ==")
        trace, caller_span = local_tp.split("-")[1], local_tp.split("-")[2]
        spans = spans_of(sink_port, trace, 2)
        srv = one(spans, kind=SERVER)
        cli = one(spans, kind=CLIENT, name="llm.upstream")
        checks.check(len(spans) == 2 and srv and cli, f"exactly 2 spans in the trace (got {len(spans)})")
        if srv and cli:
            a = srv["attributes"]
            checks.check(srv["parentSpanId"] == caller_span, "SERVER span is a child of the caller's span")
            checks.check(srv["name"] == "POST /v1/chat/completions", f"span name {srv['name']!r}")
            checks.check(a.get("vanguard.cache.result") == "miss" and a.get("http.response.status_code") == 200,
                         f"cache.result={a.get('vanguard.cache.result')} status={a.get('http.response.status_code')}")
            checks.check(a.get("vanguard.tenant") and a.get("vanguard.ring.owner") == "n1" and
                         a.get("vanguard.route") == "local" and a.get("gen_ai.request.model") == "mock-model",
                         "tenant, ring owner, route and model recorded")
            checks.check(srv["resource"].get("service.name") == "vanguard-proxy" and
                         srv["resource"].get("service.instance.id") == "n1", "resource: service.name + instance n1")
            checks.check(cli["parentSpanId"] == srv["spanId"] and
                         cli["attributes"].get("vanguard.upstream.target") == "primary" and
                         cli["attributes"].get("http.response.status_code") == 200,
                         "llm.upstream is a child of the SERVER span (primary, 200)")
            checks.check(srv["start"] <= cli["start"] and cli["end"] <= srv["end"],
                         "client span lies within the server span")
            sent = [t for t in upstream_traceparents(up_port) if t and t.split("-")[1] == trace]
            checks.check(sent == [f"00-{trace}-{cli['spanId']}-01"],
                         "upstream received traceparent = llm.upstream span (sampled)")

        # 2. Forwarded miss: one trace across both nodes ------------------------
        print("\n== forwarded request is one trace across the ring ==")
        trace, caller_span = remote_tp.split("-")[1], remote_tp.split("-")[2]
        spans = spans_of(sink_port, trace, 4)
        ingress = one(spans, kind=SERVER, parentSpanId=caller_span)
        fwd = one(spans, kind=CLIENT, name="peer.forward")
        owner = [s for s in spans if s["kind"] == SERVER and s["resource"].get("service.instance.id") == "n2"]
        llm = one(spans, kind=CLIENT, name="llm.upstream")
        checks.check(len(spans) == 4 and ingress and fwd and len(owner) == 1 and llm,
                     f"4 spans: n1 SERVER, n1 peer.forward, n2 SERVER, n2 llm.upstream (got {len(spans)})")
        if ingress and fwd and owner and llm:
            owner = owner[0]
            checks.check(ingress["resource"]["service.instance.id"] == "n1" and
                         ingress["attributes"].get("vanguard.route") == "forward" and
                         ingress["attributes"].get("vanguard.cache.result") == "forwarded",
                         "n1 SERVER span: route=forward, cache.result=forwarded")
            checks.check(fwd["parentSpanId"] == ingress["spanId"] and fwd["attributes"].get("vanguard.peer.id") == "n2"
                         and fwd["attributes"].get("http.response.status_code") == 200,
                         "peer.forward is n1's child span towards n2 (200)")
            checks.check(owner["parentSpanId"] == fwd["spanId"] and owner["attributes"].get("vanguard.hop") == 1 and
                         owner["attributes"].get("vanguard.route") == "owner_for_peer" and
                         owner["attributes"].get("vanguard.cache.result") == "miss",
                         "n2 SERVER span is a child of n1's peer.forward (hop 1, miss)")
            checks.check(llm["parentSpanId"] == owner["spanId"] and llm["resource"]["service.instance.id"] == "n2",
                         "n2 llm.upstream is a child of n2's SERVER span")
            checks.check(ingress["attributes"].get("vanguard.peer.x_cache") == "MISS", "ingress recorded owner's X-Cache")

        # 3. Cache hit: one span, no outbound call ---------------------------------
        print("\n== cache hit ==")
        tp = new_traceparent()
        st, h, _ = chat(n1, local_prompt, tp)
        spans = spans_of(sink_port, tp.split("-")[1], 1)
        checks.check(st == 200 and h.get("x-cache") == "HIT", f"repeat is a HIT ({st} {h.get('x-cache')})")
        checks.check(len(spans) == 1 and spans[0]["attributes"].get("vanguard.cache.result") == "hit",
                     f"hit trace has 1 span with cache.result=hit (got {len(spans)})")

        # 4. Unsampled caller: nothing recorded, decision propagated -----------------
        print("\n== unsampled caller ==")
        tp = new_traceparent(sampled=False)
        trace = tp.split("-")[1]
        st, h, _ = chat(n1, find_prompt("n1"), tp)
        time.sleep(0.6)
        spans = [s for s in json.loads(req(sink_port, "GET", "/spans")[2]) if s["traceId"] == trace]
        sent = [t for t in upstream_traceparents(up_port) if t and t.split("-")[1] == trace]
        checks.check(st == 200 and not spans, f"no spans exported for an unsampled trace (got {len(spans)})")
        checks.check(len(sent) == 1 and sent[0].endswith("-00") and sent[0].split("-")[2] != tp.split("-")[2],
                     "upstream still got the trace id with flags 00 (decision propagated)")

        # 5. No caller context: a new root trace ----------------------------------
        print("\n== no traceparent ==")
        before = len(upstream_traceparents(up_port))
        st, h, _ = chat(n1, find_prompt("n1"))
        new = upstream_traceparents(up_port)[before:]
        trace = new[-1].split("-")[1] if new and new[-1] else ""
        spans = spans_of(sink_port, trace, 2) if trace else []
        root = one(spans, kind=SERVER)
        checks.check(st == 200 and root is not None and root["parentSpanId"] == "",
                     "the gateway starts a root span (no parentSpanId) and propagates it")

        # 6. 4xx is not a span error; upstream 5xx is ------------------------------
        print("\n== status codes ==")
        tp = new_traceparent()
        st, _, _ = chat(n1, "no key", tp, key=None)
        spans = spans_of(sink_port, tp.split("-")[1], 1)
        checks.check(st == 401 and len(spans) == 1 and spans[0]["attributes"].get("http.response.status_code") == 401
                     and spans[0]["status"].get("code", 0) == 0, "401: recorded, span status UNSET (not an error)")
        tp = new_traceparent()
        st, _, _ = chat(n1, find_prompt("n1", marker=" __mock_error_500__"), tp)
        spans = spans_of(sink_port, tp.split("-")[1], 2)
        srv, cli = one(spans, kind=SERVER), one(spans, kind=CLIENT)
        checks.check(st == 500 and srv and cli and srv["status"].get("code") == 2 and cli["status"].get("code") == 2
                     and cli["attributes"].get("http.response.status_code") == 500,
                     f"upstream 500: SERVER and llm.upstream spans have status ERROR (HTTP {st})")

        # 7. Streaming -------------------------------------------------------------
        print("\n== streaming ==")
        tp = new_traceparent()
        st, h, body = chat(n1, find_prompt("n1", stream=True), tp, stream=True)
        spans = spans_of(sink_port, tp.split("-")[1], 2)
        srv = one(spans, kind=SERVER)
        checks.check(st == 200 and b"[DONE]" in body and srv and srv["attributes"].get("vanguard.stream") is True
                     and one(spans, kind=CLIENT) is not None, "SSE miss traced (stream=true, upstream child span)")

        # 8. Exporter metrics -------------------------------------------------------
        print("\n== /metrics ==")
        metrics = req(n1, "GET", "/metrics")[2].decode()
        vals = {l.split()[0]: float(l.split()[1]) for l in metrics.splitlines() if l.startswith("vanguard_trace")}
        checks.check("vanguard_cache_hits_total" in metrics, "Prometheus metrics still served")
        checks.check(vals.get("vanguard_trace_spans_exported_total", 0) > 0 and
                     vals.get("vanguard_trace_spans_dropped_total", 1) == 0 and
                     vals.get("vanguard_trace_export_errors_total", 1) == 0,
                     f"exporter counters: exported={vals.get('vanguard_trace_spans_exported_total')} "
                     f"dropped={vals.get('vanguard_trace_spans_dropped_total')} "
                     f"errors={vals.get('vanguard_trace_export_errors_total')}")

        # 9. Tracing off: no headers added ---------------------------------------------
        print("\n== tracing disabled ==")
        before = len(upstream_traceparents(up_port))
        st, _, _ = chat(n3, "tracing off question", new_traceparent())
        new = upstream_traceparents(up_port)[before:]
        checks.check(st == 200 and new == [None], "tracing off: upstream request carries no traceparent")
        m3 = req(n3, "GET", "/metrics")[2].decode()
        checks.check("vanguard_tracing_enabled 0" in m3, "tracing off: vanguard_tracing_enabled 0")

        # 10. Clean shutdown flushes ---------------------------------------------------
        print("\n== shutdown ==")
        codes = []
        for p in procs[2:]:
            p.send_signal(signal.SIGTERM)
        for p in procs[2:]:
            try:
                codes.append(p.wait(timeout=15))
            except subprocess.TimeoutExpired:
                codes.append(None)
        checks.check(codes == [0, 0, 0], f"proxies exit 0 on SIGTERM (got {codes})")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
    print(f"\n{len(checks.results) - checks.failed}/{len(checks.results)} checks passed")
    if checks.failed:
        for nid in ("n1", "n2", "n3"):
            path = os.path.join(tmp, f"{nid}.log")
            if os.path.exists(path):
                print(f"--- {nid}.log (tail) ---")
                print("".join(open(path).readlines()[-30:]))
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
