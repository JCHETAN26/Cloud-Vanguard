#!/usr/bin/env python3
"""Tiny OTLP/HTTP (JSON encoding) trace receiver for local development and tests.

Stdlib only. It is not a collector: it accepts what vanguard-proxy exports and
keeps it in memory so tests and benchmarks can inspect it.

  POST   /v1/traces   ExportTraceServiceRequest, Content-Type: application/json
  GET    /spans       every received span, flattened (resource attributes merged
                      under "resource"), as a JSON array
  GET    /stats       {"requests": N, "spans": M, "rejected": K}
  DELETE /spans       forget everything received so far
  GET    /healthz     liveness

Protobuf-encoded exports get 415, as an OTLP receiver that only speaks JSON
would answer. Use the real OpenTelemetry Collector
(observability/otel-collector/config.yaml) for anything beyond tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
SPANS: list[dict] = []
STATS = {"requests": 0, "spans": 0, "rejected": 0}


def attr_value(v: dict):
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    return v


def flatten(doc: dict) -> list[dict]:
    out = []
    for rs in doc.get("resourceSpans", []):
        resource = {a["key"]: attr_value(a["value"]) for a in rs.get("resource", {}).get("attributes", [])}
        for ss in rs.get("scopeSpans", []):
            scope = ss.get("scope", {})
            for s in ss.get("spans", []):
                out.append({
                    "traceId": s.get("traceId", ""),
                    "spanId": s.get("spanId", ""),
                    "parentSpanId": s.get("parentSpanId", ""),
                    "name": s.get("name", ""),
                    "kind": s.get("kind", 0),
                    "start": int(s.get("startTimeUnixNano", 0)),
                    "end": int(s.get("endTimeUnixNano", 0)),
                    "attributes": {a["key"]: attr_value(a["value"]) for a in s.get("attributes", [])},
                    "status": s.get("status", {}),
                    "resource": resource,
                    "scope": scope.get("name", ""),
                })
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    out_file = None

    def log_message(self, *args):  # quiet
        pass

    def reply(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length)
        if self.path != "/v1/traces":
            return self.reply(404, b'{"error":"not found"}')
        if "json" not in (self.headers.get("Content-Type") or ""):
            with LOCK:
                STATS["rejected"] += 1
            return self.reply(415, b'{"error":"only OTLP/JSON is supported"}')
        try:
            spans = flatten(json.loads(body))
        except (ValueError, KeyError, TypeError) as e:
            with LOCK:
                STATS["rejected"] += 1
            return self.reply(400, json.dumps({"error": str(e)}).encode())
        with LOCK:
            STATS["requests"] += 1
            STATS["spans"] += len(spans)
            SPANS.extend(spans)
            if self.out_file:
                for s in spans:
                    self.out_file.write(json.dumps(s) + "\n")
                self.out_file.flush()
        self.reply(200, b"{}")

    def do_GET(self):
        if self.path == "/spans":
            with LOCK:
                data = json.dumps(SPANS).encode()
            return self.reply(200, data)
        if self.path == "/stats":
            with LOCK:
                data = json.dumps(STATS).encode()
            return self.reply(200, data)
        if self.path == "/healthz":
            return self.reply(200, b"ok\n", "text/plain")
        self.reply(404, b'{"error":"not found"}')

    def do_DELETE(self):
        if self.path != "/spans":
            return self.reply(404, b'{"error":"not found"}')
        with LOCK:
            SPANS.clear()
            STATS.update(requests=0, spans=0, rejected=0)
        self.reply(200, b"{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4318)
    ap.add_argument("--out", default="", help="also append received spans as NDJSON to this file")
    args = ap.parse_args()
    if args.out:
        Handler.out_file = open(args.out, "a", encoding="utf-8")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"otlp_sink listening on http://{args.host}:{args.port}/v1/traces", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
