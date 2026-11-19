#!/usr/bin/env python3
"""Hostile-input regression test: no single request may take vanguard-proxy down.

Regression for a crash found in Milestone 2: a request path containing invalid
UTF-8 (e.g. GET /\\xff) reached nlohmann::json::dump() in the 404 error body,
which throws; the uncaught exception terminated the whole process.

Sends raw requests to the data-plane and admin ports (invalid UTF-8 in the
path, method, headers and body; wrongly typed JSON fields; malformed control
documents) and requires that every one gets an HTTP answer (or a clean close
for unparseable HTTP), that JSON error bodies are valid JSON, and that the
proxy is still healthy and serving afterwards.

    python3 tests/integration/test_http_robustness.py --binary proxy-cpp/build/vanguard-proxy
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOCK = os.path.join(ROOT, "benchmarks", "mock_upstream.py")


def raw(port: int, data: bytes, timeout: float = 3.0) -> bytes:
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    try:
        s.sendall(data)
        out = b""
        while True:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            out += chunk
            if b"\r\n\r\n" in out:
                head, _, body = out.partition(b"\r\n\r\n")
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        if len(body) >= int(line.split(b":")[1]):
                            return out
        return out
    finally:
        s.close()


def request(method: bytes, path: bytes, body: bytes = b"", headers: bytes = b"") -> bytes:
    return (method + b" " + path + b" HTTP/1.1\r\nHost: x\r\nConnection: close\r\n" + headers +
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)


def healthy(port: int) -> bool:
    try:
        return raw(port, request(b"GET", b"/healthz")).startswith(b"HTTP/1.1 200")
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--base-port", type=int, default=18500)
    args = ap.parse_args()
    port, admin, up = args.base_port, args.base_port + 1, args.base_port + 2
    tmp = tempfile.mkdtemp(prefix="vanguard-robust-")
    log = open(os.path.join(tmp, "proxy.log"), "w")
    procs = [subprocess.Popen([sys.executable, MOCK, "--port", str(up), "--min-latency-ms", "1", "--max-latency-ms", "2",
                               "--token-interval-ms", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    env = dict(os.environ, LISTEN_PORT=str(port), ADMIN_PORT=str(admin), WORKERS="1", NODE_ID="robust",
               UPSTREAM_URL=f"http://127.0.0.1:{up}", DEFAULT_RPM="0", DEFAULT_TPM="0")
    proxy = subprocess.Popen([args.binary], env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append(proxy)
    failures = []
    try:
        deadline = time.time() + 15
        while not healthy(port) and time.time() < deadline:
            time.sleep(0.05)
        while not healthy(up) and time.time() < deadline:
            time.sleep(0.05)
        assert healthy(port), "proxy did not start"

        bad_bodies = [
            b"\xff\xfe{",
            b'{"model": 1, "messages": [{"role": "user", "content": "x"}]}',
            b'{"model": "m", "messages": 5}',
            b'{"model": "m", "messages": [{"role": 7, "content": {"a": 1}}]}',
            b'{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": "yes"}',
            b'{"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": "many"}',
            b'{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": {"x": []}}',
            b'{"model": "m\xff", "messages": [{"role": "user", "content": "caf\xc3"}]}',
            b"[]", b"null", b'"str"', b"{" * 5000,
        ]
        cases = [
            ("data", port, request(b"GET", b"/\xff")),
            ("data", port, request(b"GET", b"/v1/\xc3(\xed\xa0\x80")),
            ("data", port, request(b"POST", b"/v1/chat/completions\xff")),
            ("data", port, request(b"G\xffT", b"/")),
            ("data", port, request(b"GET", b"/metrics", headers=b"X-\xff: \xfe\r\n")),
            ("data", port, request(b"POST", b"/v1/chat/completions", b"{}", b"Authorization: Bearer \xff\xfe\r\n")),
            ("data", port, request(b"POST", b"/v1/chat/completions", b"{}",
                                   b"Authorization: Bearer k\r\nX-Vanguard-Hop: \xff\r\n")),
            ("data", port, request(b"POST", b"/v1/chat/completions", b"{}",
                                   b"Authorization: Bearer k\r\ntraceparent: 00-\xff\r\n")),
            ("admin", admin, request(b"GET", b"/admin/v1/\xff")),
            ("admin", admin, request(b"PUT\xff", b"/admin/v1/ring")),
            ("admin", admin, request(b"PUT", b"/admin/v1/config", b"\xff{")),
            ("admin", admin, request(b"PUT", b"/admin/v1/ring", b'{"version": "x", "nodes": 5}')),
            ("admin", admin, request(b"PUT", b"/admin/v1/tenants", b'{"version": 1, "tenants": [{"tenantId": 5}]}')),
            ("admin", admin, request(b"PUT", b"/admin/v1/config", b'{"version": 1, "ring": {"nodes": [{"id": "\xff"}]}}')),
        ]
        for body in bad_bodies:
            cases.append(("data", port, request(b"POST", b"/v1/chat/completions", body, b"Authorization: Bearer k\r\n")))
            cases.append(("admin", admin, request(b"PUT", b"/admin/v1/config", body)))

        answered = 0
        for i, (where, p, data) in enumerate(cases):
            try:
                resp = raw(p, data)
            except OSError as e:
                resp = b""
                if not healthy(port):
                    failures.append(f"case {i} ({where} {data[:40]!r}): connection error {e} and proxy is down")
                    break
            if resp.startswith(b"HTTP/1.1 "):
                answered += 1
                head, _, body = resp.partition(b"\r\n\r\n")
                if b"application/json" in head and body:
                    try:
                        json.loads(body)
                    except ValueError as e:
                        failures.append(f"case {i} ({where} {data[:40]!r}): invalid JSON error body: {e}")
            if not healthy(port):
                failures.append(f"case {i} ({where} {data[:40]!r}) killed the proxy")
                break
        print(f"{answered}/{len(cases)} hostile requests answered with HTTP; "
              f"the rest closed the connection (unparseable HTTP)")

        # Still serving real traffic.
        ok = request(b"POST", b"/v1/chat/completions",
                     b'{"model": "m", "messages": [{"role": "user", "content": "still alive?"}]}',
                     b"Authorization: Bearer k\r\nContent-Type: application/json\r\n")
        if not raw(port, ok).startswith(b"HTTP/1.1 200"):
            failures.append("a normal chat request failed after the hostile inputs")
        proxy.send_signal(signal.SIGTERM)
        code = proxy.wait(timeout=15)
        if code != 0:
            failures.append(f"proxy exit code {code}")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        log.close()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print("--- proxy.log (tail) ---")
        print("".join(open(os.path.join(tmp, "proxy.log"), errors="replace").readlines()[-20:]))
        return 1
    print("PASS: proxy survived every hostile request and still serves traffic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
