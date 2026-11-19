#!/usr/bin/env python3
"""Phase 3 integration test: 3 vanguard-proxy nodes on one consistent-hash ring.

Starts a mock LLM upstream and three proxies (n0, n1, n2) on local ports with a
static PEERS ring, then verifies over real sockets:

  * every request is served by its ring owner, whichever node it enters
    (cross-node forwarding, X-Vanguard-Ingress / X-Vanguard-Node headers);
  * the nodes form one logical cache: re-asking through a different node hits,
    and the upstream sees each prompt exactly once;
  * SSE responses relay through a forwarding node byte-for-byte;
  * loop prevention: hop > 1 -> 508, unauthenticated hop -> 403, malformed -> 400;
  * ring divergence (a node pushed a wrong ring) never loops: forwarded requests
    are served by the receiving node, which never re-forwards;
  * dynamic tenant policies via the admin API (cluster-wide RPM split across
    nodes, version ordering, auth) and via CONTROL_FILE + SIGHUP;
  * an owner going down degrades to local service (no client errors);
  * every proxy exits cleanly (and, in sanitizer builds, without reports).

Used by ctest (Linux) and runnable by hand:
    python3 benchmarks/multinode_test.py --binary proxy-cpp/build/vanguard-proxy
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADMIN_TOKEN = "admin-secret"
PEER_TOKEN = "peer-secret"


class Checks:
    def __init__(self):
        self.items = []

    def check(self, ok, what):
        self.items.append((bool(ok), what))
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}", flush=True)
        return ok


class Node:
    def __init__(self, idx, port, admin_port):
        self.id, self.port, self.admin_port = f"n{idx}", port, admin_port
        self.proc = None
        self.log = None
        self._conn = None

    def conn(self):
        if self._conn is None:
            self._conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        return self._conn

    def chat(self, prompt, key="sk-multinode", stream=False, headers=None):
        body = {"model": "mock-llm-1", "messages": [{"role": "user", "content": prompt}]}
        if stream:
            body["stream"] = True
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        h.update(headers or {})
        for attempt in range(2):
            try:
                c = self.conn()
                c.request("POST", "/v1/chat/completions", json.dumps(body), h)
                r = c.getresponse()
                data = r.read()
                return r.status, {k.lower(): v for k, v in r.getheaders()}, data
            except (ConnectionError, http.client.HTTPException, OSError):
                self._conn = None
                if attempt:
                    raise
        raise RuntimeError("unreachable")

    def admin(self, method, path, body=None, token=ADMIN_TOKEN):
        c = http.client.HTTPConnection("127.0.0.1", self.admin_port, timeout=10)
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        c.request(method, path, json.dumps(body) if body is not None else None, h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data) if data else {}

    def metric(self, name):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/metrics")
        text = c.getresponse().read().decode()
        c.close()
        for line in text.splitlines():
            if line.startswith(name + " "):
                return float(line.split()[-1])
        return 0.0


def wait_healthy(port, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/healthz")
            if c.getresponse().status == 200:
                return True
        except OSError:
            time.sleep(0.05)
    return False


def upstream_requests(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/stats")
    return json.loads(c.getresponse().read())["requests"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--base-port", type=int, default=18100)
    ap.add_argument("--requests", type=int, default=300)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="vanguard-multinode-")
    up_port = args.base_port + 99
    nodes = [Node(i, args.base_port + i, args.base_port + 10 + i) for i in range(3)]
    peers = ",".join(f"{n.id}=127.0.0.1:{n.port}" for n in nodes)
    control_file = os.path.join(tmp, "n2-control.json")
    procs = []
    checks = Checks()
    status = 1
    try:
        mock = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_upstream.py"), "--port", str(up_port),
             "--min-latency-ms", "2", "--max-latency-ms", "6", "--token-interval-ms", "0", "--tokens", "8"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(mock)
        assert wait_healthy(up_port), "mock upstream did not start"

        for n in nodes:
            env = dict(os.environ, NODE_ID=n.id, LISTEN_PORT=str(n.port), ADMIN_PORT=str(n.admin_port),
                       ADMIN_TOKEN=ADMIN_TOKEN, PEER_TOKEN=PEER_TOKEN, PEERS=peers, WORKERS="2",
                       UPSTREAM_URL=f"http://127.0.0.1:{up_port}", DEFAULT_RPM="0", DEFAULT_TPM="0",
                       PEER_CONNECT_TIMEOUT_MS="200")
            if n.id == "n2":
                env["CONTROL_FILE"] = control_file
            n.log = open(os.path.join(tmp, f"{n.id}.log"), "w")
            n.proc = subprocess.Popen([args.binary], env=env, stdout=n.log, stderr=subprocess.STDOUT)
            procs.append(n.proc)
        for n in nodes:
            assert wait_healthy(n.port), f"{n.id} did not start (see {tmp}/{n.id}.log)"
        print(f"3 nodes up (logs in {tmp})")

        # 1. Forwarding to the owner -------------------------------------------
        print("\n== forwarding to the ring owner ==")
        prompts = [f"multinode question {i}" for i in range(args.requests)]
        owners, served_by_owner, forwarded, ingress_ok = {}, 0, 0, 0
        for i, p in enumerate(prompts):
            ingress = nodes[i % 3]
            st, h, _ = ingress.chat(p)
            owner, node = h.get("x-vanguard-owner"), h.get("x-vanguard-node")
            owners[owner] = owners.get(owner, 0) + 1
            served_by_owner += st == 200 and owner == node
            if owner != ingress.id:
                forwarded += 1
                ingress_ok += h.get("x-vanguard-ingress") == ingress.id
        n = len(prompts)
        checks.check(served_by_owner == n, f"{served_by_owner}/{n} requests served by their ring owner")
        checks.check(ingress_ok == forwarded and forwarded > n / 2,
                     f"{forwarded} entered a non-owner and were forwarded (X-Vanguard-Ingress set on {ingress_ok})")
        checks.check(len(owners) == 3 and all(0.2 * n < v < 0.47 * n for v in owners.values()),
                     f"keys spread over all owners: {dict(sorted(owners.items()))}")

        # 2. One logical cache --------------------------------------------------
        hits = 0
        for i, p in enumerate(prompts):
            st, h, _ = nodes[(i + 1) % 3].chat(p)  # a different ingress than before
            hits += st == 200 and h.get("x-cache") == "HIT" and h.get("x-vanguard-node") == h.get("x-vanguard-owner")
        checks.check(hits == n, f"re-asking through a different node: {hits}/{n} HIT at the owner")
        calls = upstream_requests(up_port)
        checks.check(calls == n, f"upstream saw each prompt exactly once across 3 nodes ({calls} calls)")

        # 3. SSE through a forwarding node -------------------------------------
        sse = None
        for i in range(50):
            st, h, body = nodes[0].chat(f"multinode stream {i}", stream=True)
            if h.get("x-vanguard-owner") != "n0":
                sse = (f"multinode stream {i}", h, body)
                break
        st2, h2, body2 = nodes[2 if sse and sse[1]["x-vanguard-owner"] != "n2" else 1].chat(sse[0], stream=True)
        checks.check(sse and sse[1].get("x-vanguard-ingress") == "n0" and body2 == sse[2]
                     and body2.endswith(b"data: [DONE]\n\n") and h2.get("x-cache") == "HIT",
                     "SSE relayed through a non-owner, then replayed byte-identically from the owner's cache")

        # 4. Loop prevention ---------------------------------------------------
        print("\n== loop prevention ==")
        st, h, body = nodes[0].chat("loop probe", headers={"X-Vanguard-Hop": "2", "X-Vanguard-Peer-Token": PEER_TOKEN})
        checks.check(st == 508 and b"loop_detected" in body, f"X-Vanguard-Hop: 2 -> {st} Loop Detected")
        st, h, body = nodes[0].chat("spoof probe", headers={"X-Vanguard-Hop": "1"})
        checks.check(st == 403 and b"untrusted_hop" in body, f"X-Vanguard-Hop without the peer token -> {st}")
        st, h, body = nodes[0].chat("bad hop", headers={"X-Vanguard-Hop": "x", "X-Vanguard-Peer-Token": PEER_TOKEN})
        checks.check(st == 400, f"malformed X-Vanguard-Hop -> {st}")

        # 5. Ring divergence ---------------------------------------------------
        # n0 gets a ring whose addresses are rotated: it believes n1 lives at
        # n2's port and n2 at n1's. Requests n0 forwards land on a node that does
        # not own them; that node must serve them itself, not forward again.
        print("\n== ring divergence ==")
        wrong = {"version": 5, "vnodes": 1000, "nodes": [
            {"id": "n0", "address": f"127.0.0.1:{nodes[0].port}"},
            {"id": "n1", "address": f"127.0.0.1:{nodes[2].port}"},
            {"id": "n2", "address": f"127.0.0.1:{nodes[1].port}"}]}
        st, res = nodes[0].admin("PUT", "/admin/v1/ring", wrong)
        checks.check(st == 200 and res.get("applied"), f"pushed a divergent ring to n0 ({res.get('message')})")
        fwd_before = nodes[1].metric("vanguard_forwarded_total") + nodes[2].metric("vanguard_forwarded_total")
        div_before = nodes[1].metric("vanguard_ring_divergence_total") + nodes[2].metric("vanguard_ring_divergence_total")
        statuses, misrouted = {}, 0
        for i in range(120):
            st, h, _ = nodes[0].chat(f"divergence question {i}")
            statuses[st] = statuses.get(st, 0) + 1
            misrouted += h.get("x-vanguard-node") not in ("n0", h.get("x-vanguard-owner"))
        fwd_after = nodes[1].metric("vanguard_forwarded_total") + nodes[2].metric("vanguard_forwarded_total")
        div_after = nodes[1].metric("vanguard_ring_divergence_total") + nodes[2].metric("vanguard_ring_divergence_total")
        checks.check(statuses == {200: 120}, f"all 120 requests succeeded under ring divergence ({statuses})")
        checks.check(misrouted > 20 and div_after - div_before == misrouted,
                     f"{misrouted} requests reached a non-owner and were served there (divergence counter +{div_after - div_before:.0f})")
        checks.check(fwd_after == fwd_before, "receiving nodes never re-forwarded (no ping-pong)")
        st, res = nodes[0].admin("PUT", "/admin/v1/ring", wrong)
        checks.check(st == 200 and not res.get("applied"), "same ring version again is a no-op")
        st, res = nodes[0].admin("PUT", "/admin/v1/ring", dict(wrong, version=4))
        checks.check(st == 409, f"older ring version rejected as stale ({st})")
        fixed = {"version": 6, "vnodes": 1000,
                 "nodes": [{"id": x.id, "address": f"127.0.0.1:{x.port}"} for x in nodes]}
        st, res = nodes[0].admin("PUT", "/admin/v1/ring", fixed)
        st, h, _ = nodes[0].chat("multinode question 1")
        checks.check(res.get("applied") and h.get("x-cache") == "HIT", "restored ring: n0 forwards correctly again")

        # 6. Dynamic tenant policy via the admin API ---------------------------
        print("\n== tenant policies without restart ==")
        key_hash = hashlib.sha256(b"sk-dyn").hexdigest()
        policy = {"version": "1", "tenants": [{"tenantId": "dyn", "apiKeySha256": [key_hash], "rpmLimit": "6"}]}
        st, res = nodes[0].admin("PUT", "/admin/v1/tenants", policy, token=None)
        checks.check(st == 401, f"admin API without the token -> {st}")
        c = http.client.HTTPConnection("127.0.0.1", nodes[0].port, timeout=5)
        c.request("GET", "/admin/v1/config")
        checks.check(c.getresponse().status == 404, "admin API is not exposed on the data-plane port")
        for nd in nodes:
            st, res = nd.admin("PUT", "/admin/v1/tenants", policy)
        codes = [nodes[0].chat(f"dyn {i}", key="sk-dyn")[0] for i in range(3)]
        checks.check(codes == [200, 200, 429], f"rpmLimit 6 across 3 nodes = 2/min per node: {codes}")
        st, res = nodes[0].admin("PUT", "/admin/v1/tenants", dict(policy, version="0"))
        checks.check(st == 200, "version 0 is a manual override")
        st, res = nodes[0].admin("PUT", "/admin/v1/tenants",
                                 {"version": "2", "tenants": [{"tenantId": "dyn", "apiKeySha256": [key_hash]}]})
        codes = [nodes[0].chat(f"dyn more {i}", key="sk-dyn")[0] for i in range(3)]
        checks.check(res.get("applied") and codes == [200, 200, 200], f"policy v2 (unlimited) applied live: {codes}")
        st, res = nodes[0].admin("PUT", "/admin/v1/tenants", {"version": "1", "tenants": []})
        checks.check(st == 409, f"stale tenant policy version rejected ({st})")
        st, res = nodes[0].admin("PUT", "/admin/v1/tenants", {"tenants": [{"tenantId": "bad", "semanticThreshold": 2}]})
        checks.check(st == 400, f"invalid policy rejected with 400 ({res.get('error', {}).get('message', '')})")

        # 7. CONTROL_FILE + SIGHUP ---------------------------------------------
        file_hash = hashlib.sha256(b"sk-file").hexdigest()
        with open(control_file + ".tmp", "w") as f:
            json.dump({"version": "3", "tenants": {"tenants": [
                {"tenantId": "filet", "apiKeySha256": [file_hash], "rpmLimit": "3"}]}}, f)
        os.replace(control_file + ".tmp", control_file)  # atomic, like a ConfigMap update
        nodes[2].proc.send_signal(signal.SIGHUP)
        time.sleep(1.5)
        st, cfg = nodes[2].admin("GET", "/admin/v1/config")
        codes = [nodes[2].chat(f"file {i}", key="sk-file")[0] for i in range(2)]
        checks.check(cfg.get("tenants", {}).get("version") == 3 and codes == [200, 429],
                     f"CONTROL_FILE reloaded on SIGHUP (tenants v{cfg.get('tenants', {}).get('version')}, codes {codes})")

        # 8. Owner down --------------------------------------------------------
        print("\n== owner node down ==")
        nodes[1].proc.terminate()
        nodes[1].proc.wait(timeout=10)
        statuses, lost_owner, t0 = {}, 0, time.time()
        for i in range(90):
            st, h, _ = nodes[0].chat(f"owner down question {i}")
            statuses[st] = statuses.get(st, 0) + 1
            lost_owner += h.get("x-vanguard-owner") == "n1"
        elapsed = time.time() - t0
        fallback = nodes[0].metric("vanguard_forward_local_fallback_total")
        checks.check(statuses == {200: 90}, f"90/90 served with n1 down ({statuses}, {elapsed:.2f}s total)")
        checks.check(lost_owner > 10 and fallback >= lost_owner,
                     f"{lost_owner} requests owned by n1 were served locally (fallback counter {fallback:.0f})")

        # 9. Clean shutdown ----------------------------------------------------
        for nd in (nodes[0], nodes[2]):
            nd.proc.send_signal(signal.SIGTERM)
        codes = {nd.id: nd.proc.wait(timeout=15) for nd in nodes}
        for nd in nodes:
            nd.log.flush()
        reports = []
        for nd in nodes:
            with open(os.path.join(tmp, f"{nd.id}.log")) as f:
                text = f.read()
            if "Sanitizer" in text or "runtime error:" in text:
                reports.append(nd.id)
        checks.check(all(c == 0 for c in codes.values()) and not reports,
                     f"all nodes exited cleanly {codes}, sanitizer reports: {reports or 'none'}")

        failed = sum(1 for ok, _ in checks.items if not ok)
        print(f"\n{len(checks.items) - failed}/{len(checks.items)} multi-node checks passed")
        status = 1 if failed else 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
    return status


if __name__ == "__main__":
    sys.exit(main())
