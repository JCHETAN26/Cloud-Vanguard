#!/usr/bin/env python3
"""Phase 4: CSI + QEMU warm-cache persistence, end to end, without a cluster.

Plays the role of the Kubernetes control plane: every storage operation is a
real CSI gRPC call to csi-driver (via csictl, the same RPCs external-provisioner,
external-attacher, external-snapshotter and kubelet issue). Volumes are QEMU
raw images (qemu-img), published to the "pod" as a raw block device, and
vanguard-proxy writes its A/B snapshot slots straight onto that device.

  1. CreateVolume -> ControllerPublish(node-a) -> NodeStage -> NodePublish
  2. proxy A fills its cache from the mock LLM, snapshots, is SIGKILLed
  3. NodeUnpublish/Unstage/ControllerUnpublish(node-a); publish to node-b
  4. proxy B restores from the re-attached volume: every prompt is a HIT and
     the upstream sees zero calls; new entries + graceful SIGTERM snapshot
  5. CreateSnapshot (compressed qcow2) -> CreateVolume from snapshot -> node-c:
     proxy C restores everything
  6. Corrupt the newest snapshot slot on the volume: proxy D falls back to the
     previous committed slot (crash consistency)
  7. Guards + cleanup: DeleteVolume while attached -> FailedPrecondition
  8. Optional: hot-plug a qcow2 volume into a real QEMU VM over QMP (TCG)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadgen import Checks, Conn, Job, run_jobs, summarize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "sk-csi-e2e"


class Ctl:
    def __init__(self, binary, endpoint):
        self.binary, self.endpoint = binary, endpoint
        self.timings: dict[str, list[float]] = {}

    def __call__(self, cmd, *args, ok=True):
        t0 = time.perf_counter()
        p = subprocess.run([self.binary, "--endpoint", self.endpoint, cmd, *args], capture_output=True, text=True)
        self.timings.setdefault(cmd, []).append((time.perf_counter() - t0) * 1000)
        out = json.loads(p.stdout.strip() or "{}")
        if ok and p.returncode != 0:
            raise RuntimeError(f"csictl {cmd} {args}: {out} {p.stderr}")
        return out


def wait_port(port, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


def http(port, method, path, body=None):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data) if data else {}


def upstream_calls(port):
    return http(port, "GET", "/stats")[1]["requests"]


class Proxy:
    def __init__(self, binary, name, port, admin, upstream_port, device, logdir):
        self.name, self.port, self.admin = name, port, admin
        env = dict(os.environ, NODE_ID=name, LISTEN_PORT=str(port), ADMIN_PORT=str(admin), WORKERS="2",
                   UPSTREAM_URL=f"http://127.0.0.1:{upstream_port}", SNAPSHOT_PATH=device, SNAPSHOT_MODE="block",
                   SNAPSHOT_INTERVAL_S="0", DEFAULT_RPM="0", DEFAULT_TPM="0", CACHE_TTL_S="3600")
        self.log = open(os.path.join(logdir, f"proxy-{name}.log"), "w")
        self.proc = subprocess.Popen([binary], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        if not wait_port(port):
            raise RuntimeError(f"proxy {name} did not start")

    def snapshot(self):
        st, res = http(self.admin, "POST", "/admin/v1/snapshot")
        assert st == 202, res
        want = res["request"]
        for _ in range(200):
            st, status = http(self.admin, "GET", "/admin/v1/snapshot")
            if status["saveCompleted"] >= want:
                return status["lastSave"]
            time.sleep(0.05)
        raise RuntimeError("snapshot did not complete")

    def status(self):
        return http(self.admin, "GET", "/admin/v1/snapshot")[1]

    def kill(self):
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        return self.proc.wait(timeout=30)


def body(i, stream=False):
    b = {"model": "mock-llm-1", "messages": [{"role": "user", "content": f"warm cache prompt {i}"}]}
    if stream:
        b["stream"] = True
    return json.dumps(b).encode()


async def ask(port, ids, concurrency=16):
    jobs = [Job(KEY, body(i, stream=(i % 10 == 0))) for i in ids]
    res, errs, el = await run_jobs("127.0.0.1", port, jobs, concurrency, keep_body=True)
    return res, errs, el


def attach(ctl, vol, node, root):
    dev = ctl("publish", "--id", vol, "--node", node)["devicePath"]
    staging = os.path.join(root, "kubelet", node, "staging", vol)
    target = os.path.join(root, "kubelet", node, "pods", "warm-cache")
    ctl("stage", "--id", vol, "--device", dev, "--staging", staging)
    ctl("node-publish", "--id", vol, "--staging", staging, "--target", target)
    return target, staging


def detach(ctl, vol, node, target, staging):
    ctl("node-unpublish", "--id", vol, "--target", target)
    ctl("unstage", "--id", vol, "--staging", staging)
    ctl("unpublish", "--id", vol, "--node", node)


def qmp(sock_path, cmd, args=None):
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock_path)
    f = s.makefile("rw")
    f.readline()
    f.write(json.dumps({"execute": "qmp_capabilities"}) + "\n")
    f.flush()
    f.readline()
    req = {"execute": cmd}
    if args:
        req["arguments"] = args
    f.write(json.dumps(req) + "\n")
    f.flush()
    while True:
        m = json.loads(f.readline())
        if "return" in m or "error" in m:
            s.close()
            return m


def main() -> int:
    import asyncio

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-binary", required=True)
    ap.add_argument("--csi-driver", required=True)
    ap.add_argument("--csictl", required=True)
    ap.add_argument("--base-port", type=int, default=19100)
    ap.add_argument("--prompts", type=int, default=400)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="vg-p4-")
    sockdir = tempfile.mkdtemp(prefix="vgcsi", dir="/tmp")  # unix socket path length limit
    endpoint = f"unix://{sockdir}/csi.sock"
    checks = Checks()
    results = {"env": {}, "timings": {}}
    procs = []
    n = args.prompts
    ids = list(range(n))
    up = args.base_port + 99
    try:
        mock = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_upstream.py"), "--port", str(up),
                                 "--min-latency-ms", "20", "--max-latency-ms", "40", "--token-interval-ms", "1"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(mock)
        assert wait_port(up)
        driver_log = open(os.path.join(root, "csi-driver.log"), "w")
        drv = subprocess.Popen([args.csi_driver, "--endpoint", endpoint, "--mode", "all", "--node-id", "node-a",
                                "--state-dir", os.path.join(root, "csi-state"), "--mounter", "link",
                                "--known-nodes", "node-a,node-b,node-c,node-d"],
                               stdout=driver_log, stderr=subprocess.STDOUT)
        procs.append(drv)
        for _ in range(100):
            if os.path.exists(endpoint[len("unix://"):]):
                break
            time.sleep(0.05)
        ctl = Ctl(args.csictl, endpoint)
        info = ctl("info")
        tool = open(os.path.join(root, "csi-driver.log")).read().split("via ")[1].split(",")[0]
        results["env"]["image_tool"] = tool
        print(f"CSI driver {info['name']} {info['version']}, image tool: {tool}\n")

        # 1. Provision + attach to node-a --------------------------------------
        print("== provision, attach, fill, snapshot, crash (node-a) ==")
        vol = ctl("create-volume", "--name", "pvc-warm-cache-vc-0", "--size", str(64 << 20), "--format", "raw")["volumeId"]
        target_a, staging_a = attach(ctl, vol, "node-a", root)
        checks.check(os.path.realpath(target_a).endswith(".raw"), f"volume {vol} published to node-a as a raw block device")

        pa = Proxy(args.proxy_binary, "node-a", args.base_port, args.base_port + 10, up, target_a, root)
        procs.append(pa.proc)
        before = upstream_calls(up)
        res, errs, el = asyncio.run(ask(pa.port, ids))
        fill = summarize("fill (MISS)", res, el)
        checks.check(not errs and fill["x_cache"].get("MISS") == n, f"proxy A filled {n} entries from the LLM")
        miss_bodies = {i: r.body for i, r in zip(ids, res) if r}
        snap = pa.snapshot()
        results["snapshot_a"] = snap
        checks.check(snap["ok"] and snap["entries"] == n and snap["mode"] == "block" and snap["slot"] == 0,
                     f"block snapshot: {snap['entries']} entries, {snap['bytes']} bytes, slot A gen {snap['generation']}, "
                     f"{snap['durationMs']:.1f} ms")
        pa.kill()
        checks.check(pa.proc.returncode == -9, "proxy A SIGKILLed (no shutdown snapshot)")
        detach(ctl, vol, "node-a", target_a, staging_a)

        # 2. Re-attach to node-b: warm restart ---------------------------------
        print("\n== re-attach to node-b, warm restore ==")
        target_b, staging_b = attach(ctl, vol, "node-b", root)
        t0 = time.perf_counter()
        pb = Proxy(args.proxy_binary, "node-b", args.base_port + 1, args.base_port + 11, up, target_b, root)
        procs.append(pb.proc)
        start_ms = (time.perf_counter() - t0) * 1000
        rst = pb.status()["lastRestore"]
        results["restore_b"] = rst
        checks.check(rst["ok"] and rst["loaded"] == n,
                     f"restored {rst['loaded']} entries in {rst['durationMs']:.1f} ms (proxy up in {start_ms:.0f} ms)")
        before = upstream_calls(up)
        res, errs, el = asyncio.run(ask(pb.port, ids))
        warm = summarize("after re-attach (HIT)", res, el)
        identical = sum(1 for i, r in zip(ids, res) if r and r.body == miss_bodies.get(i))
        checks.check(warm["x_cache"].get("HIT") == n and upstream_calls(up) == before,
                     f"{warm['x_cache'].get('HIT', 0)}/{n} HIT after re-attach, 0 upstream calls "
                     f"(p50 {warm['p50_ms']:.2f} ms, p99 {warm['p99_ms']:.2f} ms)")
        checks.check(identical == n, f"restored responses are byte-identical incl. SSE replays ({identical}/{n})")
        res, errs, el = asyncio.run(ask(pb.port, range(n, n + 100)))
        code = pb.stop()
        checks.check(code == 0, "proxy B SIGTERM: graceful exit with shutdown snapshot (slot B)")
        detach(ctl, vol, "node-b", target_b, staging_b)

        # 3. CSI snapshot -> new volume -> node-c --------------------------------
        print("\n== CSI snapshot (qcow2) -> restore into a new volume (node-c) ==")
        snap_id = ctl("snapshot", "--name", "vc-0-nightly", "--source", vol)["snapshotId"]
        snap_file = [f for f in os.listdir(os.path.join(root, "csi-state", "snapshots")) if not f.endswith(".json")][0]
        snap_size = os.path.getsize(os.path.join(root, "csi-state", "snapshots", snap_file))
        results["csi_snapshot"] = {"id": snap_id, "file": snap_file, "bytes_on_disk": snap_size}
        checks.check(snap_id.startswith("snap-"), f"CreateSnapshot {snap_id}: {snap_file}, {snap_size / 1024:.0f} KiB on disk "
                                                  f"for a 64 MiB volume")
        vol2 = ctl("create-volume", "--name", "pvc-warm-cache-restored", "--size", str(64 << 20), "--format", "raw",
                   "--from-snapshot", snap_id)["volumeId"]
        target_c, staging_c = attach(ctl, vol2, "node-c", root)
        pc = Proxy(args.proxy_binary, "node-c", args.base_port + 2, args.base_port + 12, up, target_c, root)
        procs.append(pc.proc)
        rst = pc.status()["lastRestore"]
        before = upstream_calls(up)
        res, errs, el = asyncio.run(ask(pc.port, range(n + 100)))
        s = summarize("from CSI snapshot", res, el)
        checks.check(rst["loaded"] == n + 100 and rst["generation"] == 2 and s["x_cache"].get("HIT") == n + 100
                     and upstream_calls(up) == before,
                     f"volume restored from snapshot: {rst['loaded']} entries (gen {rst['generation']}), "
                     f"{s['x_cache'].get('HIT', 0)}/{n + 100} HIT, 0 upstream calls")
        img = os.path.realpath(target_c)  # the raw image behind the published device
        pc.kill()
        detach(ctl, vol2, "node-c", target_c, staging_c)

        # 4. Torn newest slot -> fallback ---------------------------------------
        print("\n== crash consistency: corrupt the newest slot ==")
        cap = ((64 << 20) - 8192) // 2 // 4096 * 4096
        with open(img, "r+b") as f:  # slot B (generation 2) data region
            f.seek(8192 + cap + 64)
            f.write(b"\xde\xad\xbe\xef" * 256)
        target_d, staging_d = attach(ctl, vol2, "node-d", root)
        pd = Proxy(args.proxy_binary, "node-d", args.base_port + 3, args.base_port + 13, up, target_d, root)
        procs.append(pd.proc)
        rst = pd.status()["lastRestore"]
        before = upstream_calls(up)
        res, errs, el = asyncio.run(ask(pd.port, ids))
        hits = sum(1 for r in res if r and r.x_cache == "HIT")
        checks.check(rst["ok"] and rst["generation"] == 1 and rst["loaded"] == n and hits == n and upstream_calls(up) == before,
                     f"torn slot B detected ('{rst['error'][:60]}...'); fell back to slot A: {rst['loaded']} entries, "
                     f"{hits}/{n} HIT")

        # 5. Guards & cleanup ---------------------------------------------------
        print("\n== guards & cleanup ==")
        r = ctl("delete-volume", "--id", vol2, ok=False)
        checks.check("FailedPrecondition" in r.get("error", ""), "DeleteVolume of an attached volume -> FailedPrecondition")
        r = ctl("publish", "--id", vol2, "--node", "node-a", ok=False)
        checks.check("FailedPrecondition" in r.get("error", ""), "second single-writer attach -> FailedPrecondition")
        pd.kill()
        detach(ctl, vol2, "node-d", target_d, staging_d)
        for v in (vol, vol2):
            ctl("delete-volume", "--id", v)
        ctl("delete-snapshot", "--id", snap_id)
        left = ctl("list-volumes").get("volumes") or []
        images = [f for f in os.listdir(os.path.join(root, "csi-state", "volumes")) if not f.endswith(".json")]
        checks.check(not left and not images, "all volumes, images and snapshots deleted")

        # 6. Real QEMU VM attach over QMP ---------------------------------------
        qemu = shutil.which("qemu-system-x86_64")
        if qemu and tool == "qemu-img":
            print("\n== QMP hot-plug into a QEMU VM (TCG) ==")
            vmsock = os.path.join(sockdir, "vm-node.qmp")
            vm = subprocess.Popen([qemu, "-machine", "pc,accel=tcg", "-S", "-nodefaults", "-display", "none",
                                   "-device", "virtio-scsi-pci,id=scsi0", "-qmp", f"unix:{vmsock},server=on,wait=off"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(vm)
            for _ in range(100):
                if os.path.exists(vmsock):
                    break
                time.sleep(0.05)
            qendpoint = f"unix://{sockdir}/csi-qmp.sock"
            qdrv = subprocess.Popen([args.csi_driver, "--endpoint", qendpoint, "--mode", "controller",
                                     "--state-dir", os.path.join(root, "csi-qmp"), "--attacher", "qmp",
                                     "--qmp-socket-template", os.path.join(sockdir, "{node}.qmp")],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(qdrv)
            for _ in range(100):
                if os.path.exists(qendpoint[len("unix://"):]):
                    break
                time.sleep(0.05)
            qctl = Ctl(args.csictl, qendpoint)
            qv = qctl("create-volume", "--name", "vm-warm-cache", "--size", str(32 << 20), "--format", "qcow2")["volumeId"]
            t0 = time.perf_counter()
            dev = qctl("publish", "--id", qv, "--node", "vm-node")["devicePath"]
            attach_ms = (time.perf_counter() - t0) * 1000
            blocks = json.dumps(qmp(vmsock, "query-block"))
            checks.check(qv in blocks or "vgdev-" in blocks,
                         f"ControllerPublish hot-plugged qcow2 into the VM over QMP in {attach_ms:.0f} ms (guest {dev})")
            r = qctl("publish", "--id", qv, "--node", "unknown-vm", ok=False)
            checks.check("NotFound" in r.get("error", ""), "publish to a node without a VM -> NotFound")
            qctl("unpublish", "--id", qv, "--node", "vm-node")
            nodes = json.dumps(qmp(vmsock, "query-named-block-nodes", {"flat": True}))
            checks.check("vgblk-" not in nodes, "ControllerUnpublish removed the disk from the VM")
            qctl("delete-volume", "--id", qv)
            results["qmp_attach_ms"] = attach_ms
        else:
            print("\n(skipping QMP hot-plug: qemu-system-x86_64 / qemu-img not installed)")
            results["qmp_attach_ms"] = None

        results["timings"] = {k: round(sorted(v)[len(v) // 2], 2) for k, v in ctl.timings.items()}
        print("\nCSI RPC latency (median ms, incl. csictl process start):",
              ", ".join(f"{k} {v}" for k, v in results["timings"].items()))
        failed = sum(1 for ok, _ in checks.items if not ok)
        print("\nchecks:")
        for ok, w in checks.items:
            print(f"  [{'PASS' if ok else 'FAIL'}] {w}")
        print(f"\n{len(checks.items) - failed}/{len(checks.items)} CSI/QEMU persistence checks passed")
        results["checks"] = [{"ok": ok, "check": w} for ok, w in checks.items]
        if args.out:
            json.dump(results, open(args.out, "w"), indent=2)
        return 1 if failed else 0
    except Exception:
        print("\nchecks completed before the error:")
        for ok, w in checks.items:
            print(f"  [{'PASS' if ok else 'FAIL'}] {w}")
        raise
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        shutil.rmtree(sockdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
