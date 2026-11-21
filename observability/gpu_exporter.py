#!/usr/bin/env python3
"""Prometheus exporter for NVIDIA GPUs over NVML (nvidia-ml-py), stdlib HTTP server.

    python3 observability/gpu_exporter.py --port 9400        # GET /metrics

Metrics (one series per GPU, labels gpu/uuid/name):
  cloudvanguard_gpu_up                         1 when NVML answered on the last scrape
  cloudvanguard_gpu_memory_used_bytes          device memory in use (what nvidia-smi shows)
  cloudvanguard_gpu_memory_total_bytes
  cloudvanguard_gpu_utilization_ratio          SM busy fraction over the last sample period (0-1)
  cloudvanguard_gpu_memory_bandwidth_utilization_ratio
  cloudvanguard_gpu_power_watts
  cloudvanguard_gpu_temperature_celsius
  cloudvanguard_gpu_process_memory_used_bytes  per compute process (label pid)

NVML is read on every scrape (no background thread), so the scrape interval is the sampling rate.
NVIDIA's dcgm-exporter is the production choice on Kubernetes; this exists so the stack has GPU metrics on
any host with a driver, without containers, and with metric names that match observability/prometheus/rules.yml.
"""

from __future__ import annotations

import argparse
import http.server
import sys

PREFIX = "cloudvanguard_gpu_"
HELP = {
    "up": ("gauge", "1 if NVML answered on this scrape"),
    "memory_used_bytes": ("gauge", "GPU memory in use, bytes"),
    "memory_total_bytes": ("gauge", "GPU memory capacity, bytes"),
    "utilization_ratio": ("gauge", "Fraction of the sample period a kernel was running"),
    "memory_bandwidth_utilization_ratio": ("gauge", "Fraction of the sample period device memory was read/written"),
    "power_watts": ("gauge", "Board power draw, watts"),
    "temperature_celsius": ("gauge", "GPU core temperature, Celsius"),
    "process_memory_used_bytes": ("gauge", "GPU memory used by a compute process, bytes"),
}


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render(gpus: list[dict], up: bool = True) -> str:
    """Prometheus text format for readings shaped like read_nvml()'s output."""
    series: dict[str, list[tuple[dict, float]]] = {k: [] for k in HELP}
    series["up"].append(({}, 1.0 if up else 0.0))
    for g in gpus:
        lab = {"gpu": str(g["index"]), "uuid": g["uuid"], "name": g["name"]}
        for k in ("memory_used_bytes", "memory_total_bytes", "utilization_ratio",
                  "memory_bandwidth_utilization_ratio", "power_watts", "temperature_celsius"):
            if g.get(k) is not None:
                series[k].append((lab, float(g[k])))
        for pid, used in g.get("processes", {}).items():
            series["process_memory_used_bytes"].append(({**lab, "pid": str(pid)}, float(used)))
    lines = []
    for k, (typ, text) in HELP.items():
        if not series[k]:
            continue
        lines += [f"# HELP {PREFIX}{k} {text}", f"# TYPE {PREFIX}{k} {typ}"]
        for lab, v in series[k]:
            ls = ",".join(f'{n}="{_esc(x)}"' for n, x in lab.items())
            lines.append(f"{PREFIX}{k}{{{ls}}} {v:g}" if ls else f"{PREFIX}{k} {v:g}")
    return "\n".join(lines) + "\n"


def read_nvml() -> list[dict]:
    import pynvml as nv

    nv.nvmlInit()
    out = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        mem, util = nv.nvmlDeviceGetMemoryInfo(h), nv.nvmlDeviceGetUtilizationRates(h)
        name, uuid = nv.nvmlDeviceGetName(h), nv.nvmlDeviceGetUUID(h)
        g = {"index": i, "name": name.decode() if isinstance(name, bytes) else name,
             "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
             "memory_used_bytes": mem.used, "memory_total_bytes": mem.total,
             "utilization_ratio": util.gpu / 100.0, "memory_bandwidth_utilization_ratio": util.memory / 100.0}
        for key, fn in (("power_watts", lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000.0),
                        ("temperature_celsius", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))):
            try:
                g[key] = fn()
            except nv.NVMLError:  # not supported on every board / vGPU
                g[key] = None
        try:
            g["processes"] = {p.pid: p.usedGpuMemory for p in nv.nvmlDeviceGetComputeRunningProcesses(h)
                              if p.usedGpuMemory is not None}
        except nv.NVMLError:
            g["processes"] = {}
        out.append(g)
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/metrics":
            self.send_error(404)
            return
        try:
            body = render(read_nvml())
        except Exception as e:  # driver gone, NVML missing: report down instead of failing the scrape
            sys.stderr.write(f"gpu_exporter: NVML read failed: {e}\n")
            body = render([], up=False)
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9400)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    http.server.ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
