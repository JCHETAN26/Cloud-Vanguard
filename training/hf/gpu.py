"""GPU memory/utilization sampling through NVML (nvidia-ml-py), for training summaries and MLflow.

torch.cuda.max_memory_allocated counts tensors the caching allocator handed out. NVML's "used" is what
nvidia-smi shows: allocated + allocator cache + CUDA context + other processes. Both are reported, because
they answer different questions (what the model needs vs. what the device had to have free).
"""

from __future__ import annotations

import threading


def _nvml():
    try:
        import pynvml  # provided by the nvidia-ml-py package
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_snapshot() -> list[dict]:
    """One reading per visible GPU; [] without NVML or a GPU."""
    nv = _nvml()
    if nv is None:
        return []
    out = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        name = nv.nvmlDeviceGetName(h)
        out.append({
            "index": i, "name": name.decode() if isinstance(name, bytes) else name,
            "memory_used_mb": mem.used / 2**20, "memory_total_mb": mem.total / 2**20,
            "utilization_pct": float(util.gpu),
        })
    return out


class GPUSampler:
    """Background thread recording peak NVML memory and mean utilization per GPU while active.

    with GPUSampler(interval_s=0.2) as s: ...train...
    s.summary() -> {"0": {"peak_memory_used_mb": ..., "mean_utilization_pct": ..., "samples": n}, ...}
    """

    def __init__(self, interval_s: float = 0.25, indices: list[int] | None = None):
        self.interval_s = interval_s
        self.indices = indices
        self._peak: dict[int, float] = {}
        self._util: dict[int, list[float]] = {}
        self._names: dict[int, str] = {}
        self._total: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = _nvml() is not None

    def _tick(self) -> None:
        for g in gpu_snapshot():
            i = g["index"]
            if self.indices is not None and i not in self.indices:
                continue
            self._peak[i] = max(self._peak.get(i, 0.0), g["memory_used_mb"])
            self._util.setdefault(i, []).append(g["utilization_pct"])
            self._names[i], self._total[i] = g["name"], g["memory_total_mb"]

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GPUSampler:
        if self.available:
            self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._tick()
        return False

    def summary(self) -> dict[str, dict]:
        return {str(i): {"name": self._names[i], "memory_total_mb": self._total[i],
                         "peak_memory_used_mb": self._peak[i],
                         "mean_utilization_pct": sum(self._util[i]) / len(self._util[i]),
                         "samples": len(self._util[i])} for i in sorted(self._peak)}
