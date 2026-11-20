"""Experiment tracking with MLflow, local by default.

The tracking URI is, in order: the explicit argument, ``MLFLOW_TRACKING_URI``,
then ``sqlite:///<output_dir>/mlflow.db``. SQLite and ``file:`` stores need no
server; browse them with ``mlflow ui --backend-store-uri <uri>``.

Only rank 0 logs under DDP (other ranks get a disabled tracker). If MLflow is
not installed, metrics go to ``<output_dir>/metrics.jsonl`` instead and a
warning says so; training never fails because tracking is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from typing import Any

log = logging.getLogger(__name__)


def default_tracking_uri(output_dir: str) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{os.path.abspath(output_dir)}/mlflow.db"


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, output_dir: str, *, tracking_uri: str | None = None, experiment: str = "cloudvanguard-training",
                 run_name: str | None = None, enabled: bool = True):
        self.output_dir = output_dir
        self.tracking_uri = tracking_uri or default_tracking_uri(output_dir)
        self.experiment = experiment
        self.run_name = run_name
        self.enabled = enabled
        self.backend = "disabled"
        self.run_id: str | None = None
        self._mlflow = None
        self._jsonl = None

    def __enter__(self) -> Tracker:
        self.start()
        return self

    def __exit__(self, exc_type, *_):
        self.end("FAILED" if exc_type else "FINISHED")

    def start(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow
        except ImportError:
            warnings.warn("mlflow is not installed; logging metrics to metrics.jsonl", stacklevel=2)
            self.backend = "jsonl"
            self._jsonl = open(os.path.join(self.output_dir, "metrics.jsonl"), "a")
            self.run_id = f"jsonl-{int(time.time())}"
            return
        self._mlflow = mlflow
        if self.tracking_uri.startswith("file:") or "://" not in self.tracking_uri:
            # MLflow >= 3.x refuses the file store unless opted in; a local file store is a supported choice here.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        self.run_id = mlflow.start_run(run_name=self.run_name).info.run_id
        self.backend = "mlflow"
        log.info("MLflow run %s at %s", self.run_id, self.tracking_uri)

    def _write(self, kind: str, payload: dict, step: int | None = None) -> None:
        self._jsonl.write(json.dumps({"kind": kind, "step": step, "time": time.time(), **payload}) + "\n")
        self._jsonl.flush()

    def log_params(self, params: dict) -> None:
        flat = {k: ("" if v is None else v) for k, v in _flatten(params).items()}
        if self.backend == "mlflow":
            self._mlflow.log_params({k: str(v)[:6000] for k, v in flat.items()})
        elif self.backend == "jsonl":
            self._write("params", {"params": {k: str(v) for k, v in flat.items()}})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if self.backend == "mlflow":
            self._mlflow.log_metrics(clean, step=step)
        elif self.backend == "jsonl":
            self._write("metrics", {"metrics": clean}, step)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self.backend == "mlflow":
            self._mlflow.set_tags(tags)
        elif self.backend == "jsonl":
            self._write("tags", {"tags": tags})

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if self.backend != "mlflow":
            return
        if os.path.isdir(path):
            self._mlflow.log_artifacts(path, artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def end(self, status: str = "FINISHED") -> None:
        if self.backend == "mlflow":
            self._mlflow.end_run(status=status)
        elif self.backend == "jsonl" and self._jsonl:
            self._jsonl.close()
        self.backend = "disabled"  # further calls are no-ops; run_id stays for lookups
