import json
import sys

import pytest

from training.eval import Tracker, default_tracking_uri

mlflow = pytest.importorskip("mlflow")


def _check_run(uri, run_id, artifact_name):
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["lora.r"] == "8" and run.data.params["mode"] == "lora"
    assert run.data.tags["mode"] == "lora"
    hist = client.get_metric_history(run_id, "train/loss")
    assert [(m.step, m.value) for m in hist] == [(0, 2.5), (1, 1.5), (2, 1.0)]
    assert run.data.metrics["eval/perplexity"] == pytest.approx(3.25)
    assert artifact_name in [a.path for a in client.list_artifacts(run_id)]


@pytest.mark.parametrize("store", ["sqlite", "file"])
def test_logs_params_metrics_and_artifacts_locally(tmp_path, store):
    uri = f"sqlite:///{tmp_path}/mlflow.db" if store == "sqlite" else f"file://{tmp_path}/mlruns"
    note = tmp_path / "summary.json"
    note.write_text(json.dumps({"ok": True}))
    with Tracker(str(tmp_path), tracking_uri=uri, experiment=f"t-{store}", run_name="unit") as t:
        assert t.backend == "mlflow"
        t.log_params({"mode": "lora", "lora": {"r": 8, "targets": ["q", "v"]}, "unused": None})
        t.set_tags({"mode": "lora"})
        for step, loss in enumerate([2.5, 1.5, 1.0]):
            t.log_metrics({"train/loss": loss, "skipped": None}, step)
        t.log_metrics({"eval/perplexity": 3.25})
        t.log_artifact(str(note))
    _check_run(uri, t.run_id, "summary.json")


def test_default_uri_is_local_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert default_tracking_uri(str(tmp_path)) == f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///x")
    assert default_tracking_uri(str(tmp_path)) == "file:///x"


def test_failed_block_marks_the_run_failed(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    with pytest.raises(RuntimeError):
        with Tracker(str(tmp_path), tracking_uri=uri, experiment="fail") as t:
            raise RuntimeError("boom")
    assert mlflow.tracking.MlflowClient(uri).get_run(t.run_id).info.status == "FAILED"


def test_disabled_tracker_is_a_no_op(tmp_path):
    with Tracker(str(tmp_path / "rank1"), enabled=False) as t:
        t.log_params({"a": 1})
        t.log_metrics({"x": 1.0}, 0)
        t.log_artifact(__file__)
    assert t.backend == "disabled" and t.run_id is None
    assert not (tmp_path / "rank1").exists()


def test_falls_back_to_jsonl_without_mlflow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # import mlflow -> ImportError
    with pytest.warns(UserWarning, match="mlflow is not installed"):
        t = Tracker(str(tmp_path))
        t.start()
    assert t.backend == "jsonl"
    t.log_params({"lr": 0.1})
    t.log_metrics({"train/loss": 1.25}, 3)
    t.end()
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["params"] == {"lr": "0.1"}
    assert lines[1]["metrics"] == {"train/loss": 1.25} and lines[1]["step"] == 3
