"""End-to-end: the real CLI, in subprocesses, as a user would run it (CPU, no network)."""

import json
import os
import subprocess
import sys

import pytest

from .conftest import ROOT, free_port

pytestmark = pytest.mark.integration
MODEL = {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128}


def run(*args, extra_env=None, cmd=None):
    env = dict(os.environ, PYTHONPATH=ROOT, MLFLOW_DISABLE_AGENT_HINT="1", OMP_NUM_THREADS="2", **(extra_env or {}))
    p = subprocess.run((cmd or [sys.executable, "-m", "training"]) + list(args), cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, f"exit {p.returncode}\nstdout:\n{p.stdout[-3000:]}\nstderr:\n{p.stderr[-3000:]}"
    return p


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """prepare-data -> pre-train a base -> LoRA fine-tune on the reverse task."""
    d = tmp_path_factory.mktemp("e2e")
    run("prepare-data", "--out", str(d / "data"), "--task", "reverse", "--pretrain-rows", "2000",
        "--train-rows", "3000", "--eval-rows", "64")
    cfg = d / "base.json"
    cfg.write_text(json.dumps({"mode": "full", "model": MODEL, "steps": 60, "batch_size": 32, "lr": 3e-3,
                               "warmup_steps": 5, "log_every": 20, "max_len": 96}))
    run("train", "--config", str(cfg), "--train-data", str(d / "data/pretrain.arrow"),
        "--eval-data", str(d / "data/pretrain_eval.arrow"), "--output-dir", str(d / "base"))
    run("train", "--config", os.path.join(ROOT, "training/configs/lora_tiny.json"),
        "--base-model", str(d / "base/model"), "--train-data", str(d / "data/reverse_train.arrow"),
        "--eval-data", str(d / "data/reverse_eval.arrow"), "--output-dir", str(d / "lora"),
        "--steps", "150", "--eval-every", "50", "--eval-examples", "32", "--max-len", "96",
        "--num-workers", "2", "--prefetch-factor", "4")
    return d


def test_prepare_data_writes_arrow(pipeline):
    for name in ("pretrain.arrow", "pretrain_eval.arrow", "reverse_train.arrow", "reverse_eval.arrow"):
        assert (pipeline / "data" / name).stat().st_size > 0


def test_lora_run_outputs_summary_adapter_and_mlflow(pipeline):
    s = json.loads((pipeline / "lora/train_summary.json").read_text())
    # precision "auto" resolves per device: bf16/fp16 on a CUDA GPU, fp32 on CPU.
    import torch
    expected = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if torch.cuda.is_available() else "fp32"
    assert s["mode"] == "lora" and s["world_size"] == 1 and s["precision"] == expected
    assert 0 < s["trainable_pct"] < 20
    assert s["final_loss_mean_last10"] < s["first_loss"]
    assert s["throughput_tokens_per_s"] > 0 and s["step_ms"]["p50"] > 0 and s["peak_rss_mb"] > 0
    assert sorted(os.listdir(s["adapter_dir"]))[0] == "adapter_config.json"
    assert s["eval"]["examples"] == 32 and s["eval"]["perplexity"] > 1

    import mlflow
    client = mlflow.tracking.MlflowClient(s["tracking_uri"])
    assert s["tracking_uri"].startswith("sqlite:///")
    run_ = client.get_run(s["mlflow_run_id"])
    assert run_.info.status == "FINISHED" and run_.data.params["mode"] == "lora"
    assert len(client.get_metric_history(run_.info.run_id, "train/loss")) >= 2
    assert len(client.get_metric_history(run_.info.run_id, "eval/loss")) == 2  # steps 50 and 100
    assert "eval/exact_match" in run_.data.metrics
    assert {a.path for a in client.list_artifacts(run_.info.run_id)} >= {"train_summary.json", "adapter"}


def test_eval_cli_adapter_beats_base(pipeline):
    common = ["eval", "--backend", "torch", "--base-model", str(pipeline / "base/model"),
              "--data", str(pipeline / "data/reverse_eval.arrow"), "--limit", "32"]
    base = json.loads(run(*common).stdout)
    tuned = json.loads(run(*common, "--adapter", str(pipeline / "lora/adapter"), "--output",
                           str(pipeline / "eval.json"), "--track",
                           "--tracking-uri", f"sqlite:///{pipeline}/eval.db").stdout)
    assert tuned["perplexity"] < base["perplexity"] / 2
    assert tuned["mlflow_run_id"] and json.loads((pipeline / "eval.json").read_text())["examples"] == 32
    # The same adapter evaluated over a 4-bit base still runs (QLoRA-style inference).
    q = json.loads(run(*common, "--adapter", str(pipeline / "lora/adapter"), "--quant-bits", "4").stdout)
    assert q["perplexity"] < base["perplexity"]


def test_torchrun_two_process_qlora(pipeline):
    """torchrun (torch.distributed.run, same interpreter): 2 Gloo ranks, 4-bit base, rank-0-only logging."""
    out = pipeline / "ddp"
    run("--nproc_per_node=2", "--master_addr=127.0.0.1", f"--master_port={free_port()}", "-m", "training", "train",
        "--config", os.path.join(ROOT, "training/configs/qlora_tiny.json"), "--base-model", str(pipeline / "base/model"),
        "--train-data", str(pipeline / "data/reverse_train.arrow"), "--output-dir", str(out), "--steps", "20",
        "--batch-size", "16", "--max-len", "96", cmd=[sys.executable, "-m", "torch.distributed.run"])
    s = json.loads((out / "train_summary.json").read_text())
    assert s["world_size"] == 2 and s["quant_bits"] == 4 and s["quant_backend"] in ("torch", "bnb")
    assert s["batch_size_per_rank"] == 16 and s["tokens_seen"] > 0
    import mlflow
    client = mlflow.tracking.MlflowClient(s["tracking_uri"])  # this run's own sqlite store
    exp = client.get_experiment_by_name("cloudvanguard-training")
    assert len(client.search_runs([exp.experiment_id])) == 1  # only rank 0 logged


def test_invalid_configs_fail_fast(pipeline):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "training", "train", "--mode", "full", "--quant-bits", "4",
                        "--train-data", str(pipeline / "data/pretrain.arrow"), "--output-dir",
                        str(pipeline / "bad"), "--no-track"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "quant_bits requires mode='lora'" in p.stderr
