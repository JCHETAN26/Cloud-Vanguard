"""Command line: python -m training {info,prepare-data,prepare-hf-data,train,train-hf,eval}.

Examples (from the repository root):

  python -m training prepare-data --out runs/data --task reverse
  python -m training train --config training/configs/pretrain_tiny.json \
      --train-data runs/data/pretrain.arrow --eval-data runs/data/pretrain_eval.arrow --output-dir runs/base
  python -m training train --config training/configs/lora_tiny.json --base-model runs/base/model \
      --train-data runs/data/reverse_train.arrow --eval-data runs/data/reverse_eval.arrow --output-dir runs/lora
  torchrun --nproc_per_node=2 -m training train --config training/configs/lora_tiny.json ...
  python -m training eval --base-model runs/base/model --adapter runs/lora/adapter --data runs/data/reverse_eval.arrow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys

from . import __version__


def _cmd_info(_: argparse.Namespace) -> int:
    import torch

    from .eval.backends import vllm_available
    from .models.quant import bnb_available

    def has(mod: str) -> str | None:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "installed")
        except Exception:
            return None

    cuda = torch.cuda.is_available()
    info = {
        "training": __version__, "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": cuda, "cuda_devices": torch.cuda.device_count() if cuda else 0,
        "bf16_gpu": bool(cuda and torch.cuda.is_bf16_supported()), "cpu_threads": torch.get_num_threads(),
        "distributed_backends": {"gloo": torch.distributed.is_gloo_available(),
                                 "nccl": torch.distributed.is_nccl_available()},
        "pyarrow": has("pyarrow"), "mlflow": has("mlflow"), "safetensors": has("safetensors"),
        "bitsandbytes_usable": bnb_available(), "vllm_usable": vllm_available(),
        "transformers": has("transformers"), "peft": has("peft"), "nvml": has("pynvml"),
    }
    print(json.dumps(info, indent=2))
    return 0


def _cmd_prepare(a: argparse.Namespace) -> int:
    from .data import generate, write_arrow

    os.makedirs(a.out, exist_ok=True)
    outputs = {
        "pretrain.arrow": ("pretrain", a.pretrain_rows, a.seed),
        "pretrain_eval.arrow": ("pretrain", a.eval_rows, a.seed + 1),
        f"{a.task}_train.arrow": (a.task, a.train_rows, a.seed),
        f"{a.task}_eval.arrow": (a.task, a.eval_rows, a.seed + 1),
    }
    for name, (kind, rows, seed) in outputs.items():
        n = write_arrow(generate(kind, rows, seed), os.path.join(a.out, name))
        print(f"wrote {n:>7} rows  {os.path.join(a.out, name)}")
    return 0


_TRAIN_FLAGS = {
    "train_data": str, "eval_data": str, "output_dir": str, "mode": str, "base_model": str, "lora_r": int,
    "lora_alpha": float, "lora_dropout": float, "quant_bits": int, "quant_backend": str, "steps": int,
    "batch_size": int, "grad_accum": int, "lr": float, "weight_decay": float, "warmup_steps": int, "precision": str,
    "device": str, "seed": int, "max_len": int, "num_workers": int, "prefetch_factor": int, "pin_memory": str,
    "log_every": int, "eval_every": int, "eval_examples": int, "tracking_uri": str, "experiment": str,
    "run_name": str,
}


def _cmd_train(a: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    d: dict = {}
    if a.config:
        with open(a.config) as f:
            d.update(json.load(f))
    for k in _TRAIN_FLAGS:
        v = getattr(a, k)
        if v is not None:
            d[k] = v
    if a.lora_targets:
        d["lora_targets"] = a.lora_targets.split(",")
    if a.no_track:
        d["track"] = False
    summary = train(TrainConfig.from_dict(d))
    if summary:
        print(json.dumps({k: summary[k] for k in summary if k != "eval"} |
                         ({"eval": {k: v for k, v in summary["eval"].items() if k != "samples"}}
                          if "eval" in summary else {}), indent=2, default=str))
    return 0


def _cmd_prepare_hf(a: argparse.Namespace) -> int:
    from .hf.data import prepare

    written = prepare(a.out, a.tokenizer, train_rows=a.train_rows, eval_rows=a.eval_rows, seed=a.seed,
                      parquet_dir=a.parquet_dir or None)
    for name, n in written.items():
        print(f"wrote {n:>7} rows  {os.path.join(a.out, name)}")
    return 0


_HF_TRAIN_FLAGS = {
    "model": str, "train_data": str, "eval_data": str, "output_dir": str, "quant_bits": int, "quant_backend": str,
    "dtype": str, "lora_r": int, "lora_alpha": float, "lora_dropout": float, "steps": int, "batch_size": int,
    "grad_accum": int, "lr": float, "warmup_steps": int, "max_len": int, "device": str, "seed": int,
    "log_every": int, "eval_examples": int, "max_new_tokens": int, "tracking_uri": str, "experiment": str,
    "run_name": str,
}


def _cmd_train_hf(a: argparse.Namespace) -> int:
    from .hf.train import HFTrainConfig, train_hf

    d: dict = {}
    if a.config:
        with open(a.config) as f:
            d.update(json.load(f))
    for k in _HF_TRAIN_FLAGS:
        v = getattr(a, k)
        if v is not None:
            d[k] = v
    if a.lora_targets:
        d["lora_targets"] = a.lora_targets.split(",")
    if a.no_gradient_checkpointing:
        d["gradient_checkpointing"] = False
    if a.no_track:
        d["track"] = False
    summary = train_hf(HFTrainConfig.from_dict(d))
    if summary:
        print(json.dumps({k: summary[k] for k in summary if k != "eval"} |
                         ({"eval": {k: v for k, v in summary["eval"].items() if k != "samples"}}
                          if "eval" in summary else {}), indent=2, default=str))
    return 0


def _cmd_eval(a: argparse.Namespace) -> int:
    from .data import ArrowTokenDataset
    from .eval import OpenAIHTTPBackend, TorchBackend, Tracker, VLLMBackend, evaluate, vllm_available

    ds = ArrowTokenDataset(a.data)
    t = ds.table.slice(0, min(a.limit, len(ds)))
    examples = [{"prompt": p, "answer": ans, "text": x} for p, ans, x in
                zip(t.column("prompt").to_pylist(), t.column("answer").to_pylist(), t.column("text").to_pylist())]
    backend_name = a.backend
    if backend_name == "auto":
        backend_name = "vllm" if (a.hf_model and vllm_available()) else ("openai" if a.base_url else "torch")
    if backend_name == "torch":
        import torch

        from .models import load_adapter, load_model, quantize_model, resolve_device, resolve_precision
        if not a.base_model:
            raise SystemExit("--base-model is required for the torch backend")
        device = resolve_device(a.device)
        model = load_model(a.base_model)
        if a.quant_bits:
            quantize_model(model, a.quant_bits, backend=a.quant_backend)
        if a.adapter:
            load_adapter(model, a.adapter)
        model.to(device)
        torch.manual_seed(0)
        backend = TorchBackend(model, device=device, precision=resolve_precision(a.precision, device))
    elif backend_name == "hf":
        import torch

        from .hf import HFBackend, load_hf_model, load_tokenizer
        from .models import load_adapter, resolve_device
        if not a.hf_model:
            raise SystemExit("--hf-model is required for the hf backend")
        device = resolve_device(a.device)
        model, _ = load_hf_model(a.hf_model, device=device, dtype=a.precision if a.precision != "auto" else "bf16",
                                 quant_bits=a.quant_bits, quant_backend=a.quant_backend)
        if a.adapter:
            load_adapter(model, a.adapter)
            model.to(device)
        torch.manual_seed(0)
        backend = HFBackend(model, load_tokenizer(a.hf_model), device=device,
                            precision=a.precision if a.precision != "auto" else "bf16")
    elif backend_name == "vllm":
        if not a.hf_model:
            raise SystemExit("--hf-model is required for the vllm backend")
        backend = VLLMBackend(a.hf_model, adapter=a.adapter or None, max_model_len=a.max_model_len)
    elif backend_name == "openai":
        if not a.base_url:
            raise SystemExit("--base-url is required for the openai backend")
        backend = OpenAIHTTPBackend(a.base_url, model=a.model_name, api_key=a.api_key or os.environ.get("OPENAI_API_KEY"))
    else:
        raise SystemExit(f"unknown backend {a.backend}")
    rep = evaluate(backend, examples, max_new_tokens=a.max_new_tokens)
    backend.close()
    out = rep.to_dict()
    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        with open(a.output, "w") as f:
            json.dump(out, f, indent=2)
    if a.track:
        with Tracker(os.path.dirname(os.path.abspath(a.output or "runs/eval/x")), tracking_uri=a.tracking_uri or None,
                     experiment=a.experiment, run_name=a.run_name or f"eval-{backend.name}") as tr:
            tr.log_params({"backend": backend.name, "data": a.data, "adapter": a.adapter, "base_model": a.base_model,
                           "examples": len(examples)})
            tr.log_metrics(rep.metrics())
            out["mlflow_run_id"] = tr.run_id
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m training", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="report hardware and optional-dependency availability")

    pr = sub.add_parser("prepare-data", help="generate synthetic Arrow datasets (no network)")
    pr.add_argument("--out", required=True)
    pr.add_argument("--task", default="reverse", choices=["reverse", "upper", "add", "sort"])
    pr.add_argument("--pretrain-rows", type=int, default=20000)
    pr.add_argument("--train-rows", type=int, default=20000)
    pr.add_argument("--eval-rows", type=int, default=500)
    pr.add_argument("--seed", type=int, default=0)

    tr = sub.add_parser("train", help="train (full or LoRA/QLoRA); use torchrun for DDP")
    tr.add_argument("--config", help="JSON file with TrainConfig fields; flags override it")
    for k, typ in _TRAIN_FLAGS.items():
        tr.add_argument("--" + k.replace("_", "-"), dest=k, type=typ, default=None)
    tr.add_argument("--lora-targets", help="comma-separated module names, e.g. q_proj,v_proj")
    tr.add_argument("--no-track", action="store_true", help="disable MLflow")

    ph = sub.add_parser("prepare-hf-data", help="AG News topic classification tokenized for a Hugging Face model")
    ph.add_argument("--out", required=True)
    ph.add_argument("--tokenizer", required=True, help="Hugging Face model id or dir whose tokenizer to use")
    ph.add_argument("--train-rows", type=int, default=4000)
    ph.add_argument("--eval-rows", type=int, default=500)
    ph.add_argument("--seed", type=int, default=0)
    ph.add_argument("--parquet-dir", default="", help="directory with the AG News parquet files (offline)")

    th = sub.add_parser("train-hf", help="LoRA/QLoRA on a Hugging Face model (GPU); use torchrun for DDP")
    th.add_argument("--config", help="JSON file with HFTrainConfig fields; flags override it")
    for k, typ in _HF_TRAIN_FLAGS.items():
        th.add_argument("--" + k.replace("_", "-"), dest=k, type=typ, default=None)
    th.add_argument("--lora-targets", help="comma-separated module names")
    th.add_argument("--no-gradient-checkpointing", action="store_true")
    th.add_argument("--no-track", action="store_true", help="disable MLflow")

    ev = sub.add_parser("eval", help="evaluate a model/adapter: exact match, perplexity, latency")
    ev.add_argument("--data", required=True, help="Arrow file with prompt/answer/text columns")
    ev.add_argument("--backend", default="auto", choices=["auto", "torch", "hf", "vllm", "openai"])
    ev.add_argument("--base-model", default="", help="TinyCausalLM checkpoint dir (torch backend)")
    ev.add_argument("--adapter", default="", help="LoRA adapter dir")
    ev.add_argument("--hf-model", default="", help="Hugging Face model dir or id (hf and vllm backends)")
    ev.add_argument("--base-url", default="", help="OpenAI-compatible base URL, e.g. http://gateway:8080/v1")
    ev.add_argument("--model-name", default="default")
    ev.add_argument("--api-key", default="")
    ev.add_argument("--quant-bits", type=int, default=0)
    ev.add_argument("--quant-backend", default="auto")
    ev.add_argument("--precision", default="auto")
    ev.add_argument("--device", default="auto")
    ev.add_argument("--limit", type=int, default=200)
    ev.add_argument("--max-new-tokens", type=int, default=12)
    ev.add_argument("--max-model-len", type=int, default=2048,
                    help="vllm backend: context cap that sizes the KV cache (the model's own may be 128k)")
    ev.add_argument("--output", default="")
    ev.add_argument("--track", action="store_true", help="log the evaluation to MLflow")
    ev.add_argument("--tracking-uri", default="")
    ev.add_argument("--experiment", default="cloudvanguard-eval")
    ev.add_argument("--run-name", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose or a.cmd in ("train", "train-hf") else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return {"info": _cmd_info, "prepare-data": _cmd_prepare, "prepare-hf-data": _cmd_prepare_hf, "train": _cmd_train,
            "train-hf": _cmd_train_hf, "eval": _cmd_eval}[a.cmd](a)
