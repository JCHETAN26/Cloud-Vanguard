"""Training loop: full fine-tuning / pre-training, LoRA and QLoRA, single process or DDP."""

from __future__ import annotations

import json
import logging
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field, fields

import torch

from .data import ArrowTokenDataset, LoaderConfig, build_dataloader, set_epoch
from .eval import Tracker, TorchBackend, evaluate
from .eval.harness import percentile
from .models import (LoRAConfig, ModelConfig, TinyCausalLM, autocast, count_parameters, grad_scaler, inject_lora,
                     load_model, quantize_model, resolve_device, resolve_precision, save_adapter, save_model, unwrap,
                     weight_bytes, wrap_model)
from .models.distributed import all_reduce_max, all_reduce_sum, barrier, cleanup, init_distributed

log = logging.getLogger("training")


@dataclass
class TrainConfig:
    train_data: str = ""
    eval_data: str = ""
    output_dir: str = "runs/latest"
    mode: str = "lora"                  # "full" (all weights) or "lora"
    base_model: str = ""                # checkpoint dir; empty = fresh model from `model`
    model: dict = field(default_factory=dict)  # ModelConfig overrides for a fresh model
    # LoRA
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Quantization of the frozen base (LoRA only): 0 = off, 8 = int8, 4 = nf4
    quant_bits: int = 0
    quant_backend: str = "auto"
    # Optimisation
    steps: int = 200
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-3
    weight_decay: float = 0.0
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    precision: str = "auto"
    device: str = "auto"
    seed: int = 0
    max_len: int = 128
    # Data loading
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: str = "auto"
    # Evaluation / logging
    log_every: int = 10
    eval_every: int = 0                 # 0 = only at the end
    eval_examples: int = 200            # harness examples at the end (exact match / perplexity / latency)
    max_new_tokens: int = 12
    timing_warmup_steps: int = 3        # excluded from step-latency statistics
    tracking_uri: str = ""
    experiment: str = "cloudvanguard-training"
    run_name: str = ""
    track: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def validate(self) -> None:
        if self.mode not in ("full", "lora"):
            raise ValueError(f"mode must be 'full' or 'lora', got {self.mode!r}")
        if self.quant_bits not in (0, 4, 8):
            raise ValueError("quant_bits must be 0, 4 or 8")
        if self.quant_bits and self.mode != "lora":
            raise ValueError("quantized bases are frozen: quant_bits requires mode='lora' (QLoRA)")
        if not self.train_data:
            raise ValueError("train_data is required")
        if self.steps < 1 or self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("steps, batch_size and grad_accum must be >= 1")


def peak_memory_mb(device: torch.device) -> dict[str, float]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes on macOS, KiB on Linux
    out = {"peak_rss_mb": rss_mb}
    if device.type == "cuda":
        out["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def build_model(cfg: TrainConfig) -> tuple[torch.nn.Module, dict]:
    """Base model (fresh or from a checkpoint), quantized and LoRA-injected as configured."""
    info: dict = {}
    if cfg.base_model:
        model = load_model(cfg.base_model)
    else:
        model = TinyCausalLM(ModelConfig(**{"max_len": cfg.max_len, **cfg.model}))
    info["base_weight_bytes"] = weight_bytes(model)
    if cfg.quant_bits:
        info["quant_backend"] = quantize_model(model, cfg.quant_bits, backend=cfg.quant_backend)
    if cfg.mode == "lora":
        lcfg = LoRAConfig(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                          target_modules=list(cfg.lora_targets))
        info["lora_modules"] = len(inject_lora(model, lcfg))
    info["weight_bytes"] = weight_bytes(model)
    info["trainable_params"], info["total_params"] = count_parameters(model)
    return model, info


@torch.no_grad()
def eval_loss(model: torch.nn.Module, ds: ArrowTokenDataset, device: torch.device, precision: str,
              batch_size: int = 64, max_batches: int = 50) -> float:
    model.eval()
    loader = build_dataloader(ds, LoaderConfig(batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False))
    total, count = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        n = int((b["labels"][:, 1:] != -100).sum())
        with autocast(device, precision):
            _, loss = model(b["input_ids"].to(device), labels=b["labels"].to(device))
        total += float(loss) * n
        count += n
    model.train()
    return total / max(count, 1)


def train(cfg: TrainConfig) -> dict:
    cfg.validate()
    device = resolve_device(cfg.device)
    ctx = init_distributed(device)
    device = ctx.device
    precision = resolve_precision(cfg.precision, device)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.track and ctx.is_main:
        try:  # import before the baseline so tracking overhead is not attributed to the model
            import mlflow  # noqa: F401
        except ImportError:
            pass
    baseline_rss = peak_memory_mb(device)["peak_rss_mb"]  # interpreter + torch + mlflow, before any model/data

    model, info = build_model(cfg)
    model.to(device)
    ddp_model = wrap_model(model, ctx)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = grad_scaler(device, precision)

    train_ds = ArrowTokenDataset(cfg.train_data, max_len=cfg.max_len)
    eval_ds = ArrowTokenDataset(cfg.eval_data, max_len=cfg.max_len) if cfg.eval_data else None
    lcfg = LoaderConfig(batch_size=cfg.batch_size, seed=cfg.seed, num_workers=cfg.num_workers,
                        prefetch_factor=cfg.prefetch_factor,
                        pin_memory=cfg.pin_memory if cfg.pin_memory == "auto" else cfg.pin_memory == "true")
    loader = build_dataloader(train_ds, lcfg, rank=ctx.rank, world_size=ctx.world_size)
    if len(loader) == 0:
        raise ValueError(f"{cfg.train_data}: fewer rows than one batch per rank")

    tracker = Tracker(cfg.output_dir, tracking_uri=cfg.tracking_uri or None, experiment=cfg.experiment,
                      run_name=cfg.run_name or None, enabled=cfg.track and ctx.is_main)
    summary: dict = {}
    with tracker:
        tracker.log_params({**asdict(cfg), "world_size": ctx.world_size, "device": str(device),
                            "resolved_precision": precision, "dist_backend": ctx.backend or "none",
                            "torch_version": torch.__version__, **{f"model.{k}": v for k, v in info.items()}})
        tracker.set_tags({"mode": cfg.mode, "quant_bits": str(cfg.quant_bits), "precision": precision})
        if ctx.is_main:
            log.info("mode=%s precision=%s device=%s world=%d trainable=%d/%d (%.3f%%)", cfg.mode, precision, device,
                     ctx.world_size, info["trainable_params"], info["total_params"],
                     100 * info["trainable_params"] / info["total_params"])

        model.train()
        it, epoch = iter(loader), 0
        step_ms: list[float] = []
        data_ms: list[float] = []
        tokens_local = 0
        timed_tokens = 0
        timed_time = 0.0
        first_loss = last_loss = None
        losses: list[float] = []
        for step in range(cfg.steps):
            for g in optim.param_groups:
                g["lr"] = _lr_at(step, cfg)
            t_step = time.perf_counter()
            wait = 0.0
            step_tokens = 0
            loss_acc = 0.0
            for micro in range(cfg.grad_accum):
                t0 = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    epoch += 1
                    set_epoch(loader, epoch)
                    it = iter(loader)
                    batch = next(it)
                wait += time.perf_counter() - t0
                ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                step_tokens += int(batch["attention_mask"].sum())
                sync = micro == cfg.grad_accum - 1
                cm = ddp_model.no_sync() if (ctx.enabled and not sync) else _null()
                with cm:
                    with autocast(device, precision):
                        _, loss = ddp_model(ids, labels=labels)
                    scaler.scale(loss / cfg.grad_accum).backward()
                loss_acc += float(loss.detach()) / cfg.grad_accum
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t_step
            tokens_local += step_tokens
            if step >= cfg.timing_warmup_steps:
                step_ms.append(dt * 1000)
                data_ms.append(wait * 1000)
                timed_tokens += step_tokens
                timed_time += dt
            first_loss = loss_acc if first_loss is None else first_loss
            last_loss = loss_acc
            losses.append(loss_acc)
            if ctx.is_main and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                tracker.log_metrics({"train/loss": loss_acc, "train/lr": optim.param_groups[0]["lr"],
                                     "train/step_ms": dt * 1000, "train/tokens_per_s_rank0": step_tokens / dt}, step)
                log.info("step %d loss %.4f lr %.2e %.1f ms", step, loss_acc, optim.param_groups[0]["lr"], dt * 1000)
            if cfg.eval_every and eval_ds is not None and step and step % cfg.eval_every == 0 and ctx.is_main:
                el = eval_loss(model, eval_ds, device, precision)
                tracker.log_metrics({"eval/loss": el, "eval/perplexity": math.exp(el)}, step)
            if cfg.eval_every and ctx.enabled and step and step % cfg.eval_every == 0:
                barrier(ctx)

        # Throughput over the timed (post-warm-up) steps, summed across ranks.
        total_timed_tokens = all_reduce_sum(float(timed_tokens), ctx)
        slowest = all_reduce_max(timed_time, ctx)
        mem = peak_memory_mb(device)
        peak_rss_max = all_reduce_max(mem["peak_rss_mb"], ctx)
        summary = {
            "mode": cfg.mode, "quant_bits": cfg.quant_bits, "precision": precision, "device": str(device),
            "world_size": ctx.world_size, "steps": cfg.steps, "batch_size_per_rank": cfg.batch_size,
            "grad_accum": cfg.grad_accum, **info,
            "trainable_pct": 100.0 * info["trainable_params"] / info["total_params"],
            "first_loss": first_loss, "final_loss": last_loss,
            "final_loss_mean_last10": sum(losses[-10:]) / len(losses[-10:]),
            "tokens_seen": all_reduce_sum(float(tokens_local), ctx),
            "throughput_tokens_per_s": (total_timed_tokens / slowest) if slowest > 0 else None,
            "step_ms": {"p50": percentile(step_ms, 50), "p95": percentile(step_ms, 95),
                        "mean": sum(step_ms) / len(step_ms) if step_ms else float("nan")},
            "data_wait_ms_mean": sum(data_ms) / len(data_ms) if data_ms else float("nan"),
            **mem, "peak_rss_mb_max_rank": peak_rss_max, "baseline_rss_mb": baseline_rss,
            "peak_rss_over_baseline_mb": mem["peak_rss_mb"] - baseline_rss,
        }

        if ctx.is_main:
            if cfg.mode == "lora":
                out = os.path.join(cfg.output_dir, "adapter")
                save_adapter(model, out, base_model=os.path.abspath(cfg.base_model) if cfg.base_model else "",
                             extra={"quant_bits": cfg.quant_bits, "max_len": cfg.max_len})
                summary["adapter_dir"] = out
                summary["adapter_bytes"] = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
            else:
                out = os.path.join(cfg.output_dir, "model")
                save_model(model, out)
                summary["model_dir"] = out
            if eval_ds is not None:
                summary["eval_loss"] = eval_loss(model, eval_ds, device, precision)
                summary["eval_perplexity_teacher_forced"] = math.exp(summary["eval_loss"])
                n = min(cfg.eval_examples, len(eval_ds))
                t = eval_ds.table.slice(0, n)
                examples = [{"prompt": p, "answer": a, "text": x} for p, a, x in
                            zip(t.column("prompt").to_pylist(), t.column("answer").to_pylist(),
                                t.column("text").to_pylist())] if "prompt" in t.column_names else []
                if examples:
                    rep = evaluate(TorchBackend(unwrap(model), device=device, precision=precision), examples,
                                   max_new_tokens=cfg.max_new_tokens,
                                   generate=any(e["prompt"] for e in examples))
                    summary["eval"] = rep.to_dict()
                    tracker.log_metrics(rep.metrics(), cfg.steps)
                model.train()
            tracker.log_metrics({k: v for k, v in {
                "final/train_loss": summary["final_loss"], "final/throughput_tokens_per_s":
                summary["throughput_tokens_per_s"], "final/step_ms_p50": summary["step_ms"]["p50"],
                "final/step_ms_p95": summary["step_ms"]["p95"], "final/peak_rss_mb": summary["peak_rss_mb"],
                "final/eval_loss": summary.get("eval_loss")}.items() if v is not None and not _isnan(v)}, cfg.steps)
            summary["mlflow_run_id"] = tracker.run_id
            summary["tracking_uri"] = tracker.tracking_uri if tracker.enabled else None
            path = os.path.join(cfg.output_dir, "train_summary.json")
            with open(path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            tracker.log_artifact(path)
            if "adapter_dir" in summary:
                tracker.log_artifact(summary["adapter_dir"], artifact_path="adapter")
    barrier(ctx)
    cleanup(ctx)
    return summary


def _isnan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
