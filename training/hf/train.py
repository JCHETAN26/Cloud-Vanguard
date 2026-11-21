"""LoRA / QLoRA fine-tuning of a Hugging Face causal LM, single GPU or DDP (torchrun).

Same building blocks as training.train (the tiny-model reference loop): training.models.lora for the
adapters (saved in PEFT layout, loadable by peft and vLLM), training.models.quant for the frozen 4/8-bit
base, training.data for the memory-mapped Arrow data, training.eval for the harness and MLflow.

Memory is reported three ways, all in train_summary.json:
  weights_gpu_mb             torch.cuda.memory_allocated right after loading (+ LoRA injection)
  peak_allocated_mb          torch.cuda.max_memory_allocated over the training steps (tensors in use)
  peak_reserved_mb           torch.cuda.max_memory_reserved (allocator high-water mark)
  nvml[<gpu>].peak_memory_used_mb   what nvidia-smi shows (adds CUDA context and allocator slack)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields

import torch

from ..data import ArrowTokenDataset, LoaderConfig, build_dataloader, set_epoch
from ..eval import Tracker, evaluate
from ..eval.harness import percentile
from ..models import LoRAConfig, count_parameters, inject_lora, resolve_device, save_adapter, weight_bytes
from ..models.distributed import all_reduce_max, all_reduce_sum, barrier, cleanup, init_distributed
from ..models.precision import autocast
from .gpu import GPUSampler
from .model import HFBackend, load_hf_model, load_tokenizer

log = logging.getLogger("training.hf")

ALL_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class HFTrainConfig:
    model: str = ""                     # Hugging Face hub id or local directory
    train_data: str = ""
    eval_data: str = ""
    output_dir: str = "runs/hf"
    quant_bits: int = 0                 # 0 = bf16 LoRA, 4 = QLoRA NF4, 8 = int8
    quant_backend: str = "auto"
    dtype: str = "bf16"                 # dtype of the (non-quantized) weights and of compute
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: list(ALL_PROJ))
    steps: int = 250
    batch_size: int = 16                # per rank
    grad_accum: int = 1
    lr: float = 2e-4
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    max_len: int = 256
    gradient_checkpointing: bool = True
    device: str = "auto"
    seed: int = 0
    log_every: int = 10
    eval_examples: int = 500
    max_new_tokens: int = 6
    timing_warmup_steps: int = 3
    tracking_uri: str = ""
    experiment: str = "cloudvanguard-hf"
    run_name: str = ""
    track: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> HFTrainConfig:
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def validate(self) -> None:
        if not self.model or not self.train_data:
            raise ValueError("model and train_data are required")
        if self.quant_bits not in (0, 4, 8):
            raise ValueError("quant_bits must be 0, 4 or 8")
        if self.steps < 1 or self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("steps, batch_size and grad_accum must be >= 1")


def _lr_at(step: int, cfg: HFTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def _mb(n: float) -> float:
    return n / 2**20


def _wrap_ddp(model: torch.nn.Module, ctx) -> torch.nn.Module:
    """DDP over the adapter only. Frozen weights (including bitsandbytes 4-bit params, whose quantization
    state DDP cannot broadcast) are identical on every rank by construction, so they are excluded from
    DDP's initial broadcast and buffer sync."""
    if not ctx.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    model._ddp_params_and_buffers_to_ignore = (
        [n for n, p in model.named_parameters() if not p.requires_grad] + [n for n, _ in model.named_buffers()])
    ids = [ctx.device.index] if ctx.device.type == "cuda" else None
    return DDP(model, device_ids=ids, broadcast_buffers=False)


def train_hf(cfg: HFTrainConfig) -> dict:
    cfg.validate()
    device = resolve_device(cfg.device)
    ctx = init_distributed(device)
    device = ctx.device
    precision = cfg.dtype  # autocast dtype; LoRA weights stay fp32
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    tok = load_tokenizer(cfg.model)
    model, info = load_hf_model(cfg.model, device=device, dtype=cfg.dtype, quant_bits=cfg.quant_bits,
                                quant_backend=cfg.quant_backend, gradient_checkpointing=cfg.gradient_checkpointing)
    lcfg = LoRAConfig(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_targets))
    info["lora_modules"] = len(inject_lora(model, lcfg))
    model.to(device)  # the new lora_A/B layers
    info["trainable_params"], info["total_params"] = count_parameters(model)
    info["weight_bytes"] = weight_bytes(model)
    if cuda:
        torch.cuda.synchronize(device)
        info["weights_gpu_mb"] = _mb(torch.cuda.memory_allocated(device))
        info["load_peak_allocated_mb"] = _mb(torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    ddp_model = _wrap_ddp(model, ctx)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    train_ds = ArrowTokenDataset(cfg.train_data, max_len=cfg.max_len)
    eval_ds = ArrowTokenDataset(cfg.eval_data, max_len=cfg.max_len) if cfg.eval_data else None
    loader = build_dataloader(train_ds, LoaderConfig(batch_size=cfg.batch_size, seed=cfg.seed),
                              rank=ctx.rank, world_size=ctx.world_size)
    if len(loader) == 0:
        raise ValueError(f"{cfg.train_data}: fewer rows than one batch per rank")

    tracker = Tracker(cfg.output_dir, tracking_uri=cfg.tracking_uri or None, experiment=cfg.experiment,
                      run_name=cfg.run_name or None, enabled=cfg.track and ctx.is_main)
    summary: dict = {}
    sampler = GPUSampler(interval_s=0.25, indices=[device.index] if cuda else None)
    with tracker:
        tracker.log_params({**asdict(cfg), "world_size": ctx.world_size, "device": str(device),
                            "precision": precision, "torch_version": torch.__version__,
                            **{f"model.{k}": v for k, v in info.items()}})
        tracker.set_tags({"quant_bits": str(cfg.quant_bits), "hf_model": cfg.model, "world_size": str(ctx.world_size)})
        if ctx.is_main:
            log.info("%s quant_bits=%d world=%d trainable=%d/%d (%.3f%%) weights_gpu=%s MB", cfg.model,
                     cfg.quant_bits, ctx.world_size, info["trainable_params"], info["total_params"],
                     100 * info["trainable_params"] / info["total_params"], info.get("weights_gpu_mb"))
        model.train()
        it, epoch = iter(loader), 0
        step_ms: list[float] = []
        timed_tokens, timed_time, tokens_local = 0, 0.0, 0
        losses: list[float] = []
        t_train = time.perf_counter()
        with sampler:
            for step in range(cfg.steps):
                for g in optim.param_groups:
                    g["lr"] = _lr_at(step, cfg)
                t_step = time.perf_counter()
                step_tokens, loss_acc = 0, 0.0
                for micro in range(cfg.grad_accum):
                    try:
                        batch = next(it)
                    except StopIteration:
                        epoch += 1
                        set_epoch(loader, epoch)
                        it = iter(loader)
                        batch = next(it)
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    step_tokens += int(batch["attention_mask"].sum())
                    sync = micro == cfg.grad_accum - 1
                    cm = ddp_model.no_sync() if (ctx.enabled and not sync) else _null()
                    with cm:
                        with autocast(device, precision):
                            loss = ddp_model(input_ids=ids, attention_mask=mask, labels=labels).loss
                        (loss / cfg.grad_accum).backward()
                    loss_acc += float(loss.detach()) / cfg.grad_accum
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.synchronize(device)
                dt = time.perf_counter() - t_step
                tokens_local += step_tokens
                if step >= cfg.timing_warmup_steps:
                    step_ms.append(dt * 1000)
                    timed_tokens += step_tokens
                    timed_time += dt
                losses.append(loss_acc)
                if ctx.is_main and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                    m = {"train/loss": loss_acc, "train/lr": optim.param_groups[0]["lr"], "train/step_ms": dt * 1000}
                    if cuda:
                        m["gpu/allocated_mb"] = _mb(torch.cuda.memory_allocated(device))
                        m["gpu/peak_allocated_mb"] = _mb(torch.cuda.max_memory_allocated(device))
                    tracker.log_metrics(m, step)
                    log.info("step %d loss %.4f %.0f ms", step, loss_acc, dt * 1000)
        train_s = time.perf_counter() - t_train

        total_timed_tokens = all_reduce_sum(float(timed_tokens), ctx)
        slowest = all_reduce_max(timed_time, ctx)
        mem: dict = {}
        if cuda:
            mem = {"peak_allocated_mb": _mb(torch.cuda.max_memory_allocated(device)),
                   "peak_reserved_mb": _mb(torch.cuda.max_memory_reserved(device))}
            mem["peak_allocated_mb_max_rank"] = all_reduce_max(mem["peak_allocated_mb"], ctx)
        summary = {
            "model": cfg.model, "quant_bits": cfg.quant_bits, "dtype": cfg.dtype, "precision": precision,
            "device": str(device), "world_size": ctx.world_size, "steps": cfg.steps,
            "batch_size_per_rank": cfg.batch_size, "grad_accum": cfg.grad_accum,
            "global_batch": cfg.batch_size * cfg.grad_accum * ctx.world_size, "max_len": cfg.max_len,
            "gradient_checkpointing": cfg.gradient_checkpointing, "lora_r": cfg.lora_r, **info,
            "trainable_pct": 100.0 * info["trainable_params"] / info["total_params"],
            "first_loss": losses[0], "final_loss_mean_last10": sum(losses[-10:]) / len(losses[-10:]),
            "tokens_seen": all_reduce_sum(float(tokens_local), ctx), "train_s": train_s,
            "throughput_tokens_per_s": (total_timed_tokens / slowest) if slowest > 0 else None,
            "step_ms": {"p50": percentile(step_ms, 50), "p95": percentile(step_ms, 95)},
            **mem, "nvml": sampler.summary(),
            "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
            "torch_version": torch.__version__,
        }
        if ctx.enabled:  # every rank's NVML peak, gathered as one number per rank on rank 0's summary
            summary["nvml_peak_memory_used_mb_max_rank"] = all_reduce_max(
                max((v["peak_memory_used_mb"] for v in summary["nvml"].values()), default=0.0), ctx)

        if ctx.is_main:
            out = os.path.join(cfg.output_dir, "adapter")
            save_adapter(model, out, base_model=cfg.model,
                         extra={"quant_bits": cfg.quant_bits, "max_len": cfg.max_len, "hf_model": cfg.model})
            summary["adapter_dir"] = out
            summary["adapter_bytes"] = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
            if eval_ds is not None and cfg.eval_examples:
                n = min(cfg.eval_examples, len(eval_ds))
                t = eval_ds.table.slice(0, n)
                examples = [{"prompt": p, "answer": a, "text": x} for p, a, x in
                            zip(t.column("prompt").to_pylist(), t.column("answer").to_pylist(),
                                t.column("text").to_pylist())]
                rep = evaluate(HFBackend(model, tok, device=device, precision=precision, max_len=cfg.max_len),
                               examples, max_new_tokens=cfg.max_new_tokens)
                summary["eval"] = rep.to_dict()
                tracker.log_metrics(rep.metrics(), cfg.steps)
            tracker.log_metrics({k: v for k, v in {
                "final/train_loss": summary["final_loss_mean_last10"],
                "final/throughput_tokens_per_s": summary["throughput_tokens_per_s"],
                "final/peak_allocated_mb": summary.get("peak_allocated_mb"),
                "final/weights_gpu_mb": summary.get("weights_gpu_mb")}.items() if v is not None}, cfg.steps)
            summary["mlflow_run_id"] = tracker.run_id
            path = os.path.join(cfg.output_dir, "train_summary.json")
            with open(path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            tracker.log_artifact(path)
            tracker.log_artifact(out, artifact_path="adapter")
    barrier(ctx)
    cleanup(ctx)
    return summary


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
