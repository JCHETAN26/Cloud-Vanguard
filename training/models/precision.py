"""Mixed precision: pick what the hardware supports and build the autocast context + grad scaler."""

from __future__ import annotations

import contextlib
import warnings

import torch

PRECISIONS = ("auto", "fp32", "bf16", "fp16")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; using CPU", stacklevel=2)
        return torch.device("cpu")
    return dev


def resolve_precision(requested: str, device: torch.device) -> str:
    """auto: bf16 on CUDA GPUs that support it, else fp16 on CUDA, else fp32 on CPU.
    fp16 on CPU is refused (no fast kernels, no loss scaling) and falls back to fp32;
    bf16 on CPU is honoured (CPU autocast supports bfloat16)."""
    if requested not in PRECISIONS:
        raise ValueError(f"precision {requested!r}: expected one of {PRECISIONS}")
    cuda = device.type == "cuda"
    if requested == "auto":
        if cuda:
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "fp32"
    if requested == "fp16" and not cuda:
        warnings.warn("fp16 mixed precision needs a CUDA device; using fp32", stacklevel=2)
        return "fp32"
    if requested == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        warnings.warn("this GPU has no bf16 support; using fp16 with loss scaling", stacklevel=2)
        return "fp16"
    return requested


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler:
    """Loss scaling is only needed (and only enabled) for fp16."""
    return torch.amp.GradScaler(device.type, enabled=precision == "fp16")
