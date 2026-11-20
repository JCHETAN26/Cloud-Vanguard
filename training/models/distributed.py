"""DDP / multi-process hooks.

Launch with torchrun (``torchrun --nproc_per_node=N -m training train ...``):
torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT and
init_distributed() picks them up. NCCL is used on CUDA, Gloo on CPU. Without
those variables everything runs single-process and the hooks are no-ops.

With LoRA only the adapter parameters require gradients, so DDP all-reduces
just the adapter gradients (a few hundred KB per step instead of the model).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = ""
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def init_distributed(device: torch.device, *, timeout_s: int = 1800) -> DistContext:
    """timeout_s bounds collectives and barriers: rank 0 runs the final evaluation while the other ranks
    wait at a barrier, so it must cover the evaluation, not just a training step."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return DistContext(device=device)
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if device.type == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world, timeout=timedelta(seconds=timeout_s))
    return DistContext(rank=rank, world_size=world, local_rank=local, backend=backend, device=device)


def wrap_model(model: nn.Module, ctx: DistContext) -> nn.Module:
    if not ctx.enabled:
        return model
    ids = [ctx.device.index] if ctx.device.type == "cuda" else None
    # broadcast_buffers=False: the only buffers are frozen quantized weights, identical on every
    # rank; the default would re-broadcast them from rank 0 before every forward.
    return DDP(model, device_ids=ids, broadcast_buffers=False)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def all_reduce_sum(value: float, ctx: DistContext) -> float:
    if not ctx.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(t)
    return float(t.item())


def all_reduce_max(value: float, ctx: DistContext) -> float:
    if not ctx.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def barrier(ctx: DistContext) -> None:
    if ctx.enabled:
        dist.barrier()


def cleanup(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()
