"""DataLoader construction: padding/label masking, pinned memory, prefetching, DDP sharding."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler

from .tokenizer import PAD

IGNORE_INDEX = -100


def collate(batch: list[dict], pad_id: int = PAD) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence. labels = input_ids with padding and prompt tokens set to -100."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for r, b in enumerate(batch):
        x = b["input_ids"]
        ids[r, : len(x)] = x
        mask[r, : len(x)] = 1
        start = min(int(b.get("prompt_len", 1)), len(x))
        labels[r, start: len(x)] = x[start:]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def resolve_pin_memory(requested: bool | str) -> bool:
    """'auto' pins when CUDA is present. Pinning without an accelerator is a no-op that warns, so it is refused."""
    has_accel = torch.cuda.is_available()
    if requested == "auto":
        return has_accel
    if requested and not has_accel:
        warnings.warn("pin_memory requested but no CUDA device is available; disabled", stacklevel=2)
        return False
    return bool(requested)


@dataclass
class LoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 0
    num_workers: int = 0
    prefetch_factor: int = 2  # batches buffered per worker (only with workers)
    pin_memory: bool | str = "auto"
    drop_last: bool = True


def build_dataloader(ds: Dataset, cfg: LoaderConfig, *, rank: int = 0, world_size: int = 1) -> DataLoader:
    """A DataLoader over ``ds``. With ``world_size > 1`` each rank gets a disjoint shard
    (DistributedSampler); call ``loader.sampler.set_epoch(e)`` each epoch to reshuffle."""
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=cfg.shuffle,
                                     seed=cfg.seed, drop_last=cfg.drop_last)
    elif cfg.shuffle:
        sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(cfg.seed))
    else:
        sampler = SequentialSampler(ds)
    kwargs = {}
    if cfg.num_workers > 0:
        kwargs = {"prefetch_factor": cfg.prefetch_factor, "persistent_workers": True}
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
                      collate_fn=collate, pin_memory=resolve_pin_memory(cfg.pin_memory),
                      drop_last=cfg.drop_last, **kwargs)


def set_epoch(loader: DataLoader, epoch: int) -> None:
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    elif isinstance(loader.sampler, RandomSampler) and loader.sampler.generator is not None:
        loader.sampler.generator.manual_seed(1_000_003 * (epoch + 1))
