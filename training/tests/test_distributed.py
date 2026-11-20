"""DDP across real processes (Gloo on CPU): data-parallel LoRA training must equal single-process training."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from training.models import DistContext, LoRAConfig, ModelConfig, TinyCausalLM, inject_lora, lora_state_dict, wrap_model

from .conftest import SMALL, free_port

STEPS, BATCH, SEQ = 4, 8, 16


def make_model():
    torch.manual_seed(0)
    m = TinyCausalLM(ModelConfig(**SMALL))
    inject_lora(m, LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj", "up_proj"]))
    return m


def data():
    # Equal-length rows with every position a target: the mean of two half-batch losses is the full-batch loss.
    return torch.randint(3, 98, (STEPS, BATCH, SEQ), generator=torch.Generator().manual_seed(42))


def train(model, batches):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for ids in batches:
        _, loss = model(ids, labels=ids)
        loss.backward()
        opt.step()
        opt.zero_grad()


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = make_model()
        ddp = wrap_model(model, DistContext(rank=rank, world_size=world, backend="gloo"))
        assert ddp is not model and not ddp.broadcast_buffers
        shard = BATCH // world
        train(ddp, [b[rank * shard:(rank + 1) * shard] for b in data()])
        torch.save(lora_state_dict(model), f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_ddp_lora_matches_single_process(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    out = str(tmp_path / "state")
    mp.spawn(_worker, args=(2, free_port(), out), nprocs=2, join=True)
    single = make_model()
    before = lora_state_dict(single)
    train(single, data())
    expected = lora_state_dict(single)
    r0, r1 = torch.load(f"{out}.0"), torch.load(f"{out}.1")
    changed = False
    for k in expected:
        assert torch.equal(r0[k], r1[k]), f"ranks diverged on {k}"
        assert torch.allclose(r0[k], expected[k], atol=1e-5), f"DDP != single process on {k}"
        changed |= not torch.equal(before[k], expected[k])
    assert changed
