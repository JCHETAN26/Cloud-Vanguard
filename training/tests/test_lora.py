import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from training.models import (LoRAConfig, LoRALinear, NF4Linear, count_parameters, inject_lora, merge_lora,
                             quantize_model)
from training.models.lora import lora_modules

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def batch(seed=0, b=4, t=12):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 98, (b, t), generator=g)
    return ids, ids.clone()


def test_injects_exactly_the_targeted_layers(small_model):
    names = inject_lora(small_model, LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]))
    assert names == [f"layers.{i}.self_attn.{p}" for i in range(2) for p in ("q_proj", "v_proj")]
    assert all(isinstance(small_model.get_submodule(n), LoRALinear) for n in names)
    assert isinstance(small_model.layers[0].self_attn.k_proj, nn.Linear)


def test_regex_targets_and_unmatched_targets(small_model):
    m = copy.deepcopy(small_model)
    names = inject_lora(m, LoRAConfig(r=2, target_modules=r"layers\.1\.mlp\.(up|down)_proj"))
    assert names == ["layers.1.mlp.up_proj", "layers.1.mlp.down_proj"]
    with pytest.raises(ValueError, match="no linear module"):
        inject_lora(copy.deepcopy(small_model), LoRAConfig(target_modules=["qproj"]))
    for bad in (dict(r=0), dict(alpha=0), dict(dropout=1.0)):
        with pytest.raises(ValueError):
            LoRAConfig(**bad)


def test_freezes_base_and_counts_trainable_parameters(small_model):
    total_before = sum(p.numel() for p in small_model.parameters())
    r = 4
    inject_lora(small_model, LoRAConfig(r=r, target_modules=ATTN))
    trainable, total = count_parameters(small_model)
    d = small_model.config.d_model
    expected = small_model.config.n_layers * len(ATTN) * r * (d + d)  # r * (in + out) per wrapped layer
    assert trainable == expected
    assert total == total_before + expected
    for name, p in small_model.named_parameters():
        assert p.requires_grad == (".lora_A." in name or ".lora_B." in name), name


def test_zero_initialised_B_leaves_the_model_unchanged(small_model):
    ids, _ = batch()
    before, _ = small_model(ids)
    inject_lora(small_model, LoRAConfig(r=8, target_modules=ATTN + ["up_proj", "down_proj", "gate_proj"]))
    after, _ = small_model(ids)
    assert torch.equal(before, after)


def test_forward_backward_updates_only_adapters(small_model):
    inject_lora(small_model, LoRAConfig(r=4, alpha=8, target_modules=ATTN))
    frozen = {n: p.detach().clone() for n, p in small_model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in small_model.parameters() if p.requires_grad], lr=1e-2)
    ids, labels = batch()
    losses = []
    for _ in range(25):
        _, loss = small_model(ids, labels=labels)
        loss.backward()
        for n, p in small_model.named_parameters():
            assert (p.grad is not None) == p.requires_grad, n
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] - 0.1  # nats; random tokens, so a modest but real drop
    for n, p in small_model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), f"frozen weight {n} changed"
    lin = small_model.layers[0].self_attn.q_proj
    assert lin.lora_B.weight.abs().sum() > 0


def test_lora_output_is_W_plus_scaled_BA():
    torch.manual_seed(1)
    base = nn.Linear(16, 24, bias=True)
    m = LoRALinear(base, r=3, alpha=12)
    nn.init.normal_(m.lora_B.weight)
    x = torch.randn(5, 16)
    w_eff = base.weight + (12 / 3) * m.lora_B.weight @ m.lora_A.weight
    assert torch.allclose(m(x), F.linear(x, w_eff, base.bias), atol=1e-5)
    assert torch.allclose(m.delta_weight(), w_eff - base.weight, atol=1e-6)


def test_merge_unmerge_and_merge_lora_are_exact(small_model):
    inject_lora(small_model, LoRAConfig(r=4, target_modules=ATTN))
    for m in lora_modules(small_model).values():
        nn.init.normal_(m.lora_B.weight, std=0.1)
    ids, _ = batch()
    ref, _ = small_model(ids)
    lin = small_model.layers[0].self_attn.q_proj
    w0 = lin.base_layer.weight.detach().clone()
    lin.merge()
    assert torch.allclose(small_model(ids)[0], ref, atol=1e-5)
    lin.unmerge()
    assert torch.allclose(lin.base_layer.weight, w0, atol=1e-6)
    merged = merge_lora(small_model)
    assert not lora_modules(merged) and isinstance(merged.layers[0].self_attn.q_proj, nn.Linear)
    assert torch.allclose(merged(ids)[0], ref, atol=1e-5)


def test_merge_lora_dequantizes_a_quantized_base(small_model):
    quantize_model(small_model, 4, backend="torch")
    inject_lora(small_model, LoRAConfig(r=4, target_modules=ATTN))
    assert isinstance(small_model.layers[0].self_attn.q_proj.base_layer, NF4Linear)
    for m in lora_modules(small_model).values():
        nn.init.normal_(m.lora_B.weight, std=0.1)
    ids, _ = batch()
    ref, _ = small_model(ids)
    with pytest.raises(TypeError):
        small_model.layers[0].self_attn.q_proj.merge()
    merged = merge_lora(small_model)
    assert torch.allclose(merged(ids)[0], ref, atol=1e-4)


def test_dropout_applies_only_in_training():
    torch.manual_seed(0)
    m = LoRALinear(nn.Linear(8, 8), r=2, alpha=2, dropout=0.5)
    nn.init.normal_(m.lora_B.weight)
    x = torch.randn(64, 8)
    m.train()
    assert not torch.allclose(m(x), m(x))
    m.eval()
    assert torch.equal(m(x), m(x))
