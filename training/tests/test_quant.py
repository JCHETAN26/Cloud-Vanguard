import pytest
import torch
from torch import nn

from training.models import (Int8Linear, LoRAConfig, NF4Linear, inject_lora, quantize_model, resolve_backend,
                             weight_bytes)
from training.models.quant import NF4_CODE, bnb_available


def rel_err(a, b):
    return float((a - b).norm() / b.norm())


def test_int8_per_channel_round_trip():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=True)
    q = Int8Linear(lin)
    assert q.qweight.dtype == torch.int8 and q.scale.shape == (128,)
    assert rel_err(q.dequantized_weight(), lin.weight) < 0.01
    x = torch.randn(8, 256)
    assert rel_err(q(x), lin(x)) < 0.01
    # 1 byte per weight + one fp32 scale per row (+ bias) vs 4 bytes per weight.
    assert weight_bytes(q) == 256 * 128 + 128 * 4 + 128 * 4


def test_nf4_codes_packing_and_error():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False)
    q = NF4Linear(lin, block_size=64)
    assert q.qweight.dtype == torch.uint8 and q.qweight.numel() == 256 * 128 // 2
    w = q.dequantized_weight()
    # Every dequantized value is a code-book entry times its block's absmax.
    blocks = w.reshape(-1, 64) / q.absmax[:, None]
    assert torch.isclose(blocks.unsqueeze(-1), NF4_CODE, atol=1e-6).any(-1).all()
    # The block maximum is represented exactly (code +-1.0).
    assert torch.allclose(w.reshape(-1, 64).abs().amax(1), lin.weight.reshape(-1, 64).abs().amax(1), atol=1e-6)
    assert rel_err(w, lin.weight) < 0.12  # 4-bit: ~10% relative error for Gaussian-like weights
    assert weight_bytes(q) == 256 * 128 // 2 + (256 * 128 // 64) * 4 + 16 * 4


def test_nf4_handles_sizes_that_do_not_fill_a_block():
    torch.manual_seed(0)
    lin = nn.Linear(7, 5, bias=True)  # 35 weights: odd, < one block
    q = NF4Linear(lin)
    assert q.dequantized_weight().shape == (5, 7)
    assert rel_err(q.dequantized_weight(), lin.weight) < 0.15
    x = torch.randn(3, 7)
    assert q(x).shape == (3, 5)


def test_quantize_model_freezes_and_skips_lm_head(small_model):
    before = weight_bytes(small_model)
    backend = quantize_model(small_model, 8, backend="auto")
    assert backend == ("bnb" if bnb_available() else "torch")
    assert isinstance(small_model.lm_head, nn.Linear)
    if backend == "bnb":  # bitsandbytes swaps in its own LLM.int8() layer
        import bitsandbytes as bnb
        assert isinstance(small_model.layers[0].mlp.up_proj, bnb.nn.Linear8bitLt)
    else:
        assert isinstance(small_model.layers[0].mlp.up_proj, Int8Linear)
    assert not any(p.requires_grad for p in small_model.parameters())
    assert weight_bytes(small_model) < before
    with pytest.raises(ValueError):
        quantize_model(small_model, 3)


def test_backend_resolution_falls_back_without_cuda():
    if bnb_available():
        pytest.skip("bitsandbytes + CUDA present")
    assert resolve_backend("auto") == "torch"
    with pytest.warns(UserWarning, match="bitsandbytes"):
        assert resolve_backend("bnb") == "torch"
    with pytest.raises(ValueError):
        resolve_backend("gptq")


@pytest.mark.parametrize("bits", [8, 4])
def test_qlora_trains_adapters_over_frozen_quantized_weights(small_model, bits):
    quantize_model(small_model, bits, backend="torch")
    inject_lora(small_model, LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]))
    qbufs = {n: b.clone() for n, b in small_model.named_buffers() if n.endswith("qweight")}
    assert qbufs
    params = [p for p in small_model.parameters() if p.requires_grad]
    assert params and all(".lora_" in n for n, p in small_model.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=1e-2)
    ids = torch.randint(3, 98, (4, 16), generator=torch.Generator().manual_seed(0))
    losses = []
    for _ in range(25):
        _, loss = small_model(ids, labels=ids)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] - 0.1  # nats; random tokens, so a modest but real drop
    for n, b in small_model.named_buffers():
        if n in qbufs:
            assert torch.equal(b, qbufs[n])


@pytest.mark.skipif(not bnb_available(), reason="bitsandbytes needs CUDA; not available here")
@pytest.mark.parametrize("bits", [8, 4])
def test_bitsandbytes_backend(small_model, bits):
    small_model.cuda()
    assert quantize_model(small_model, bits, backend="bnb") == "bnb"
    inject_lora(small_model, LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]))
    ids = torch.randint(3, 98, (2, 8), device="cuda")
    _, loss = small_model(ids, labels=ids)
    loss.backward()
