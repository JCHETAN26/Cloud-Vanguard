import pytest
import torch

from training.models import LoRAConfig, autocast, grad_scaler, inject_lora, resolve_device, resolve_precision


def test_precision_resolution_on_cpu():
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) == "fp32"
    assert resolve_precision("bf16", cpu) == "bf16"
    assert resolve_precision("fp32", cpu) == "fp32"
    with pytest.warns(UserWarning, match="fp16"):
        assert resolve_precision("fp16", cpu) == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("int8", cpu)
    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            assert resolve_device("cuda").type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_precision_resolution_on_cuda():
    dev = torch.device("cuda")
    assert resolve_precision("auto", dev) == ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    assert grad_scaler(dev, "fp16").is_enabled()


def test_bf16_autocast_lora_step_on_cpu(small_model):
    cpu = torch.device("cpu")
    inject_lora(small_model, LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]))
    scaler = grad_scaler(cpu, "bf16")
    assert not scaler.is_enabled()  # loss scaling is only for fp16
    ids = torch.randint(3, 98, (4, 16), generator=torch.Generator().manual_seed(0))
    seen = []
    h = small_model.layers[0].self_attn.q_proj.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    with autocast(cpu, "bf16"):
        logits, loss = small_model(ids, labels=ids)
    h.remove()
    assert seen == [torch.bfloat16] and logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32 and torch.isfinite(loss)  # loss computed in fp32
    scaler.scale(loss).backward()
    grads = [p.grad for p in small_model.parameters() if p.requires_grad]
    assert all(g is not None and g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
