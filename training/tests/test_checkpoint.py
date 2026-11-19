import copy
import json
import os

import pytest
import torch
from torch import nn

from training.models import (LoRAConfig, ModelConfig, TinyCausalLM, inject_lora, load_adapter, load_model,
                             lora_state_dict, save_adapter, save_model)
from training.models.lora import lora_modules

from .conftest import SMALL

TARGETS = ["q_proj", "v_proj", "down_proj"]


def ids():
    return torch.randint(3, 98, (3, 20), generator=torch.Generator().manual_seed(5))


def randomize_adapters(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in lora_modules(model).values():
        with torch.no_grad():
            m.lora_B.weight.copy_(torch.randn(m.lora_B.weight.shape, generator=g) * 0.1)


def test_base_checkpoint_round_trip(small_model, tmp_path):
    save_model(small_model, tmp_path / "m")
    loaded = load_model(tmp_path / "m")
    assert loaded.config == small_model.config
    assert torch.equal(loaded(ids())[0], small_model(ids())[0])
    meta = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert meta["architecture"] == "TinyCausalLM" and meta["d_model"] == SMALL["d_model"]


def test_adapter_round_trip_reproduces_outputs(small_model, tmp_path):
    base = copy.deepcopy(small_model)
    inject_lora(small_model, LoRAConfig(r=4, alpha=8, target_modules=TARGETS))
    randomize_adapters(small_model)
    ref, _ = small_model(ids())
    weights = save_adapter(small_model, tmp_path / "a", base_model="tiny")
    files = sorted(os.listdir(tmp_path / "a"))
    assert files[0] == "adapter_config.json" and files[1].startswith("adapter_model.")
    cfg = json.loads((tmp_path / "a" / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4 and cfg["lora_alpha"] == 8
    assert cfg["target_modules"] == sorted(TARGETS)
    state = lora_state_dict(small_model)
    assert all(k.startswith("base_model.model.layers.") and ".lora_" in k for k in state)
    assert len(state) == 2 * len(TARGETS) * SMALL["n_layers"]
    # The adapter is a small fraction of the model.
    assert os.path.getsize(weights) < 0.2 * sum(p.numel() * 4 for p in base.parameters())

    load_adapter(base, tmp_path / "a")
    assert torch.equal(base(ids())[0], ref)


def test_adapter_refuses_a_model_it_does_not_fit(small_model, tmp_path):
    inject_lora(small_model, LoRAConfig(r=4, target_modules=TARGETS))
    save_adapter(small_model, tmp_path / "a")
    wider = TinyCausalLM(ModelConfig(**{**SMALL, "d_model": 48, "n_heads": 2}))
    with pytest.raises(ValueError, match="shape"):
        load_adapter(wider, tmp_path / "a")
    other = TinyCausalLM(ModelConfig(**SMALL))
    inject_lora(other, LoRAConfig(r=4, target_modules=["k_proj"]))
    with pytest.raises(ValueError, match="does not fit"):
        load_adapter(other, tmp_path / "a")


def _tiny_llama():
    tr = pytest.importorskip("transformers")
    cfg = tr.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return tr.LlamaForCausalLM(cfg).eval()


def test_adapter_is_loadable_by_peft(tmp_path):
    """Our PEFT-layout adapter, trained on a Hugging Face Llama, loads in peft and gives the same logits."""
    peft = pytest.importorskip("peft")
    model = _tiny_llama()
    base = copy.deepcopy(model)
    inject_lora(model, LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"]))
    randomize_adapters(model)
    x = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours = model(x).logits
    save_adapter(model, tmp_path / "a", base_model="tiny-llama")
    pm = peft.PeftModel.from_pretrained(base, str(tmp_path / "a")).eval()
    with torch.no_grad():
        theirs = pm(x).logits
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_peft_adapter_loads_into_our_lora(tmp_path):
    peft = pytest.importorskip("peft")
    model = _tiny_llama()
    pm = peft.get_peft_model(copy.deepcopy(model),
                             peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p, std=0.1)
    pm.eval()
    x = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        theirs = pm(x).logits
    pm.save_pretrained(str(tmp_path / "p"))
    load_adapter(model, tmp_path / "p")
    with torch.no_grad():
        assert torch.allclose(model(x).logits, theirs, atol=1e-5)
