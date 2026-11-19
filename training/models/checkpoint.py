"""Base-model checkpoints (config + weights) for the in-repo TinyCausalLM."""

from __future__ import annotations

import json
import os

import torch

from .tiny_lm import ModelConfig, TinyCausalLM

CONFIG = "model_config.json"
WEIGHTS = "model.safetensors"
WEIGHTS_BIN = "model.bin"


def save_model(model: TinyCausalLM, path: str | os.PathLike, extra: dict | None = None) -> str:
    """Save float weights. Quantization is applied at load time, so checkpoints stay full precision."""
    model = getattr(model, "module", model)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, CONFIG), "w") as f:
        json.dump({**model.config.to_dict(), "architecture": "TinyCausalLM", **(extra or {})}, f, indent=2,
                  sort_keys=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    try:
        from safetensors.torch import save_file
        out = os.path.join(path, WEIGHTS)
        save_file(state, out)
    except ImportError:
        out = os.path.join(path, WEIGHTS_BIN)
        torch.save(state, out)
    return out


def load_model(path: str | os.PathLike, map_location: str | torch.device = "cpu") -> TinyCausalLM:
    with open(os.path.join(path, CONFIG)) as f:
        meta = json.load(f)
    fields = ModelConfig.__dataclass_fields__
    model = TinyCausalLM(ModelConfig(**{k: v for k, v in meta.items() if k in fields}))
    st = os.path.join(path, WEIGHTS)
    if os.path.exists(st):
        from safetensors.torch import load_file
        state = load_file(st, device=str(map_location))
    else:
        state = torch.load(os.path.join(path, WEIGHTS_BIN), map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model
