"""LoRA: frozen base weights plus a trainable low-rank update.

For a frozen linear layer W (out x in) LoRA learns

    h = W x + (alpha / r) * B A x,   A: r x in,  B: out x r,  r << min(in, out)

so the effective weight is W + dW with dW = (alpha / r) * B A. A starts with
Kaiming-uniform noise and B with zeros, so an injected model computes exactly
what the base model computes until the first optimizer step.

The wrapped base can be an ``nn.Linear`` or a quantized linear (training.models.quant):
that combination is QLoRA. Saved adapters use the Hugging Face PEFT layout
(``adapter_config.json`` + ``adapter_model.safetensors`` with keys
``base_model.model.<module>.lora_A.weight``), so an adapter trained on a Hugging
Face model here loads with ``peft.PeftModel.from_pretrained`` and with vLLM's
``LoRARequest``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_WEIGHTS_BIN = "adapter_model.bin"
ADAPTER_EXTRA = "cloudvanguard_adapter.json"
_PREFIX = "base_model.model."


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # Module-name suffixes (Llama naming), or one regex matched against the full module name.
    target_modules: list[str] | str = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError("LoRA rank r must be >= 1")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def matches(self, name: str) -> bool:
        if isinstance(self.target_modules, str):
            return re.fullmatch(self.target_modules, name) is not None
        leaf = name.rsplit(".", 1)[-1]
        return leaf in self.target_modules


class LoRALinear(nn.Module):
    """Wraps a (frozen) linear-like module that exposes in_features/out_features."""

    def __init__(self, base_layer: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.scaling = alpha / r
        w = getattr(base_layer, "weight", None)
        device = w.device if isinstance(w, torch.Tensor) else None
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, r, bias=False, device=device)
        self.lora_B = nn.Linear(r, self.out_features, bias=False, device=device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_layer(x)
        if self.merged:
            return y
        a = self.lora_A.weight
        delta = self.lora_B(self.lora_A(self.lora_dropout(x).to(a.dtype)))
        return y + (delta * self.scaling).to(y.dtype)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    def merge(self) -> None:
        """Fold dW into the base weight (float bases only)."""
        if self.merged:
            return
        if not isinstance(self.base_layer, nn.Linear):
            raise TypeError("merge() needs a float nn.Linear base; use merge_lora(), which dequantizes")
        with torch.no_grad():
            self.base_layer.weight += self.delta_weight().to(self.base_layer.weight.dtype)
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight -= self.delta_weight().to(self.base_layer.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:g}"


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, module)


def _is_linear_like(m: nn.Module) -> bool:
    return hasattr(m, "in_features") and hasattr(m, "out_features") and not isinstance(m, LoRALinear)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Wrap every targeted linear layer in a LoRALinear and freeze everything else.

    Returns the wrapped module names. Raises if nothing matched, so a typo in
    ``target_modules`` cannot silently train zero parameters."""
    names = [n for n, m in model.named_modules()
             if n and cfg.matches(n) and _is_linear_like(m) and ".lora_" not in n]
    if not names:
        raise ValueError(f"no linear module matches target_modules={cfg.target_modules!r}")
    for n in names:
        _set_module(model, n, LoRALinear(model.get_submodule(n), cfg.r, cfg.alpha, cfg.dropout))
    mark_only_lora_trainable(model)
    model.lora_config = cfg  # type: ignore[attr-defined]
    return names


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = ".lora_A." in name or ".lora_B." in name


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts. Quantized weights count by element, not by byte."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        n = getattr(m, "quantized_numel", None)
        if n is not None:
            total += n
    return trainable, total


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter weights in PEFT key layout: CPU copies, not views (on CPU, detach().cpu() would alias the live
    parameters and change with the next optimizer step)."""
    model = getattr(model, "module", model)  # DDP
    out = {}
    for name, m in lora_modules(model).items():
        out[f"{_PREFIX}{name}.lora_A.weight"] = m.lora_A.weight.detach().to("cpu", copy=True)
        out[f"{_PREFIX}{name}.lora_B.weight"] = m.lora_B.weight.detach().to("cpu", copy=True)
    return out


def save_adapter(model: nn.Module, path: str | os.PathLike, *, base_model: str = "",
                 extra: dict | None = None) -> str:
    """Write a PEFT-format adapter directory. Returns the weights file path."""
    model = getattr(model, "module", model)
    cfg: LoRAConfig = model.lora_config
    os.makedirs(path, exist_ok=True)
    targets = cfg.target_modules if isinstance(cfg.target_modules, str) else sorted(set(cfg.target_modules))
    meta = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": cfg.r, "lora_alpha": cfg.alpha,
        "lora_dropout": cfg.dropout, "target_modules": targets, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True, "base_model_name_or_path": base_model, "modules_to_save": None,
        "init_lora_weights": True, "use_rslora": False, "use_dora": False,
    }
    with open(os.path.join(path, ADAPTER_CONFIG), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    if extra:  # our own metadata lives beside the PEFT config, which must stay strictly PEFT
        with open(os.path.join(path, ADAPTER_EXTRA), "w") as f:
            json.dump(extra, f, indent=2, sort_keys=True)
    state = lora_state_dict(model)
    try:
        from safetensors.torch import save_file
        out = os.path.join(path, ADAPTER_WEIGHTS)
        save_file(state, out, metadata={"format": "pt"})
    except ImportError:
        out = os.path.join(path, ADAPTER_WEIGHTS_BIN)
        torch.save(state, out)
    return out


def read_adapter_config(path: str | os.PathLike) -> LoRAConfig:
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        meta = json.load(f)
    if meta.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"{path}: not a LoRA adapter (peft_type={meta.get('peft_type')})")
    return LoRAConfig(r=int(meta["r"]), alpha=float(meta["lora_alpha"]), dropout=float(meta.get("lora_dropout", 0)),
                      target_modules=meta["target_modules"])


def load_adapter(model: nn.Module, path: str | os.PathLike) -> LoRAConfig:
    """Inject (if needed) and load a saved adapter. Every adapter tensor must land on a module and vice versa."""
    cfg = read_adapter_config(path)
    if not lora_modules(model):
        inject_lora(model, cfg)
    st = os.path.join(path, ADAPTER_WEIGHTS)
    if os.path.exists(st):
        from safetensors.torch import load_file
        state = load_file(st)
    else:
        state = torch.load(os.path.join(path, ADAPTER_WEIGHTS_BIN), map_location="cpu", weights_only=True)
    expected = lora_state_dict(model)
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"adapter {path} does not fit this model: missing={missing[:4]} unexpected={unexpected[:4]}")
    mods = lora_modules(model)
    with torch.no_grad():
        for key, tensor in state.items():
            name, which = key[len(_PREFIX):].rsplit(".", 2)[0], key.rsplit(".", 2)[1]
            w = getattr(mods[name], which).weight
            if w.shape != tensor.shape:
                raise ValueError(f"adapter {path}: {key} has shape {tuple(tensor.shape)}, model expects {tuple(w.shape)}")
            w.copy_(tensor.to(w.dtype))
    return cfg


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear with a plain nn.Linear holding W + dW (quantized bases are dequantized).
    The result has no adapter overhead at inference and can be exported as an ordinary checkpoint."""
    for name, m in list(lora_modules(model).items()):
        base = m.base_layer
        if isinstance(base, nn.Linear):
            weight, bias = base.weight.detach(), base.bias
        else:
            weight, bias = base.dequantized_weight(), getattr(base, "bias", None)
        merged = nn.Linear(m.in_features, m.out_features, bias=bias is not None, device=weight.device,
                           dtype=torch.float32)
        with torch.no_grad():
            merged.weight.copy_(weight.float() + m.delta_weight().float())
            if bias is not None:
                merged.bias.copy_(bias.detach().float())
        _set_module(model, name, merged)
    if hasattr(model, "lora_config"):
        del model.lora_config
    return model


def trainable_names(model: nn.Module) -> Iterable[str]:
    return (n for n, p in model.named_parameters() if p.requires_grad)
