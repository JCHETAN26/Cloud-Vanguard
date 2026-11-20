"""Weight-only quantization of the frozen base model (the "Q" in QLoRA).

Backends:

* ``bnb``: bitsandbytes ``Linear8bitLt`` (LLM.int8()) / ``Linear4bit`` (NF4).
  Used when bitsandbytes imports and a CUDA device is present.
* ``torch``: a portable implementation in plain PyTorch that runs anywhere,
  including CPU-only CI:
    - int8: symmetric per-output-channel absmax (scale = max|W_row| / 127).
    - nf4:  4-bit NormalFloat (Dettmers et al., QLoRA, 2023): blocks of 64
      weights are scaled by their absmax and each value is mapped to the
      nearest of 16 quantiles of N(0, 1); two codes are packed per byte.
  Weights are dequantized on the fly in forward. This saves weight memory, not
  compute: on CPU it is slower than float matmuls (measured in the benchmark).

``auto`` picks ``bnb`` when usable and falls back to ``torch`` otherwise.
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

# NF4 code book from the QLoRA reference implementation (bitsandbytes create_normal_map).
NF4_CODE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


class Int8Linear(nn.Module):
    """Frozen int8 weight with per-output-channel scales."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        self.register_buffer("qweight", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().float().clone())
        self.quantized_numel = w.numel()

    def dequantized_weight(self) -> torch.Tensor:
        return self.qweight.float() * self.scale[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight().to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8 per-channel"


class NF4Linear(nn.Module):
    """Frozen 4-bit NormalFloat weight, block-wise absmax scaling, two codes per byte."""

    def __init__(self, linear: nn.Linear, block_size: int = 64):
        super().__init__()
        w = linear.weight.detach().float()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.block_size = block_size
        flat = w.reshape(-1)
        n = flat.numel()
        pad = (-n) % block_size
        blocks = F.pad(flat, (0, pad)).view(-1, block_size)
        absmax = blocks.abs().amax(dim=1).clamp(min=1e-12)
        normed = blocks / absmax[:, None]
        codes = (normed.unsqueeze(-1) - NF4_CODE.to(w.device)).abs().argmin(-1).to(torch.uint8).view(-1)
        if codes.numel() % 2:
            codes = torch.cat([codes, codes.new_zeros(1)])
        packed = (codes[0::2] << 4) | codes[1::2]
        self.register_buffer("qweight", packed)
        self.register_buffer("absmax", absmax)
        self.register_buffer("code", NF4_CODE.clone().to(w.device))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().float().clone())
        self.numel = n
        self.quantized_numel = n

    def dequantized_weight(self) -> torch.Tensor:
        hi, lo = self.qweight >> 4, self.qweight & 0x0F
        codes = torch.stack([hi, lo], dim=1).view(-1)[: self.absmax.numel() * self.block_size].long()
        vals = self.code[codes].view(-1, self.block_size) * self.absmax[:, None]
        return vals.view(-1)[: self.numel].view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight().to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, nf4 block={self.block_size}"


def bnb_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:  # ImportError or a broken CUDA setup inside bitsandbytes
        return False
    return True


def resolve_backend(requested: str) -> str:
    if requested not in ("auto", "bnb", "torch"):
        raise ValueError(f"quantization backend {requested!r}: expected auto, bnb or torch")
    if requested == "torch":
        return "torch"
    if bnb_available():
        return "bnb"
    if requested == "bnb":
        warnings.warn("bitsandbytes needs a CUDA device and the bitsandbytes package; "
                      "falling back to the portable torch quantizer", stacklevel=2)
    return "torch"


def _bnb_linear(linear: nn.Linear, bits: int) -> nn.Module:
    import bitsandbytes as bnb

    if bits == 8:
        q = bnb.nn.Linear8bitLt(linear.in_features, linear.out_features, bias=linear.bias is not None,
                                has_fp16_weights=False, threshold=6.0)
        q.weight = bnb.nn.Int8Params(linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False)
    else:
        q = bnb.nn.Linear4bit(linear.in_features, linear.out_features, bias=linear.bias is not None,
                              compute_dtype=torch.bfloat16, quant_type="nf4")
        q.weight = bnb.nn.Params4bit(linear.weight.data.clone(), requires_grad=False, quant_type="nf4")
    if linear.bias is not None:
        q.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
    return q.cuda()  # bitsandbytes quantizes on transfer to the GPU


def quantize_model(model: nn.Module, bits: int, *, backend: str = "auto",
                   skip: tuple[str, ...] = ("lm_head",), block_size: int = 64) -> str:
    """Replace every nn.Linear (except names ending in ``skip``) with a frozen quantized layer.

    Call before inject_lora: adapters then wrap the quantized layers (QLoRA).
    Returns the backend actually used."""
    if bits not in (4, 8):
        raise ValueError(f"quantization bits must be 4 or 8, got {bits}")
    chosen = resolve_backend(backend)
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and not n.endswith(skip) and ".lora_" not in n]
    for name in targets:
        lin = model.get_submodule(name)
        if chosen == "bnb":
            q = _bnb_linear(lin, bits)
        else:
            q = Int8Linear(lin) if bits == 8 else NF4Linear(lin, block_size)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, q)
    for p in model.parameters():  # embeddings, norms and lm_head stay float but frozen
        p.requires_grad = False
    log.info("quantized %d linear layers to %d-bit with the %s backend", len(targets), bits, chosen)
    return chosen


def weight_bytes(model: nn.Module) -> int:
    """Bytes held by parameters and buffers (what the weights cost in memory)."""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total
