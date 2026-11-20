"""A small Llama-style causal LM trained from scratch (no downloads).

Module names follow the Llama convention (q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj) so the same LoRA target lists work here and on
Hugging Face Llama/Mistral/Qwen models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..data.loader import IGNORE_INDEX


@dataclass
class ModelConfig:
    vocab_size: int = 98
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 344
    max_len: int = 128
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_heads

        def heads(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, h, d // h).transpose(1, 2)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        # Right padding + causal mask: real tokens never attend to padding.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = nn.Embedding(cfg.max_len, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits, loss). ``labels`` uses -100 for ignored positions; the shift happens here.
        ``attention_mask`` is accepted for interface parity; right padding needs no mask under causality."""
        t = input_ids.shape[1]
        if t > self.config.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.config.max_len}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_positions(pos)[None]
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None) -> torch.Tensor:
        """Greedy decoding for one prompt (shape [1, t]). Returns only the new tokens."""
        out = input_ids
        for _ in range(max_new_tokens):
            if out.shape[1] >= self.config.max_len:
                break
            logits, _ = self(out)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and int(nxt) == eos_id:
                break
        return out[:, input_ids.shape[1]:]
