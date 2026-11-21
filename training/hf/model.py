"""Load a Hugging Face causal LM (optionally NF4/int8-quantized) and evaluate it through the harness."""

from __future__ import annotations

import logging
import time

import torch
from torch import nn

from ..eval.backends import Backend, Generation
from ..models.quant import quantize_model

log = logging.getLogger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_hf_model(model_id: str, *, device: torch.device, dtype: str = "bf16", quant_bits: int = 0,
                  quant_backend: str = "auto", gradient_checkpointing: bool = False) -> tuple[nn.Module, dict]:
    """Returns (model on ``device``, info).

    Unquantized: weights are loaded straight onto the device in ``dtype``.
    Quantized: weights are loaded on the CPU in ``dtype`` and each nn.Linear (except lm_head) is replaced by
    training.models.quant: with bitsandbytes on CUDA that is Linear4bit(nf4) / Linear8bitLt, which moves
    each layer to the current CUDA device as it is quantized, so the full-precision model never sits on the
    GPU. The host needs RAM for one full-precision copy per process (about 30 GB for a 14.8B model in bf16).
    """
    from transformers import AutoModelForCausalLM

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    torch_dtype = DTYPES[dtype]
    t0 = time.perf_counter()
    on_gpu = device.type == "cuda" and not quant_bits
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True,
        device_map={"": device} if on_gpu else None)
    info: dict = {"hf_model": model_id, "dtype": dtype, "load_s": None,
                  "total_params_base": sum(p.numel() for p in model.parameters())}
    if quant_bits:
        info["quant_backend"] = quantize_model(model, quant_bits, backend=quant_backend)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    if gradient_checkpointing:
        # Non-reentrant checkpointing works with a fully frozen base (no need for inputs to require grad)
        # and with DDP.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    info["load_s"] = time.perf_counter() - t0
    return model, info


class HFBackend(Backend):
    """Greedy generation and completion scoring with a (possibly quantized, LoRA-wrapped) HF model."""

    name = "hf"
    supports_scoring = True

    def __init__(self, model: nn.Module, tokenizer, *, device: torch.device, precision: str = "bf16",
                 max_len: int | None = None):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.precision = precision
        self.max_len = max_len

    def _autocast(self):
        from ..models.precision import autocast
        return autocast(self.device, self.precision)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        self.model.eval()
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        self._sync()
        t0 = time.perf_counter()
        with self._autocast():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                      eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        self._sync()
        dt = time.perf_counter() - t0
        new = out[0, enc["input_ids"].shape[1]:]
        return Generation(self.tok.decode(new, skip_special_tokens=True), dt, int(new.numel()))

    @torch.no_grad()
    def score(self, prompt: str, completion: str) -> tuple[float, int]:
        """NLL of completion + EOS given the prompt: the tokens the trainer computes loss on."""
        self.model.eval()
        p_ids = self.tok(prompt).input_ids
        ids = self.tok(prompt + completion).input_ids + [self.tok.eos_token_id]
        if self.max_len:
            ids = ids[: self.max_len]
        start = len(p_ids)
        if start >= len(ids):
            return 0.0, 0
        x = torch.tensor([ids], device=self.device)
        with self._autocast():
            logits = self.model(input_ids=x).logits
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        nll = -logp.gather(1, x[0, 1:, None])[:, 0]
        scored = nll[start - 1:]
        return float(scored.sum()), int(scored.numel())
