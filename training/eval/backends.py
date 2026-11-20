"""Inference backends for the evaluation harness.

* ``TorchBackend``: the in-process model (base + adapter, optionally quantized).
  Always available; this is the CPU / no-GPU path used by the tests.
* ``VLLMBackend``: vLLM's offline engine for a Hugging Face model directory,
  with an optional PEFT-format LoRA adapter (what training.models.lora.save_adapter
  writes) served through ``LoRARequest``. Needs ``vllm`` and a GPU; it is
  imported lazily and never required.
* ``OpenAIHTTPBackend``: any OpenAI-compatible ``/v1/chat/completions``
  endpoint (``vllm serve``, or the CloudVanguard gateway in front of it).
  Measures end-to-end latency as clients see it; cannot score perplexity.

``score(prompt, completion)`` returns the summed negative log-likelihood (nats)
of ``completion`` given ``prompt`` and the number of scored tokens.
"""

from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import torch

from ..data.tokenizer import CharTokenizer
from ..models.precision import autocast


@dataclass
class Generation:
    text: str
    latency_s: float
    new_tokens: int


class Backend:
    name = "base"
    supports_scoring = False

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        raise NotImplementedError

    def score(self, prompt: str, completion: str) -> tuple[float, int]:
        raise NotImplementedError(f"{self.name} backend cannot score likelihoods")

    def close(self) -> None:
        pass


class TorchBackend(Backend):
    name = "torch"
    supports_scoring = True

    def __init__(self, model: torch.nn.Module, *, device: torch.device | None = None, precision: str = "fp32",
                 tokenizer: CharTokenizer | None = None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.precision = precision
        self.tok = tokenizer or CharTokenizer()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        ids = torch.tensor([self.tok.encode(prompt, bos=True)], device=self.device)
        self._sync()
        t0 = time.perf_counter()
        with autocast(self.device, self.precision):
            new = self.model.generate(ids, max_new_tokens, eos_id=self.tok.eos_id)
        self._sync()
        dt = time.perf_counter() - t0
        return Generation(self.tok.decode(new[0].tolist()), dt, int(new.shape[1]))

    @torch.no_grad()
    def score(self, prompt: str, completion: str) -> tuple[float, int]:
        ids = self.tok.encode(prompt + completion, bos=True, eos=True)
        max_len = getattr(getattr(self.model, "config", None), "max_len", None)
        if max_len:
            ids = ids[:max_len]  # same truncation as training (ArrowTokenDataset max_len)
        start = 1 + len(prompt)
        if start >= len(ids):
            return 0.0, 0
        x = torch.tensor([ids], device=self.device)
        with autocast(self.device, self.precision):
            logits, _ = self.model(x)
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = x[0, 1:]
        nll = -logp.gather(1, targets[:, None])[:, 0]
        scored = nll[start - 1:]
        return float(scored.sum()), int(scored.numel())


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


class VLLMBackend(Backend):
    """Offline vLLM engine. ``model`` is a Hugging Face model directory or hub id; ``adapter`` a PEFT LoRA dir.

    Scoring uses ``prompt_logprobs``: the completion's tokens are those after the
    tokenized prompt, so a BPE merge across the prompt/completion boundary shifts
    one token between the two (documented approximation)."""

    name = "vllm"
    supports_scoring = True

    def __init__(self, model: str, *, adapter: str | None = None, dtype: str = "auto", max_lora_rank: int = 64,
                 gpu_memory_utilization: float = 0.85, max_model_len: int | None = None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self.llm = LLM(model=model, dtype=dtype, enable_lora=adapter is not None, max_lora_rank=max_lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        self.lora = LoRARequest("eval-adapter", 1, adapter) if adapter else None

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        params = self._SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        t0 = time.perf_counter()
        out = self.llm.generate([prompt], params, lora_request=self.lora, use_tqdm=False)[0].outputs[0]
        return Generation(out.text, time.perf_counter() - t0, len(out.token_ids))

    def score(self, prompt: str, completion: str) -> tuple[float, int]:
        n_prompt = len(self.llm.get_tokenizer()(prompt).input_ids)
        params = self._SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        out = self.llm.generate([prompt + completion], params, lora_request=self.lora, use_tqdm=False)[0]
        ids, lps = out.prompt_token_ids, out.prompt_logprobs
        nll = -sum(lps[i][ids[i]].logprob for i in range(max(n_prompt, 1), len(ids)))
        return float(nll), max(0, len(ids) - max(n_prompt, 1))


class OpenAIHTTPBackend(Backend):
    """OpenAI-compatible chat completions over one persistent HTTP/1.1 connection."""

    name = "openai"

    def __init__(self, base_url: str, *, model: str = "default", api_key: str | None = None, timeout_s: float = 60.0):
        u = urlsplit(base_url.rstrip("/"))
        if u.scheme not in ("http", "https"):
            raise ValueError(f"base_url must be http(s)://...: {base_url!r}")
        self.host, self.port, self.https = u.hostname, u.port or (443 if u.scheme == "https" else 80), u.scheme == "https"
        self.path = (u.path or "") + "/chat/completions"
        self.model, self.api_key, self.timeout_s = model, api_key, timeout_s
        self._conn: http.client.HTTPConnection | None = None

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            self._conn = cls(self.host, self.port, timeout=self.timeout_s)
        return self._conn

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_new_tokens, "temperature": 0}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in (0, 1):  # one retry on a stale keep-alive connection
            conn = self._connection()
            t0 = time.perf_counter()
            try:
                conn.request("POST", self.path, body, headers)
                resp = conn.getresponse()
                data = resp.read()
            except (ConnectionError, http.client.HTTPException):
                self.close()
                if attempt:
                    raise
                continue
            dt = time.perf_counter() - t0
            if resp.status != 200:
                raise RuntimeError(f"{self.path}: HTTP {resp.status}: {data[:200]!r}")
            doc = json.loads(data)
            text = doc["choices"][0]["message"]["content"] or ""
            n = int(doc.get("usage", {}).get("completion_tokens") or len(text.split()))
            return Generation(text, dt, n)
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
