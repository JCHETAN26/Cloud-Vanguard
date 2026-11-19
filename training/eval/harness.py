"""Evaluation harness: task accuracy, perplexity and latency for any backend."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .backends import Backend


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); nan for no values."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class EvalReport:
    backend: str
    examples: int
    exact_match: float | None
    perplexity: float | None
    mean_nll: float | None
    scored_tokens: int
    latency_ms: dict[str, float]
    generated_tokens_per_s: float | None
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def metrics(self, prefix: str = "eval/") -> dict[str, float]:
        out = {f"{prefix}{k}": v for k, v in (("exact_match", self.exact_match), ("perplexity", self.perplexity),
                                                ("mean_nll", self.mean_nll),
                                                ("gen_tokens_per_s", self.generated_tokens_per_s)) if v is not None}
        out.update({f"{prefix}latency_{k}_ms": v for k, v in self.latency_ms.items()})
        return out


def evaluate(backend: Backend, examples: Sequence[dict], *, max_new_tokens: int = 16, generate: bool = True,
             score: bool = True, warmup: int = 2, keep_samples: int = 5) -> EvalReport:
    """``examples``: dicts with ``prompt``, ``answer`` and ``text`` (= prompt + completion).

    * exact match: greedy completion (stripped) == answer, for examples that have a prompt;
    * perplexity: exp(total NLL / total tokens) of each completion given its prompt (backends that score);
    * latency: per-request wall time of generation, after ``warmup`` unmeasured requests."""
    if not examples:
        raise ValueError("no evaluation examples")
    nll_sum, n_tok = 0.0, 0
    if score and backend.supports_scoring:
        for ex in examples:
            s, n = backend.score(ex["prompt"], ex["text"][len(ex["prompt"]):])
            nll_sum += s
            n_tok += n
    latencies: list[float] = []
    correct = graded = gen_tokens = 0
    gen_time = 0.0
    samples = []
    if generate:
        for ex in examples[: min(warmup, len(examples))]:
            backend.generate(ex["prompt"], max_new_tokens)
        for ex in examples:
            g = backend.generate(ex["prompt"], max_new_tokens)
            latencies.append(g.latency_s * 1000)
            gen_tokens += g.new_tokens
            gen_time += g.latency_s
            if ex["prompt"]:
                graded += 1
                ok = g.text.strip() == ex["answer"]
                correct += ok
                if len(samples) < keep_samples:
                    samples.append({"prompt": ex["prompt"], "expected": ex["answer"], "got": g.text.strip(), "ok": ok})
    lat = {}
    if latencies:
        lat = {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
               "p99": percentile(latencies, 99), "mean": statistics.fmean(latencies)}
    return EvalReport(
        backend=backend.name, examples=len(examples),
        exact_match=(correct / graded) if graded else None,
        perplexity=math.exp(nll_sum / n_tok) if n_tok else None,
        mean_nll=(nll_sum / n_tok) if n_tok else None,
        scored_tokens=n_tok, latency_ms=lat,
        generated_tokens_per_s=(gen_tokens / gen_time) if gen_time > 0 else None,
        samples=samples,
    )
