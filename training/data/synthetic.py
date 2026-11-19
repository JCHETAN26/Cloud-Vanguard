"""Deterministic synthetic corpora so training, tests and benchmarks run with no network.

Two kinds of data:

* ``pretrain``: templated sentences about the gateway's domain. Every token is
  a training target. Used to give the base model general character statistics.
* task data (``reverse``, ``upper``, ``add``, ``sort``): prompt => answer pairs
  with a single correct answer, so fine-tuning can be scored by exact match.
  Only answer tokens are training targets (``prompt_len`` masks the prompt).

Every record has the columns written to Arrow by training.data.arrow:
``text``, ``prompt``, ``answer``, ``input_ids`` (list<int32>) and ``prompt_len``.
"""

from __future__ import annotations

import random
import string
from collections.abc import Iterator

from .tokenizer import CharTokenizer

TASKS = ("reverse", "upper", "add", "sort")

_SUBJECTS = ["the gateway", "a tenant", "the cache", "each shard", "the operator", "a replica", "the ring",
             "the proxy", "an adapter", "the model", "a request", "the collector"]
_VERBS = ["serves", "stores", "routes", "evicts", "traces", "rejects", "replays", "warms", "shards", "batches",
          "streams", "measures"]
_OBJECTS = ["every prompt", "hot keys", "cold tokens", "a snapshot", "the quota", "slow peers", "new vectors",
            "stale answers", "its spans", "the budget", "short replies", "each stream"]
_TAILS = ["", " quickly", " at night", " per tenant", " under load", " in order", " twice", " again"]


def _sentence(rng: random.Random) -> str:
    s = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}{rng.choice(_TAILS)}."
    return s[0].upper() + s[1:]


def _word(rng: random.Random, lo: int, hi: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi)))


def task_pair(task: str, rng: random.Random) -> tuple[str, str]:
    """One (prompt, answer) pair. The prompt ends with ' =>' and the answer starts with a space."""
    if task == "reverse":
        w = _word(rng, 3, 6)
        return f"rev {w} =>", " " + w[::-1]
    if task == "upper":
        w = _word(rng, 3, 8)
        return f"up {w} =>", " " + w.upper()
    if task == "add":
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        return f"add {a}+{b} =>", f" {a + b}"
    if task == "sort":
        d = "".join(rng.choice(string.digits) for _ in range(rng.randint(3, 6)))
        return f"sort {d} =>", " " + "".join(sorted(d))
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def generate(kind: str, rows: int, seed: int = 0, tokenizer: CharTokenizer | None = None) -> Iterator[dict]:
    """Yield ``rows`` records of ``kind`` ('pretrain' or a task name). Same seed, same records."""
    tok = tokenizer or CharTokenizer()
    rng = random.Random(f"{kind}:{seed}")
    for _ in range(rows):
        if kind == "pretrain":
            text = " ".join(_sentence(rng) for _ in range(rng.randint(1, 3)))
            prompt, answer = "", text
        else:
            prompt, answer = task_pair(kind, rng)
            text = prompt + answer
        ids = tok.encode(text, bos=True, eos=True)
        # BOS + prompt characters are context, not targets (pretrain: only BOS).
        yield {"text": text, "prompt": prompt, "answer": answer.strip(), "input_ids": ids,
               "prompt_len": 1 + len(prompt)}
