"""Character-level tokenizer: fixed vocabulary, no files, no downloads.

The in-repo models are trained from scratch on synthetic data, so a byte/char
vocabulary keeps the whole pipeline offline and deterministic. Hugging Face
models bring their own tokenizer (see training.eval.backends).
"""

from __future__ import annotations

from collections.abc import Iterable

PAD, BOS, EOS = 0, 1, 2
_FIRST, _LAST = 32, 126  # printable ASCII
_OFFSET = 3


class CharTokenizer:
    pad_id, bos_id, eos_id = PAD, BOS, EOS
    vocab_size = _OFFSET + (_LAST - _FIRST + 1)  # 98

    def encode(self, text: str, *, bos: bool = True, eos: bool = False) -> list[int]:
        ids = [BOS] if bos else []
        for ch in text:
            c = ord(ch)
            ids.append(c - _FIRST + _OFFSET if _FIRST <= c <= _LAST else ord("?") - _FIRST + _OFFSET)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: Iterable[int], *, stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS and stop_at_eos:
                break
            if i >= _OFFSET:
                out.append(chr(i - _OFFSET + _FIRST))
        return "".join(out)
