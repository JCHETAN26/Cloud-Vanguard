"""AG News topic classification as an instruction task, tokenized with the model's own tokenizer into the
Arrow schema of training.data (text / prompt / answer / input_ids / prompt_len).

Source: the public AG News dataset (fancyzhx/ag_news on the Hugging Face Hub: 120k train / 7.6k test,
4 balanced classes). Answers are one word, so exact match is a clean accuracy metric and generation is
cheap even for a 14B model. Train rows come from the train split and eval rows from the test split.
"""

from __future__ import annotations

import os
import random

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
REPO = "fancyzhx/ag_news"
FILES = {"train": "data/train-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}
MAX_ARTICLE_CHARS = 600

PROMPT = ("Classify the news article into one of these topics: World, Sports, Business, Sci/Tech.\n\n"
          "Article: {article}\nTopic:")


def format_prompt(article: str) -> str:
    article = " ".join(article.split())[:MAX_ARTICLE_CHARS]
    return PROMPT.format(article=article)


def load_split(split: str, parquet_dir: str | None = None) -> list[tuple[str, int]]:
    import pyarrow.parquet as pq

    if parquet_dir:
        path = os.path.join(parquet_dir, os.path.basename(FILES[split]))
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, FILES[split], repo_type="dataset")
    t = pq.read_table(path, columns=["text", "label"])
    return list(zip(t.column("text").to_pylist(), t.column("label").to_pylist()))


def encode(tok, article: str, label: int) -> dict:
    prompt = format_prompt(article)
    answer = LABELS[label]
    p_ids = tok(prompt).input_ids
    ids = tok(prompt + " " + answer).input_ids + [tok.eos_token_id]
    if ids[: len(p_ids)] != p_ids:  # BPE merged across the boundary; never expected with a leading space
        raise ValueError(f"prompt tokens are not a prefix of prompt+answer for label {answer!r}")
    return {"text": prompt + " " + answer, "prompt": prompt, "answer": answer, "input_ids": ids,
            "prompt_len": len(p_ids)}


def sample(rows: list[tuple[str, int]], n: int, seed: int) -> list[tuple[str, int]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def prepare(out_dir: str, tokenizer_id: str, *, train_rows: int = 4000, eval_rows: int = 500, seed: int = 0,
            parquet_dir: str | None = None) -> dict[str, int]:
    from ..data import write_arrow
    from .model import load_tokenizer

    tok = load_tokenizer(tokenizer_id)
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name, split, n in (("agnews_train.arrow", "train", train_rows), ("agnews_eval.arrow", "test", eval_rows)):
        recs = (encode(tok, a, y) for a, y in sample(load_split(split, parquet_dir), n, seed))
        written[name] = write_arrow(recs, os.path.join(out_dir, name))
    return written
