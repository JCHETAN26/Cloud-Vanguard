import os

import pyarrow as pa
import pytest
import torch

from training.data import (IGNORE_INDEX, ArrowTokenDataset, CharTokenizer, LoaderConfig, build_dataloader, collate,
                           generate, resolve_pin_memory, task_pair, write_arrow)
from training.data.synthetic import TASKS


def test_tokenizer_round_trip_and_special_tokens():
    tok = CharTokenizer()
    ids = tok.encode("rev abc => cba", bos=True, eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "rev abc => cba"
    assert max(ids) < tok.vocab_size
    assert tok.decode(tok.encode("é", bos=False)) == "?"  # out-of-vocabulary maps to '?'
    assert tok.decode([tok.encode("a", bos=False)[0], tok.eos_id, tok.encode("b", bos=False)[0]]) == "a"


def test_synthetic_tasks_are_correct_and_deterministic():
    import random
    rng = random.Random(7)
    for _ in range(200):
        p, a = task_pair("reverse", rng)
        assert a.strip() == p.split()[1][::-1]
        p, a = task_pair("upper", rng)
        assert a.strip() == p.split()[1].upper()
        p, a = task_pair("add", rng)
        x, y = p.split()[1].split("+")
        assert int(a) == int(x) + int(y)
        p, a = task_pair("sort", rng)
        assert a.strip() == "".join(sorted(p.split()[1]))
    assert list(generate("reverse", 50, 3)) == list(generate("reverse", 50, 3))
    assert list(generate("reverse", 50, 3)) != list(generate("reverse", 50, 4))
    with pytest.raises(ValueError):
        task_pair("nope", rng)
    assert set(TASKS) == {"reverse", "upper", "add", "sort"}
    rec = next(generate("upper", 1, 0))
    assert rec["prompt_len"] == 1 + len(rec["prompt"]) and rec["text"] == rec["prompt"] + " " + rec["answer"]


def test_arrow_is_memory_mapped_and_rows_are_zero_copy(tmp_path):
    records = list(generate("pretrain", 3000, 0))
    path = tmp_path / "p.arrow"
    assert write_arrow(records, path, batch_rows=500) == 3000  # 6 record batches
    before = pa.total_allocated_bytes()
    ds = ArrowTokenDataset(path)
    assert ds.is_memory_mapped and len(ds) == 3000
    assert ds.table.column("input_ids").num_chunks == 6
    for i in (0, 499, 500, 1234, 2999, -1):
        assert ds[i]["input_ids"].tolist() == records[i]["input_ids"]
        assert ds[i]["prompt_len"] == records[i]["prompt_len"]
    for i in range(len(ds)):
        ds[i]
    # Reading every row allocated nothing in Arrow's pool: buffers live in the mapped file.
    assert pa.total_allocated_bytes() - before < 64 * 1024
    with pytest.raises(IndexError):
        ds[3000]


def test_arrow_stream_format_with_text_only(tmp_path):
    """Hugging Face datasets save IPC *stream* files; a text-only table is tokenized on the fly."""
    path = tmp_path / "data-00000-of-00001.arrow"
    table = pa.table({"text": ["hello world", "abc"]})
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    ds = ArrowTokenDataset(path, max_len=5)
    tok = CharTokenizer()
    assert ds[0]["input_ids"].tolist() == tok.encode("hello world", eos=True)[:5]
    assert ds[1]["input_ids"].tolist() == tok.encode("abc", eos=True)
    assert ds[1]["prompt_len"] == 1


def test_arrow_rejects_tables_without_tokens(tmp_path):
    path = tmp_path / "bad.arrow"
    t = pa.table({"x": [1, 2]})
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, t.schema) as w:
        w.write_table(t)
    with pytest.raises(ValueError, match="input_ids"):
        ArrowTokenDataset(path)


def test_write_arrow_is_atomic(tmp_path):
    def broken():
        yield from generate("reverse", 10, 0)
        raise RuntimeError("source failed")

    path = tmp_path / "x.arrow"
    with pytest.raises(RuntimeError):
        write_arrow(broken(), path, batch_rows=4)
    assert os.listdir(tmp_path) == []


def test_collate_pads_and_masks_prompt_tokens():
    tok = CharTokenizer()
    a = {"input_ids": torch.tensor(tok.encode("rev ab => ba", eos=True)), "prompt_len": 1 + len("rev ab =>")}
    b = {"input_ids": torch.tensor(tok.encode("hi", eos=True)), "prompt_len": 1}
    out = collate([a, b])
    assert out["input_ids"].shape == (2, len(a["input_ids"]))
    la, lb = out["labels"][0].tolist(), out["labels"][1].tolist()
    assert la[: a["prompt_len"]] == [IGNORE_INDEX] * a["prompt_len"]
    assert tok.decode(la[a["prompt_len"]:], stop_at_eos=False) == " ba"
    assert lb[:1] == [IGNORE_INDEX] and lb[3] == tok.eos_id and set(lb[4:]) == {IGNORE_INDEX}
    assert out["attention_mask"][1].tolist() == [1] * 4 + [0] * (len(la) - 4)


def test_worker_prefetching_loader_matches_in_process_loader(data_dir):
    ds = ArrowTokenDataset(data_dir / "pretrain.arrow", max_len=64)
    plain = list(build_dataloader(ds, LoaderConfig(batch_size=16, seed=3)))
    workers = list(build_dataloader(ds, LoaderConfig(batch_size=16, seed=3, num_workers=2, prefetch_factor=4)))
    assert len(plain) == len(workers) == len(ds) // 16
    for x, y in zip(plain, workers):
        assert torch.equal(x["input_ids"], y["input_ids"]) and torch.equal(x["labels"], y["labels"])


def test_distributed_sampler_gives_disjoint_shards(data_dir):
    ds = ArrowTokenDataset(data_dir / "reverse_eval.arrow")
    seen = []
    for rank in range(2):
        loader = build_dataloader(ds, LoaderConfig(batch_size=4, seed=1, drop_last=False), rank=rank, world_size=2)
        seen.append({i for i in loader.sampler})
    assert not seen[0] & seen[1]
    assert seen[0] | seen[1] == set(range(len(ds)))


def test_pin_memory_resolution():
    if torch.cuda.is_available():
        assert resolve_pin_memory("auto") is True
        loader = build_dataloader([{"input_ids": torch.tensor([1, 5, 2]), "prompt_len": 1}] * 4,
                                  LoaderConfig(batch_size=2, pin_memory=True))
        assert next(iter(loader))["input_ids"].is_pinned()
    else:
        assert resolve_pin_memory("auto") is False
        with pytest.warns(UserWarning, match="pin_memory"):
            assert resolve_pin_memory(True) is False
    assert resolve_pin_memory(False) is False
