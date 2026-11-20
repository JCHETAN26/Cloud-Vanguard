"""Memory-mapped Apache Arrow datasets.

Files are Arrow IPC. Both the random-access *file* format (what write_arrow
produces) and the *stream* format (what Hugging Face ``datasets.save_to_disk``
writes as ``data-*.arrow``) are read through ``pyarrow.memory_map``: column
buffers point into the page cache, so opening a multi-GB dataset costs no heap
and DataLoader workers share the same physical pages.

Rows are read zero-copy: ``input_ids`` offsets/values are numpy views over the
mapped buffers; only the slice for the requested row is copied into a tensor.
"""

from __future__ import annotations

import bisect
import os
from collections.abc import Iterable, Sequence

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer

SCHEMA = pa.schema([
    ("text", pa.string()),
    ("prompt", pa.string()),
    ("answer", pa.string()),
    ("input_ids", pa.list_(pa.int32())),
    ("prompt_len", pa.int32()),
])


def write_arrow(records: Iterable[dict], path: str | os.PathLike, *, batch_rows: int = 4096) -> int:
    """Write records (dicts with SCHEMA's columns) to an Arrow IPC file atomically. Returns the row count."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    rows = 0
    buf: dict[str, list] = {name: [] for name in SCHEMA.names}
    try:
        with pa.OSFile(tmp, "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
            def flush() -> None:
                if buf["text"]:
                    writer.write_batch(pa.record_batch([pa.array(buf[n], type=SCHEMA.field(n).type)
                                                        for n in SCHEMA.names], schema=SCHEMA))
                    for v in buf.values():
                        v.clear()

            for rec in records:
                for name in SCHEMA.names:
                    buf[name].append(rec[name])
                rows += 1
                if len(buf["text"]) >= batch_rows:
                    flush()
            flush()
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return rows


def _open_table(path: str) -> tuple[pa.MemoryMappedFile, pa.Table]:
    source = pa.memory_map(path, "r")
    try:
        table = pa.ipc.open_file(source).read_all()
    except pa.ArrowInvalid:  # stream format (e.g. Hugging Face datasets)
        source.seek(0)
        table = pa.ipc.open_stream(source).read_all()
    return source, table


class ArrowTokenDataset(Dataset):
    """Token sequences from a memory-mapped Arrow file.

    Needs an ``input_ids`` list column, or a ``text`` column that is tokenized
    on the fly with the char tokenizer. ``prompt_len`` (optional) masks prompt
    tokens out of the loss. Sequences longer than ``max_len`` are truncated.

    The mapping is opened lazily and dropped when pickled, so each DataLoader
    worker maps the file itself instead of receiving a copy of the data.
    """

    def __init__(self, path: str | os.PathLike, *, max_len: int | None = None):
        self.path = os.fspath(path)
        self.max_len = max_len
        self._source = None
        self._table: pa.Table | None = None
        self._open()

    def _open(self) -> None:
        self._source, self._table = _open_table(self.path)
        t = self._table
        if "input_ids" not in t.column_names and "text" not in t.column_names:
            raise ValueError(f"{self.path}: needs an 'input_ids' or 'text' column (has {t.column_names})")
        self._chunks = []
        starts = [0]
        col = "input_ids" if "input_ids" in t.column_names else "text"
        for chunk in t.column(col).chunks:
            if col == "input_ids":
                # Views over the mapped buffers (offsets index into the full child array).
                offsets = chunk.offsets.to_numpy(zero_copy_only=True)
                values = chunk.values.to_numpy(zero_copy_only=True)
                self._chunks.append((offsets, values))
            else:
                self._chunks.append(chunk)
            starts.append(starts[-1] + len(chunk))
        self._starts = starts
        self._prompt_len = None
        if "prompt_len" in t.column_names:
            self._prompt_len = t.column("prompt_len").to_numpy()  # zero-copy when single-chunk
        self._tok = CharTokenizer()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        for k in ("_source", "_table", "_chunks", "_prompt_len", "_starts"):
            state[k] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open()

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def is_memory_mapped(self) -> bool:
        return isinstance(self._source, pa.MemoryMappedFile)

    def column(self, name: str) -> list:
        return self._table.column(name).to_pylist()

    def __len__(self) -> int:
        return self._starts[-1]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        c = bisect.bisect_right(self._starts, i) - 1
        j = i - self._starts[c]
        chunk = self._chunks[c]
        if isinstance(chunk, tuple):
            offsets, values = chunk
            ids = np.array(values[offsets[j]:offsets[j + 1]], dtype=np.int64)  # copies only this row
        else:
            ids = np.asarray(self._tok.encode(chunk[j].as_py(), bos=True, eos=True), dtype=np.int64)
        if self.max_len is not None:
            ids = ids[: self.max_len]
        plen = int(self._prompt_len[i]) if self._prompt_len is not None else 1
        return {"input_ids": torch.from_numpy(ids), "prompt_len": plen}


def dataset_tokens(ds: Sequence) -> int:
    """Total token count (for throughput accounting)."""
    if isinstance(ds, ArrowTokenDataset) and ds._chunks and isinstance(ds._chunks[0], tuple):
        return int(sum(o[-1] - o[0] for o, _ in ds._chunks))
    return sum(len(ds[i]["input_ids"]) for i in range(len(ds)))
