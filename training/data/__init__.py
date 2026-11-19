"""Data pipeline: memory-mapped Arrow datasets, synthetic corpora, DataLoaders."""

from .arrow import SCHEMA, ArrowTokenDataset, dataset_tokens, write_arrow
from .loader import IGNORE_INDEX, LoaderConfig, build_dataloader, collate, resolve_pin_memory, set_epoch
from .synthetic import TASKS, generate, task_pair
from .tokenizer import CharTokenizer

__all__ = ["SCHEMA", "ArrowTokenDataset", "dataset_tokens", "write_arrow", "IGNORE_INDEX", "LoaderConfig",
           "build_dataloader", "collate", "resolve_pin_memory", "set_epoch", "TASKS", "generate", "task_pair",
           "CharTokenizer"]
