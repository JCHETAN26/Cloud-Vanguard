"""CloudVanguard Milestone 3: PEFT/LoRA fine-tuning, quantization, data loading and evaluation.

Subpackages:
  training.data    Arrow (memory-mapped) datasets, synthetic corpora, DataLoader construction
  training.models  tiny causal LM, LoRA adapters, weight quantization, mixed precision, DDP, checkpoints
  training.eval    MLflow tracking and the evaluation harness (torch / vLLM / OpenAI-HTTP backends)

Entry point: python -m training --help
"""

__version__ = "0.3.0"
