"""Hugging Face model path: real pretrained causal LMs (e.g. Qwen2.5-14B) through the same LoRA, quantization,
DDP, MLflow and evaluation code as the tiny model.

The tiny path (training.train) is the CPU-verifiable reference; this path is for GPUs. Both use
training.models.lora (adapters in PEFT layout) and training.models.quant (bitsandbytes NF4/int8 on CUDA).
transformers is imported lazily so the rest of the package works without it.
"""

from .gpu import GPUSampler, gpu_snapshot
from .model import HFBackend, load_hf_model, load_tokenizer

__all__ = ["GPUSampler", "gpu_snapshot", "HFBackend", "load_hf_model", "load_tokenizer"]
