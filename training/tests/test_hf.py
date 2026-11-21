"""The Hugging Face path (training.hf) on CPU with a tiny random Qwen2 model built offline.

The GPU runs (Qwen2.5-14B, bitsandbytes NF4, NCCL DDP, vLLM) live in benchmarks/gpu/; these tests pin the
logic those runs depend on: tokenization/masking, the training loop, the portable NF4 path, DDP wiring,
adapter interop with peft, and the evaluation backend."""

import json
import os
import random
import subprocess
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from .conftest import free_port  # noqa: E402
from training.data import ArrowTokenDataset, write_arrow  # noqa: E402
from training.hf.data import LABELS, encode, format_prompt  # noqa: E402
from training.hf.train import HFTrainConfig, train_hf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEYWORDS = {
    0: ["minister", "election", "treaty", "embassy", "parliament", "summit"],
    1: ["match", "goal", "coach", "league", "season", "striker"],
    2: ["shares", "profit", "market", "earnings", "merger", "investors"],
    3: ["software", "chip", "internet", "researchers", "satellite", "browser"],
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def article(label: int, rng: random.Random) -> str:
    words = rng.sample(KEYWORDS[label], 3) + rng.sample(["the", "a", "on", "today", "said", "new", "after"], 3)
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


@pytest.fixture(scope="module")
def tiny_hf(tmp_path_factory):
    """A 2-layer Qwen2 with a byte-level BPE tokenizer trained on the task's own text, saved like a hub model."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen2")
    rng = random.Random(0)
    corpus = [format_prompt(article(y, rng)) + " " + LABELS[y] for y in range(4) for _ in range(50)]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=False)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def task_data(tiny_hf, tmp_path_factory):
    from training.hf.model import load_tokenizer

    tok = load_tokenizer(tiny_hf)
    d = tmp_path_factory.mktemp("agnews_like")
    rng = random.Random(1)
    write_arrow((encode(tok, article(y, rng), y) for y in [i % 4 for i in range(400)]), d / "train.arrow")
    write_arrow((encode(tok, article(y, rng), y) for y in [i % 4 for i in range(16)]), d / "eval.arrow")
    return d


def test_encode_masks_prompt_and_appends_eos(tiny_hf):
    from training.hf.model import load_tokenizer

    tok = load_tokenizer(tiny_hf)
    rec = encode(tok, "Shares rose after the merger.", 2)
    assert rec["answer"] == "Business" and rec["text"] == rec["prompt"] + " Business"
    assert rec["input_ids"][: rec["prompt_len"]] == tok(rec["prompt"]).input_ids
    assert rec["input_ids"][-1] == tok.eos_token_id
    assert tok.decode(rec["input_ids"][rec["prompt_len"]:-1]).strip() == "Business"


def _cfg(tiny_hf, task_data, out, **kw) -> HFTrainConfig:
    base = dict(model=tiny_hf, train_data=str(task_data / "train.arrow"), eval_data=str(task_data / "eval.arrow"),
                output_dir=str(out), dtype="fp32", device="cpu", steps=60, batch_size=16, lr=1e-2, warmup_steps=5,
                lora_r=8, lora_alpha=16, lora_dropout=0.0, eval_examples=8, max_new_tokens=4, log_every=20,
                max_len=128, track=False)
    base.update(kw)
    return HFTrainConfig(**base)


def test_lora_trains_saves_and_loads_in_peft(tiny_hf, task_data, tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    s = train_hf(_cfg(tiny_hf, task_data, tmp_path / "lora"))
    # A random frozen base gives near-flat logits: LoRA moves the argmax long before the loss gets small.
    assert s["final_loss_mean_last10"] < s["first_loss"] - 0.5, s
    assert 0 < s["trainable_pct"] < 50 and s["lora_modules"] == 2 * 7
    assert s["eval"]["exact_match"] is not None and s["eval"]["perplexity"] > 0
    # The adapter is a PEFT adapter for the HF base: same logits through peft as through our LoRA.
    from training.hf.model import load_hf_model
    from training.models import load_adapter
    ours, _ = load_hf_model(tiny_hf, device=torch.device("cpu"), dtype="fp32")
    load_adapter(ours, s["adapter_dir"])
    theirs = peft.PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tiny_hf, dtype=torch.float32),
                                            s["adapter_dir"])
    x = torch.tensor([ArrowTokenDataset(task_data / "eval.arrow")[0]["input_ids"].tolist()])
    with torch.no_grad():
        torch.testing.assert_close(ours(input_ids=x).logits, theirs(input_ids=x).logits, rtol=1e-4, atol=1e-4)
    cfg = load_json(os.path.join(s["adapter_dir"], "adapter_config.json"))
    assert cfg["base_model_name_or_path"] == tiny_hf and cfg["r"] == 8


def test_learns_the_task(tiny_hf, task_data, tmp_path):
    s = train_hf(_cfg(tiny_hf, task_data, tmp_path / "learn", steps=300, eval_examples=16))
    assert s["eval"]["exact_match"] >= 0.75, s["eval"]["samples"]


def test_qlora_nf4_portable_backend(tiny_hf, task_data, tmp_path):
    s = train_hf(_cfg(tiny_hf, task_data, tmp_path / "nf4", quant_bits=4, quant_backend="torch"))
    assert s["quant_backend"] == "torch"
    assert s["final_loss_mean_last10"] < s["first_loss"] - 0.5, s
    full = train_hf(_cfg(tiny_hf, task_data, tmp_path / "full", steps=1, eval_examples=0))
    assert s["weight_bytes"] < full["weight_bytes"]


def test_cli_train_hf_ddp_gloo_and_eval(tiny_hf, task_data, tmp_path):
    """torchrun 2 ranks (Gloo): the frozen-param DDP ignore list, sharded loader and rank-0 save/eval."""
    out = tmp_path / "ddp"
    env = {**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_addr=127.0.0.1",
           f"--master_port={free_port()}", "-m", "training",
           "train-hf", "--model", tiny_hf, "--train-data", str(task_data / "train.arrow"), "--eval-data",
           str(task_data / "eval.arrow"), "--output-dir", str(out), "--dtype", "fp32", "--device", "cpu",
           "--steps", "20", "--batch-size", "8", "--eval-examples", "4", "--max-new-tokens", "4", "--no-track"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    s = load_json(out / "train_summary.json")
    assert s["world_size"] == 2 and s["global_batch"] == 16
    ev = tmp_path / "eval.json"
    r = subprocess.run([sys.executable, "-m", "training", "eval", "--backend", "hf", "--hf-model", tiny_hf,
                        "--adapter", s["adapter_dir"], "--data", str(task_data / "eval.arrow"), "--limit", "4",
                        "--precision", "fp32", "--device", "cpu", "--max-new-tokens", "4", "--output", str(ev)],
                       env=env, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = load_json(ev)
    assert rep["backend"] == "hf" and rep["examples"] == 4 and rep["perplexity"] > 0
