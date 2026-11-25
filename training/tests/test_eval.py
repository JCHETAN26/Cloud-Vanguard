import math

import pytest
import torch

from training.data import ArrowTokenDataset, CharTokenizer, LoaderConfig, build_dataloader
from training.eval import (Backend, Generation, OpenAIHTTPBackend, TorchBackend, VLLMBackend, evaluate, percentile,
                           vllm_available)

EXAMPLES = [
    {"prompt": "rev abc =>", "answer": "cba", "text": "rev abc => cba"},
    {"prompt": "rev xy =>", "answer": "yx", "text": "rev xy => yx"},
    {"prompt": "rev hello =>", "answer": "olleh", "text": "rev hello => olleh"},
    {"prompt": "rev pq =>", "answer": "qp", "text": "rev pq => qp"},
]


class Scripted(Backend):
    """Answers correctly except for one prompt; fixed NLL per token; fixed latency."""
    name = "scripted"
    supports_scoring = True

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, max_new_tokens):
        self.calls += 1
        answer = prompt.split()[1][::-1]
        return Generation(" " + (answer if "hello" not in prompt else "wrong"), 0.010, len(answer))

    def score(self, prompt, completion):
        return 0.5 * len(completion), len(completion)


def test_percentile():
    assert percentile([], 50) != percentile([], 50)  # nan
    assert percentile([5.0], 99) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(101)), 95) == 95


def test_harness_metrics_from_a_scripted_backend():
    b = Scripted()
    rep = evaluate(b, EXAMPLES, warmup=2)
    assert b.calls == len(EXAMPLES) + 2  # warm-up requests are not measured
    assert rep.exact_match == 0.75
    assert rep.mean_nll == pytest.approx(0.5) and rep.perplexity == pytest.approx(math.exp(0.5))
    assert rep.scored_tokens == sum(len(e["text"]) - len(e["prompt"]) for e in EXAMPLES)
    assert rep.latency_ms["p50"] == pytest.approx(10.0) and rep.latency_ms["p99"] == pytest.approx(10.0)
    assert rep.samples[2] == {"i": 2, "prompt": "rev hello =>", "expected": "olleh", "got": "wrong", "ok": False}
    m = rep.metrics()
    assert m["eval/exact_match"] == 0.75 and "eval/latency_p95_ms" in m
    with pytest.raises(ValueError):
        evaluate(b, [])


def test_harness_keeps_every_prediction_when_asked():
    many = EXAMPLES * 3
    assert len(evaluate(Scripted(), many, keep_samples=2).samples) == 2
    kept = evaluate(Scripted(), many, keep_samples=None).samples
    assert [s["i"] for s in kept] == list(range(len(many)))


def test_torch_backend_score_matches_the_training_loss(small_model, data_dir):
    """Perplexity from the harness is the same quantity the trainer minimises."""
    ds = ArrowTokenDataset(data_dir / "reverse_eval.arrow")
    loader = build_dataloader(ds, LoaderConfig(batch_size=1, shuffle=False, drop_last=False, pin_memory=False))
    batch = next(iter(loader))
    with torch.no_grad():
        _, loss = small_model(batch["input_ids"], labels=batch["labels"])
    t = ds.table
    prompt, text = t.column("prompt")[0].as_py(), t.column("text")[0].as_py()
    nll, n = TorchBackend(small_model).score(prompt, text[len(prompt):])
    assert n == int((batch["labels"][:, 1:] != -100).sum())
    assert nll / n == pytest.approx(float(loss), rel=1e-5)


def test_torch_backend_generates_and_times(small_model):
    b = TorchBackend(small_model)
    g = b.generate("rev abc =>", 5)
    assert g.latency_s > 0 and 0 < g.new_tokens <= 5
    rep = evaluate(b, EXAMPLES, max_new_tokens=5)
    assert rep.backend == "torch" and rep.exact_match == 0.0  # untrained model
    assert rep.perplexity > 1 and rep.latency_ms["p50"] > 0 and rep.generated_tokens_per_s > 0


def test_torch_backend_truncates_to_the_context_window(small_model):
    long = "x" * 500
    nll, n = TorchBackend(small_model).score("", long)
    assert n == small_model.config.max_len - 1


def test_openai_http_backend_against_mock_upstream(mock_upstream):
    b = OpenAIHTTPBackend(mock_upstream, model="mock", api_key="k")
    rep = evaluate(b, EXAMPLES, max_new_tokens=4)
    assert rep.backend == "openai" and rep.perplexity is None and rep.scored_tokens == 0
    assert rep.exact_match == 0.0  # the mock answers with unrelated words
    assert rep.latency_ms["p50"] >= 20.0  # the mock waits 20 ms per request
    assert rep.generated_tokens_per_s > 0
    with pytest.raises(NotImplementedError):
        b.score("a", "b")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        b.generate("__mock_error_500__", 4)
    b.close()
    with pytest.raises(ValueError):
        OpenAIHTTPBackend("ftp://x")


@pytest.mark.skipif(not vllm_available(), reason="vLLM needs the vllm package and a CUDA GPU; not available here")
def test_vllm_backend(tmp_path):
    tr = pytest.importorskip("transformers")
    cfg = tr.LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    tr.LlamaForCausalLM(cfg).save_pretrained(tmp_path / "m")
    tok = tr.AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.save_pretrained(tmp_path / "m")
    rep = evaluate(VLLMBackend(str(tmp_path / "m"), max_model_len=128), EXAMPLES, max_new_tokens=4)
    assert rep.backend == "vllm" and rep.perplexity is not None


def test_tokenizer_is_the_one_the_backend_uses(small_model):
    assert TorchBackend(small_model).tok.vocab_size == CharTokenizer.vocab_size == small_model.config.vocab_size
