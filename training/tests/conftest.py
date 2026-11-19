import os
import socket
import subprocess
import sys
import time

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from training.data import generate, write_arrow  # noqa: E402
from training.models import ModelConfig, TinyCausalLM  # noqa: E402

SMALL = dict(vocab_size=98, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64)


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end runs of the CLI (slower)")


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return TinyCausalLM(ModelConfig(**SMALL))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_arrow(generate("pretrain", 600, 0), d / "pretrain.arrow", batch_rows=256)
    write_arrow(generate("reverse", 1200, 0), d / "reverse_train.arrow", batch_rows=256)
    write_arrow(generate("reverse", 64, 1), d / "reverse_eval.arrow")
    return d


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_upstream():
    port = free_port()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "benchmarks", "mock_upstream.py"), "--port", str(port),
                             "--min-latency-ms", "20", "--max-latency-ms", "20", "--tokens", "4"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    proc.kill()
    proc.wait()
