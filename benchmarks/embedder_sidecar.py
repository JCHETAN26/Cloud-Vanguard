#!/usr/bin/env python3
"""OpenAI-compatible embedding sidecar for the CloudVanguard semantic cache.

Serves a real sentence-embedding model (default BAAI/bge-small-en-v1.5, 384
dims) through fastembed's ONNX runtime on CPU: no PyTorch, ~70 MB model.

  POST /v1/embeddings   {"model": "...", "input": "text" | ["text", ...]}
                        -> {"object": "list", "data": [{"embedding": [...], "index": 0}], ...}
  POST /similarity      {"pairs": [["a", "b"], ...]} -> {"cosine": [...]}  (calibration helper)
  GET  /healthz
  GET  /stats

The gateway calls this on every exact-cache miss when EMBEDDER=remote.

    pip install fastembed
    python3 embedder_sidecar.py --port 9200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

STATS = {"requests": 0, "texts": 0, "busy_ms": 0.0}


def http_response(status: int, body: bytes, content_type: str = "application/json") -> bytes:
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(status, "OK")
    return (
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\nConnection: keep-alive\r\n\r\n"
    ).encode() + body


class Encoder:
    def __init__(self, model_name: str, threads: int):
        from fastembed import TextEmbedding  # imported lazily so --help works without it

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, threads=threads)
        self.dim = len(self.embed(["warmup"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self.model.embed(texts)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def handle(reader, writer, enc: Encoder, pool: ThreadPoolExecutor) -> None:
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                return
            lines = head.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
            n = int(headers.get("content-length", "0") or 0)
            body = await reader.readexactly(n) if n else b""

            if method == "POST" and target.endswith("/embeddings"):
                try:
                    req = json.loads(body)
                    inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
                except (ValueError, KeyError, TypeError):
                    writer.write(http_response(400, b'{"error":{"message":"bad request"}}'))
                    await writer.drain()
                    continue
                t0 = time.perf_counter()
                vecs = await loop.run_in_executor(pool, enc.embed, inputs)
                STATS["busy_ms"] += (time.perf_counter() - t0) * 1000
                STATS["requests"] += 1
                STATS["texts"] += len(inputs)
                out = {
                    "object": "list",
                    "model": enc.model_name,
                    "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }
                writer.write(http_response(200, json.dumps(out).encode()))
            elif method == "POST" and target == "/similarity":
                pairs = json.loads(body)["pairs"]
                flat = [t for p in pairs for t in p]
                vecs = await loop.run_in_executor(pool, enc.embed, flat)
                sims = [cosine(vecs[2 * i], vecs[2 * i + 1]) for i in range(len(pairs))]
                writer.write(http_response(200, json.dumps({"model": enc.model_name, "cosine": sims}).encode()))
            elif target == "/healthz":
                writer.write(http_response(200, b"ok\n", "text/plain"))
            elif target == "/stats":
                writer.write(http_response(200, json.dumps(dict(STATS, model=enc.model_name, dim=enc.dim)).encode()))
            else:
                writer.write(http_response(404, b'{"error":{"message":"not found"}}'))
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--threads", type=int, default=2, help="ONNX intra-op threads")
    ap.add_argument("--workers", type=int, default=2, help="concurrent embedding calls")
    args = ap.parse_args()

    t0 = time.perf_counter()
    enc = Encoder(args.model, args.threads)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    server = await asyncio.start_server(lambda r, w: handle(r, w, enc, pool), args.host, args.port, backlog=1024)
    print(f"embedder sidecar on {args.host}:{args.port}: {args.model} ({enc.dim} dims, loaded in "
          f"{time.perf_counter() - t0:.1f}s)", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
