#!/usr/bin/env python3
"""Mock OpenAI-compatible LLM upstream for CloudVanguard benchmarks.

Stdlib only (asyncio). Speaks HTTP/1.1 with keep-alive and implements:

  POST /v1/chat/completions   JSON or SSE (stream: true) completions
  GET  /healthz               liveness
  GET  /stats                 request counters (used to prove cache hits never
                              reach the upstream)

Latency model: each completion waits uniform(--min-latency-ms, --max-latency-ms)
before the first byte (time-to-first-token). Streams then emit one SSE event
per token every --token-interval-ms. Completion text is a deterministic
function of the prompt, so identical prompts get identical content.

Fault injection: a last user message containing "__mock_error_500__" returns a
500, which lets the benchmark verify that errors are never cached.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time

WORDS = (
    "cache gateway token stream latency tenant shard ring replay quota "
    "kernel epoll socket buffer vector model prompt answer cluster node"
).split()

STATS = {"requests": 0, "stream_requests": 0, "errors_injected": 0, "in_flight": 0, "max_in_flight": 0}


def completion_words(messages: list, n: int) -> list[str]:
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).digest()
    return [WORDS[digest[i % len(digest)] % len(WORDS)] for i in range(n)]


def last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def http_response(status: int, reason: str, body: bytes, content_type: str = "application/json") -> bytes:
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: keep-alive\r\n\r\n"
    )
    return head.encode() + body


def chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


async def handle_chat(body: bytes, writer: asyncio.StreamWriter, args: argparse.Namespace, rng: random.Random) -> None:
    try:
        req = json.loads(body)
        messages = req["messages"]
        model = req.get("model", "mock-model")
    except (ValueError, KeyError, TypeError):
        writer.write(http_response(400, "Bad Request", b'{"error":{"message":"bad request"}}'))
        return

    STATS["requests"] += 1
    stream = bool(req.get("stream", False))
    if stream:
        STATS["stream_requests"] += 1

    await asyncio.sleep(rng.uniform(args.min_latency_ms, args.max_latency_ms) / 1000.0)

    if "__mock_error_500__" in last_user_text(messages):
        STATS["errors_injected"] += 1
        writer.write(http_response(500, "Internal Server Error", b'{"error":{"message":"injected failure"}}'))
        return

    n_tokens = min(int(req.get("max_tokens") or args.tokens), args.tokens)
    words = completion_words(messages, n_tokens)
    cid = "chatcmpl-" + hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:24]
    created = int(time.time())

    if not stream:
        payload = {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(words)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)},
        }
        writer.write(http_response(200, "OK", json.dumps(payload).encode()))
        return

    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
        b"Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
    )

    def event(delta: dict, finish: str | None = None) -> bytes:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(obj)}\n\n".encode()

    writer.write(chunk(event({"role": "assistant", "content": ""})))
    await writer.drain()
    for i, w in enumerate(words):
        if args.token_interval_ms > 0:
            await asyncio.sleep(args.token_interval_ms / 1000.0)
        writer.write(chunk(event({"content": (" " if i else "") + w})))
        await writer.drain()
    writer.write(chunk(event({}, "stop")))
    writer.write(chunk(b"data: [DONE]\n\n"))
    writer.write(b"0\r\n\r\n")


async def handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, args, rng) -> None:
    try:
        while True:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                return
            lines = head.decode("latin-1").split("\r\n")
            try:
                method, target, _ = lines[0].split(" ", 2)
            except ValueError:
                return
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length", "0") or 0)
            body = await reader.readexactly(length) if length else b""

            if method == "POST" and target.endswith("/chat/completions"):
                STATS["in_flight"] += 1
                STATS["max_in_flight"] = max(STATS["max_in_flight"], STATS["in_flight"])
                try:
                    await handle_chat(body, writer, args, rng)
                    await writer.drain()
                finally:
                    STATS["in_flight"] -= 1
            else:
                if target == "/healthz":
                    writer.write(http_response(200, "OK", b"ok\n", "text/plain"))
                elif target == "/stats":
                    writer.write(http_response(200, "OK", json.dumps(STATS).encode()))
                else:
                    writer.write(http_response(404, "Not Found", b'{"error":{"message":"not found"}}'))
                await writer.drain()
            if headers.get("connection", "").lower() == "close":
                return
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--min-latency-ms", type=float, default=50.0)
    p.add_argument("--max-latency-ms", type=float, default=200.0)
    p.add_argument("--token-interval-ms", type=float, default=5.0)
    p.add_argument("--tokens", type=int, default=24, help="completion length in tokens (words)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    server = await asyncio.start_server(lambda r, w: handle_conn(r, w, args, rng), args.host, args.port, backlog=4096)
    print(
        f"mock upstream on {args.host}:{args.port} "
        f"(latency {args.min_latency_ms:.0f}-{args.max_latency_ms:.0f} ms, {args.tokens} tokens, "
        f"{args.token_interval_ms:.0f} ms/token)",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
