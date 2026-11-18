# vanguard-proxy (Phase 1)

C++20, Linux `epoll` (edge-triggered), no framework. Dependencies: nlohmann/json,
plus GoogleTest for tests. CMake uses the system packages if present and fetches
them otherwise.

## Request path

```
client ──► epoll worker (SO_REUSEPORT listener, one per thread)
            │ parse HTTP/1.1 (incremental, pipelining-safe)
            │ auth: Authorization: Bearer <tenant key> ──► 401
            │ RPM token bucket (all requests)            ──► 429 + Retry-After
            │ validate body, key = SHA-256(tenant, model, stream, canon(messages), canon(params))
            │ ring owner = consistent-hash(key)          (X-Vanguard-Owner)
            ├─ cache HIT ──► memcpy stored JSON / pre-chunked SSE into socket   X-Cache: HIT
            └─ MISS: TPM token bucket (estimated tokens) ──► 429
                     non-blocking upstream conn (per-worker keep-alive pool)
                     JSON: buffer, respond, cache if 200
                     SSE:  relay each upstream chunk immediately, capture events,
                           cache only if the stream completed with data: [DONE]
```

## Layout

| File | Responsibility |
|---|---|
| `src/server.cpp` | epoll loops, connection state machines, upstream pool, routing (Linux only) |
| `include/vanguard/http_parser.hpp` | incremental request parser and response parser (CL / chunked / until-close) |
| `include/vanguard/lru_cache.hpp` | sharded, byte-bounded, TTL-aware LRU (`shared_ptr<const V>` values) |
| `include/vanguard/hash_ring.hpp` | consistent-hash ring, MurmurHash3, 150 vnodes by default |
| `include/vanguard/rate_limiter.hpp` | per-tenant RPM + TPM token buckets, sharded mutexes |
| `include/vanguard/llm_request.hpp` | chat request validation, canonical cache key, token estimate |
| `include/vanguard/sse.hpp` | SSE event splitter, `[DONE]` detection, HTTP chunk framing |
| `include/vanguard/{tenants,config,metrics}.hpp` | tenant registry, env/flag config, Prometheus metrics |

## Endpoints

* `POST /v1/chat/completions` (OpenAI-compatible, `stream: true|false`)
* `GET /healthz`
* `GET /metrics` (Prometheus text format: cache, rate limit, upstream counters, latency histograms)

Response headers: `X-Cache: HIT|MISS|BYPASS`, `X-Vanguard-Cache-Key` (16-hex prefix),
`X-Vanguard-Owner` (ring owner). `Cache-Control: no-cache` skips lookup and
`no-store` skips the fill.

## Configuration

Set each option as an environment variable or as the matching flag
(`UPSTREAM_URL` ↔ `--upstream-url=`). Run `vanguard-proxy --help` for the full list. Key options:
`LISTEN_PORT` (8080), `WORKERS` (#cores), `UPSTREAM_URL`, `UPSTREAM_API_KEY`,
`TENANTS_FILE`, `DEFAULT_RPM`/`DEFAULT_TPM`, `CACHE_MAX_MB` (256), `CACHE_SHARDS` (16),
`CACHE_TTL_S` (3600), `NODE_ID`, `RING_NODES`, `RING_VNODES` (150).

Example tenants file: [`../benchmarks/tenants.bench.json`](../benchmarks/tenants.bench.json).
Without one, any bearer token is accepted as its own tenant (`anon-<sha256 prefix>`).

## Design decisions and trade-offs

* **One event loop per core, no shared accept.** Each worker owns its listener
  (SO_REUSEPORT), its connections and its upstream pool, so the hot path takes
  no locks except the cache-shard and limiter-shard mutexes.
* **ET with EPOLLOUT registered once.** Under edge-triggering, having EPOLLOUT
  always armed is free and avoids an `epoll_ctl(MOD)` per partial write.
* **Connections are addressed by id, not pointer.** Closed connections are reaped
  after the event batch, so stale events in the same batch are harmless.
* **Streams are relayed as they arrive** (TTFT on a miss ≈ upstream TTFT) and
  cached only after a complete `[DONE]`. A cached stream is stored already
  chunk-framed, so replaying it is one buffer append. A client that disconnects
  mid-stream aborts the upstream request to stop token spend. A non-streaming miss
  runs to completion so the cache still fills.
* **Backpressure:** a slow streaming client above 4 MiB of queued output pauses
  upstream reads, which resume below 1 MiB.
* **RPM counts every request; TPM counts only misses.** Cache hits cost no upstream
  tokens, so a tenant that is out of token budget can still read cached answers.
* **Cache key includes generation params** (temperature, max_tokens, tools…) and
  the stream flag. It excludes `user`, so end users within a tenant share entries.
  Tenants never share entries.
* **The ring is advisory in Phase 1.** It computes and reports the key owner,
  while local shard selection uses `murmur3(key) % shards`. Phase 3 uses the owner
  for peer forwarding.
* **Sharded LRU = approximate global LRU.** Capacity is split evenly across shards.
* **Known Phase 1 gaps:** plain-HTTP upstream only (no TLS), IPv4 listener, no
  request coalescing for concurrent identical misses (cache stampede), a TPM
  estimate that is not reconciled with the upstream's reported `usage`, and chunked
  request bodies rejected with 411.
