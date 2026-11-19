# vanguard-proxy (Phases 1–2)

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
            ├─ embed final user message (lexical in-process, or async sidecar)
            │  semantic index (scope = everything else, exact) cosine >= 0.94
            │  + guard (numbers, negations) ──► serve that entry       X-Cache: SEMANTIC-HIT
            └─ MISS: TPM token bucket (estimated tokens) ──► 429
                     route: primary breaker closed? ──► primary, else fallback (503 if both open)
                     5xx / connect error / drop / no headers in 300 ms ──► retry once on fallback
                     non-blocking upstream conn (per-worker, per-upstream keep-alive pool)
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
| `include/vanguard/hash_ring.hpp` | consistent-hash ring, MurmurHash3, 1000 vnodes by default |
| `include/vanguard/rate_limiter.hpp` | per-tenant RPM + TPM token buckets, sharded mutexes |
| `include/vanguard/llm_request.hpp` | chat request validation, canonical cache key, token estimate |
| `include/vanguard/sse.hpp` | SSE event splitter, `[DONE]` detection, HTTP chunk framing |
| `include/vanguard/circuit_breaker.hpp` | closed/open/half-open breaker with exponential backoff; `FailoverRouter` policy |
| `include/vanguard/embedding.hpp` | prompt normalization + guard, lexical encoder, `/v1/embeddings` client helpers |
| `include/vanguard/semantic_index.hpp` | scope-partitioned exact cosine index, per-scope + global bounds |
| `include/vanguard/{tenants,config,metrics}.hpp` | tenant registry, env/flag config, Prometheus metrics |

## Endpoints

* `POST /v1/chat/completions` (OpenAI-compatible, `stream: true|false`)
* `GET /healthz`
* `GET /metrics` (Prometheus text format: cache, rate limit, upstream counters, latency histograms)

Response headers: `X-Cache: HIT|SEMANTIC-HIT|MISS|BYPASS`, `X-Vanguard-Similarity` (semantic hits),
`X-Vanguard-Upstream: primary|fallback`, `X-Vanguard-Failover: <reason>`, `X-Vanguard-Cache-Key` (16-hex prefix),
`X-Vanguard-Owner` (ring owner). `Cache-Control: no-cache` skips lookup and
`no-store` skips the fill.

## Configuration

Set each option as an environment variable or as the matching flag
(`UPSTREAM_URL` ↔ `--upstream-url=`). Run `vanguard-proxy --help` for the full list. Key options:
`LISTEN_PORT` (8080), `WORKERS` (#cores), `UPSTREAM_URL`, `UPSTREAM_API_KEY`,
`TENANTS_FILE`, `DEFAULT_RPM`/`DEFAULT_TPM`, `CACHE_MAX_MB` (256), `CACHE_SHARDS` (16),
`CACHE_TTL_S` (3600), `NODE_ID`, `RING_NODES`, `RING_VNODES` (1000).

Phase 2: `FALLBACK_UPSTREAM_URL`, `FALLBACK_UPSTREAM_API_KEY`, `FALLBACK_MODEL`,
`UPSTREAM_TTFB_TIMEOUT_MS` (300), `FALLBACK_TTFB_TIMEOUT_MS` (10000), `BREAKER_FAILURE_THRESHOLD` (5),
`BREAKER_OPEN_MS` (1000), `BREAKER_MAX_OPEN_MS` (30000), `EMBEDDER` (`lexical`|`remote`|`off`),
`EMBEDDER_URL`, `EMBEDDER_MODEL`, `EMBEDDER_TIMEOUT_MS` (150), `SEMANTIC_THRESHOLD` (0.94),
`SEMANTIC_MAX_CHARS` (2000), `SEMANTIC_MAX_ENTRIES` (100000), `SEMANTIC_MAX_PER_SCOPE` (4096).

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
* **Semantic matching only varies the final user message.** Tenant, model, stream flag,
  system prompt, earlier turns and generation params form an exact *scope*. A guard
  additionally requires identical numbers and negation count, so "2+2" never matches
  "2+3" and "is X safe" never matches "is X not safe".
* **Two encoders.** The lexical one (default, ~20 µs, no dependencies) catches rewording
  such as casing, filler, framing and inflection, but not true paraphrase. The remote one
  calls any OpenAI-compatible `/v1/embeddings` service asynchronously on the event loop
  (e.g. `benchmarks/embedder_sidecar.py`, bge-small). It has its own breaker and timeout;
  if the sidecar is slow or down, requests degrade to exact-match-only and are still served.
* **Exact brute-force scan per scope.** The scan is exact, with no approximate-index recall
  loss, and costs ~1 ms at the per-scope cap. Scopes larger than `SEMANTIC_MAX_PER_SCOPE`
  overwrite their oldest vectors. A single very large scope needs an approximate index
  (HNSW) or int8 vectors, which is not built yet.
* **A semantic hit aliases its exact key**, so repeating the same new phrasing is an exact hit.
* **Breaker per upstream, shared by all workers.** It opens after N consecutive failures,
  then half-open admits one probe. A failed probe doubles the open period (capped); a
  successful one closes the breaker and resets the backoff. Results from requests that
  started before a state change are ignored.
* **Per-request failover happens once, before any byte reaches the client.** A stream that
  dies mid-relay is cut, not restarted elsewhere. Upstream 429 is passed through, not
  treated as a failure. Responses from the fallback are cached under the same key.
* **Known Phase 1 gaps:** plain-HTTP upstream only (no TLS), IPv4 listener, no
  request coalescing for concurrent identical misses (cache stampede), a TPM
  estimate that is not reconciled with the upstream's reported `usage`, and chunked
  request bodies rejected with 411.
