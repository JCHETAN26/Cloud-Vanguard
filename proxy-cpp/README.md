# vanguard-proxy (Phases 1–4, Milestone 1 tracing)

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

## Multi-node (Phase 3)

N gateways form one logical cache. Each request is routed by its exact cache key on a 1000-vnode consistent-hash
ring to the **owner**:

```
client -> any node (ingress) -- owner elsewhere --> same request + X-Vanguard-Hop: 1 + X-Vanguard-Peer-Token --> owner
                                                     owner serves (cache/semantic/upstream) <- response relayed
```

| Situation | Behaviour |
|---|---|
| Ingress is the owner | serve locally |
| Owner elsewhere, reachable | forward once; the response (JSON or SSE) is relayed byte-for-byte with `X-Vanguard-Ingress` |
| Owner unknown / unresolved / circuit open / connect fails | **serve locally** (`vanguard_forward_local_fallback_total`) |
| Forwarded to us, but our ring names another owner (rolling update) | **serve locally, never re-forward** (`vanguard_ring_divergence_total`) |
| `X-Vanguard-Hop` > 1 | **508 Loop Detected** |
| `X-Vanguard-Hop` without a valid `X-Vanguard-Peer-Token` | **403** (clients can't skip rate limits) |

**Rate limits and response headers.** Tenant limits are cluster-wide: each of N members enforces limit/N. RPM is
charged only at the ingress hop, and TPM at the node that calls the upstream. Every chat response carries
`X-Vanguard-Owner` (the ring owner) and `X-Vanguard-Node` (the node that served it).

**Control plane.** Configuration arrives on the admin listener (`ADMIN_PORT`, default 9901,
`ADMIN_TOKEN`) or from `CONTROL_FILE` (polled every second; reloaded immediately on SIGHUP), using the messages
in [`proto/vanguard/control/v1/control.proto`](../proto/vanguard/control/v1/control.proto):
- **Ring updates** replace the topology atomically. Hostnames are resolved by a control thread, off the event
  loop, and re-resolved every 15 s, since pod IPs change on restart.
- **Tenant policy sets** replace dynamic tenants atomically. A tenant authenticates by the SHA-256 of its API
  key, and can override `semanticMode`, `semanticThreshold` and `ttlSeconds`.
- **Versions are monotonic**, so an older update can't overwrite a newer one.
- **Workers pick up new snapshots** with one atomic generation check per event-loop iteration.

Static setup without an operator: `PEERS=n0=10.0.0.1:8080,n1=10.0.0.2:8080 NODE_ID=n0`.

## Warm-cache snapshots (Phase 4)

`SNAPSHOT_PATH` points at a file (filesystem volume) or a **raw block device** (CSI `volumeMode: Block`, set
`SNAPSHOT_MODE=block`). The cache is written as a CRC-checked record stream:
- **On disk:** a filesystem volume is written to a temp file, fsynced and renamed. A raw block device uses two
  alternating slots; the data goes to the older slot and is fsynced, and only then is that slot's header (generation,
  length, CRCs) written and fsynced. Loading picks the newest slot whose checksums verify, so a crash at any point
  leaves the previous snapshot readable.
- **When:** every `SNAPSHOT_INTERVAL_S` seconds, on SIGTERM, and on `POST /admin/v1/snapshot`. A restore happens before
  the proxy serves its first request (or on `POST /admin/v1/restore`). TTLs keep counting through downtime.
- **Off the hot path:** all snapshot I/O runs on its own thread; `GET /admin/v1/snapshot` reports progress.

## Tracing (Milestone 1)

OpenTelemetry traces over OTLP/HTTP (JSON encoding), enabled with `OTEL_EXPORTER_OTLP_ENDPOINT`. One SERVER span
per chat request, plus CLIENT spans for each LLM upstream attempt, embedding call and peer forward. W3C
`traceparent` is continued from the caller and injected into every outbound request, so a request forwarded to its
ring owner is a single trace across nodes. Span schema, settings and trade-offs:
[`observability/README.md`](../observability/README.md). Implementation: `include/vanguard/tracing.hpp`,
`src/tracing.cpp`.

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
| `include/vanguard/snapshot.hpp` | warm-cache snapshot format, file mode and A/B block-slot mode |
| `include/vanguard/control.hpp` | control messages (JSON), versioned ring/tenant snapshots, forwarding decision, peer resolution |
| `include/vanguard/circuit_breaker.hpp` | closed/open/half-open breaker with exponential backoff; `FailoverRouter` policy |
| `include/vanguard/embedding.hpp` | prompt normalization + guard, lexical encoder, `/v1/embeddings` client helpers |
| `include/vanguard/semantic_index.hpp` | scope-partitioned exact cosine index, per-scope + global bounds |
| `include/vanguard/{tenants,config,metrics}.hpp` | tenant registry, env/flag config, Prometheus metrics |

## Endpoints

* `POST /v1/chat/completions` (OpenAI-compatible, `stream: true|false`)
* `GET /healthz`
* `GET /metrics` (Prometheus text format: cache, rate limit, upstream counters, latency histograms, trace exporter counters)

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
`BREAKER_OPEN_MS` (1000), `BREAKER_MAX_OPEN_MS` (30000), `SEMANTIC_MODE` (`lexical`|`neural`|`off`;
`EMBEDDER=remote` still accepted), `EMBEDDER_URL`, `EMBEDDER_MODEL`, `EMBEDDER_TIMEOUT_MS` (150),
`SEMANTIC_THRESHOLD` (default 0.94 lexical, **0.96 neural**),
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
* **Salient-token guard.** Calibrating bge-small showed different questions scoring above
  0.95 ("reverse a list in Python" vs "… in JavaScript" 0.954; "center a div" vs "… vertically"
  0.955), overlapping real paraphrases (0.875–0.985). So neural mode defaults to 0.96, and every
  match above the threshold must also pass a lexical check: each prompt's salient tokens
  (content words minus a small generic/framing vocabulary, numbers and negations) must have a
  counterpart in the other, with inflections allowed ("differ"/"difference") but not
  "java"/"javascript". This trades recall for safety: 3 of 12 hand-written paraphrases hit,
  with 0 false positives.
* **Two encoders.** The lexical one (default, ~20 µs, no dependencies) catches rewording
  such as casing, filler, framing and inflection, but not true paraphrase. The remote one
  (`SEMANTIC_MODE=neural`) calls any OpenAI-compatible `/v1/embeddings` service asynchronously on the event loop
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
