# observability/

Telemetry for the gateway: **Prometheus metrics** (existing, `GET /metrics` on the data-plane port) and
**OpenTelemetry traces** (Milestone 1, OTLP/HTTP).

| Path | What |
|---|---|
| `otel-collector/config.yaml` | Collector pipeline: OTLP receiver (4318 HTTP, 4317 gRPC) → memory_limiter → batch → debug (add Jaeger/Tempo) |
| `prometheus/prometheus.yml` | Scrape config: local proxy + Kubernetes pods rendered by the operator (`app.kubernetes.io/name=vanguard-proxy`, port `http`) |
| `prometheus/rules.yml` | Recording rules (hit ratio, P50/P99, upstream failure ratio) and alerts (circuit open, no upstream, hit P99 > 5 ms, dropped spans) |
| `tools/otlp_sink.py` | Stdlib OTLP/JSON receiver that keeps spans in memory (tests, benchmarks, local debugging) |
| `scripts/fetch-tools.sh` | Downloads pinned `otelcol` 0.111.0 and `promtool` 2.54.1 into `.bin/` (gitignored) |
| `scripts/verify-collector.sh` | Real Collector: `otelcol validate`, then proxy → collector, asserts our trace id and spans arrive |
| `scripts/verify-prometheus.sh` | `promtool check config/rules`, and `promtool check metrics` on a live `/metrics` scrape |

## Tracing in the proxy

Enabled by pointing the proxy at an OTLP/HTTP endpoint (standard OpenTelemetry variable names):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318   # traces go to <url>/v1/traces
OTEL_TRACES_SAMPLER=parentbased_traceidratio             # default parentbased_always_on
OTEL_TRACES_SAMPLER_ARG=0.1
OTEL_SERVICE_NAME=vanguard-proxy                         # service.instance.id = NODE_ID
OTEL_BSP_SCHEDULE_DELAY=5000 OTEL_BSP_MAX_QUEUE_SIZE=2048 OTEL_BSP_MAX_EXPORT_BATCH_SIZE=512
```

No endpoint means tracing is off: no spans, no extra headers, no exporter thread.

Spans per chat request:

```
POST /v1/chat/completions   SERVER  vanguard.cache.result=hit|semantic_hit|miss|bypass|forwarded,
  │                                 vanguard.tenant, vanguard.route, vanguard.ring.owner, vanguard.hop,
  │                                 gen_ai.request.model, http.response.status_code, ...
  ├─ embedding              CLIENT  neural semantic mode only
  ├─ llm.upstream           CLIENT  one per attempt (primary, then fallback on failover); error.type on failure
  └─ peer.forward           CLIENT  when the key's ring owner is another node; that node's SERVER span is its child
```

Context propagation is W3C Trace Context: an incoming `traceparent` is continued (parent-based sampling honours
its sampled flag) and every outbound request carries a new `traceparent`, so the LLM server (vLLM supports
OTLP tracing) and peer gateways join the same trace. 5xx responses mark spans as errors; 4xx do not (HTTP
semantic conventions for server spans).

### Implementation notes and trade-offs

- **No opentelemetry-cpp dependency.** The proxy implements the pieces it needs (`include/vanguard/tracing.hpp`):
  W3C parsing/formatting, the six standard samplers, a bounded batch span processor and an OTLP/HTTP **JSON**
  exporter. It keeps the build dependency-free (opentelemetry-cpp's OTLP exporter pulls in protobuf + gRPC or
  libcurl). Cost: no OTLP/protobuf or gRPC export, no TLS to the collector (run a local collector/agent), no
  retry with backoff on 429/503 (a failed batch is dropped and counted), no metrics or logs signals over OTLP.
- **Never blocks the event loop.** Ending a span is a move into a mutex-protected vector; serialization and the
  HTTP POST happen on the exporter thread. When the queue is full, spans are dropped and counted
  (`vanguard_trace_spans_dropped_total`).
- **Exporter metrics** on `/metrics`: `vanguard_tracing_enabled`, `vanguard_trace_spans_{started,exported,dropped,failed}_total`,
  `vanguard_trace_export_{batches,errors}_total`, `vanguard_trace_queue_spans`.

### Measured overhead (`make bench-m1`)

One run on a 4-vCPU Codespace (AMD EPYC 9V74), 2 proxy workers, with `hey` (32 connections, 10 s, all cache hits),
the mock LLM and the OTLP receiver on the same machine. Raw data: `benchmarks/reports/milestone1_codespace_4vcpu.json`.

| config | hit req/s (`hey`) | `hey` P50 / P95 / P99 | spans exported / dropped |
|---|---|---|---|
| tracing off | 49,397 | 0.5 / 1.5 / 2.8 ms | – |
| 100% sampled, collector unreachable (proxy-side cost only) | 46,742 | 0.6 / 1.6 / 2.6 ms | 0 / 0 (all 491,458 failed export, as intended) |
| 100% sampled → Python test receiver | 36,417 | 0.6 / 2.5 / 3.6 ms | 214,312 / 114,544 |
| 10% sampled → Python test receiver | 48,721 | 0.5 / 1.5 / 2.6 ms | 51,017 / 0 |
| tracing off (repeat) | 52,366 | 0.5 / 1.4 / 2.0 ms | – |

The two tracing-off runs differ by 6%, so smaller differences are noise. The proxy's own histogram puts every
configuration's hit P99 in its lowest bucket (≤ 0.1 ms). The large drop at 100% sampling comes from the
stdlib Python receiver competing for the same 4 vCPUs and not keeping up (spans are dropped, the proxy is not
blocked). Use ratio sampling, or run the collector on separate cores, for full-rate traffic.
