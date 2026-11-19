#include "vanguard/metrics.hpp"

#include <cstdio>

namespace vanguard {
namespace {

void line(std::string& out, std::string_view name, std::string_view labels, double value) {
  char num[64];
  std::snprintf(num, sizeof(num), "%.17g", value);
  out.append(name);
  if (!labels.empty()) {
    out.push_back('{');
    out.append(labels);
    out.push_back('}');
  }
  out.push_back(' ');
  out.append(num);
  out.push_back('\n');
}

void counter(std::string& out, std::string_view name, std::string_view help, std::uint64_t v) {
  out.append("# HELP ").append(name).append(" ").append(help).append("\n");
  out.append("# TYPE ").append(name).append(" counter\n");
  line(out, name, {}, static_cast<double>(v));
}

void gauge(std::string& out, std::string_view name, std::string_view help, double v) {
  out.append("# HELP ").append(name).append(" ").append(help).append("\n");
  out.append("# TYPE ").append(name).append(" gauge\n");
  line(out, name, {}, v);
}

std::uint64_t load(const std::atomic<std::uint64_t>& a) { return a.load(std::memory_order_relaxed); }

}  // namespace

void LatencyHistogram::observe(std::chrono::nanoseconds d) noexcept {
  const double s = std::chrono::duration<double>(d).count();
  std::size_t i = 0;
  while (i < kBounds.size() && s > kBounds[i]) ++i;
  buckets_[i].fetch_add(1, std::memory_order_relaxed);
  sum_ns_.fetch_add(static_cast<std::uint64_t>(d.count() > 0 ? d.count() : 0), std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
}

void LatencyHistogram::render(std::string& out, std::string_view name, std::string_view labels) const {
  const std::string bucket_name = std::string(name) + "_bucket";
  const std::string prefix = labels.empty() ? std::string() : std::string(labels) + ",";
  std::uint64_t cumulative = 0;
  char le[64];
  for (std::size_t i = 0; i <= kBounds.size(); ++i) {
    cumulative += buckets_[i].load(std::memory_order_relaxed);
    if (i < kBounds.size()) {
      std::snprintf(le, sizeof(le), "le=\"%g\"", kBounds[i]);
    } else {
      std::snprintf(le, sizeof(le), "le=\"+Inf\"");
    }
    line(out, bucket_name, prefix + le, static_cast<double>(cumulative));
  }
  line(out, std::string(name) + "_sum", labels,
       static_cast<double>(sum_ns_.load(std::memory_order_relaxed)) / 1e9);
  line(out, std::string(name) + "_count", labels,
       static_cast<double>(count_.load(std::memory_order_relaxed)));
}

void Metrics::count_status(int status) noexcept {
  std::size_t i = 0;
  while (i + 1 < kTrackedStatuses.size() && kTrackedStatuses[i] != status) ++i;
  statuses_[i].fetch_add(1, std::memory_order_relaxed);  // last slot = "other"
}

std::string Metrics::render(const CacheStats& cache, std::string_view node_id, std::size_t ring_nodes,
                            std::size_t ring_points, std::size_t semantic_entries,
                            const std::vector<BreakerReport>& breakers) const {
  std::string out;
  out.reserve(8192);

  out.append("# HELP vanguard_info Static information about this gateway instance.\n");
  out.append("# TYPE vanguard_info gauge\n");
  line(out, "vanguard_info", "node_id=\"" + std::string(node_id) + "\",version=\"0.1.0\"", 1);

  counter(out, "vanguard_connections_accepted_total", "Client connections accepted.", load(connections_accepted));
  gauge(out, "vanguard_connections_active", "Open client connections.",
        static_cast<double>(connections_active.load(std::memory_order_relaxed)));
  counter(out, "vanguard_requests_total", "HTTP requests parsed.", load(requests_total));

  out.append("# HELP vanguard_responses_total Responses sent, by status code.\n");
  out.append("# TYPE vanguard_responses_total counter\n");
  for (std::size_t i = 0; i < kTrackedStatuses.size(); ++i) {
    const std::string code = kTrackedStatuses[i] == 0 ? "other" : std::to_string(kTrackedStatuses[i]);
    line(out, "vanguard_responses_total", "code=\"" + code + "\"",
         static_cast<double>(statuses_[i].load(std::memory_order_relaxed)));
  }

  counter(out, "vanguard_cache_hits_total", "Chat completions served from cache.", load(cache_hits));
  counter(out, "vanguard_cache_hits_stream_total", "Streaming (SSE) completions replayed from cache.",
          load(cache_hits_stream));
  counter(out, "vanguard_cache_misses_total", "Chat completions not found in cache.", load(cache_misses));
  counter(out, "vanguard_cache_bypass_total", "Requests that skipped lookup (Cache-Control: no-cache).",
          load(cache_bypass));
  counter(out, "vanguard_cache_fills_total", "Upstream responses stored in cache.", load(cache_fills));
  counter(out, "vanguard_cache_fill_skipped_total",
          "Upstream responses not cached (non-200, incomplete stream, too large, no-store).",
          load(cache_fill_skipped));
  gauge(out, "vanguard_cache_entries", "Entries currently cached.", static_cast<double>(cache.entries));
  gauge(out, "vanguard_cache_bytes", "Bytes currently cached (accounted).", static_cast<double>(cache.bytes));
  gauge(out, "vanguard_cache_capacity_bytes", "Cache capacity in bytes.", static_cast<double>(cache.capacity_bytes));
  counter(out, "vanguard_cache_evictions_total", "LRU evictions.", cache.evictions);
  counter(out, "vanguard_cache_expirations_total", "Entries dropped on TTL expiry.", cache.expirations);

  counter(out, "vanguard_rate_limited_rpm_total", "Requests rejected by the RPM bucket.", load(rate_limited_rpm));
  counter(out, "vanguard_rate_limited_tpm_total", "Requests rejected by the TPM bucket.", load(rate_limited_tpm));
  counter(out, "vanguard_auth_failures_total", "Requests rejected for missing/unknown API key.", load(auth_failures));
  counter(out, "vanguard_bad_requests_total", "Malformed HTTP or chat requests.", load(bad_requests));

  counter(out, "vanguard_upstream_requests_total", "Requests forwarded upstream (cache misses).",
          load(upstream_requests));
  counter(out, "vanguard_upstream_connects_total", "New upstream TCP connections opened.", load(upstream_connects));
  counter(out, "vanguard_upstream_retries_total", "Requests retried after a stale pooled connection.",
          load(upstream_retries));
  counter(out, "vanguard_upstream_errors_total", "Upstream connect/protocol failures.", load(upstream_errors));
  counter(out, "vanguard_upstream_timeouts_total", "Upstream requests that timed out.", load(upstream_timeouts));
  counter(out, "vanguard_client_aborts_total", "Clients that disconnected before their response completed.",
          load(client_aborts));
  counter(out, "vanguard_estimated_tokens_forwarded_total", "Estimated tokens sent upstream (TPM accounting).",
          load(estimated_tokens_forwarded));

  counter(out, "vanguard_semantic_lookups_total", "Semantic index lookups after an exact miss.",
          load(semantic_lookups));
  counter(out, "vanguard_semantic_hits_total", "Requests served by a semantically similar cached prompt.",
          load(semantic_hits));
  counter(out, "vanguard_semantic_inserts_total", "Prompt vectors added to the semantic index.", load(semantic_inserts));
  counter(out, "vanguard_semantic_stale_total", "Semantic matches whose cache entry had been evicted.",
          load(semantic_stale));
  counter(out, "vanguard_semantic_skipped_total", "Exact misses that skipped semantic lookup (embedder unavailable).",
          load(semantic_skipped));
  counter(out, "vanguard_semantic_salient_rejections_total",
          "Matches above the threshold rejected because a salient token differed.",
          load(semantic_salient_rejections));
  gauge(out, "vanguard_semantic_entries", "Vectors in the semantic index.", static_cast<double>(semantic_entries));

  counter(out, "vanguard_failovers_total", "Requests routed to the fallback upstream.", load(failovers));
  counter(out, "vanguard_no_upstream_available_total", "Requests rejected because every circuit was open.",
          load(no_upstream_available));
  const char* const kUpstreamCounters[][2] = {
      {"requests", "Attempts sent to each upstream."},
      {"successes", "Attempts that returned non-5xx response headers."},
      {"failures", "Attempts that failed (connect, 5xx, drop, protocol)."},
      {"timeouts", "Attempts that timed out."},
      {"circuit_rejections", "Attempts not sent because the circuit was open."}};
  for (std::size_t m = 0; m < 5; ++m) {
    const std::string name = std::string("vanguard_upstream_") + kUpstreamCounters[m][0] + "_total";
    out.append("# HELP ").append(name).append(" ").append(kUpstreamCounters[m][1]).append("\n");
    out.append("# TYPE ").append(name).append(" counter\n");
    for (std::size_t t = 0; t < breakers.size() && t < kTargets; ++t) {
      if (!breakers[t].configured) continue;
      const auto& u = upstream[t];
      const std::atomic<std::uint64_t>* fields[] = {&u.requests, &u.successes, &u.failures, &u.timeouts,
                                                    &u.circuit_rejections};
      line(out, name, "upstream=\"" + breakers[t].name + "\"", static_cast<double>(load(*fields[m])));
    }
  }
  out.append("# HELP vanguard_circuit_state Breaker state per upstream (0 closed, 1 open, 2 half-open).\n");
  out.append("# TYPE vanguard_circuit_state gauge\n");
  for (const auto& b : breakers) {
    if (b.configured) line(out, "vanguard_circuit_state", "upstream=\"" + b.name + "\"", b.state);
  }
  out.append("# HELP vanguard_circuit_open_seconds Length of the current/next open period (backoff).\n");
  out.append("# TYPE vanguard_circuit_open_seconds gauge\n");
  for (const auto& b : breakers) {
    if (b.configured) line(out, "vanguard_circuit_open_seconds", "upstream=\"" + b.name + "\"", b.open_seconds);
  }
  out.append("# HELP vanguard_circuit_trips_total Transitions into the open state.\n");
  out.append("# TYPE vanguard_circuit_trips_total counter\n");
  for (const auto& b : breakers) {
    if (b.configured) {
      line(out, "vanguard_circuit_trips_total", "upstream=\"" + b.name + "\"", static_cast<double>(b.trips));
    }
  }

  gauge(out, "vanguard_ring_nodes", "Nodes on the consistent-hash ring.", static_cast<double>(ring_nodes));
  gauge(out, "vanguard_ring_points", "Virtual node points on the ring.", static_cast<double>(ring_points));

  out.append("# HELP vanguard_request_duration_seconds Chat completion latency inside the proxy.\n");
  out.append("# TYPE vanguard_request_duration_seconds histogram\n");
  latency_hit.render(out, "vanguard_request_duration_seconds", "cache=\"hit\",stream=\"false\"");
  latency_hit_stream.render(out, "vanguard_request_duration_seconds", "cache=\"hit\",stream=\"true\"");
  latency_miss.render(out, "vanguard_request_duration_seconds", "cache=\"miss\",stream=\"false\"");
  latency_miss_stream.render(out, "vanguard_request_duration_seconds", "cache=\"miss\",stream=\"true\"");
  latency_semantic_hit.render(out, "vanguard_request_duration_seconds", "cache=\"semantic_hit\",stream=\"any\"");
  out.append("# HELP vanguard_embedding_duration_seconds Remote embedding call latency.\n");
  out.append("# TYPE vanguard_embedding_duration_seconds histogram\n");
  latency_embedding.render(out, "vanguard_embedding_duration_seconds", "");
  return out;
}

}  // namespace vanguard
