#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vanguard/lru_cache.hpp"

namespace vanguard {

inline void inc(std::atomic<std::uint64_t>& c, std::uint64_t n = 1) noexcept {
  c.fetch_add(n, std::memory_order_relaxed);
}

// Lock-free fixed-bucket latency histogram (Prometheus-compatible).
class LatencyHistogram {
 public:
  // Upper bounds in seconds; the final implicit bucket is +Inf.
  static constexpr std::array<double, 16> kBounds = {0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
                                                     0.01,   0.025,   0.05,   0.1,   0.25,   0.5,
                                                     1.0,    2.5,     5.0,    10.0};

  void observe(std::chrono::nanoseconds d) noexcept;
  // Appends `<name>_bucket{<labels>,le=...}`, `_sum` and `_count` lines.
  void render(std::string& out, std::string_view name, std::string_view labels) const;

 private:
  std::array<std::atomic<std::uint64_t>, kBounds.size() + 1> buckets_{};
  std::atomic<std::uint64_t> sum_ns_{0};
  std::atomic<std::uint64_t> count_{0};
};

// Per-upstream counters (primary, fallback, embedder).
struct UpstreamTargetMetrics {
  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> successes{0};
  std::atomic<std::uint64_t> failures{0};   // connect errors, 5xx, drops, protocol errors
  std::atomic<std::uint64_t> timeouts{0};   // time-to-first-byte or idle timeouts
  std::atomic<std::uint64_t> circuit_rejections{0};
};

// Point-in-time breaker state for /metrics.
struct BreakerReport {
  std::string name;
  bool configured = false;
  int state = 0;  // 0 closed, 1 open, 2 half-open
  std::uint32_t consecutive_failures = 0;
  double open_seconds = 0;
  std::uint64_t trips = 0;
};

// Control-plane state for /metrics.
struct ControlReport {
  std::string node_id;
  std::size_t ring_nodes = 0;
  std::size_t ring_points = 0;
  std::int64_t ring_version = 0;
  std::int64_t tenants_version = 0;
};

// Process-wide counters shared by all worker threads (relaxed atomics).
struct Metrics {
  static constexpr std::size_t kTargets = 3;  // primary, fallback, embedder

  std::atomic<std::uint64_t> connections_accepted{0};
  std::atomic<std::int64_t> connections_active{0};
  std::atomic<std::uint64_t> requests_total{0};
  std::atomic<std::uint64_t> cache_hits{0};
  std::atomic<std::uint64_t> cache_hits_stream{0};
  std::atomic<std::uint64_t> cache_misses{0};
  std::atomic<std::uint64_t> cache_bypass{0};
  std::atomic<std::uint64_t> cache_fills{0};
  std::atomic<std::uint64_t> cache_fill_skipped{0};
  std::atomic<std::uint64_t> rate_limited_rpm{0};
  std::atomic<std::uint64_t> rate_limited_tpm{0};
  std::atomic<std::uint64_t> auth_failures{0};
  std::atomic<std::uint64_t> bad_requests{0};
  std::atomic<std::uint64_t> upstream_requests{0};
  std::atomic<std::uint64_t> upstream_connects{0};
  std::atomic<std::uint64_t> upstream_retries{0};
  std::atomic<std::uint64_t> upstream_errors{0};
  std::atomic<std::uint64_t> upstream_timeouts{0};
  std::atomic<std::uint64_t> client_aborts{0};
  std::atomic<std::uint64_t> estimated_tokens_forwarded{0};

  std::array<UpstreamTargetMetrics, kTargets> upstream;
  std::atomic<std::uint64_t> failovers{0};
  std::atomic<std::uint64_t> no_upstream_available{0};

  // Request coalescing.
  std::atomic<std::uint64_t> coalesce_leaders{0};   // misses that made the upstream call for their key
  std::atomic<std::uint64_t> coalesce_waiters{0};   // misses that waited on another request's call
  std::atomic<std::uint64_t> coalesce_served{0};    // waiters answered from the leader's cache fill
  std::atomic<std::uint64_t> coalesce_retries{0};   // waiters whose leader produced no cache entry
  std::atomic<std::uint64_t> coalesce_solo{0};      // misses that skipped coalescing (waiter cap reached)

  std::atomic<std::uint64_t> semantic_lookups{0};
  std::atomic<std::uint64_t> semantic_hits{0};
  std::atomic<std::uint64_t> semantic_inserts{0};
  std::atomic<std::uint64_t> semantic_stale{0};
  std::atomic<std::uint64_t> semantic_skipped{0};  // embedding unavailable
  std::atomic<std::uint64_t> semantic_salient_rejections{0};
  std::atomic<std::uint64_t> semantic_mode_downgrades{0};  // tenant asked neural, no sidecar configured

  // Multi-node routing.
  std::atomic<std::uint64_t> forwarded{0};               // sent to the owner node
  std::atomic<std::uint64_t> forward_received{0};        // served here for another ingress
  std::atomic<std::uint64_t> forward_local_fallback{0};  // owner unreachable: served here
  std::atomic<std::uint64_t> forward_failures{0};
  std::atomic<std::uint64_t> ring_divergence{0};         // forwarded to us but our ring disagrees
  std::atomic<std::uint64_t> hop_rejected_loop{0};
  std::atomic<std::uint64_t> hop_rejected_untrusted{0};
  std::atomic<std::uint64_t> admin_updates{0};
  std::atomic<std::uint64_t> admin_rejected{0};

  // Warm-cache snapshots.
  std::atomic<std::uint64_t> snapshot_saves{0};
  std::atomic<std::uint64_t> snapshot_restores{0};
  std::atomic<std::uint64_t> snapshot_failures{0};
  std::atomic<std::uint64_t> snapshot_last_entries{0};
  std::atomic<std::uint64_t> snapshot_last_bytes{0};
  std::atomic<std::uint64_t> snapshot_restored_entries{0};

  LatencyHistogram latency_hit;
  LatencyHistogram latency_hit_stream;
  LatencyHistogram latency_miss;
  LatencyHistogram latency_miss_stream;
  LatencyHistogram latency_semantic_hit;
  LatencyHistogram latency_embedding;
  LatencyHistogram latency_forwarded;

  void count_status(int status) noexcept;

  [[nodiscard]] std::string render(const CacheStats& cache, const ControlReport& control,
                                   std::size_t semantic_entries,
                                   const std::vector<BreakerReport>& breakers) const;

 private:
  static constexpr std::array<int, 18> kTrackedStatuses = {200, 400, 401, 403, 404, 405, 408, 409, 411,
                                                           413, 429, 431, 500, 502, 503, 504, 508, 0};
  std::array<std::atomic<std::uint64_t>, kTrackedStatuses.size()> statuses_{};
};

}  // namespace vanguard
