#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

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

// Process-wide counters shared by all worker threads (relaxed atomics).
struct Metrics {
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

  LatencyHistogram latency_hit;
  LatencyHistogram latency_hit_stream;
  LatencyHistogram latency_miss;
  LatencyHistogram latency_miss_stream;

  void count_status(int status) noexcept;

  [[nodiscard]] std::string render(const CacheStats& cache, std::string_view node_id,
                                   std::size_t ring_nodes, std::size_t ring_points) const;

 private:
  static constexpr std::array<int, 16> kTrackedStatuses = {200, 400, 401, 404, 405, 408, 411, 413,
                                                           429, 431, 500, 502, 503, 504, 505, 0};
  std::array<std::atomic<std::uint64_t>, kTrackedStatuses.size()> statuses_{};
};

}  // namespace vanguard
