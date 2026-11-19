#pragma once

#include <sys/socket.h>

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "vanguard/cached_response.hpp"
#include "vanguard/circuit_breaker.hpp"
#include "vanguard/config.hpp"
#include "vanguard/control.hpp"
#include "vanguard/embedding.hpp"
#include "vanguard/hash_ring.hpp"
#include "vanguard/lru_cache.hpp"
#include "vanguard/metrics.hpp"
#include "vanguard/rate_limiter.hpp"
#include "vanguard/semantic_index.hpp"
#include "vanguard/snapshot.hpp"
#include "vanguard/tenants.hpp"
#include "vanguard/tracing.hpp"

namespace vanguard {

// Index into SharedState::targets (matches FailoverRouter::Target for 0 and 1).
// kPeerTarget marks connections to other gateway nodes (addresses come from
// the ring topology, not from `targets`).
enum TargetIndex : int { kPrimaryTarget = 0, kFallbackTarget = 1, kEmbedderTarget = 2, kPeerTarget = 3 };

// One upstream endpoint: where it is, how to talk to it, and its breaker.
struct UpstreamTarget {
  std::string name;  // "primary" | "fallback" | "embedder"
  bool configured = false;
  UpstreamUrl url;
  std::string host_header;  // "host:port"
  sockaddr_storage addr{};
  socklen_t addr_len = 0;
  std::string api_key;
  std::string model_override;  // rewrite "model" in forwarded bodies
  unsigned ttfb_timeout_ms = 300;
  std::unique_ptr<CircuitBreaker> breaker = std::make_unique<CircuitBreaker>();
};

// Resolves the URL once at startup (blocking getaddrinfo never runs on an
// event loop).
std::optional<UpstreamTarget> resolve_target(std::string name, const std::string& url, std::string api_key,
                                             unsigned ttfb_timeout_ms, const CircuitBreaker::Options& breaker,
                                             std::string& error);

// State shared by every worker thread. Everything here is internally
// synchronized; per-connection state lives in each worker's event loop.
struct SharedState {
  SharedState(Config c, ControlPlane::Options control_options, TenantRegistry base_tenants,
              std::array<UpstreamTarget, 3> upstreams);

  const Config cfg;
  const std::array<UpstreamTarget, 3> targets;
  const FailoverRouter router;
  const LexicalEmbedder lexical;
  ControlPlane control;  // ring topology + tenant registry (hot-reloadable)
  RateLimiter limiter;
  ShardedLruCache<CachedResponse> cache;
  SemanticIndex semantic;
  Metrics metrics;
  tracing::Tracer tracer;  // OTLP span export (disabled without an endpoint)

  // Wakes the control thread (e.g. to resolve new peer hostnames right away).
  void wake_control();
  std::mutex wake_mu;
  std::condition_variable wake_cv;
  bool wake = false;

  [[nodiscard]] std::vector<BreakerReport> breaker_reports() const;
  [[nodiscard]] ControlReport control_report() const;

  // Warm-cache snapshots: all I/O happens on the snapshot thread; requests
  // are queued (admin API) and their completion is observable via status.
  std::uint64_t request_snapshot_save();
  std::uint64_t request_snapshot_restore();
  SnapshotSaveStats run_snapshot_save();
  SnapshotLoadStats run_snapshot_restore();
  [[nodiscard]] std::string snapshot_status_json() const;
  mutable std::mutex snap_mu;
  std::condition_variable snap_cv;
  std::uint64_t save_requested = 0, save_completed = 0;
  std::uint64_t restore_requested = 0, restore_completed = 0;
  SnapshotSaveStats last_save;
  SnapshotLoadStats last_restore;
  std::int64_t last_save_unix = 0, last_restore_unix = 0;
};

// Starts cfg.workers epoll event loops (one SO_REUSEPORT listener each) and
// blocks until SIGINT/SIGTERM. Returns a process exit code.
int run_server(SharedState& state);

}  // namespace vanguard
