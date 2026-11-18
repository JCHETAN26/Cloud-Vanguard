#pragma once

#include <sys/socket.h>

#include <atomic>
#include <optional>
#include <string>

#include "vanguard/cached_response.hpp"
#include "vanguard/config.hpp"
#include "vanguard/hash_ring.hpp"
#include "vanguard/lru_cache.hpp"
#include "vanguard/metrics.hpp"
#include "vanguard/rate_limiter.hpp"
#include "vanguard/tenants.hpp"

namespace vanguard {

struct UpstreamTarget {
  UpstreamUrl url;
  std::string host_header;  // "host:port"
  sockaddr_storage addr{};
  socklen_t addr_len = 0;
};

// Resolves UPSTREAM_URL once at startup (blocking getaddrinfo is fine here;
// it never runs on an event loop).
std::optional<UpstreamTarget> resolve_upstream(const Config& cfg, std::string& error);

// State shared by every worker thread. Everything here is internally
// synchronized; the per-connection state lives in each worker's event loop.
struct SharedState {
  SharedState(Config c, TenantRegistry t, UpstreamTarget u);

  const Config cfg;
  const TenantRegistry tenants;
  const UpstreamTarget upstream;
  RateLimiter limiter;
  ShardedLruCache<CachedResponse> cache;
  HashRing ring;
  Metrics metrics;
};

// Starts cfg.workers epoll event loops (one SO_REUSEPORT listener each) and
// blocks until SIGINT/SIGTERM. Returns a process exit code.
int run_server(SharedState& state);

}  // namespace vanguard
