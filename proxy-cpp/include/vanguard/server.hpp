#pragma once

#include <sys/socket.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vanguard/cached_response.hpp"
#include "vanguard/circuit_breaker.hpp"
#include "vanguard/config.hpp"
#include "vanguard/embedding.hpp"
#include "vanguard/hash_ring.hpp"
#include "vanguard/lru_cache.hpp"
#include "vanguard/metrics.hpp"
#include "vanguard/rate_limiter.hpp"
#include "vanguard/semantic_index.hpp"
#include "vanguard/tenants.hpp"

namespace vanguard {

// Index into SharedState::targets (matches FailoverRouter::Target for 0 and 1).
enum TargetIndex : int { kPrimaryTarget = 0, kFallbackTarget = 1, kEmbedderTarget = 2 };

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
  SharedState(Config c, TenantRegistry t, std::array<UpstreamTarget, 3> upstreams);

  const Config cfg;
  const TenantRegistry tenants;
  const std::array<UpstreamTarget, 3> targets;
  const FailoverRouter router;
  const LexicalEmbedder lexical;
  RateLimiter limiter;
  ShardedLruCache<CachedResponse> cache;
  SemanticIndex semantic;
  HashRing ring;
  Metrics metrics;

  [[nodiscard]] std::vector<BreakerReport> breaker_reports() const;
};

// Starts cfg.workers epoll event loops (one SO_REUSEPORT listener each) and
// blocks until SIGINT/SIGTERM. Returns a process exit code.
int run_server(SharedState& state);

}  // namespace vanguard
