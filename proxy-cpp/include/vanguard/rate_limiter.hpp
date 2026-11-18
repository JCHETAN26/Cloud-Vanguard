#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace vanguard {

using SteadyTime = std::chrono::steady_clock::time_point;

// Per-tenant limits. A value <= 0 means "unlimited".
struct TenantLimits {
  double rpm = 0;  // requests per minute
  double tpm = 0;  // estimated tokens per minute

  friend bool operator==(const TenantLimits&, const TenantLimits&) = default;
};

struct RateDecision {
  bool allowed = true;
  // Seconds until the request could succeed. +infinity if it never can
  // (the cost exceeds the bucket's capacity).
  double retry_after_s = 0;
};

// Classic token bucket: holds at most `capacity` tokens and refills
// continuously at `refill_per_sec`. Time is injected so tests are deterministic.
class TokenBucket {
 public:
  TokenBucket() = default;
  TokenBucket(double capacity, double refill_per_sec, SteadyTime now) noexcept
      : capacity_(capacity), rate_(refill_per_sec), tokens_(capacity), last_(now) {}

  RateDecision try_take(double n, SteadyTime now) noexcept;
  // Change capacity/rate while preserving the current fill level (clamped).
  void reconfigure(double capacity, double refill_per_sec, SteadyTime now) noexcept;
  [[nodiscard]] double available(SteadyTime now) noexcept;

 private:
  void refill(SteadyTime now) noexcept;

  double capacity_ = 0;
  double rate_ = 0;
  double tokens_ = 0;
  SteadyTime last_{};
};

// Thread-safe, sharded per-tenant limiter with two independent buckets:
//   RPM: 1 token per request, burst = one minute's allowance.
//   TPM: estimated tokens per upstream call, burst = one minute's allowance.
// Buckets are created lazily on a tenant's first request and pick up limit
// changes (e.g. a policy update) on the next call.
class RateLimiter {
 public:
  explicit RateLimiter(std::size_t shards = 64);

  RateDecision check_request(std::string_view tenant, const TenantLimits& limits, SteadyTime now);
  RateDecision check_tokens(std::string_view tenant, const TenantLimits& limits, double tokens,
                            SteadyTime now);

  [[nodiscard]] std::size_t tenant_count() const;

 private:
  struct TenantState {
    TokenBucket rpm;
    TokenBucket tpm;
    TenantLimits applied;
  };
  struct Shard {
    mutable std::mutex mu;
    std::unordered_map<std::string, TenantState> tenants;
  };

  Shard& shard_for(std::string_view tenant);
  static TenantState& state_locked(Shard& shard, std::string_view tenant, const TenantLimits& limits,
                                   SteadyTime now);

  std::size_t num_shards_;
  std::unique_ptr<Shard[]> shards_;
};

}  // namespace vanguard
