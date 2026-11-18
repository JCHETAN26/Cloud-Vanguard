#include "vanguard/rate_limiter.hpp"

#include <algorithm>
#include <limits>

#include "vanguard/murmur3.hpp"

namespace vanguard {

void TokenBucket::refill(SteadyTime now) noexcept {
  if (now <= last_) return;
  const double elapsed = std::chrono::duration<double>(now - last_).count();
  tokens_ = std::min(capacity_, tokens_ + elapsed * rate_);
  last_ = now;
}

RateDecision TokenBucket::try_take(double n, SteadyTime now) noexcept {
  refill(now);
  if (tokens_ >= n) {
    tokens_ -= n;
    return {true, 0};
  }
  if (n > capacity_ || rate_ <= 0) {
    return {false, std::numeric_limits<double>::infinity()};
  }
  return {false, (n - tokens_) / rate_};
}

void TokenBucket::reconfigure(double capacity, double refill_per_sec, SteadyTime now) noexcept {
  refill(now);
  capacity_ = capacity;
  rate_ = refill_per_sec;
  tokens_ = std::min(tokens_, capacity_);
}

double TokenBucket::available(SteadyTime now) noexcept {
  refill(now);
  return tokens_;
}

RateLimiter::RateLimiter(std::size_t shards)
    : num_shards_(shards == 0 ? 1 : shards), shards_(std::make_unique<Shard[]>(num_shards_)) {}

RateLimiter::Shard& RateLimiter::shard_for(std::string_view tenant) {
  return shards_[murmur3_64(tenant) % num_shards_];
}

RateLimiter::TenantState& RateLimiter::state_locked(Shard& shard, std::string_view tenant,
                                                    const TenantLimits& limits, SteadyTime now) {
  auto it = shard.tenants.find(std::string(tenant));
  if (it == shard.tenants.end()) {
    TenantState st;
    st.rpm = TokenBucket(limits.rpm, limits.rpm / 60.0, now);
    st.tpm = TokenBucket(limits.tpm, limits.tpm / 60.0, now);
    st.applied = limits;
    it = shard.tenants.emplace(std::string(tenant), st).first;
  } else if (!(it->second.applied == limits)) {
    it->second.rpm.reconfigure(limits.rpm, limits.rpm / 60.0, now);
    it->second.tpm.reconfigure(limits.tpm, limits.tpm / 60.0, now);
    it->second.applied = limits;
  }
  return it->second;
}

RateDecision RateLimiter::check_request(std::string_view tenant, const TenantLimits& limits,
                                        SteadyTime now) {
  if (limits.rpm <= 0) return {true, 0};
  auto& shard = shard_for(tenant);
  std::lock_guard lock(shard.mu);
  return state_locked(shard, tenant, limits, now).rpm.try_take(1.0, now);
}

RateDecision RateLimiter::check_tokens(std::string_view tenant, const TenantLimits& limits,
                                       double tokens, SteadyTime now) {
  if (limits.tpm <= 0) return {true, 0};
  auto& shard = shard_for(tenant);
  std::lock_guard lock(shard.mu);
  return state_locked(shard, tenant, limits, now).tpm.try_take(tokens, now);
}

std::size_t RateLimiter::tenant_count() const {
  std::size_t n = 0;
  for (std::size_t i = 0; i < num_shards_; ++i) {
    std::lock_guard lock(shards_[i].mu);
    n += shards_[i].tenants.size();
  }
  return n;
}

}  // namespace vanguard
