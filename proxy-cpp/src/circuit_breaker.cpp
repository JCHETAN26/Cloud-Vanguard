#include "vanguard/circuit_breaker.hpp"

#include <algorithm>

namespace vanguard {

namespace {
CircuitBreaker::Options sanitize(CircuitBreaker::Options o) {
  o.failure_threshold = std::max<std::uint32_t>(o.failure_threshold, 1);
  o.half_open_probes = std::max<std::uint32_t>(o.half_open_probes, 1);
  o.backoff_multiplier = std::max(o.backoff_multiplier, 1.0);
  o.max_open = std::max(o.max_open, o.base_open);
  return o;
}
}  // namespace

CircuitBreaker::CircuitBreaker(Options options) : opt_(sanitize(options)), open_for_(opt_.base_open) {}

void CircuitBreaker::open_locked(Clock::time_point now) {
  state_ = State::kOpen;
  open_until_ = now + open_for_;
  probes_in_flight_ = 0;
  ++trips_;
}

CircuitBreaker::Permit CircuitBreaker::try_acquire(Clock::time_point now) {
  std::lock_guard lock(mu_);
  if (state_ == State::kOpen) {
    if (now < open_until_) {
      ++rejections_;
      return {false, false};
    }
    state_ = State::kHalfOpen;
    probes_in_flight_ = 0;
  }
  if (state_ == State::kHalfOpen) {
    if (probes_in_flight_ >= opt_.half_open_probes) {
      ++rejections_;
      return {false, false};
    }
    ++probes_in_flight_;
    return {true, true};
  }
  return {true, false};
}

void CircuitBreaker::record_success(bool probe, Clock::time_point /*now*/) {
  std::lock_guard lock(mu_);
  if (state_ == State::kClosed) {
    consecutive_ = 0;
  } else if (state_ == State::kHalfOpen && probe) {
    state_ = State::kClosed;
    consecutive_ = 0;
    probes_in_flight_ = 0;
    open_for_ = opt_.base_open;  // fully recovered: reset the backoff
  }
  // Open, or a non-probe result while half-open: stale, ignore.
}

void CircuitBreaker::record_failure(bool probe, Clock::time_point now) {
  std::lock_guard lock(mu_);
  if (state_ == State::kClosed) {
    if (++consecutive_ >= opt_.failure_threshold) {
      open_for_ = opt_.base_open;
      open_locked(now);
    }
  } else if (state_ == State::kHalfOpen && probe) {
    const auto next = std::chrono::duration_cast<Millis>(open_for_ * opt_.backoff_multiplier);
    open_for_ = std::min(next, opt_.max_open);
    open_locked(now);
  }
}

void CircuitBreaker::release_probe() {
  std::lock_guard lock(mu_);
  if (state_ == State::kHalfOpen && probes_in_flight_ > 0) --probes_in_flight_;
}

CircuitBreaker::Snapshot CircuitBreaker::snapshot(Clock::time_point now) const {
  std::lock_guard lock(mu_);
  Snapshot s;
  s.state = (state_ == State::kOpen && now >= open_until_) ? State::kHalfOpen : state_;
  s.consecutive_failures = consecutive_;
  s.open_for = open_for_;
  s.retry_after = state_ == State::kOpen && now < open_until_
                      ? std::chrono::duration_cast<Millis>(open_until_ - now)
                      : Millis{0};
  s.trips = trips_;
  s.rejections = rejections_;
  return s;
}

std::string_view to_string(CircuitBreaker::State s) noexcept {
  switch (s) {
    case CircuitBreaker::State::kClosed: return "closed";
    case CircuitBreaker::State::kOpen: return "open";
    case CircuitBreaker::State::kHalfOpen: return "half_open";
  }
  return "unknown";
}

FailoverRouter::Decision FailoverRouter::route(bool primary_failed, CircuitBreaker::Clock::time_point now) const {
  std::string_view reason;
  if (primary_failed) {
    reason = "primary-failed";
  } else {
    const auto p = primary_->try_acquire(now);
    if (p.allowed) return {kPrimary, p.probe, false, {}};
    reason = "primary-circuit-open";
  }
  if (fallback_ == nullptr) return {kNone, false, false, reason};
  const auto f = fallback_->try_acquire(now);
  if (f.allowed) return {kFallback, f.probe, true, reason};
  return {kNone, false, false, "all-circuits-open"};
}

}  // namespace vanguard
