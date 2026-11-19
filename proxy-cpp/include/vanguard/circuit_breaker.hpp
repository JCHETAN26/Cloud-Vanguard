#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string_view>

namespace vanguard {

// Thread-safe circuit breaker (Closed -> Open -> Half-Open -> Closed).
//
//  Closed     every request is allowed; `failure_threshold` consecutive
//             failures trip the breaker.
//  Open       requests are rejected until `open_until`.
//  Half-Open  after the open period, up to `half_open_probes` requests are let
//             through as probes. A successful probe closes the breaker; a
//             failed probe re-opens it for `backoff_multiplier` times longer
//             than last time (exponential backoff, capped at `max_open`).
//
// Results from requests that started before a state change ("stale" results)
// are ignored, so a burst of in-flight failures cannot extend the backoff.
// Time is injected so tests are deterministic.
class CircuitBreaker {
 public:
  using Clock = std::chrono::steady_clock;
  using Millis = std::chrono::milliseconds;

  enum class State { kClosed = 0, kOpen = 1, kHalfOpen = 2 };

  struct Options {
    std::uint32_t failure_threshold = 5;
    Millis base_open{1000};
    Millis max_open{30000};
    double backoff_multiplier = 2.0;
    std::uint32_t half_open_probes = 1;
  };

  struct Permit {
    bool allowed = false;
    bool probe = false;  // caller must report the outcome (or release_probe)
  };

  struct Snapshot {
    State state = State::kClosed;
    std::uint32_t consecutive_failures = 0;
    Millis open_for{0};               // length of the current/next open period
    Millis retry_after{0};            // time until a probe will be admitted
    std::uint64_t trips = 0;          // transitions into Open
    std::uint64_t rejections = 0;
  };

  CircuitBreaker() : CircuitBreaker(Options{}) {}
  explicit CircuitBreaker(Options options);

  Permit try_acquire(Clock::time_point now);
  void record_success(bool probe, Clock::time_point now);
  void record_failure(bool probe, Clock::time_point now);
  // A probe ended without an outcome (e.g. the client disconnected).
  void release_probe();

  [[nodiscard]] Snapshot snapshot(Clock::time_point now) const;
  [[nodiscard]] State state(Clock::time_point now) const { return snapshot(now).state; }

 private:
  void open_locked(Clock::time_point now);

  const Options opt_;
  mutable std::mutex mu_;
  State state_ = State::kClosed;
  std::uint32_t consecutive_ = 0;
  std::uint32_t probes_in_flight_ = 0;
  Millis open_for_;
  Clock::time_point open_until_{};
  std::uint64_t trips_ = 0;
  std::uint64_t rejections_ = 0;
};

std::string_view to_string(CircuitBreaker::State s) noexcept;

// Chooses between a primary and an optional fallback upstream, each guarded
// by its own breaker. Pure routing policy (no I/O) so it can be unit tested.
class FailoverRouter {
 public:
  enum Target : int { kNone = -1, kPrimary = 0, kFallback = 1 };

  struct Decision {
    Target target = kNone;
    bool probe = false;
    bool failover = false;         // routed away from the primary
    std::string_view reason;       // why (when failover or kNone)
  };

  FailoverRouter(CircuitBreaker* primary, CircuitBreaker* fallback)
      : primary_(primary), fallback_(fallback) {}

  // `primary_failed` is true when this request already failed on the primary.
  Decision route(bool primary_failed, CircuitBreaker::Clock::time_point now) const;

  // After an attempt failed: may the request be retried on the fallback?
  // Only before any response bytes reached the client, and only once.
  [[nodiscard]] bool can_fail_over(Target failed_target, bool already_failed_over,
                                   bool response_started) const noexcept {
    return failed_target == kPrimary && fallback_ != nullptr && !already_failed_over && !response_started;
  }

  // Upstream statuses that count against the breaker and trigger failover.
  static constexpr bool is_failure_status(int status) noexcept { return status >= 500; }

  [[nodiscard]] bool has_fallback() const noexcept { return fallback_ != nullptr; }

 private:
  CircuitBreaker* primary_;
  CircuitBreaker* fallback_;
};

}  // namespace vanguard
