#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "vanguard/circuit_breaker.hpp"

using namespace std::chrono_literals;
using vanguard::CircuitBreaker;
using vanguard::FailoverRouter;
using State = CircuitBreaker::State;

namespace {

const auto kT0 = CircuitBreaker::Clock::time_point{} + 1h;

CircuitBreaker::Options opts(std::uint32_t threshold = 3) {
  CircuitBreaker::Options o;
  o.failure_threshold = threshold;
  o.base_open = 1000ms;
  o.max_open = 8000ms;
  o.backoff_multiplier = 2.0;
  o.half_open_probes = 1;
  return o;
}

void fail(CircuitBreaker& b, int n, CircuitBreaker::Clock::time_point t) {
  for (int i = 0; i < n; ++i) {
    const auto p = b.try_acquire(t);
    ASSERT_TRUE(p.allowed);
    b.record_failure(p.probe, t);
  }
}

}  // namespace

TEST(CircuitBreaker, TripsAfterConsecutiveFailures) {
  CircuitBreaker b(opts(3));
  fail(b, 2, kT0);
  EXPECT_EQ(b.state(kT0), State::kClosed);
  fail(b, 1, kT0);
  EXPECT_EQ(b.state(kT0), State::kOpen);
  EXPECT_FALSE(b.try_acquire(kT0).allowed);
  EXPECT_EQ(b.snapshot(kT0).trips, 1u);
  EXPECT_EQ(b.snapshot(kT0).retry_after, 1000ms);
}

TEST(CircuitBreaker, SuccessResetsTheFailureStreak) {
  CircuitBreaker b(opts(3));
  fail(b, 2, kT0);
  b.record_success(false, kT0);
  fail(b, 2, kT0);
  EXPECT_EQ(b.state(kT0), State::kClosed);
}

TEST(CircuitBreaker, OpenRejectsUntilBackoffThenAdmitsOneProbe) {
  CircuitBreaker b(opts(1));
  fail(b, 1, kT0);
  EXPECT_FALSE(b.try_acquire(kT0 + 999ms).allowed);
  EXPECT_EQ(b.state(kT0 + 1000ms), State::kHalfOpen);
  const auto probe = b.try_acquire(kT0 + 1000ms);
  EXPECT_TRUE(probe.allowed);
  EXPECT_TRUE(probe.probe);
  // Only one probe at a time.
  EXPECT_FALSE(b.try_acquire(kT0 + 1001ms).allowed);
}

TEST(CircuitBreaker, ProbeSuccessClosesAndResetsBackoff) {
  CircuitBreaker b(opts(1));
  fail(b, 1, kT0);
  auto p = b.try_acquire(kT0 + 1s);
  b.record_failure(p.probe, kT0 + 1s);  // open for 2s now
  p = b.try_acquire(kT0 + 3s);
  ASSERT_TRUE(p.probe);
  b.record_success(p.probe, kT0 + 3s);
  EXPECT_EQ(b.state(kT0 + 3s), State::kClosed);
  // The next trip starts again from the base duration.
  fail(b, 1, kT0 + 4s);
  EXPECT_EQ(b.snapshot(kT0 + 4s).retry_after, 1000ms);
}

TEST(CircuitBreaker, FailedProbesBackOffExponentiallyWithCap) {
  CircuitBreaker b(opts(1));
  fail(b, 1, kT0);
  auto t = kT0;
  std::vector<long long> periods;
  for (int i = 0; i < 6; ++i) {
    t += b.snapshot(t).retry_after;  // wait exactly until half-open
    const auto p = b.try_acquire(t);
    ASSERT_TRUE(p.allowed && p.probe) << "probe " << i;
    b.record_failure(true, t);
    periods.push_back(b.snapshot(t).retry_after.count());
  }
  EXPECT_EQ(periods, (std::vector<long long>{2000, 4000, 8000, 8000, 8000, 8000}));
  EXPECT_EQ(b.snapshot(t).trips, 7u);
}

TEST(CircuitBreaker, StaleResultsAreIgnored) {
  CircuitBreaker b(opts(2));
  // Two requests start while closed...
  const auto a = b.try_acquire(kT0);
  const auto c = b.try_acquire(kT0);
  const auto d = b.try_acquire(kT0);
  b.record_failure(a.probe, kT0);
  b.record_failure(c.probe, kT0);  // trips
  // ...a third in-flight failure must not extend the open period,
  b.record_failure(d.probe, kT0 + 500ms);
  EXPECT_EQ(b.snapshot(kT0 + 500ms).retry_after, 500ms);
  // and a late success must not close the breaker.
  b.record_success(false, kT0 + 600ms);
  EXPECT_EQ(b.state(kT0 + 600ms), State::kOpen);
}

TEST(CircuitBreaker, ReleasedProbeFreesTheSlot) {
  CircuitBreaker b(opts(1));
  fail(b, 1, kT0);
  const auto p = b.try_acquire(kT0 + 1s);
  ASSERT_TRUE(p.probe);
  EXPECT_FALSE(b.try_acquire(kT0 + 1s).allowed);
  b.release_probe();  // client went away before the probe completed
  EXPECT_TRUE(b.try_acquire(kT0 + 1s).probe);
}

TEST(CircuitBreaker, ExactlyOneProbeUnderConcurrency) {
  CircuitBreaker b(opts(1));
  fail(b, 1, kT0);
  std::atomic<int> probes{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < 1000; ++j) {
        if (b.try_acquire(kT0 + 2s).allowed) probes.fetch_add(1);
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(probes.load(), 1);
}

// ---------------------------------------------------------------------------
// Failover routing
// ---------------------------------------------------------------------------

TEST(FailoverRouter, HealthyPrimaryTakesTraffic) {
  CircuitBreaker primary(opts()), fallback(opts());
  FailoverRouter router(&primary, &fallback);
  const auto d = router.route(false, kT0);
  EXPECT_EQ(d.target, FailoverRouter::kPrimary);
  EXPECT_FALSE(d.failover);
}

TEST(FailoverRouter, FailedPrimaryAttemptRetriesOnFallbackOnce) {
  CircuitBreaker primary(opts()), fallback(opts());
  FailoverRouter router(&primary, &fallback);
  EXPECT_TRUE(router.can_fail_over(FailoverRouter::kPrimary, false, false));
  const auto d = router.route(/*primary_failed=*/true, kT0);
  EXPECT_EQ(d.target, FailoverRouter::kFallback);
  EXPECT_TRUE(d.failover);
  EXPECT_EQ(d.reason, "primary-failed");
  // Never twice, never off the fallback, never once bytes reached the client.
  EXPECT_FALSE(router.can_fail_over(FailoverRouter::kPrimary, true, false));
  EXPECT_FALSE(router.can_fail_over(FailoverRouter::kFallback, false, false));
  EXPECT_FALSE(router.can_fail_over(FailoverRouter::kPrimary, false, true));
}

TEST(FailoverRouter, OpenPrimaryRoutesStraightToFallback) {
  CircuitBreaker primary(opts(3)), fallback(opts());
  FailoverRouter router(&primary, &fallback);
  fail(primary, 3, kT0);
  const auto d = router.route(false, kT0 + 10ms);
  EXPECT_EQ(d.target, FailoverRouter::kFallback);
  EXPECT_EQ(d.reason, "primary-circuit-open");
}

TEST(FailoverRouter, RecoveryReturnsTrafficToPrimaryViaProbe) {
  CircuitBreaker primary(opts(1)), fallback(opts());
  FailoverRouter router(&primary, &fallback);
  fail(primary, 1, kT0);
  // After the backoff the next request probes the primary...
  const auto probe = router.route(false, kT0 + 1s);
  ASSERT_EQ(probe.target, FailoverRouter::kPrimary);
  ASSERT_TRUE(probe.probe);
  // ...concurrent requests keep using the fallback while the probe is out...
  EXPECT_EQ(router.route(false, kT0 + 1s).target, FailoverRouter::kFallback);
  primary.record_success(true, kT0 + 1s);
  // ...and once it succeeds the primary is back.
  const auto after = router.route(false, kT0 + 1s);
  EXPECT_EQ(after.target, FailoverRouter::kPrimary);
  EXPECT_FALSE(after.probe);
}

TEST(FailoverRouter, NoFallbackOrBothOpenMeansNoRoute) {
  CircuitBreaker primary(opts(1)), fallback(opts(1));
  fail(primary, 1, kT0);
  EXPECT_EQ(FailoverRouter(&primary, nullptr).route(false, kT0).target, FailoverRouter::kNone);
  fail(fallback, 1, kT0);
  const auto d = FailoverRouter(&primary, &fallback).route(false, kT0);
  EXPECT_EQ(d.target, FailoverRouter::kNone);
  EXPECT_EQ(d.reason, "all-circuits-open");
}

TEST(FailoverRouter, OnlyServerErrorsCountAsFailures) {
  EXPECT_TRUE(FailoverRouter::is_failure_status(500));
  EXPECT_TRUE(FailoverRouter::is_failure_status(503));
  EXPECT_FALSE(FailoverRouter::is_failure_status(429));
  EXPECT_FALSE(FailoverRouter::is_failure_status(400));
  EXPECT_FALSE(FailoverRouter::is_failure_status(200));
}
