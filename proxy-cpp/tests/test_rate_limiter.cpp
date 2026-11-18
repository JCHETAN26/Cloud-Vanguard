#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "vanguard/rate_limiter.hpp"

using namespace std::chrono_literals;
using vanguard::RateLimiter;
using vanguard::SteadyTime;
using vanguard::TenantLimits;
using vanguard::TokenBucket;

namespace {
const SteadyTime kT0 = SteadyTime{} + 1h;
}

TEST(TokenBucket, BurstThenDenyWithRetryAfter) {
  TokenBucket b(5, 1.0, kT0);  // 5 tokens, 1 token/s
  for (int i = 0; i < 5; ++i) EXPECT_TRUE(b.try_take(1, kT0).allowed);
  const auto d = b.try_take(1, kT0);
  EXPECT_FALSE(d.allowed);
  EXPECT_DOUBLE_EQ(d.retry_after_s, 1.0);
}

TEST(TokenBucket, RefillsContinuouslyAndCapsAtCapacity) {
  TokenBucket b(10, 2.0, kT0);
  EXPECT_TRUE(b.try_take(10, kT0).allowed);
  EXPECT_NEAR(b.available(kT0 + 2500ms), 5.0, 1e-9);
  EXPECT_NEAR(b.available(kT0 + 1h), 10.0, 1e-9);
}

TEST(TokenBucket, RequestLargerThanCapacityCanNeverSucceed) {
  TokenBucket b(100, 10, kT0);
  const auto d = b.try_take(101, kT0);
  EXPECT_FALSE(d.allowed);
  EXPECT_TRUE(std::isinf(d.retry_after_s));
}

TEST(TokenBucket, ClockGoingBackwardsDoesNotMintTokens) {
  TokenBucket b(1, 1, kT0);
  EXPECT_TRUE(b.try_take(1, kT0).allowed);
  EXPECT_FALSE(b.try_take(1, kT0 - 10s).allowed);
}

TEST(RateLimiter, RpmLimitsPerTenant) {
  RateLimiter rl;
  const TenantLimits limits{60, 0};  // 60 rpm => burst 60, refill 1/s
  for (int i = 0; i < 60; ++i) ASSERT_TRUE(rl.check_request("acme", limits, kT0).allowed);
  const auto denied = rl.check_request("acme", limits, kT0);
  EXPECT_FALSE(denied.allowed);
  EXPECT_NEAR(denied.retry_after_s, 1.0, 1e-9);
  // A different tenant has its own bucket.
  EXPECT_TRUE(rl.check_request("globex", limits, kT0).allowed);
  // One second later acme has exactly one more request.
  EXPECT_TRUE(rl.check_request("acme", limits, kT0 + 1s).allowed);
  EXPECT_FALSE(rl.check_request("acme", limits, kT0 + 1s).allowed);
}

TEST(RateLimiter, TpmIsIndependentOfRpm) {
  RateLimiter rl;
  const TenantLimits limits{1000, 600};  // 600 tokens/min => 10 tokens/s
  EXPECT_TRUE(rl.check_tokens("t", limits, 500, kT0).allowed);
  const auto d = rl.check_tokens("t", limits, 200, kT0);
  EXPECT_FALSE(d.allowed);
  EXPECT_NEAR(d.retry_after_s, 10.0, 1e-9);  // need 100 more tokens at 10/s
  EXPECT_TRUE(rl.check_request("t", limits, kT0).allowed);
}

TEST(RateLimiter, ZeroMeansUnlimited) {
  RateLimiter rl;
  const TenantLimits unlimited{0, 0};
  for (int i = 0; i < 10000; ++i) ASSERT_TRUE(rl.check_request("u", unlimited, kT0).allowed);
  EXPECT_TRUE(rl.check_tokens("u", unlimited, 1e12, kT0).allowed);
  EXPECT_EQ(rl.tenant_count(), 0u);  // unlimited tenants never allocate state
}

TEST(RateLimiter, LimitChangesApplyWithoutResettingUsage) {
  RateLimiter rl;
  for (int i = 0; i < 10; ++i) ASSERT_TRUE(rl.check_request("t", {10, 0}, kT0).allowed);
  EXPECT_FALSE(rl.check_request("t", {10, 0}, kT0).allowed);
  // Raising the limit does not instantly refill the bucket...
  EXPECT_FALSE(rl.check_request("t", {600, 0}, kT0).allowed);
  // ...but the new, faster refill rate (10/s) applies immediately.
  EXPECT_TRUE(rl.check_request("t", {600, 0}, kT0 + 100ms).allowed);
}

TEST(RateLimiter, ExactUnderConcurrency) {
  RateLimiter rl(8);
  const TenantLimits limits{1000, 0};
  std::atomic<int> allowed{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 500; ++i) {
        if (rl.check_request("hot-tenant", limits, kT0).allowed) allowed.fetch_add(1);
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(allowed.load(), 1000);  // 4000 attempts at a frozen clock -> exactly the burst
}
