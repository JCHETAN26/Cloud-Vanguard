#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "vanguard/lru_cache.hpp"

using namespace std::chrono_literals;
using vanguard::LruShard;
using vanguard::ShardedLruCache;
using vanguard::SteadyClock;

namespace {

std::shared_ptr<const std::string> val(std::string s) { return std::make_shared<const std::string>(std::move(s)); }
const auto kNever = SteadyClock::time_point::max();
const auto kT0 = SteadyClock::time_point{} + 1h;

}  // namespace

TEST(LruShard, EvictsLeastRecentlyUsedByBytes) {
  LruShard<std::string> shard(30);
  ASSERT_TRUE(shard.put("a", val("A"), 10, kNever));
  ASSERT_TRUE(shard.put("b", val("B"), 10, kNever));
  ASSERT_TRUE(shard.put("c", val("C"), 10, kNever));
  // Inserting d (10 bytes) into a full 30-byte shard evicts the LRU entry: a.
  ASSERT_TRUE(shard.put("d", val("D"), 10, kNever));
  EXPECT_EQ(shard.get("a", kT0), nullptr);
  EXPECT_NE(shard.get("b", kT0), nullptr);
  EXPECT_EQ(shard.stats().evictions, 1u);
}

TEST(LruShard, GetPromotesToMostRecentlyUsed) {
  LruShard<std::string> shard(30);
  shard.put("a", val("A"), 10, kNever);
  shard.put("b", val("B"), 10, kNever);
  shard.put("c", val("C"), 10, kNever);
  ASSERT_NE(shard.get("a", kT0), nullptr);  // a becomes MRU; b is now LRU
  shard.put("d", val("D"), 10, kNever);
  EXPECT_NE(shard.get("a", kT0), nullptr);
  EXPECT_EQ(shard.get("b", kT0), nullptr);
  EXPECT_EQ(shard.keys_mru_order(), (std::vector<std::string>{"a", "d", "c"}));
}

TEST(LruShard, LargeEntryEvictsSeveral) {
  LruShard<std::string> shard(100);
  for (int i = 0; i < 10; ++i) shard.put("k" + std::to_string(i), val("x"), 10, kNever);
  ASSERT_TRUE(shard.put("big", val("BIG"), 35, kNever));
  const auto s = shard.stats();
  EXPECT_EQ(s.evictions, 4u);  // 100 - 35 = 65 -> keep 6 of the 10-byte entries
  EXPECT_EQ(s.entries, 7u);
  EXPECT_EQ(s.bytes, 95u);
  EXPECT_EQ(shard.get("k0", kT0), nullptr);
  EXPECT_NE(shard.get("k4", kT0), nullptr);
}

TEST(LruShard, OversizedEntryIsRejectedWithoutEvicting) {
  LruShard<std::string> shard(50);
  shard.put("a", val("A"), 10, kNever);
  EXPECT_FALSE(shard.put("huge", val("H"), 51, kNever));
  EXPECT_NE(shard.get("a", kT0), nullptr);
  EXPECT_EQ(shard.stats().rejections, 1u);
}

TEST(LruShard, ReplacingKeyUpdatesChargeAndValue) {
  LruShard<std::string> shard(100);
  shard.put("a", val("v1"), 40, kNever);
  shard.put("a", val("v2"), 20, kNever);
  EXPECT_EQ(*shard.get("a", kT0), "v2");
  EXPECT_EQ(shard.stats().bytes, 20u);
  EXPECT_EQ(shard.stats().entries, 1u);
}

TEST(LruShard, ExpiredEntriesAreMisses) {
  LruShard<std::string> shard(100);
  shard.put("a", val("A"), 10, kT0 + 10s);
  EXPECT_NE(shard.get("a", kT0 + 9s), nullptr);
  EXPECT_EQ(shard.get("a", kT0 + 10s), nullptr);
  const auto s = shard.stats();
  EXPECT_EQ(s.expirations, 1u);
  EXPECT_EQ(s.entries, 0u);
  EXPECT_EQ(s.bytes, 0u);
}

TEST(LruShard, EvictedValueStaysAliveForHolders) {
  LruShard<std::string> shard(10);
  shard.put("a", val("payload"), 10, kNever);
  auto held = shard.get("a", kT0);
  shard.put("b", val("B"), 10, kNever);  // evicts a
  EXPECT_EQ(shard.get("a", kT0), nullptr);
  EXPECT_EQ(*held, "payload");  // a response mid-write is unaffected
}

TEST(ShardedLruCache, SplitsCapacityAndAggregatesStats) {
  ShardedLruCache<std::string> cache(16 * 1000, 16, 0s);
  EXPECT_EQ(cache.shard_count(), 16u);
  for (int i = 0; i < 100; ++i) cache.put("key-" + std::to_string(i), val("v"), 10);
  for (int i = 0; i < 100; ++i) EXPECT_NE(cache.get("key-" + std::to_string(i)), nullptr);
  EXPECT_EQ(cache.get("absent"), nullptr);
  const auto s = cache.stats();
  EXPECT_EQ(s.entries, 100u);
  EXPECT_EQ(s.hits, 100u);
  EXPECT_EQ(s.misses, 1u);
  EXPECT_EQ(s.capacity_bytes, 16u * 1000u);
}

TEST(ShardedLruCache, TtlApplies) {
  ShardedLruCache<std::string> cache(1000, 4, 5s);
  cache.put("a", val("A"), 10, kT0);
  EXPECT_NE(cache.get("a", kT0 + 4s), nullptr);
  EXPECT_EQ(cache.get("a", kT0 + 6s), nullptr);
}

TEST(ShardedLruCache, ConcurrentMixedWorkloadStaysWithinCapacity) {
  ShardedLruCache<std::string> cache(64 * 1024, 8, 0s);
  std::atomic<std::uint64_t> hits{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 20000; ++i) {
        const std::string key = "k" + std::to_string((i * 7 + t) % 3000);
        if (auto v = cache.get(key)) {
          hits.fetch_add(1);
          ASSERT_EQ(*v, key);
        } else {
          cache.put(key, val(key), 64);
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  const auto s = cache.stats();
  EXPECT_LE(s.bytes, s.capacity_bytes);
  EXPECT_GT(hits.load(), 0u);
  EXPECT_GT(s.evictions, 0u);
}
