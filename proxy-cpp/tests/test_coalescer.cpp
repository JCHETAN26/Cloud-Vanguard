#include "vanguard/coalescer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using vanguard::CoalesceMailbox;
using vanguard::CoalesceParticipant;
using vanguard::RequestCoalescer;
using Role = RequestCoalescer::Role;

TEST(Coalescer, FirstLeadsOthersWait) {
  RequestCoalescer co;
  EXPECT_EQ(co.join("k", {0, 1}), Role::kLeader);
  EXPECT_EQ(co.join("k", {1, 7}), Role::kWaiter);
  EXPECT_EQ(co.join("k", {0, 2}), Role::kWaiter);
  EXPECT_EQ(co.in_flight(), 1u);
  const auto woken = co.release("k", {0, 1});
  ASSERT_EQ(woken.size(), 2u);
  EXPECT_EQ(woken[0], (CoalesceParticipant{1, 7}));
  EXPECT_EQ(woken[1], (CoalesceParticipant{0, 2}));
  EXPECT_EQ(co.in_flight(), 0u);
}

TEST(Coalescer, KeysAreIndependent) {
  RequestCoalescer co;
  EXPECT_EQ(co.join("a", {0, 1}), Role::kLeader);
  EXPECT_EQ(co.join("b", {0, 2}), Role::kLeader);
  EXPECT_EQ(co.join("a", {0, 3}), Role::kWaiter);
  EXPECT_TRUE(co.release("b", {0, 2}).empty());
  EXPECT_EQ(co.release("a", {0, 1}).size(), 1u);
}

TEST(Coalescer, AfterReleaseTheNextCallerLeads) {
  RequestCoalescer co;
  ASSERT_EQ(co.join("k", {0, 1}), Role::kLeader);
  co.release("k", {0, 1});
  EXPECT_EQ(co.join("k", {2, 5}), Role::kLeader);
}

TEST(Coalescer, OnlyTheLeaderCanRelease) {
  RequestCoalescer co;
  ASSERT_EQ(co.join("k", {0, 1}), Role::kLeader);
  ASSERT_EQ(co.join("k", {0, 2}), Role::kWaiter);
  EXPECT_TRUE(co.release("k", {0, 2}).empty());  // a waiter cannot end the flight
  EXPECT_TRUE(co.release("k", {1, 1}).empty());  // same client id, other worker
  EXPECT_EQ(co.in_flight(), 1u);
  EXPECT_EQ(co.release("k", {0, 1}).size(), 1u);
  EXPECT_TRUE(co.release("k", {0, 1}).empty());  // double release is harmless
}

TEST(Coalescer, WaiterCapSendsTheRestSolo) {
  RequestCoalescer co(2);
  ASSERT_EQ(co.join("k", {0, 1}), Role::kLeader);
  EXPECT_EQ(co.join("k", {0, 2}), Role::kWaiter);
  EXPECT_EQ(co.join("k", {0, 3}), Role::kWaiter);
  EXPECT_EQ(co.join("k", {0, 4}), Role::kSolo);
  EXPECT_EQ(co.release("k", {0, 1}).size(), 2u);
}

TEST(Coalescer, ConcurrentJoinsElectExactlyOneLeader) {
  RequestCoalescer co;
  constexpr int kThreads = 8, kPerThread = 200;
  std::atomic<int> leaders{0}, waiters{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i) {
        const auto role = co.join("hot", {t, static_cast<std::uint64_t>(i + 1)});
        (role == Role::kLeader ? leaders : waiters).fetch_add(1);
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(leaders.load(), 1);
  EXPECT_EQ(waiters.load(), kThreads * kPerThread - 1);
}

TEST(CoalesceMailbox, DrainEmptiesInOrder) {
  CoalesceMailbox box;
  auto outcome = std::make_shared<const vanguard::CoalesceOutcome>(vanguard::CoalesceOutcome{200, "application/json", "{}", ""});
  box.push({3, "k1", outcome});
  box.push({4, "k2", nullptr});
  const auto items = box.drain();
  ASSERT_EQ(items.size(), 2u);
  EXPECT_EQ(items[0].client_id, 3u);
  EXPECT_EQ(items[0].outcome->status, 200);
  EXPECT_EQ(items[1].key, "k2");
  EXPECT_EQ(items[1].outcome, nullptr);
  EXPECT_TRUE(box.drain().empty());
}
