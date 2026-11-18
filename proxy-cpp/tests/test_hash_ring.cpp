#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "vanguard/hash_ring.hpp"

using vanguard::HashRing;

namespace {

std::vector<std::string> make_keys(std::size_t n) {
  std::vector<std::string> keys;
  keys.reserve(n);
  for (std::size_t i = 0; i < n; ++i) keys.push_back("tenant-" + std::to_string(i % 97) + ":prompt-" + std::to_string(i));
  return keys;
}

// HashRing owns a mutex and is deliberately non-movable.
std::unique_ptr<HashRing> make_ring(std::size_t nodes, std::size_t vnodes) {
  auto ring = std::make_unique<HashRing>(vnodes);
  for (std::size_t i = 0; i < nodes; ++i) ring->add_node("node-" + std::to_string(i));
  return ring;
}

struct LoadStats {
  double cv;            // stddev / mean of per-node key counts
  double max_rel_dev;   // max |count - mean| / mean
};

LoadStats load_stats(const HashRing& ring, const std::vector<std::string>& keys) {
  std::map<std::string, std::size_t> counts;
  for (const auto& n : ring.nodes()) counts[n] = 0;
  for (const auto& k : keys) ++counts[*ring.node_for(k)];
  const double mean = static_cast<double>(keys.size()) / static_cast<double>(counts.size());
  double var = 0;
  double max_dev = 0;
  for (const auto& [node, c] : counts) {
    const double d = static_cast<double>(c) - mean;
    var += d * d;
    max_dev = std::max(max_dev, std::abs(d) / mean);
  }
  var /= static_cast<double>(counts.size());
  return {std::sqrt(var) / mean, max_dev};
}

}  // namespace

TEST(HashRing, EmptyRingHasNoOwner) {
  HashRing ring;
  EXPECT_FALSE(ring.node_for("anything").has_value());
}

TEST(HashRing, SingleNodeOwnsEverything) {
  HashRing ring;
  ring.add_node("only");
  for (const auto& k : make_keys(1000)) EXPECT_EQ(*ring.node_for(k), "only");
  EXPECT_EQ(ring.point_count(), HashRing::kDefaultVnodes);
}

TEST(HashRing, AddRemoveAreIdempotent) {
  HashRing ring(10);
  EXPECT_TRUE(ring.add_node("a"));
  EXPECT_FALSE(ring.add_node("a"));
  EXPECT_EQ(ring.point_count(), 10u);
  EXPECT_TRUE(ring.remove_node("a"));
  EXPECT_FALSE(ring.remove_node("a"));
  EXPECT_EQ(ring.point_count(), 0u);
}

TEST(HashRing, OwnershipIsIndependentOfInsertionOrder) {
  HashRing a(150), b(150);
  for (const char* n : {"n1", "n2", "n3", "n4"}) a.add_node(n);
  for (const char* n : {"n4", "n2", "n1", "n3"}) b.add_node(n);
  for (const auto& k : make_keys(5000)) EXPECT_EQ(*a.node_for(k), *b.node_for(k));
}

// Load balance with the default 150 vnodes. Theory: each node's share has a
// coefficient of variation of roughly 1/sqrt(vnodes) (~8% for 150).
TEST(HashRing, DefaultVnodesSpreadLoadEvenly) {
  const auto keys = make_keys(200000);
  const auto ring = make_ring(10, 150);
  const auto s = load_stats(*ring, keys);
  std::printf("[ring] 10 nodes x 150 vnodes, 200k keys: CV=%.2f%% max deviation=%.2f%%\n", s.cv * 100,
              s.max_rel_dev * 100);
  EXPECT_LT(s.cv, 0.10);
  EXPECT_LT(s.max_rel_dev, 0.20);
}

TEST(HashRing, MoreVnodesTightenBalance) {
  const auto keys = make_keys(200000);
  const auto s = load_stats(*make_ring(10, 1000), keys);
  std::printf("[ring] 10 nodes x 1000 vnodes, 200k keys: CV=%.2f%% max deviation=%.2f%%\n", s.cv * 100,
              s.max_rel_dev * 100);
  EXPECT_LT(s.cv, 0.05);
}

// The core consistent-hashing property: adding a node moves ~1/(N+1) of the
// keys, all of them *to the new node*, and none between existing nodes.
TEST(HashRing, AddingNodeRedistributesMinimallyAndUniformly) {
  const auto keys = make_keys(200000);
  auto ring_ptr = make_ring(10, 150);
  auto& ring = *ring_ptr;
  std::vector<std::string> before;
  before.reserve(keys.size());
  for (const auto& k : keys) before.push_back(*ring.node_for(k));

  ring.add_node("node-new");
  std::size_t moved = 0;
  std::map<std::string, std::size_t> moved_from;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto now = *ring.node_for(keys[i]);
    if (now != before[i]) {
      ASSERT_EQ(now, "node-new") << "key moved between two pre-existing nodes";
      ++moved;
      ++moved_from[before[i]];
    }
  }
  const double fraction = static_cast<double>(moved) / static_cast<double>(keys.size());
  const double ideal = 1.0 / 11.0;
  std::printf("[ring] add 11th node: moved %.2f%% of keys (ideal %.2f%%, error %.2f%% of ideal)\n",
              fraction * 100, ideal * 100, std::abs(fraction - ideal) / ideal * 100);
  // Absolute redistribution error must be < 5 percentage points of the keyspace.
  EXPECT_LT(std::abs(fraction - ideal), 0.05);
  // ...and the new node should take keys from every existing node, not just one.
  EXPECT_EQ(moved_from.size(), 10u);
}

TEST(HashRing, RemovingNodeOnlyMovesItsKeys) {
  const auto keys = make_keys(100000);
  auto ring_ptr = make_ring(8, 150);
  auto& ring = *ring_ptr;
  std::vector<std::string> before;
  for (const auto& k : keys) before.push_back(*ring.node_for(k));
  ring.remove_node("node-3");
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto now = *ring.node_for(keys[i]);
    if (before[i] != "node-3") {
      EXPECT_EQ(now, before[i]);
    } else {
      EXPECT_NE(now, "node-3");
    }
  }
}

TEST(HashRing, ConcurrentLookupsDuringMembershipChanges) {
  HashRing ring(50);
  ring.add_node("base");
  std::atomic<bool> stop{false};
  std::atomic<std::size_t> lookups{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&, t] {
      std::size_t i = 0;
      while (!stop.load()) {
        auto owner = ring.node_for("k" + std::to_string(t) + "-" + std::to_string(i++));
        ASSERT_TRUE(owner.has_value());
        lookups.fetch_add(1);
      }
    });
  }
  for (int i = 0; i < 200; ++i) {
    ring.add_node("n" + std::to_string(i % 7));
    ring.remove_node("n" + std::to_string((i + 3) % 7));
  }
  stop = true;
  for (auto& r : readers) r.join();
  EXPECT_GT(lookups.load(), 0u);
}
