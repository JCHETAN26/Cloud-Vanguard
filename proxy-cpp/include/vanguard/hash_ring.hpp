#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vanguard {

// Consistent hashing ring with virtual nodes. Each physical node is placed at
// `vnodes` points on a 64-bit ring (MurmurHash3 of "<node>#<i>"); a key maps
// to the first point clockwise from MurmurHash3(key). Adding or removing a
// node only moves the keys owned by that node's points.
//
// Thread-safe: lookups take a shared lock, membership changes an exclusive one.
class HashRing {
 public:
  // 1000 points per node keeps per-node load within ~3% CV (1/sqrt(V));
  // 150 points gives ~9%. Ring rebuilds stay ~ms-scale at 50 nodes.
  static constexpr std::size_t kDefaultVnodes = 1000;

  explicit HashRing(std::size_t vnodes = kDefaultVnodes);

  // Returns false if the node was already present (no-op).
  bool add_node(const std::string& node);
  // Returns false if the node was not present.
  bool remove_node(const std::string& node);

  [[nodiscard]] std::optional<std::string> node_for(std::string_view key) const;

  [[nodiscard]] std::vector<std::string> nodes() const;
  [[nodiscard]] std::size_t node_count() const;
  [[nodiscard]] std::size_t point_count() const;
  [[nodiscard]] std::size_t vnodes_per_node() const noexcept { return vnodes_; }

 private:
  void rebuild_locked();

  const std::size_t vnodes_;
  mutable std::shared_mutex mu_;
  std::vector<std::string> nodes_;                          // sorted, unique
  std::vector<std::pair<std::uint64_t, std::uint32_t>> ring_;  // (point hash, node index)
};

}  // namespace vanguard
