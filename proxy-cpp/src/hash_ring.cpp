#include "vanguard/hash_ring.hpp"

#include <algorithm>
#include <mutex>

#include "vanguard/murmur3.hpp"

namespace vanguard {

HashRing::HashRing(std::size_t vnodes) : vnodes_(vnodes == 0 ? 1 : vnodes) {}

bool HashRing::add_node(const std::string& node) {
  std::unique_lock lock(mu_);
  const auto it = std::lower_bound(nodes_.begin(), nodes_.end(), node);
  if (it != nodes_.end() && *it == node) return false;
  nodes_.insert(it, node);
  rebuild_locked();
  return true;
}

bool HashRing::remove_node(const std::string& node) {
  std::unique_lock lock(mu_);
  const auto it = std::lower_bound(nodes_.begin(), nodes_.end(), node);
  if (it == nodes_.end() || *it != node) return false;
  nodes_.erase(it);
  rebuild_locked();
  return true;
}

// Membership changes are rare (control-plane driven), so a full rebuild keeps
// the code simple: O(N * V log(N * V)), ~1 ms for 50 nodes x 150 vnodes.
void HashRing::rebuild_locked() {
  ring_.clear();
  ring_.reserve(nodes_.size() * vnodes_);
  std::string label;
  for (std::uint32_t n = 0; n < nodes_.size(); ++n) {
    for (std::size_t v = 0; v < vnodes_; ++v) {
      label.assign(nodes_[n]);
      label.push_back('#');
      label.append(std::to_string(v));
      ring_.emplace_back(murmur3_64(label), n);
    }
  }
  // Ties on the (astronomically unlikely) equal hash are broken by node index,
  // which is itself derived from sorted names, so every replica of the ring
  // agrees on ownership regardless of insertion order.
  std::sort(ring_.begin(), ring_.end());
}

std::optional<std::string> HashRing::node_for(std::string_view key) const {
  const std::uint64_t h = murmur3_64(key);
  std::shared_lock lock(mu_);
  if (ring_.empty()) return std::nullopt;
  auto it = std::lower_bound(ring_.begin(), ring_.end(), h,
                             [](const auto& point, std::uint64_t value) { return point.first < value; });
  if (it == ring_.end()) it = ring_.begin();  // wrap around
  return nodes_[it->second];
}

std::vector<std::string> HashRing::nodes() const {
  std::shared_lock lock(mu_);
  return nodes_;
}

std::size_t HashRing::node_count() const {
  std::shared_lock lock(mu_);
  return nodes_.size();
}

std::size_t HashRing::point_count() const {
  std::shared_lock lock(mu_);
  return ring_.size();
}

}  // namespace vanguard
