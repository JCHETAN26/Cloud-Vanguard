#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vanguard/murmur3.hpp"

namespace vanguard {

using SteadyClock = std::chrono::steady_clock;

struct CacheStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t insertions = 0;
  std::uint64_t evictions = 0;
  std::uint64_t expirations = 0;
  std::uint64_t rejections = 0;  // entry larger than a whole shard
  std::size_t entries = 0;
  std::size_t bytes = 0;
  std::size_t capacity_bytes = 0;

  CacheStats& operator+=(const CacheStats& o) noexcept {
    hits += o.hits;
    misses += o.misses;
    insertions += o.insertions;
    evictions += o.evictions;
    expirations += o.expirations;
    rejections += o.rejections;
    entries += o.entries;
    bytes += o.bytes;
    capacity_bytes += o.capacity_bytes;
    return *this;
  }
};

// One LRU shard: a byte-bounded recency list plus a hash index, guarded by a
// single mutex. Values are immutable and handed out as shared_ptr<const V>, so
// the lock is held only for the index lookup + list splice; callers serialize
// the response (memcpy into the socket buffer) outside the lock, and an
// eviction never invalidates a response that is still being written.
template <typename V>
class LruShard {
 public:
  using ValuePtr = std::shared_ptr<const V>;

  explicit LruShard(std::size_t capacity_bytes) : capacity_(capacity_bytes) {}

  ValuePtr get(std::string_view key, SteadyClock::time_point now) {
    std::lock_guard lock(mu_);
    const auto it = index_.find(key);
    if (it == index_.end()) {
      ++stats_.misses;
      return nullptr;
    }
    const auto node = it->second;
    if (node->expires <= now) {
      index_.erase(it);
      used_ -= node->charge;
      lru_.erase(node);
      ++stats_.expirations;
      ++stats_.misses;
      return nullptr;
    }
    lru_.splice(lru_.begin(), lru_, node);  // mark most-recently-used
    ++stats_.hits;
    return node->value;
  }

  // `charge` is the entry's accounted size in bytes. Returns false (and stores
  // nothing) if the entry can never fit in this shard.
  bool put(std::string key, ValuePtr value, std::size_t charge, SteadyClock::time_point expires) {
    std::lock_guard lock(mu_);
    if (charge > capacity_) {
      ++stats_.rejections;
      return false;
    }
    if (const auto it = index_.find(key); it != index_.end()) {
      const auto node = it->second;
      index_.erase(it);
      used_ -= node->charge;
      lru_.erase(node);
    }
    while (used_ + charge > capacity_ && !lru_.empty()) evict_lru_locked();
    lru_.push_front(Entry{std::move(key), std::move(value), charge, expires});
    // The index key views the string owned by the list node; std::list nodes
    // never move, and the index entry is always erased before its node.
    index_.emplace(std::string_view(lru_.front().key), lru_.begin());
    used_ += charge;
    ++stats_.insertions;
    return true;
  }

  bool erase(std::string_view key) {
    std::lock_guard lock(mu_);
    const auto it = index_.find(key);
    if (it == index_.end()) return false;
    const auto node = it->second;
    index_.erase(it);
    used_ -= node->charge;
    lru_.erase(node);
    return true;
  }

  CacheStats stats() const {
    std::lock_guard lock(mu_);
    CacheStats s = stats_;
    s.entries = index_.size();
    s.bytes = used_;
    s.capacity_bytes = capacity_;
    return s;
  }

  // Keys from most- to least-recently used (for tests / debugging).
  std::vector<std::string> keys_mru_order() const {
    std::lock_guard lock(mu_);
    std::vector<std::string> out;
    out.reserve(lru_.size());
    for (const auto& e : lru_) out.push_back(e.key);
    return out;
  }

 private:
  struct Entry {
    std::string key;
    ValuePtr value;
    std::size_t charge;
    SteadyClock::time_point expires;
  };
  using List = std::list<Entry>;

  void evict_lru_locked() {
    auto& victim = lru_.back();
    index_.erase(std::string_view(victim.key));
    used_ -= victim.charge;
    lru_.pop_back();
    ++stats_.evictions;
  }

  mutable std::mutex mu_;
  const std::size_t capacity_;
  std::size_t used_ = 0;
  List lru_;
  std::unordered_map<std::string_view, typename List::iterator> index_;
  CacheStats stats_;
};

// N independent LRU shards; a key's shard is MurmurHash3(key) mod N. Sharding
// keeps lock hold times short and spreads contention across worker threads.
// Total capacity is split evenly, so LRU order is per shard (approximate
// global LRU), the standard trade-off for concurrent caches.
template <typename V>
class ShardedLruCache {
 public:
  using ValuePtr = std::shared_ptr<const V>;

  ShardedLruCache(std::size_t capacity_bytes, std::size_t num_shards, std::chrono::seconds ttl)
      : ttl_(ttl) {
    if (num_shards == 0) num_shards = 1;
    const std::size_t per_shard = capacity_bytes / num_shards > 0 ? capacity_bytes / num_shards : 1;
    shards_.reserve(num_shards);
    for (std::size_t i = 0; i < num_shards; ++i) {
      shards_.push_back(std::make_unique<LruShard<V>>(per_shard));
    }
  }

  ValuePtr get(std::string_view key, SteadyClock::time_point now = SteadyClock::now()) {
    return shard_for(key).get(key, now);
  }

  // `ttl` overrides the cache-wide TTL for this entry (0 = never expires).
  bool put(std::string key, ValuePtr value, std::size_t charge,
           SteadyClock::time_point now = SteadyClock::now(),
           std::optional<std::chrono::seconds> ttl = std::nullopt) {
    const auto effective = ttl.value_or(ttl_);
    const auto expires = effective.count() > 0 ? now + effective : SteadyClock::time_point::max();
    auto& shard = shard_for(key);
    return shard.put(std::move(key), std::move(value), charge, expires);
  }

  bool erase(std::string_view key) { return shard_for(key).erase(key); }

  [[nodiscard]] std::size_t shard_index(std::string_view key) const noexcept {
    return static_cast<std::size_t>(murmur3_64(key) % shards_.size());
  }
  [[nodiscard]] std::size_t shard_count() const noexcept { return shards_.size(); }
  [[nodiscard]] const LruShard<V>& shard(std::size_t i) const { return *shards_[i]; }

  [[nodiscard]] CacheStats stats() const {
    CacheStats total;
    for (const auto& s : shards_) total += s->stats();
    return total;
  }

 private:
  LruShard<V>& shard_for(std::string_view key) { return *shards_[shard_index(key)]; }

  const std::chrono::seconds ttl_;
  std::vector<std::unique_ptr<LruShard<V>>> shards_;
};

}  // namespace vanguard
