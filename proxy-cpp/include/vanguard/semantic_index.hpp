#pragma once

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

namespace vanguard {

struct SemanticMatch {
  std::string cache_key;  // exact-cache key of the matched entry
  float similarity = 0;
};

// Vector index over cached prompts, partitioned into "scopes". A scope is a
// hash of everything that must match exactly for two requests to share an
// answer (tenant, model, stream flag, all messages except the final user
// message, generation params). Within a scope, lookup is an exact brute-force
// cosine scan over unit vectors. Scopes are small (one per system prompt /
// conversation prefix), so a scan is a few microseconds to well under a
// millisecond at the per-scope cap, and results are exact (no ANN recall loss).
//
// Memory is bounded two ways: each scope keeps at most `max_per_scope`
// entries (oldest overwritten first), and each shard evicts its
// least-recently-used scopes once it holds more than max_entries / shards.
// Entries point at exact-cache keys; if that entry has since been evicted from
// the response cache, the caller erases the stale vector.
class SemanticIndex {
 public:
  struct Options {
    std::size_t shards = 16;
    std::size_t max_entries = 100000;
    std::size_t max_per_scope = 2048;
  };

  SemanticIndex() : SemanticIndex(Options{}) {}
  explicit SemanticIndex(Options options);

  // Best entry in `scope` with an equal guard and similarity >= threshold.
  std::optional<SemanticMatch> lookup(std::string_view scope, std::uint64_t guard,
                                      const std::vector<float>& query, float threshold);
  void insert(std::string_view scope, std::uint64_t guard, std::vector<float> vec, std::string cache_key);
  bool erase(std::string_view scope, std::string_view cache_key);

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t scope_count() const;

 private:
  struct Entry {
    std::string cache_key;
    std::uint64_t guard = 0;
    std::vector<float> vec;
  };
  struct Scope {
    std::string key;
    std::vector<Entry> entries;
    std::size_t next = 0;  // ring position once the scope is full
  };
  struct Shard {
    mutable std::mutex mu;
    std::list<Scope> lru;  // front = most recently used
    std::unordered_map<std::string_view, std::list<Scope>::iterator> index;
    std::size_t entries = 0;
  };

  Shard& shard_for(std::string_view scope);

  Options opt_;
  std::size_t per_shard_cap_;
  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace vanguard
