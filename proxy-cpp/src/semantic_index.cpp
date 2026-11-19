#include "vanguard/semantic_index.hpp"

#include <algorithm>

#include "vanguard/embedding.hpp"
#include "vanguard/murmur3.hpp"

namespace vanguard {

SemanticIndex::SemanticIndex(Options options) : opt_(options) {
  opt_.shards = std::max<std::size_t>(opt_.shards, 1);
  opt_.max_per_scope = std::max<std::size_t>(opt_.max_per_scope, 1);
  per_shard_cap_ = std::max<std::size_t>(opt_.max_entries / opt_.shards, 1);
  shards_.reserve(opt_.shards);
  for (std::size_t i = 0; i < opt_.shards; ++i) shards_.push_back(std::make_unique<Shard>());
}

SemanticIndex::Shard& SemanticIndex::shard_for(std::string_view scope) {
  return *shards_[murmur3_64(scope) % shards_.size()];
}

std::optional<SemanticMatch> SemanticIndex::lookup(std::string_view scope, std::uint64_t guard,
                                                   const std::vector<float>& query, float threshold) {
  if (query.empty()) return std::nullopt;
  auto& shard = shard_for(scope);
  std::lock_guard lock(shard.mu);
  const auto it = shard.index.find(scope);
  if (it == shard.index.end()) return std::nullopt;
  shard.lru.splice(shard.lru.begin(), shard.lru, it->second);

  const Entry* best = nullptr;
  float best_sim = -2.0f;
  for (const auto& e : it->second->entries) {
    if (e.guard != guard || e.vec.size() != query.size()) continue;
    const float sim = dot(e.vec.data(), query.data(), query.size());
    if (sim > best_sim) {
      best_sim = sim;
      best = &e;
    }
  }
  if (best == nullptr || best_sim < threshold) return std::nullopt;
  return SemanticMatch{best->cache_key, best_sim};
}

void SemanticIndex::insert(std::string_view scope, std::uint64_t guard, std::vector<float> vec,
                           std::string cache_key) {
  if (vec.empty()) return;
  auto& shard = shard_for(scope);
  std::lock_guard lock(shard.mu);
  auto it = shard.index.find(scope);
  if (it == shard.index.end()) {
    shard.lru.push_front(Scope{std::string(scope), {}, 0});
    it = shard.index.emplace(std::string_view(shard.lru.front().key), shard.lru.begin()).first;
  } else {
    shard.lru.splice(shard.lru.begin(), shard.lru, it->second);
  }
  Scope& s = *it->second;

  Entry entry{std::move(cache_key), guard, std::move(vec)};
  const auto same = std::find_if(s.entries.begin(), s.entries.end(),
                                 [&](const Entry& e) { return e.cache_key == entry.cache_key; });
  if (same != s.entries.end()) {
    *same = std::move(entry);
  } else if (s.entries.size() < opt_.max_per_scope) {
    s.entries.push_back(std::move(entry));
    ++shard.entries;
  } else {
    s.entries[s.next] = std::move(entry);  // overwrite the oldest
    s.next = (s.next + 1) % s.entries.size();
  }

  // Evict whole least-recently-used scopes (never the one just written).
  while (shard.entries > per_shard_cap_ && shard.lru.size() > 1) {
    Scope& victim = shard.lru.back();
    shard.entries -= victim.entries.size();
    shard.index.erase(std::string_view(victim.key));
    shard.lru.pop_back();
  }
}

bool SemanticIndex::erase(std::string_view scope, std::string_view cache_key) {
  auto& shard = shard_for(scope);
  std::lock_guard lock(shard.mu);
  const auto it = shard.index.find(scope);
  if (it == shard.index.end()) return false;
  auto& entries = it->second->entries;
  const auto e = std::find_if(entries.begin(), entries.end(),
                              [&](const Entry& x) { return x.cache_key == cache_key; });
  if (e == entries.end()) return false;
  entries.erase(e);
  --shard.entries;
  it->second->next = 0;
  if (entries.empty()) {
    const auto list_it = it->second;
    shard.index.erase(it);
    shard.lru.erase(list_it);
  }
  return true;
}

std::size_t SemanticIndex::size() const {
  std::size_t n = 0;
  for (const auto& s : shards_) {
    std::lock_guard lock(s->mu);
    n += s->entries;
  }
  return n;
}

std::size_t SemanticIndex::scope_count() const {
  std::size_t n = 0;
  for (const auto& s : shards_) {
    std::lock_guard lock(s->mu);
    n += s->index.size();
  }
  return n;
}

}  // namespace vanguard
