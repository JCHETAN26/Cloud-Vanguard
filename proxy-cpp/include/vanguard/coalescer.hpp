#pragma once

// Request coalescing ("single flight") for cache misses.
//
// When many clients ask for the same uncached completion at once, only the
// first one (the leader) calls the upstream; the others wait for it and are
// then served from the cache the leader filled. The table is shared by every
// worker thread of a node: identical requests land on different workers
// (SO_REUSEPORT), so a per-worker table would still send duplicates upstream.
// Across nodes, the hash ring already routes a key to one owner.
//
// This class is pure bookkeeping: who leads a key and who waits on it. Waking
// a waiter on its own event loop is the server's job (see CoalesceMailbox).
//
// Waiters receive the leader's response as-is, errors included: an upstream
// failure seen by the leader is not retried once per waiter (that would turn
// one failure into a stampede). Only a leader that ends without a response
// (its client disconnected) hands the call over: its waiters join again and
// one of them leads.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vanguard {

struct CoalesceParticipant {
  int worker = -1;              // worker (event loop) that owns the client connection
  std::uint64_t client_id = 0;  // connection id, unique within that worker
  bool operator==(const CoalesceParticipant& o) const noexcept {
    return worker == o.worker && client_id == o.client_id;
  }
};

class RequestCoalescer {
 public:
  enum class Role {
    kLeader,  // no call in flight for this key: the caller makes it
    kWaiter,  // a call is in flight: the caller waits to be woken
    kSolo,    // too many waiters already: the caller goes upstream on its own
  };

  explicit RequestCoalescer(std::size_t max_waiters_per_key = 1024) : max_waiters_(max_waiters_per_key) {}

  Role join(const std::string& key, CoalesceParticipant self) {
    std::lock_guard lock(mu_);
    auto [it, inserted] = flights_.try_emplace(key);
    if (inserted) {
      it->second.leader = self;
      return Role::kLeader;
    }
    if (it->second.waiters.size() >= max_waiters_) return Role::kSolo;
    it->second.waiters.push_back(self);
    return Role::kWaiter;
  }

  // The leader's call is over (success, failure or abort). Removes the entry
  // and returns the waiters to wake; they then look in the cache. A release
  // by anyone other than the current leader is ignored and returns nothing.
  std::vector<CoalesceParticipant> release(const std::string& key, CoalesceParticipant leader) {
    std::lock_guard lock(mu_);
    const auto it = flights_.find(key);
    if (it == flights_.end() || !(it->second.leader == leader)) return {};
    auto waiters = std::move(it->second.waiters);
    flights_.erase(it);
    return waiters;
  }

  [[nodiscard]] std::size_t in_flight() const {
    std::lock_guard lock(mu_);
    return flights_.size();
  }

 private:
  struct Flight {
    CoalesceParticipant leader;
    std::vector<CoalesceParticipant> waiters;
  };
  const std::size_t max_waiters_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, Flight> flights_;
};

// The leader's response, shared read-only with every waiter.
struct CoalesceOutcome {
  int status = 0;
  std::string content_type;
  std::string body;
  std::string headers;  // pass-through header lines ("Name: value\r\n"), e.g. Retry-After
};

struct CoalesceWakeup {
  std::uint64_t client_id = 0;
  std::string key;
  std::shared_ptr<const CoalesceOutcome> outcome;  // null: the leader ended without a response
};

// Per-worker inbox of woken waiters. Producers push under the mutex and then
// signal the worker's wake fd (an eventfd on Linux).
struct CoalesceMailbox {
  std::mutex mu;
  std::vector<CoalesceWakeup> items;
  int wake_fd = -1;

  void push(CoalesceWakeup w) {
    std::lock_guard lock(mu);
    items.push_back(std::move(w));
  }
  std::vector<CoalesceWakeup> drain() {
    std::lock_guard lock(mu);
    return std::exchange(items, {});
  }
};

}  // namespace vanguard
