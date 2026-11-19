#pragma once

// Control plane for vanguard-proxy: dynamic ring topology and tenant policies.
//
// Wire format is the proto3 JSON mapping of proto/vanguard/control/v1/control.proto.
// Updates arrive from the admin HTTP listener (operator pushes) or from
// CONTROL_FILE (an operator-managed ConfigMap, polled). Both go through
// ControlPlane::apply_*, which validates, enforces version monotonicity and
// publishes immutable snapshots. Worker threads pick snapshots up lock-free
// (one relaxed atomic load per request batch) via Published<T>.

#include <sys/socket.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vanguard/circuit_breaker.hpp"
#include "vanguard/hash_ring.hpp"
#include "vanguard/tenants.hpp"

namespace vanguard {

// ---------------------------------------------------------------------------
// Messages (mirror control.proto)
// ---------------------------------------------------------------------------

struct PeerNode {
  std::string id;
  std::string address;  // host:port
};

struct RingUpdate {
  std::int64_t version = 0;
  std::uint32_t vnodes = 0;  // 0 = default (1000)
  std::vector<PeerNode> nodes;
};

struct TenantPolicyUpdate {
  std::string tenant_id;
  std::vector<std::string> api_key_sha256;
  double rpm_limit = 0;
  double tpm_limit = 0;
  std::optional<std::string> semantic_mode;
  std::optional<double> semantic_threshold;
  std::optional<unsigned> ttl_seconds;
  std::int64_t source_generation = 0;
};

struct TenantPolicySetUpdate {
  std::int64_t version = 0;
  std::vector<TenantPolicyUpdate> tenants;
};

struct ControlDocumentUpdate {
  std::int64_t version = 0;
  std::optional<RingUpdate> ring;
  std::optional<TenantPolicySetUpdate> tenants;
};

std::optional<RingUpdate> parse_ring_update(std::string_view json, std::string& error);
std::optional<TenantPolicySetUpdate> parse_tenant_set_update(std::string_view json, std::string& error);
std::optional<ControlDocumentUpdate> parse_control_document(std::string_view json, std::string& error);

// "id=host:port,id2=host:port" (the static PEERS setting).
std::optional<std::vector<PeerNode>> parse_peer_list(std::string_view spec, std::string& error);

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

// A shared, immutable value that can be replaced at runtime. Readers cache
// the pointer and only re-load it when generation() changes.
template <typename T>
class Published {
 public:
  explicit Published(std::shared_ptr<const T> v) : value_(std::move(v)) {}
  [[nodiscard]] std::shared_ptr<const T> load() const {
    std::lock_guard lock(mu_);
    return value_;
  }
  void store(std::shared_ptr<const T> v) {
    {
      std::lock_guard lock(mu_);
      value_ = std::move(v);
    }
    generation_.fetch_add(1, std::memory_order_release);
  }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_.load(std::memory_order_acquire); }

 private:
  mutable std::mutex mu_;
  std::shared_ptr<const T> value_;
  std::atomic<std::uint64_t> generation_{1};
};

struct PeerInfo {
  std::string id;
  std::string address;
  std::string host;
  std::uint16_t port = 0;
  sockaddr_storage addr{};
  socklen_t addr_len = 0;
  bool resolved = false;
  std::shared_ptr<CircuitBreaker> breaker;  // survives topology swaps
};

struct Topology {
  std::int64_t version = 0;
  std::string self_id;
  std::size_t vnodes = HashRing::kDefaultVnodes;
  std::shared_ptr<const HashRing> ring;
  std::vector<PeerInfo> peers;  // every ring member (including self)

  [[nodiscard]] const PeerInfo* find(std::string_view id) const;
  [[nodiscard]] std::size_t members() const noexcept { return peers.empty() ? 1 : peers.size(); }
  // Owner of `key`, or self when the ring is empty.
  [[nodiscard]] std::string owner(std::string_view key) const;
};

// ---------------------------------------------------------------------------
// Forwarding policy (pure; unit-tested)
// ---------------------------------------------------------------------------

inline constexpr int kMaxHops = 1;

enum class RouteAction {
  kServeLocal,          // we own the key
  kForward,             // send to the owner, X-Vanguard-Hop: 1
  kServeLocalDivergent, // forwarded to us but our ring names another owner: never re-forward
  kServeLocalFallback,  // owner unknown/unresolved/circuit open: degrade to local
  kRejectLoop,          // hop count above kMaxHops: 508 Loop Detected
};

// `hop` is the X-Vanguard-Hop value (0 when absent).
RouteAction decide_route(int hop, bool owner_is_self, bool owner_reachable) noexcept;

// Parses X-Vanguard-Hop; empty header = 0. Returns false if malformed.
bool parse_hop_header(std::string_view header, int& hop) noexcept;

// ---------------------------------------------------------------------------
// ControlPlane
// ---------------------------------------------------------------------------

// Resolves host:port into a socket address. The fast resolver must never
// block (numeric addresses only); the slow one may (getaddrinfo) and is only
// called from the control thread.
using Resolver = std::function<bool(const std::string& host, std::uint16_t port, sockaddr_storage& out,
                                    socklen_t& len)>;
bool resolve_numeric(const std::string& host, std::uint16_t port, sockaddr_storage& out, socklen_t& len);
bool resolve_dns(const std::string& host, std::uint16_t port, sockaddr_storage& out, socklen_t& len);

struct ApplyOutcome {
  int http_status = 200;  // 200 applied/no-op, 400 invalid, 409 stale
  bool applied = false;
  std::int64_t version = 0;
  std::string message;
};

class ControlPlane {
 public:
  struct Options {
    std::string self_id = "local";
    std::size_t default_vnodes = HashRing::kDefaultVnodes;
    std::vector<PeerNode> static_peers;  // initial ring (PEERS / RING_NODES)
    CircuitBreaker::Options peer_breaker{};
    Resolver fast_resolver = resolve_numeric;
    Resolver slow_resolver = resolve_dns;
  };

  ControlPlane(Options options, TenantRegistry base_tenants);

  ApplyOutcome apply_ring(const RingUpdate& update);
  ApplyOutcome apply_tenants(const TenantPolicySetUpdate& update);
  ApplyOutcome apply_document(const ControlDocumentUpdate& update);

  [[nodiscard]] const Published<Topology>& topology() const noexcept { return topology_; }
  [[nodiscard]] const Published<TenantRegistry>& tenants() const noexcept { return tenants_; }
  [[nodiscard]] std::int64_t ring_version() const;
  [[nodiscard]] std::int64_t tenants_version() const;
  [[nodiscard]] const std::string& self_id() const noexcept { return opt_.self_id; }

  // Resolve peers the fast resolver could not (hostnames), or every peer when
  // `refresh_all` (pod IPs change on restart). Publishes a new topology if any
  // address changed. Blocking: call from the control thread only.
  bool resolve_peers(bool refresh_all);

  // Current state as a ControlDocument JSON (GET /admin/v1/config).
  [[nodiscard]] std::string to_json() const;

 private:
  std::shared_ptr<Topology> build_topology_locked(std::int64_t version, std::size_t vnodes,
                                                  const std::vector<PeerNode>& nodes);
  ApplyOutcome apply_ring_locked(const RingUpdate& update);
  ApplyOutcome apply_tenants_locked(const TenantPolicySetUpdate& update);

  Options opt_;
  const TenantRegistry base_tenants_;
  mutable std::mutex mu_;  // serializes writers
  std::int64_t ring_version_ = 0;
  std::int64_t tenants_version_ = 0;
  RingUpdate ring_spec_;
  TenantPolicySetUpdate tenant_spec_;
  std::unordered_map<std::string, std::shared_ptr<CircuitBreaker>> breakers_;
  Published<Topology> topology_;
  Published<TenantRegistry> tenants_;
};

}  // namespace vanguard
