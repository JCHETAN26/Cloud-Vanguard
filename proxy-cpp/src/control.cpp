#include "vanguard/control.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

#include <algorithm>
#include <cstring>
#include <set>

#include <nlohmann/json.hpp>

#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

using nlohmann::json;

constexpr std::size_t kMaxRingNodes = 1024;
constexpr std::uint32_t kMaxVnodes = 100000;
constexpr unsigned kMaxTtlSeconds = 30u * 24 * 3600;

// proto3 JSON: 64-bit integers may be JSON numbers or decimal strings.
bool read_int64(const json& obj, const char* name, std::int64_t& out, std::string& error) {
  const auto it = obj.find(name);
  if (it == obj.end() || it->is_null()) return true;
  if (it->is_number_integer()) {
    out = it->get<std::int64_t>();
    return true;
  }
  if (it->is_string()) {
    const auto& s = it->get_ref<const std::string&>();
    std::uint64_t v = 0;
    const bool neg = !s.empty() && s[0] == '-';
    if (parse_u64(neg ? std::string_view(s).substr(1) : std::string_view(s), v) && v <= INT64_MAX) {
      out = neg ? -static_cast<std::int64_t>(v) : static_cast<std::int64_t>(v);
      return true;
    }
  }
  error = std::string("'") + name + "' must be an integer";
  return false;
}

bool valid_id(std::string_view id) {
  if (id.empty() || id.size() > 253) return false;
  return std::all_of(id.begin(), id.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ||
           c == '.';
  });
}

bool split_host_port(std::string_view address, std::string& host, std::uint16_t& port) {
  const auto colon = address.rfind(':');
  if (colon == std::string_view::npos || colon == 0) return false;
  std::uint64_t p = 0;
  if (!parse_u64(address.substr(colon + 1), p) || p == 0 || p > 65535) return false;
  host = std::string(address.substr(0, colon));
  if (host.size() > 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
  port = static_cast<std::uint16_t>(p);
  return !host.empty();
}

bool is_sha256_hex(std::string_view s) {
  return s.size() == 64 &&
         std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::optional<RingUpdate> ring_from_json(const json& j, std::string& error) {
  if (!j.is_object()) {
    error = "ring must be an object";
    return std::nullopt;
  }
  RingUpdate r;
  if (!read_int64(j, "version", r.version, error)) return std::nullopt;
  std::int64_t vnodes = 0;
  if (!read_int64(j, "vnodes", vnodes, error)) return std::nullopt;
  if (vnodes < 0 || vnodes > kMaxVnodes) {
    error = "vnodes must be in [0, 100000]";
    return std::nullopt;
  }
  r.vnodes = static_cast<std::uint32_t>(vnodes);
  const auto nodes = j.find("nodes");
  if (nodes != j.end() && !nodes->is_null()) {
    if (!nodes->is_array() || nodes->size() > kMaxRingNodes) {
      error = "nodes must be an array of at most 1024 entries";
      return std::nullopt;
    }
    std::set<std::string> seen;
    for (const auto& n : *nodes) {
      const auto id = n.is_object() ? n.find("id") : n.end();
      const auto addr = n.is_object() ? n.find("address") : n.end();
      if (!n.is_object() || id == n.end() || !id->is_string() || addr == n.end() || !addr->is_string()) {
        error = "each node needs string 'id' and 'address'";
        return std::nullopt;
      }
      PeerNode p{id->get<std::string>(), addr->get<std::string>()};
      std::string host;
      std::uint16_t port = 0;
      if (!valid_id(p.id)) {
        error = "invalid node id '" + p.id + "'";
        return std::nullopt;
      }
      if (!split_host_port(p.address, host, port)) {
        error = "node '" + p.id + "' address must be host:port";
        return std::nullopt;
      }
      if (!seen.insert(p.id).second) {
        error = "duplicate node id '" + p.id + "'";
        return std::nullopt;
      }
      r.nodes.push_back(std::move(p));
    }
  }
  if (r.version < 0) {
    error = "version must be >= 0";
    return std::nullopt;
  }
  return r;
}

std::optional<TenantPolicySetUpdate> tenants_from_json(const json& j, std::string& error) {
  if (!j.is_object()) {
    error = "tenants must be an object";
    return std::nullopt;
  }
  TenantPolicySetUpdate set;
  if (!read_int64(j, "version", set.version, error)) return std::nullopt;
  if (set.version < 0) {
    error = "version must be >= 0";
    return std::nullopt;
  }
  const auto list = j.find("tenants");
  if (list == j.end() || list->is_null()) return set;
  if (!list->is_array()) {
    error = "'tenants' must be an array";
    return std::nullopt;
  }
  std::set<std::string> ids;
  std::set<std::string> keys;
  for (const auto& t : *list) {
    if (!t.is_object()) {
      error = "each tenant policy must be an object";
      return std::nullopt;
    }
    TenantPolicyUpdate p;
    const auto id = t.find("tenantId");
    if (id == t.end() || !id->is_string() || !valid_id(id->get_ref<const std::string&>())) {
      error = "tenant policy needs a valid 'tenantId'";
      return std::nullopt;
    }
    p.tenant_id = id->get<std::string>();
    if (!ids.insert(p.tenant_id).second) {
      error = "duplicate tenantId '" + p.tenant_id + "'";
      return std::nullopt;
    }
    if (const auto k = t.find("apiKeySha256"); k != t.end() && !k->is_null()) {
      if (!k->is_array()) {
        error = "apiKeySha256 must be an array";
        return std::nullopt;
      }
      for (const auto& h : *k) {
        if (!h.is_string() || !is_sha256_hex(h.get_ref<const std::string&>())) {
          error = "tenant '" + p.tenant_id + "': apiKeySha256 entries must be 64 lower-case hex chars";
          return std::nullopt;
        }
        if (!keys.insert(h.get<std::string>()).second) {
          error = "API key hash assigned to more than one tenant";
          return std::nullopt;
        }
        p.api_key_sha256.push_back(h.get<std::string>());
      }
    }
    std::int64_t rpm = 0, tpm = 0;
    if (!read_int64(t, "rpmLimit", rpm, error) || !read_int64(t, "tpmLimit", tpm, error) ||
        !read_int64(t, "sourceGeneration", p.source_generation, error)) {
      error = "tenant '" + p.tenant_id + "': " + error;
      return std::nullopt;
    }
    if (rpm < 0 || tpm < 0) {
      error = "tenant '" + p.tenant_id + "': limits must be >= 0";
      return std::nullopt;
    }
    p.rpm_limit = static_cast<double>(rpm);
    p.tpm_limit = static_cast<double>(tpm);
    if (const auto m = t.find("semanticMode"); m != t.end() && !m->is_null()) {
      const std::string mode = m->is_string() ? m->get<std::string>() : "";
      if (mode != "lexical" && mode != "neural" && mode != "off") {
        error = "tenant '" + p.tenant_id + "': semanticMode must be lexical, neural or off";
        return std::nullopt;
      }
      p.semantic_mode = mode;
    }
    if (const auto th = t.find("semanticThreshold"); th != t.end() && !th->is_null()) {
      const double v = th->is_number() ? th->get<double>() : -1;
      if (!(v > 0 && v <= 1)) {
        error = "tenant '" + p.tenant_id + "': semanticThreshold must be in (0, 1]";
        return std::nullopt;
      }
      p.semantic_threshold = v;
    }
    if (const auto ttl = t.find("ttlSeconds"); ttl != t.end() && !ttl->is_null()) {
      std::int64_t v = -1;
      if (!read_int64(t, "ttlSeconds", v, error) || v < 0 || v > kMaxTtlSeconds) {
        error = "tenant '" + p.tenant_id + "': ttlSeconds must be in [0, 2592000]";
        return std::nullopt;
      }
      p.ttl_seconds = static_cast<unsigned>(v);
    }
    set.tenants.push_back(std::move(p));
  }
  return set;
}

json parse_object(std::string_view text, std::string& error) {
  json j = json::parse(text.begin(), text.end(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) {
    error = "body must be a JSON object";
    return json();
  }
  return j;
}

}  // namespace

std::optional<RingUpdate> parse_ring_update(std::string_view text, std::string& error) {
  const json j = parse_object(text, error);
  if (j.is_null()) return std::nullopt;
  return ring_from_json(j, error);
}

std::optional<TenantPolicySetUpdate> parse_tenant_set_update(std::string_view text, std::string& error) {
  const json j = parse_object(text, error);
  if (j.is_null()) return std::nullopt;
  return tenants_from_json(j, error);
}

std::optional<ControlDocumentUpdate> parse_control_document(std::string_view text, std::string& error) {
  const json j = parse_object(text, error);
  if (j.is_null()) return std::nullopt;
  ControlDocumentUpdate doc;
  if (!read_int64(j, "version", doc.version, error)) return std::nullopt;
  if (const auto r = j.find("ring"); r != j.end() && !r->is_null()) {
    doc.ring = ring_from_json(*r, error);
    if (!doc.ring) {
      error = "ring: " + error;
      return std::nullopt;
    }
    if (doc.ring->version == 0) doc.ring->version = doc.version;
  }
  if (const auto t = j.find("tenants"); t != j.end() && !t->is_null()) {
    doc.tenants = tenants_from_json(*t, error);
    if (!doc.tenants) {
      error = "tenants: " + error;
      return std::nullopt;
    }
    if (doc.tenants->version == 0) doc.tenants->version = doc.version;
  }
  return doc;
}

std::optional<std::vector<PeerNode>> parse_peer_list(std::string_view spec, std::string& error) {
  std::vector<PeerNode> out;
  std::set<std::string> seen;
  while (!spec.empty()) {
    const auto comma = spec.find(',');
    const auto item = trim(spec.substr(0, comma));
    spec = comma == std::string_view::npos ? std::string_view{} : spec.substr(comma + 1);
    if (item.empty()) continue;
    const auto eq = item.find('=');
    std::string host;
    std::uint16_t port = 0;
    if (eq == std::string_view::npos || !valid_id(item.substr(0, eq)) ||
        !split_host_port(item.substr(eq + 1), host, port)) {
      error = "PEERS entries must look like id=host:port (got '" + std::string(item) + "')";
      return std::nullopt;
    }
    PeerNode p{std::string(item.substr(0, eq)), std::string(item.substr(eq + 1))};
    if (!seen.insert(p.id).second) {
      error = "duplicate peer id '" + p.id + "'";
      return std::nullopt;
    }
    out.push_back(std::move(p));
  }
  return out;
}

// ---------------------------------------------------------------------------

const PeerInfo* Topology::find(std::string_view id) const {
  for (const auto& p : peers) {
    if (p.id == id) return &p;
  }
  return nullptr;
}

std::string Topology::owner(std::string_view key) const {
  if (!ring) return self_id;
  return ring->node_for(key).value_or(self_id);
}

RouteAction decide_route(int hop, bool owner_is_self, bool owner_reachable) noexcept {
  if (hop > kMaxHops) return RouteAction::kRejectLoop;
  if (owner_is_self) return RouteAction::kServeLocal;
  // Already forwarded once: whatever our ring says, never forward again. This
  // is what keeps divergent rings (mid rolling update) from ping-ponging.
  if (hop >= 1) return RouteAction::kServeLocalDivergent;
  return owner_reachable ? RouteAction::kForward : RouteAction::kServeLocalFallback;
}

bool parse_hop_header(std::string_view header, int& hop) noexcept {
  header = trim(header);
  if (header.empty()) {
    hop = 0;
    return true;
  }
  std::uint64_t v = 0;
  if (!parse_u64(header, v) || v > 1000) return false;
  hop = static_cast<int>(v);
  return true;
}

bool resolve_numeric(const std::string& host, std::uint16_t port, sockaddr_storage& out, socklen_t& len) {
  std::memset(&out, 0, sizeof(out));
  auto* v4 = reinterpret_cast<sockaddr_in*>(&out);
  if (::inet_pton(AF_INET, host.c_str(), &v4->sin_addr) == 1) {
    v4->sin_family = AF_INET;
    v4->sin_port = htons(port);
    len = sizeof(sockaddr_in);
    return true;
  }
  auto* v6 = reinterpret_cast<sockaddr_in6*>(&out);
  if (::inet_pton(AF_INET6, host.c_str(), &v6->sin6_addr) == 1) {
    v6->sin6_family = AF_INET6;
    v6->sin6_port = htons(port);
    len = sizeof(sockaddr_in6);
    return true;
  }
  return false;
}

bool resolve_dns(const std::string& host, std::uint16_t port, sockaddr_storage& out, socklen_t& len) {
  if (resolve_numeric(host, port, out, len)) return true;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string p = std::to_string(port);
  if (::getaddrinfo(host.c_str(), p.c_str(), &hints, &res) != 0 || res == nullptr) return false;
  std::memcpy(&out, res->ai_addr, res->ai_addrlen);
  len = static_cast<socklen_t>(res->ai_addrlen);
  ::freeaddrinfo(res);
  return true;
}

// ---------------------------------------------------------------------------
// ControlPlane
// ---------------------------------------------------------------------------

ControlPlane::ControlPlane(Options options, TenantRegistry base_tenants)
    : opt_(std::move(options)),
      base_tenants_(std::move(base_tenants)),
      topology_(std::make_shared<const Topology>()),
      tenants_(std::make_shared<const TenantRegistry>(base_tenants_)) {
  std::lock_guard lock(mu_);
  ring_spec_.vnodes = static_cast<std::uint32_t>(opt_.default_vnodes);
  ring_spec_.nodes = opt_.static_peers;
  if (ring_spec_.nodes.empty()) ring_spec_.nodes.push_back(PeerNode{opt_.self_id, ""});
  topology_.store(build_topology_locked(0, opt_.default_vnodes, ring_spec_.nodes));
}

std::shared_ptr<Topology> ControlPlane::build_topology_locked(std::int64_t version, std::size_t vnodes,
                                                              const std::vector<PeerNode>& nodes) {
  auto topo = std::make_shared<Topology>();
  topo->version = version;
  topo->self_id = opt_.self_id;
  topo->vnodes = vnodes == 0 ? opt_.default_vnodes : vnodes;
  auto ring = std::make_shared<HashRing>(topo->vnodes);
  for (const auto& n : nodes) {
    ring->add_node(n.id);
    PeerInfo p;
    p.id = n.id;
    p.address = n.address;
    if (!n.address.empty() && split_host_port(n.address, p.host, p.port)) {
      p.resolved = opt_.fast_resolver(p.host, p.port, p.addr, p.addr_len);
    }
    auto& breaker = breakers_[n.id];
    if (!breaker) breaker = std::make_shared<CircuitBreaker>(opt_.peer_breaker);
    p.breaker = breaker;
    topo->peers.push_back(std::move(p));
  }
  std::sort(topo->peers.begin(), topo->peers.end(), [](const PeerInfo& a, const PeerInfo& b) { return a.id < b.id; });
  topo->ring = std::move(ring);
  return topo;
}

namespace {
ApplyOutcome check_version(std::int64_t incoming, std::int64_t current, const char* what) {
  if (incoming != 0 && incoming < current) {
    return {409, false, current,
            std::string("stale ") + what + " version " + std::to_string(incoming) + " < active " +
                std::to_string(current)};
  }
  if (incoming != 0 && incoming == current) {
    return {200, false, current, std::string(what) + " version " + std::to_string(current) + " already active"};
  }
  return {0, true, incoming, {}};
}
}  // namespace

ApplyOutcome ControlPlane::apply_ring_locked(const RingUpdate& update) {
  auto check = check_version(update.version, ring_version_, "ring");
  if (check.http_status != 0) return check;
  const std::int64_t version = update.version == 0 ? ring_version_ : update.version;
  auto topo = build_topology_locked(version, update.vnodes, update.nodes);
  std::string note;
  if (topo->find(opt_.self_id) == nullptr && !update.nodes.empty()) {
    note = "; note: this node (" + opt_.self_id + ") is not a ring member and will forward every request";
  }
  const auto unresolved = std::count_if(topo->peers.begin(), topo->peers.end(),
                                        [](const PeerInfo& p) { return !p.resolved; });
  ring_spec_ = update;
  ring_version_ = version;
  topology_.store(std::move(topo));
  return {200, true, version,
          "ring applied: " + std::to_string(update.nodes.size()) + " nodes, " +
              std::to_string(unresolved) + " pending DNS resolution" + note};
}

ApplyOutcome ControlPlane::apply_tenants_locked(const TenantPolicySetUpdate& update) {
  auto check = check_version(update.version, tenants_version_, "tenants");
  if (check.http_status != 0) return check;
  const std::int64_t version = update.version == 0 ? tenants_version_ : update.version;
  std::vector<TenantOverride> overrides;
  overrides.reserve(update.tenants.size());
  for (const auto& p : update.tenants) {
    TenantOverride o;
    o.info.id = p.tenant_id;
    o.info.limits = TenantLimits{p.rpm_limit, p.tpm_limit};
    o.info.semantic_mode = p.semantic_mode;
    o.info.semantic_threshold = p.semantic_threshold;
    o.info.ttl_seconds = p.ttl_seconds;
    o.api_key_sha256 = p.api_key_sha256;
    overrides.push_back(std::move(o));
  }
  tenants_.store(std::make_shared<const TenantRegistry>(base_tenants_.with_overrides(overrides)));
  tenant_spec_ = update;
  tenants_version_ = version;
  return {200, true, version, "tenants applied: " + std::to_string(update.tenants.size()) + " policies"};
}

ApplyOutcome ControlPlane::apply_ring(const RingUpdate& update) {
  std::lock_guard lock(mu_);
  return apply_ring_locked(update);
}

ApplyOutcome ControlPlane::apply_tenants(const TenantPolicySetUpdate& update) {
  std::lock_guard lock(mu_);
  return apply_tenants_locked(update);
}

ApplyOutcome ControlPlane::apply_document(const ControlDocumentUpdate& doc) {
  std::lock_guard lock(mu_);
  // Validate versions for both parts before changing anything, so a document
  // is applied all-or-nothing.
  if (doc.ring && check_version(doc.ring->version, ring_version_, "ring").http_status == 409) {
    return check_version(doc.ring->version, ring_version_, "ring");
  }
  if (doc.tenants && check_version(doc.tenants->version, tenants_version_, "tenants").http_status == 409) {
    return check_version(doc.tenants->version, tenants_version_, "tenants");
  }
  ApplyOutcome out{200, false, doc.version, {}};
  if (doc.ring) {
    const auto r = apply_ring_locked(*doc.ring);
    out.applied = out.applied || r.applied;
    out.message += r.message;
  }
  if (doc.tenants) {
    const auto t = apply_tenants_locked(*doc.tenants);
    out.applied = out.applied || t.applied;
    out.message += (out.message.empty() ? "" : "; ") + t.message;
  }
  out.version = std::max(ring_version_, tenants_version_);
  return out;
}

std::int64_t ControlPlane::ring_version() const {
  std::lock_guard lock(mu_);
  return ring_version_;
}

std::int64_t ControlPlane::tenants_version() const {
  std::lock_guard lock(mu_);
  return tenants_version_;
}

bool ControlPlane::resolve_peers(bool refresh_all) {
  // Snapshot what to resolve, resolve without holding the lock (DNS can be
  // slow), then publish if this topology is still current.
  const auto current = topology_.load();
  std::vector<std::pair<std::size_t, PeerInfo>> work;
  for (std::size_t i = 0; i < current->peers.size(); ++i) {
    const auto& p = current->peers[i];
    if (p.host.empty()) continue;
    if (!p.resolved || refresh_all) work.emplace_back(i, p);
  }
  if (work.empty()) return false;
  bool changed = false;
  for (auto& [i, p] : work) {
    sockaddr_storage addr{};
    socklen_t len = 0;
    const bool ok = opt_.slow_resolver(p.host, p.port, addr, len);
    const auto& before = current->peers[i];
    if (ok != before.resolved || (ok && (len != before.addr_len || std::memcmp(&addr, &before.addr, len) != 0))) {
      p.resolved = ok;
      p.addr = addr;
      p.addr_len = len;
      changed = true;
    }
  }
  if (!changed) return false;
  std::lock_guard lock(mu_);
  if (topology_.load() != current) return false;  // superseded meanwhile; next pass will redo it
  auto next = std::make_shared<Topology>(*current);
  for (auto& [i, p] : work) next->peers[i] = p;
  topology_.store(std::move(next));
  return true;
}

std::string ControlPlane::to_json() const {
  std::lock_guard lock(mu_);
  json doc;
  doc["version"] = std::max(ring_version_, tenants_version_);
  doc["nodeId"] = opt_.self_id;
  json nodes = json::array();
  for (const auto& p : topology_.load()->peers) {
    nodes.push_back({{"id", p.id}, {"address", p.address}, {"resolved", p.resolved}});
  }
  doc["ring"] = {{"version", ring_version_},
                 {"vnodes", topology_.load()->vnodes},
                 {"nodes", nodes}};
  json tenants = json::array();
  for (const auto& t : tenant_spec_.tenants) {
    json o = {{"tenantId", t.tenant_id},
              {"apiKeySha256", t.api_key_sha256},
              {"rpmLimit", static_cast<std::int64_t>(t.rpm_limit)},
              {"tpmLimit", static_cast<std::int64_t>(t.tpm_limit)},
              {"sourceGeneration", t.source_generation}};
    if (t.semantic_mode) o["semanticMode"] = *t.semantic_mode;
    if (t.semantic_threshold) o["semanticThreshold"] = *t.semantic_threshold;
    if (t.ttl_seconds) o["ttlSeconds"] = *t.ttl_seconds;
    tenants.push_back(std::move(o));
  }
  doc["tenants"] = {{"version", tenants_version_}, {"tenants", tenants}};
  return doc.dump();
}

}  // namespace vanguard
