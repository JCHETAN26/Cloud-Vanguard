#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <vector>

#include "vanguard/control.hpp"
#include "vanguard/sha256.hpp"

using vanguard::ControlPlane;
using vanguard::decide_route;
using vanguard::RouteAction;

namespace {

const char* kRing3 = R"({"version":"3","vnodes":1000,"nodes":[
  {"id":"vc-0","address":"10.0.0.10:8080"},{"id":"vc-1","address":"10.0.0.11:8080"},
  {"id":"vc-2","address":"vc-2.vc-peers.default.svc.cluster.local:8080"}]})";

ControlPlane make(const std::string& self, vanguard::TenantRegistry base = vanguard::TenantRegistry::open({600, 0})) {
  ControlPlane::Options o;
  o.self_id = self;
  // Deterministic "DNS": numeric hosts as-is, every hostname -> 127.0.0.1.
  o.slow_resolver = [](const std::string& host, std::uint16_t port, sockaddr_storage& out, socklen_t& len) {
    return vanguard::resolve_numeric(host, port, out, len) || vanguard::resolve_numeric("127.0.0.1", port, out, len);
  };
  return ControlPlane(std::move(o), std::move(base));
}

vanguard::RingUpdate ring(const char* json) {
  std::string err;
  auto r = vanguard::parse_ring_update(json, err);
  EXPECT_TRUE(r) << err;
  return r.value_or(vanguard::RingUpdate{});
}

}  // namespace

// ---------------------------------------------------------------------------
// Forwarding / loop prevention policy
// ---------------------------------------------------------------------------

TEST(Routing, ForwardOnlyFromTheIngressHop) {
  EXPECT_EQ(decide_route(0, true, true), RouteAction::kServeLocal);
  EXPECT_EQ(decide_route(0, false, true), RouteAction::kForward);
  EXPECT_EQ(decide_route(0, false, false), RouteAction::kServeLocalFallback);
  // Forwarded once: serve here even if our ring disagrees (rolling update).
  EXPECT_EQ(decide_route(1, true, true), RouteAction::kServeLocal);
  EXPECT_EQ(decide_route(1, false, true), RouteAction::kServeLocalDivergent);
  EXPECT_EQ(decide_route(1, false, false), RouteAction::kServeLocalDivergent);
  // Anything beyond one hop is a loop.
  EXPECT_EQ(decide_route(2, true, true), RouteAction::kRejectLoop);
  EXPECT_EQ(decide_route(7, false, true), RouteAction::kRejectLoop);
}

TEST(Routing, HopHeaderParsing) {
  int hop = -1;
  EXPECT_TRUE(vanguard::parse_hop_header("", hop));
  EXPECT_EQ(hop, 0);
  EXPECT_TRUE(vanguard::parse_hop_header(" 1 ", hop));
  EXPECT_EQ(hop, 1);
  EXPECT_FALSE(vanguard::parse_hop_header("-1", hop));
  EXPECT_FALSE(vanguard::parse_hop_header("one", hop));
  EXPECT_FALSE(vanguard::parse_hop_header("99999999999", hop));
}

// With any assignment of rings to nodes (including divergent ones during a
// rolling update), a request is handled after at most one forward.
TEST(Routing, DivergentRingsNeverLoop) {
  std::vector<std::string> nodes = {"a", "b", "c"};
  // Each node believes a different (rotated) owner for every key.
  auto believed_owner = [&](int node, int key) { return (key + node) % 3; };
  for (int key = 0; key < 300; ++key) {
    for (int ingress = 0; ingress < 3; ++ingress) {
      int at = ingress;
      int hop = 0;
      int forwards = 0;
      for (int step = 0; step < 10; ++step) {
        const int owner = believed_owner(at, key);
        const auto action = decide_route(hop, owner == at, true);
        if (action != RouteAction::kForward) {
          EXPECT_NE(action, RouteAction::kRejectLoop);
          break;
        }
        at = owner;
        ++hop;
        ++forwards;
      }
      EXPECT_LE(forwards, 1);
    }
  }
}

// Three gateways given the same ring agree on every key's owner, and keys
// spread across all of them.
TEST(Routing, NodesSharingARingAgreeOnOwners) {
  auto a = make("vc-0"), b = make("vc-1"), c = make("vc-2");
  for (auto* cp : {&a, &b, &c}) ASSERT_TRUE(cp->apply_ring(ring(kRing3)).applied);
  std::map<std::string, int> spread;
  for (int i = 0; i < 30000; ++i) {
    const std::string key = vanguard::Sha256::hex("prompt-" + std::to_string(i));
    const auto oa = a.topology().load()->owner(key);
    EXPECT_EQ(oa, b.topology().load()->owner(key));
    EXPECT_EQ(oa, c.topology().load()->owner(key));
    ++spread[oa];
  }
  ASSERT_EQ(spread.size(), 3u);
  for (const auto& [node, n] : spread) EXPECT_NEAR(n, 10000, 1000) << node;
}

// ---------------------------------------------------------------------------
// Control plane: parsing, versions, resolution, tenant policies
// ---------------------------------------------------------------------------

TEST(ControlPlane, StartsWithStaticPeers) {
  ControlPlane::Options o;
  o.self_id = "n1";
  std::string err;
  o.static_peers = *vanguard::parse_peer_list("n1=127.0.0.1:8181, n2=127.0.0.1:8182", err);
  ControlPlane cp(std::move(o), vanguard::TenantRegistry::open({}));
  const auto t = cp.topology().load();
  ASSERT_EQ(t->peers.size(), 2u);
  EXPECT_TRUE(t->find("n2")->resolved);
  EXPECT_EQ(t->members(), 2u);
  EXPECT_FALSE(vanguard::parse_peer_list("n1=nohost", err));
  EXPECT_FALSE(vanguard::parse_peer_list("n1=h:1,n1=h:2", err));
}

TEST(ControlPlane, RingVersionsAreMonotonic) {
  auto cp = make("vc-0");
  const auto first = cp.apply_ring(ring(kRing3));
  EXPECT_TRUE(first.applied);
  EXPECT_EQ(cp.ring_version(), 3);
  const auto same = cp.apply_ring(ring(kRing3));
  EXPECT_FALSE(same.applied);
  EXPECT_EQ(same.http_status, 200);
  const auto stale = cp.apply_ring(ring(R"({"version":2,"nodes":[{"id":"x","address":"1.2.3.4:1"}]})"));
  EXPECT_EQ(stale.http_status, 409);
  EXPECT_EQ(cp.topology().load()->peers.size(), 3u) << "stale update must not apply";
  // Version 0 = manual override: applied, version unchanged.
  const auto manual = cp.apply_ring(ring(R"({"nodes":[{"id":"vc-0","address":"1.2.3.4:1"}]})"));
  EXPECT_TRUE(manual.applied);
  EXPECT_EQ(cp.ring_version(), 3);
  EXPECT_EQ(cp.topology().load()->peers.size(), 1u);
}

TEST(ControlPlane, HostnamesResolveOffTheEventLoop) {
  auto cp = make("vc-0");
  cp.apply_ring(ring(kRing3));
  const auto before = cp.topology().load();
  EXPECT_TRUE(before->find("vc-0")->resolved);   // numeric: resolved immediately
  EXPECT_FALSE(before->find("vc-2")->resolved);  // hostname: pending
  const auto gen = cp.topology().generation();
  EXPECT_TRUE(cp.resolve_peers(false));
  EXPECT_GT(cp.topology().generation(), gen);
  EXPECT_TRUE(cp.topology().load()->find("vc-2")->resolved);
  EXPECT_FALSE(cp.resolve_peers(true)) << "nothing changed";
}

TEST(ControlPlane, PeerBreakersSurviveRingUpdates) {
  auto cp = make("vc-0");
  cp.apply_ring(ring(kRing3));
  const auto breaker = cp.topology().load()->find("vc-1")->breaker;
  cp.apply_ring(ring(R"({"version":4,"nodes":[{"id":"vc-0","address":"10.0.0.10:8080"},{"id":"vc-1","address":"10.0.0.99:8080"}]})"));
  EXPECT_EQ(cp.topology().load()->find("vc-1")->breaker, breaker);
}

TEST(ControlPlane, RejectsInvalidMessages) {
  std::string err;
  EXPECT_FALSE(vanguard::parse_ring_update("[]", err));
  EXPECT_FALSE(vanguard::parse_ring_update(R"({"nodes":[{"id":"a","address":"nohostport"}]})", err));
  EXPECT_FALSE(vanguard::parse_ring_update(R"({"nodes":[{"id":"a","address":"h:1"},{"id":"a","address":"h:2"}]})", err));
  EXPECT_FALSE(vanguard::parse_ring_update(R"({"vnodes":200000})", err));
  EXPECT_FALSE(vanguard::parse_tenant_set_update(R"({"tenants":[{"tenantId":"t","semanticThreshold":1.5}]})", err));
  EXPECT_FALSE(vanguard::parse_tenant_set_update(R"({"tenants":[{"tenantId":"t","semanticMode":"gpu"}]})", err));
  EXPECT_FALSE(vanguard::parse_tenant_set_update(R"({"tenants":[{"tenantId":"t","apiKeySha256":["abc"]}]})", err));
  EXPECT_FALSE(vanguard::parse_tenant_set_update(R"({"tenants":[{"tenantId":"t"},{"tenantId":"t"}]})", err));
  EXPECT_FALSE(vanguard::parse_tenant_set_update(R"({"tenants":[{"tenantId":"t","rpmLimit":-5}]})", err));
}

TEST(ControlPlane, TenantPoliciesApplyWithoutRestart) {
  std::string err;
  auto base = vanguard::TenantRegistry::from_json(
      R"({"tenants":[{"id":"acme","api_keys":["sk-static"],"rpm":100}]})", {600, 0}, err);
  ASSERT_TRUE(base) << err;
  auto cp = make("vc-0", *base);
  const std::string dyn_hash = vanguard::Sha256::hex("sk-dynamic");
  const std::string body = R"({"version":"5","tenants":[
      {"tenantId":"acme","rpmLimit":"1200","tpmLimit":"50000","semanticMode":"neural","semanticThreshold":0.97,"ttlSeconds":600,"sourceGeneration":"2"},
      {"tenantId":"beta","apiKeySha256":[")" + dyn_hash + R"("],"rpmLimit":60}]})";
  auto set = vanguard::parse_tenant_set_update(body, err);
  ASSERT_TRUE(set) << err;
  const auto gen = cp.tenants().generation();
  EXPECT_TRUE(cp.apply_tenants(*set).applied);
  EXPECT_GT(cp.tenants().generation(), gen);

  const auto reg = cp.tenants().load();
  const auto acme = reg->resolve("sk-static");  // static key, dynamic settings
  ASSERT_TRUE(acme);
  EXPECT_EQ(acme->limits.rpm, 1200);
  EXPECT_EQ(acme->limits.tpm, 50000);
  EXPECT_EQ(acme->semantic_mode.value_or(""), "neural");
  EXPECT_DOUBLE_EQ(acme->semantic_threshold.value_or(0), 0.97);
  EXPECT_EQ(acme->ttl_seconds.value_or(0), 600u);
  const auto beta = reg->resolve("sk-dynamic");  // authenticated by key hash only
  ASSERT_TRUE(beta);
  EXPECT_EQ(beta->id, "beta");
  EXPECT_EQ(beta->limits.rpm, 60);

  // The set is replaced wholesale: dropping a policy reverts to static config.
  EXPECT_TRUE(cp.apply_tenants(*vanguard::parse_tenant_set_update(R"({"version":6,"tenants":[]})", err)).applied);
  EXPECT_EQ(cp.tenants().load()->resolve("sk-static")->limits.rpm, 100);
  EXPECT_FALSE(cp.tenants().load()->resolve("sk-dynamic").has_value()) << "removed policy revokes its keys";
}

TEST(ControlPlane, DocumentsApplyAllOrNothing) {
  auto cp = make("vc-0");
  std::string err;
  auto doc = vanguard::parse_control_document(
      std::string(R"({"version":"10","ring":)") + kRing3 + R"(,"tenants":{"tenants":[{"tenantId":"t","rpmLimit":5}]}})", err);
  ASSERT_TRUE(doc) << err;
  EXPECT_EQ(doc->tenants->version, 10) << "parts inherit the document version";
  const auto out = cp.apply_document(*doc);
  EXPECT_TRUE(out.applied) << out.message;
  EXPECT_EQ(cp.ring_version(), 3);
  EXPECT_EQ(cp.tenants_version(), 10);
  // A document whose tenant part is stale changes nothing, not even the ring.
  auto stale = vanguard::parse_control_document(
      R"({"version":"9","ring":{"version":11,"nodes":[{"id":"z","address":"1.1.1.1:1"}]},"tenants":{"tenants":[]}})", err);
  ASSERT_TRUE(stale) << err;
  EXPECT_EQ(cp.apply_document(*stale).http_status, 409);
  EXPECT_EQ(cp.ring_version(), 3);
  // GET round-trips through the same parser.
  auto round = vanguard::parse_control_document(cp.to_json(), err);
  ASSERT_TRUE(round) << err;
  EXPECT_EQ(round->ring->nodes.size(), 3u);
  EXPECT_EQ(round->tenants->tenants.at(0).tenant_id, "t");
}

// ---------------------------------------------------------------------------
// Cross-language contract: the operator's rendered document (Go, protojson)
// must be accepted by the data plane. The file is produced by
// operator-go/internal/render TestGoldenControlDocument.
// ---------------------------------------------------------------------------

TEST(ControlPlaneGolden, AcceptsTheOperatorsRenderedDocument) {
  std::ifstream in(std::string(VANGUARD_REPO_ROOT) + "/operator-go/internal/render/testdata/control_golden.json");
  ASSERT_TRUE(in.good()) << "golden file missing";
  std::stringstream ss;
  ss << in.rdbuf();
  std::string err;
  const auto doc = vanguard::parse_control_document(ss.str(), err);
  ASSERT_TRUE(doc) << err;
  EXPECT_EQ(doc->version, 42);
  ASSERT_TRUE(doc->ring && doc->tenants);
  EXPECT_EQ(doc->ring->vnodes, 1000u);
  ASSERT_EQ(doc->ring->nodes.size(), 3u);
  EXPECT_EQ(doc->ring->nodes[0].address, "vc-0.vc-peers.prod.svc.cluster.local:8080");

  auto cp = make("vc-1", vanguard::TenantRegistry::from_json(R"({"tenants":[]})", {600, 0}, err).value());
  const auto out = cp.apply_document(*doc);
  ASSERT_TRUE(out.applied) << out.message;
  EXPECT_EQ(cp.topology().load()->self_id, "vc-1");
  EXPECT_NE(cp.topology().load()->find("vc-1"), nullptr);
  const auto acme = cp.tenants().load()->resolve("sk-acme-2");  // raw key never left the operator
  ASSERT_TRUE(acme);
  EXPECT_EQ(acme->id, "acme");
  EXPECT_EQ(acme->limits.rpm, 1200);
  EXPECT_EQ(acme->limits.tpm, 50000);
  EXPECT_EQ(acme->semantic_mode.value_or(""), "neural");
  EXPECT_DOUBLE_EQ(acme->semantic_threshold.value_or(0), 0.97);
  EXPECT_EQ(acme->ttl_seconds.value_or(0), 600u);
  EXPECT_FALSE(cp.tenants().load()->resolve("sk-unknown"));
}
