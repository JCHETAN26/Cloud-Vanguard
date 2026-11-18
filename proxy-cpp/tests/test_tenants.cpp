#include <gtest/gtest.h>

#include "vanguard/tenants.hpp"

using vanguard::bearer_token;
using vanguard::TenantRegistry;

TEST(BearerToken, Extraction) {
  EXPECT_EQ(bearer_token("Bearer sk-abc").value_or(""), "sk-abc");
  EXPECT_EQ(bearer_token("  bearer   sk-abc  ").value_or(""), "sk-abc");
  EXPECT_FALSE(bearer_token("").has_value());
  EXPECT_FALSE(bearer_token("Basic dXNlcjpwYXNz").has_value());
  EXPECT_FALSE(bearer_token("Bearer ").has_value());
  EXPECT_FALSE(bearer_token("Bearersk-abc").has_value());
}

TEST(TenantRegistry, OpenModeDerivesStableAnonymousIds) {
  const auto r = TenantRegistry::open({100, 1000});
  const auto a = r.resolve("sk-one");
  const auto b = r.resolve("sk-one");
  const auto c = r.resolve("sk-two");
  ASSERT_TRUE(a && b && c);
  EXPECT_EQ(a->id, b->id);
  EXPECT_NE(a->id, c->id);
  EXPECT_EQ(a->id.rfind("anon-", 0), 0u);
  EXPECT_EQ(a->id.find("sk-one"), std::string::npos) << "raw key must not leak into the id";
  EXPECT_EQ(a->limits.rpm, 100);
}

TEST(TenantRegistry, FromJson) {
  std::string err;
  const auto r = TenantRegistry::from_json(R"({
    "default_limits": {"rpm": 50},
    "tenants": [
      {"id": "acme", "api_keys": ["k1", "k2"], "tpm": 9000},
      {"id": "globex", "api_keys": ["k3"], "rpm": 0}
    ]})",
                                           {600, 100000}, err);
  ASSERT_TRUE(r.has_value()) << err;
  EXPECT_FALSE(r->allow_unknown());
  EXPECT_EQ(r->resolve("k2")->id, "acme");
  EXPECT_EQ(r->resolve("k2")->limits.rpm, 50);       // from default_limits
  EXPECT_EQ(r->resolve("k2")->limits.tpm, 9000);     // per-tenant override
  EXPECT_EQ(r->resolve("k3")->limits.rpm, 0);        // unlimited
  EXPECT_EQ(r->resolve("k3")->limits.tpm, 100000);   // global default
  EXPECT_FALSE(r->resolve("unknown").has_value());
}

TEST(TenantRegistry, RejectsInvalidConfig) {
  std::string err;
  EXPECT_FALSE(TenantRegistry::from_json("[]", {}, err));
  EXPECT_FALSE(TenantRegistry::from_json(R"({"tenants":[{"api_keys":["k"]}]})", {}, err));
  EXPECT_FALSE(TenantRegistry::from_json(R"({"tenants":[{"id":"a","api_keys":[]}]})", {}, err));
  EXPECT_FALSE(TenantRegistry::from_json(
      R"({"tenants":[{"id":"a","api_keys":["k"]},{"id":"b","api_keys":["k"]}]})", {}, err));
  EXPECT_NE(err.find("duplicate"), std::string::npos);
}
