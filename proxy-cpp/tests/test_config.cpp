#include <gtest/gtest.h>

#include "vanguard/config.hpp"

using vanguard::load_config;
using vanguard::parse_upstream_url;

TEST(UpstreamUrl, Parses) {
  std::string err;
  auto u = parse_upstream_url("http://127.0.0.1:9000", err);
  ASSERT_TRUE(u) << err;
  EXPECT_EQ(u->host, "127.0.0.1");
  EXPECT_EQ(u->port, 9000);
  EXPECT_EQ(u->base_path, "");

  u = parse_upstream_url("http://vllm.svc/openai/", err);
  ASSERT_TRUE(u) << err;
  EXPECT_EQ(u->host, "vllm.svc");
  EXPECT_EQ(u->port, 80);
  EXPECT_EQ(u->base_path, "/openai");

  EXPECT_FALSE(parse_upstream_url("https://api.openai.com", err));
  EXPECT_FALSE(parse_upstream_url("ftp://x", err));
  EXPECT_FALSE(parse_upstream_url("http://:80", err));
  EXPECT_FALSE(parse_upstream_url("http://h:99999", err));
}

TEST(Config, FlagsOverrideDefaults) {
  const char* argv[] = {"vanguard-proxy", "--listen-port=9090", "--ring-nodes", "a, b,,c",
                        "--cache-max-mb=64", "--default-rpm=1.5"};
  std::string err;
  const auto cfg = load_config(6, argv, err);
  ASSERT_TRUE(cfg) << err;
  EXPECT_EQ(cfg->listen_port, 9090);
  EXPECT_EQ(cfg->ring_nodes, (std::vector<std::string>{"a", "b", "c"}));
  EXPECT_EQ(cfg->cache_max_mb, 64u);
  EXPECT_DOUBLE_EQ(cfg->default_rpm, 1.5);
}

TEST(Config, RejectsBadInput) {
  std::string err;
  const char* bad_port[] = {"p", "--listen-port=0"};
  EXPECT_FALSE(load_config(2, bad_port, err));
  const char* unknown[] = {"p", "--nope=1"};
  EXPECT_FALSE(load_config(2, unknown, err));
  const char* help[] = {"p", "--help"};
  EXPECT_FALSE(load_config(2, help, err));
  EXPECT_EQ(err, "help");
}
