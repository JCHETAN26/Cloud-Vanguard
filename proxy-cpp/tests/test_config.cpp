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

TEST(Config, Phase2Defaults) {
  const char* argv[] = {"p"};
  std::string err;
  const auto cfg = load_config(1, argv, err);
  ASSERT_TRUE(cfg) << err;
  EXPECT_EQ(cfg->ring_vnodes, 1000u);
  EXPECT_EQ(cfg->upstream_ttfb_timeout_ms, 300u);
  EXPECT_DOUBLE_EQ(cfg->semantic_threshold, 0.94);
  EXPECT_EQ(cfg->semantic_mode, "lexical");
  EXPECT_TRUE(cfg->fallback_upstream_url.empty());
}

TEST(Config, RejectsBadPhase2Input) {
  std::string err;
  const char* threshold[] = {"p", "--semantic-threshold=1.5"};
  EXPECT_FALSE(load_config(2, threshold, err));
  const char* mode[] = {"p", "--semantic-mode=gpu"};
  EXPECT_FALSE(load_config(2, mode, err));
  const char* fallback[] = {"p", "--fallback-upstream-url=https://api.example.com"};
  EXPECT_FALSE(load_config(2, fallback, err));
  const char* backoff[] = {"p", "--breaker-open-ms=5000", "--breaker-max-open-ms=1000"};
  EXPECT_FALSE(load_config(3, backoff, err));
}

TEST(Config, NeuralModeDefaultsToStricterThreshold) {
  std::string err;
  const char* neural[] = {"p", "--semantic-mode=neural"};
  auto cfg = load_config(2, neural, err);
  ASSERT_TRUE(cfg) << err;
  EXPECT_DOUBLE_EQ(cfg->semantic_threshold, 0.96);
  const char* explicit_threshold[] = {"p", "--semantic-mode=neural", "--semantic-threshold=0.98"};
  cfg = load_config(3, explicit_threshold, err);
  ASSERT_TRUE(cfg) << err;
  EXPECT_DOUBLE_EQ(cfg->semantic_threshold, 0.98);
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
