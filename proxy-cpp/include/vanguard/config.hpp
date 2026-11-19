#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vanguard {

// Runtime configuration. Every field can be set by environment variable
// (e.g. UPSTREAM_URL) or by the equivalent flag (--upstream-url=...); flags win.
struct Config {
  std::string listen_addr = "0.0.0.0";
  std::uint16_t listen_port = 8080;
  unsigned workers = 0;  // 0 = std::thread::hardware_concurrency()

  // Primary upstream (e.g. self-hosted vLLM).
  std::string upstream_url = "http://127.0.0.1:9000";
  std::string upstream_api_key;  // sent upstream as "Authorization: Bearer ..."
  unsigned upstream_ttfb_timeout_ms = 300;  // no response headers within this => failure
  unsigned upstream_timeout_ms = 60000;     // max silence once the response has started
  std::size_t upstream_max_idle = 256;      // pooled keep-alive connections per worker

  // Optional fallback upstream (e.g. a commercial API). Empty = no failover.
  std::string fallback_upstream_url;
  std::string fallback_upstream_api_key;
  std::string fallback_model;  // rewrite "model" for the fallback (empty = keep)
  unsigned fallback_ttfb_timeout_ms = 10000;

  // Circuit breaker (one per upstream, shared by all workers).
  unsigned breaker_failure_threshold = 5;
  unsigned breaker_open_ms = 1000;       // first open period
  unsigned breaker_max_open_ms = 30000;  // exponential backoff cap

  // Semantic cache.
  std::string embedder = "lexical";  // lexical | remote | off
  std::string embedder_url = "http://127.0.0.1:9200";
  std::string embedder_model = "BAAI/bge-small-en-v1.5";
  unsigned embedder_timeout_ms = 150;
  double semantic_threshold = 0.94;
  std::size_t semantic_max_chars = 2000;
  std::size_t semantic_max_entries = 100000;
  std::size_t semantic_max_per_scope = 4096;  // brute-force scan bound (~1 ms at 512 dims)
  std::size_t semantic_dim = 512;  // lexical encoder dimensions

  std::string tenants_file;  // empty = accept any bearer token with default limits
  double default_rpm = 600;
  double default_tpm = 200000;
  std::uint64_t default_completion_tokens = 256;

  std::size_t cache_max_mb = 256;
  std::size_t cache_shards = 16;
  unsigned cache_ttl_s = 3600;  // 0 = no expiry
  std::size_t cache_max_entry_kb = 4096;

  std::string node_id = "local";
  std::vector<std::string> ring_nodes;  // empty = {node_id}
  std::size_t ring_vnodes = 1000;

  unsigned client_idle_timeout_s = 60;
  std::size_t max_body_kb = 4096;
};

struct UpstreamUrl {
  std::string host;
  std::uint16_t port = 80;
  std::string base_path;  // no trailing slash; prepended to the request path
};

std::optional<UpstreamUrl> parse_upstream_url(std::string_view url, std::string& error);

// Returns nullopt with `error` set on invalid input, or with error == "help"
// when --help was requested.
std::optional<Config> load_config(int argc, const char* const* argv, std::string& error);

std::string config_usage();
std::string config_summary(const Config& cfg);

}  // namespace vanguard
