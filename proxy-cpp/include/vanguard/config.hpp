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
  //   lexical  in-process near-duplicate encoder (default)
  //   neural   async OpenAI-compatible /v1/embeddings sidecar (EMBEDDER_URL)
  //   off      exact-match caching only
  // EMBEDDER=lexical|remote|off is accepted as a deprecated alias.
  std::string semantic_mode = "lexical";
  std::string embedder_url = "http://127.0.0.1:9200";
  std::string embedder_model = "BAAI/bge-small-en-v1.5";
  unsigned embedder_timeout_ms = 150;
  // 0 = per-mode default: 0.94 lexical, 0.96 neural. Calibration with
  // bge-small showed different questions scoring up to 0.955 (e.g. "reverse a
  // list in Python" vs "... in JavaScript"), so neural needs >= 0.96.
  double semantic_threshold = 0;
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
  std::vector<std::string> ring_nodes;  // ids only (legacy); prefer PEERS
  std::string peers;                    // "id=host:port,..." static ring with forwarding addresses
  std::size_t ring_vnodes = 1000;

  // Control plane (Phase 3).
  std::string admin_addr = "127.0.0.1";  // operator deployments set 0.0.0.0 + ADMIN_TOKEN
  std::uint16_t admin_port = 9901;       // 0 = disabled
  std::string admin_token;               // required as Bearer on /admin/v1/* when set
  std::string peer_token;                // authenticates X-Vanguard-Hop between nodes
  std::string control_file;              // ControlDocument JSON to load + poll (ConfigMap mount)
  unsigned peer_connect_timeout_ms = 250;
  unsigned peer_timeout_ms = 60000;      // max wait for the owner's response headers

  unsigned client_idle_timeout_s = 60;
  std::size_t max_body_kb = 4096;
};

struct UpstreamUrl {
  std::string host;
  std::uint16_t port = 80;
  std::string base_path;  // no trailing slash; prepended to the request path
};

inline constexpr double kLexicalDefaultThreshold = 0.94;
inline constexpr double kNeuralDefaultThreshold = 0.96;

std::optional<UpstreamUrl> parse_upstream_url(std::string_view url, std::string& error);

// Returns nullopt with `error` set on invalid input, or with error == "help"
// when --help was requested.
std::optional<Config> load_config(int argc, const char* const* argv, std::string& error);

std::string config_usage();
std::string config_summary(const Config& cfg);

}  // namespace vanguard
