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

  std::string upstream_url = "http://127.0.0.1:9000";
  std::string upstream_api_key;  // sent upstream as "Authorization: Bearer ..."
  unsigned upstream_timeout_ms = 60000;  // max idle time waiting on upstream bytes
  std::size_t upstream_max_idle = 256;   // pooled keep-alive connections per worker

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
