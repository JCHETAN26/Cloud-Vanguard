#include <cstdio>
#include <string>

#include "vanguard/config.hpp"
#include "vanguard/server.hpp"
#include "vanguard/tenants.hpp"

int main(int argc, char** argv) {
  using namespace vanguard;

  std::string err;
  auto cfg = load_config(argc, argv, err);
  if (!cfg) {
    if (err == "help") {
      std::fputs(config_usage().c_str(), stdout);
      return 0;
    }
    std::fprintf(stderr, "vanguard-proxy: config error: %s\n", err.c_str());
    return 2;
  }

  const TenantLimits defaults{cfg->default_rpm, cfg->default_tpm};
  std::optional<TenantRegistry> tenants;
  if (cfg->tenants_file.empty()) {
    tenants = TenantRegistry::open(defaults);
  } else {
    tenants = TenantRegistry::from_file(cfg->tenants_file, defaults, err);
    if (!tenants) {
      std::fprintf(stderr, "vanguard-proxy: %s\n", err.c_str());
      return 2;
    }
  }

  CircuitBreaker::Options breaker;
  breaker.failure_threshold = cfg->breaker_failure_threshold;
  breaker.base_open = CircuitBreaker::Millis(cfg->breaker_open_ms);
  breaker.max_open = CircuitBreaker::Millis(cfg->breaker_max_open_ms);

  std::array<UpstreamTarget, 3> targets;
  auto primary = resolve_target("primary", cfg->upstream_url, cfg->upstream_api_key, cfg->upstream_ttfb_timeout_ms,
                                breaker, err);
  if (!primary) {
    std::fprintf(stderr, "vanguard-proxy: %s\n", err.c_str());
    return 2;
  }
  targets[kPrimaryTarget] = std::move(*primary);
  targets[kFallbackTarget].name = "fallback";
  if (!cfg->fallback_upstream_url.empty()) {
    auto fallback = resolve_target("fallback", cfg->fallback_upstream_url, cfg->fallback_upstream_api_key,
                                   cfg->fallback_ttfb_timeout_ms, breaker, err);
    if (!fallback) {
      std::fprintf(stderr, "vanguard-proxy: %s\n", err.c_str());
      return 2;
    }
    fallback->model_override = cfg->fallback_model;
    targets[kFallbackTarget] = std::move(*fallback);
  }
  targets[kEmbedderTarget].name = "embedder";
  if (cfg->semantic_mode == "neural") {
    auto embedder = resolve_target("embedder", cfg->embedder_url, "", cfg->embedder_timeout_ms, breaker, err);
    if (!embedder) {
      std::fprintf(stderr, "vanguard-proxy: %s\n", err.c_str());
      return 2;
    }
    targets[kEmbedderTarget] = std::move(*embedder);
  }

  std::fprintf(stderr, "vanguard-proxy configuration:\n%s", config_summary(*cfg).c_str());
  SharedState state(std::move(*cfg), std::move(*tenants), std::move(targets));
  return run_server(state);
}
