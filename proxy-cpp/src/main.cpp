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

  auto upstream = resolve_upstream(*cfg, err);
  if (!upstream) {
    std::fprintf(stderr, "vanguard-proxy: %s\n", err.c_str());
    return 2;
  }

  std::fprintf(stderr, "vanguard-proxy configuration:\n%s", config_summary(*cfg).c_str());
  SharedState state(std::move(*cfg), std::move(*tenants), std::move(*upstream));
  return run_server(state);
}
