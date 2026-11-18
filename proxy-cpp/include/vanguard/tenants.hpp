#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "vanguard/rate_limiter.hpp"

namespace vanguard {

struct TenantInfo {
  std::string id;
  TenantLimits limits;
};

// Extracts <token> from "Bearer <token>" (scheme is case-insensitive).
std::optional<std::string_view> bearer_token(std::string_view authorization);

// Maps API keys to tenants. Loaded once at startup in Phase 1; the Phase 4
// operator will push TenantCachePolicy updates instead.
//
// File format (JSON):
//   {
//     "allow_unknown": false,
//     "default_limits": {"rpm": 600, "tpm": 200000},
//     "tenants": [{"id": "acme", "api_keys": ["sk-..."], "rpm": 1200, "tpm": 0}]
//   }
// With allow_unknown, any bearer token is accepted as its own tenant (id =
// "anon-" + first 12 hex chars of SHA-256(token)) with default limits.
class TenantRegistry {
 public:
  static TenantRegistry open(TenantLimits defaults);
  static std::optional<TenantRegistry> from_json(std::string_view json, TenantLimits defaults,
                                                 std::string& error);
  static std::optional<TenantRegistry> from_file(const std::string& path, TenantLimits defaults,
                                                 std::string& error);

  [[nodiscard]] std::optional<TenantInfo> resolve(std::string_view api_key) const;
  [[nodiscard]] std::size_t tenant_key_count() const noexcept { return by_key_.size(); }
  [[nodiscard]] bool allow_unknown() const noexcept { return allow_unknown_; }

 private:
  bool allow_unknown_ = true;
  TenantLimits defaults_{};
  std::unordered_map<std::string, TenantInfo> by_key_;
};

}  // namespace vanguard
