#include "vanguard/tenants.hpp"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "vanguard/sha256.hpp"
#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

bool read_limit(const nlohmann::json& obj, const char* name, double fallback, double& out,
                std::string& error) {
  const auto it = obj.find(name);
  if (it == obj.end() || it->is_null()) {
    out = fallback;
    return true;
  }
  if (!it->is_number()) {
    error = std::string("'") + name + "' must be a number";
    return false;
  }
  out = it->get<double>();
  return true;
}

}  // namespace

std::optional<std::string_view> bearer_token(std::string_view authorization) {
  authorization = trim(authorization);
  if (authorization.size() < 8 || !iequals(authorization.substr(0, 6), "bearer") ||
      authorization[6] != ' ') {
    return std::nullopt;
  }
  const auto token = trim(authorization.substr(7));
  if (token.empty()) return std::nullopt;
  return token;
}

TenantRegistry TenantRegistry::open(TenantLimits defaults) {
  TenantRegistry r;
  r.allow_unknown_ = true;
  r.defaults_ = defaults;
  return r;
}

std::optional<TenantRegistry> TenantRegistry::from_json(std::string_view json, TenantLimits defaults,
                                                        std::string& error) {
  const auto j = nlohmann::json::parse(json.begin(), json.end(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) {
    error = "tenants config must be a JSON object";
    return std::nullopt;
  }
  TenantRegistry r;
  r.allow_unknown_ = j.value("allow_unknown", false);
  r.defaults_ = defaults;
  if (const auto d = j.find("default_limits"); d != j.end()) {
    if (!d->is_object()) {
      error = "'default_limits' must be an object";
      return std::nullopt;
    }
    if (!read_limit(*d, "rpm", defaults.rpm, r.defaults_.rpm, error) ||
        !read_limit(*d, "tpm", defaults.tpm, r.defaults_.tpm, error)) {
      return std::nullopt;
    }
  }
  const auto tenants = j.find("tenants");
  if (tenants != j.end() && !tenants->is_array()) {
    error = "'tenants' must be an array";
    return std::nullopt;
  }
  if (tenants != j.end()) {
    for (const auto& t : *tenants) {
      const auto id = t.is_object() ? t.find("id") : t.end();
      if (!t.is_object() || id == t.end() || !id->is_string() || id->get_ref<const std::string&>().empty()) {
        error = "each tenant needs a non-empty string 'id'";
        return std::nullopt;
      }
      TenantInfo info{id->get<std::string>(), r.defaults_, {}, {}, {}};
      if (!read_limit(t, "rpm", r.defaults_.rpm, info.limits.rpm, error) ||
          !read_limit(t, "tpm", r.defaults_.tpm, info.limits.tpm, error)) {
        error = "tenant '" + info.id + "': " + error;
        return std::nullopt;
      }
      const auto keys = t.find("api_keys");
      if (keys == t.end() || !keys->is_array() || keys->empty()) {
        error = "tenant '" + info.id + "' needs a non-empty 'api_keys' array";
        return std::nullopt;
      }
      for (const auto& k : *keys) {
        if (!k.is_string() || k.get_ref<const std::string&>().empty()) {
          error = "tenant '" + info.id + "': api keys must be non-empty strings";
          return std::nullopt;
        }
        if (!r.by_key_.emplace(k.get<std::string>(), info).second) {
          error = "duplicate api key in tenant '" + info.id + "'";
          return std::nullopt;
        }
      }
    }
  }
  return r;
}

std::optional<TenantRegistry> TenantRegistry::from_file(const std::string& path, TenantLimits defaults,
                                                        std::string& error) {
  std::ifstream in(path);
  if (!in) {
    error = "cannot open tenants file: " + path;
    return std::nullopt;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  return from_json(ss.str(), defaults, error);
}

std::optional<TenantInfo> TenantRegistry::resolve(std::string_view api_key) const {
  if (const auto it = by_key_.find(std::string(api_key)); it != by_key_.end()) return it->second;
  if (!by_key_hash_.empty()) {
    if (const auto it = by_key_hash_.find(Sha256::hex(api_key)); it != by_key_hash_.end()) return it->second;
  }
  if (!allow_unknown_) return std::nullopt;
  // Never use the raw key as an id: ids appear in headers, logs and metrics.
  return TenantInfo{"anon-" + Sha256::hex(api_key).substr(0, 12), defaults_, {}, {}, {}};
}

TenantRegistry TenantRegistry::with_overrides(const std::vector<TenantOverride>& overrides) const {
  TenantRegistry r = *this;
  for (const auto& o : overrides) {
    for (auto& [key, info] : r.by_key_) {
      if (info.id == o.info.id) info = o.info;
    }
    for (const auto& h : o.api_key_sha256) r.by_key_hash_[h] = o.info;
  }
  return r;
}

}  // namespace vanguard
