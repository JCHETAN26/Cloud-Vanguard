#include "vanguard/config.hpp"

#include <cstdlib>
#include <functional>
#include <limits>

#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

using Setter = std::function<bool(Config&, std::string_view, std::string&)>;
using Getter = std::function<std::string(const Config&)>;

struct Option {
  const char* env;
  const char* help;
  Setter set;
  Getter get;
};

template <typename T>
Setter uint_setter(T Config::*field, std::uint64_t min = 0,
                   std::uint64_t max = std::numeric_limits<T>::max()) {
  return [=](Config& c, std::string_view v, std::string& err) {
    std::uint64_t n = 0;
    if (!parse_u64(v, n) || n < min || n > max) {
      err = "expected an integer in [" + std::to_string(min) + ", " + std::to_string(max) + "]";
      return false;
    }
    c.*field = static_cast<T>(n);
    return true;
  };
}

template <typename T>
Getter uint_getter(T Config::*field) {
  return [=](const Config& c) { return std::to_string(c.*field); };
}

Setter double_setter(double Config::*field) {
  return [=](Config& c, std::string_view v, std::string& err) {
    const std::string s(v);
    char* end = nullptr;
    const double d = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || d < 0) {
      err = "expected a non-negative number";
      return false;
    }
    c.*field = d;
    return true;
  };
}

Getter double_getter(double Config::*field) {
  return [=](const Config& c) {
    std::string s = std::to_string(c.*field);
    s.erase(s.find_last_not_of('0') + 1);
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
  };
}

Setter string_setter(std::string Config::*field) {
  return [=](Config& c, std::string_view v, std::string&) {
    c.*field = std::string(v);
    return true;
  };
}

Getter string_getter(std::string Config::*field) {
  return [=](const Config& c) { return c.*field; };
}

std::vector<std::string> split_csv(std::string_view s) {
  std::vector<std::string> out;
  while (!s.empty()) {
    const auto comma = s.find(',');
    const auto item = trim(s.substr(0, comma));
    if (!item.empty()) out.emplace_back(item);
    if (comma == std::string_view::npos) break;
    s.remove_prefix(comma + 1);
  }
  return out;
}

const std::vector<Option>& options() {
  static const std::vector<Option> kOptions = {
      {"LISTEN_ADDR", "IPv4 address to bind", string_setter(&Config::listen_addr),
       string_getter(&Config::listen_addr)},
      {"LISTEN_PORT", "TCP port to listen on", uint_setter(&Config::listen_port, 1, 65535),
       uint_getter(&Config::listen_port)},
      {"WORKERS", "epoll worker threads (0 = #cores)", uint_setter(&Config::workers, 0, 1024),
       uint_getter(&Config::workers)},
      {"UPSTREAM_URL", "OpenAI-compatible upstream base URL (http:// only)",
       string_setter(&Config::upstream_url), string_getter(&Config::upstream_url)},
      {"UPSTREAM_API_KEY", "bearer token sent to the upstream", string_setter(&Config::upstream_api_key),
       [](const Config& c) { return c.upstream_api_key.empty() ? std::string("<unset>") : std::string("<set>"); }},
      {"UPSTREAM_TTFB_TIMEOUT_MS", "primary: max wait for response headers before failing over",
       uint_setter(&Config::upstream_ttfb_timeout_ms, 1, 3600000), uint_getter(&Config::upstream_ttfb_timeout_ms)},
      {"FALLBACK_UPSTREAM_URL", "fallback upstream base URL (empty = no failover)",
       string_setter(&Config::fallback_upstream_url), string_getter(&Config::fallback_upstream_url)},
      {"FALLBACK_UPSTREAM_API_KEY", "bearer token sent to the fallback", string_setter(&Config::fallback_upstream_api_key),
       [](const Config& c) { return c.fallback_upstream_api_key.empty() ? std::string("<unset>") : std::string("<set>"); }},
      {"FALLBACK_MODEL", "model name to send to the fallback (empty = unchanged)",
       string_setter(&Config::fallback_model), string_getter(&Config::fallback_model)},
      {"FALLBACK_TTFB_TIMEOUT_MS", "fallback: max wait for response headers",
       uint_setter(&Config::fallback_ttfb_timeout_ms, 1, 3600000), uint_getter(&Config::fallback_ttfb_timeout_ms)},
      {"BREAKER_FAILURE_THRESHOLD", "consecutive failures that open a circuit",
       uint_setter(&Config::breaker_failure_threshold, 1, 1000000), uint_getter(&Config::breaker_failure_threshold)},
      {"BREAKER_OPEN_MS", "first open period (doubles after each failed probe)",
       uint_setter(&Config::breaker_open_ms, 1, 3600000), uint_getter(&Config::breaker_open_ms)},
      {"BREAKER_MAX_OPEN_MS", "cap for the exponential backoff",
       uint_setter(&Config::breaker_max_open_ms, 1, 86400000), uint_getter(&Config::breaker_max_open_ms)},
      {"SEMANTIC_MODE", "semantic cache encoder: lexical | neural | off", string_setter(&Config::semantic_mode),
       string_getter(&Config::semantic_mode)},
      {"EMBEDDER_URL", "OpenAI-compatible /v1/embeddings sidecar (SEMANTIC_MODE=neural)",
       string_setter(&Config::embedder_url), string_getter(&Config::embedder_url)},
      {"EMBEDDER_MODEL", "model name sent to the embedding sidecar", string_setter(&Config::embedder_model),
       string_getter(&Config::embedder_model)},
      {"EMBEDDER_TIMEOUT_MS", "max wait for an embedding before skipping semantic lookup",
       uint_setter(&Config::embedder_timeout_ms, 1, 60000), uint_getter(&Config::embedder_timeout_ms)},
      {"SEMANTIC_THRESHOLD", "min cosine similarity for a semantic hit (default 0.94 lexical, 0.96 neural)",
       double_setter(&Config::semantic_threshold),
       [](const Config& c) {
         return c.semantic_threshold <= 0 ? std::string("auto") : double_getter(&Config::semantic_threshold)(c);
       }},
      {"SEMANTIC_MAX_CHARS", "longer final user messages skip semantic matching",
       uint_setter(&Config::semantic_max_chars, 1, 1u << 24), uint_getter(&Config::semantic_max_chars)},
      {"SEMANTIC_MAX_ENTRIES", "vectors kept in the semantic index",
       uint_setter(&Config::semantic_max_entries, 1, 1u << 30), uint_getter(&Config::semantic_max_entries)},
      {"SEMANTIC_MAX_PER_SCOPE", "vectors kept per scope (system prompt); bounds the scan cost",
       uint_setter(&Config::semantic_max_per_scope, 1, 1u << 24), uint_getter(&Config::semantic_max_per_scope)},
      {"SEMANTIC_DIM", "lexical encoder dimensions", uint_setter(&Config::semantic_dim, 16, 8192),
       uint_getter(&Config::semantic_dim)},
      {"UPSTREAM_TIMEOUT_MS", "max idle wait for upstream bytes",
       uint_setter(&Config::upstream_timeout_ms, 100, 3600000), uint_getter(&Config::upstream_timeout_ms)},
      {"UPSTREAM_MAX_IDLE", "pooled idle upstream connections per worker",
       uint_setter(&Config::upstream_max_idle, 0, 65536), uint_getter(&Config::upstream_max_idle)},
      {"TENANTS_FILE", "JSON tenant/key/limit config (empty = accept any key)",
       string_setter(&Config::tenants_file), string_getter(&Config::tenants_file)},
      {"DEFAULT_RPM", "default requests/minute per tenant (0 = unlimited)", double_setter(&Config::default_rpm),
       double_getter(&Config::default_rpm)},
      {"DEFAULT_TPM", "default est. tokens/minute per tenant (0 = unlimited)",
       double_setter(&Config::default_tpm), double_getter(&Config::default_tpm)},
      {"DEFAULT_COMPLETION_TOKENS", "completion budget assumed when max_tokens is unset",
       uint_setter(&Config::default_completion_tokens), uint_getter(&Config::default_completion_tokens)},
      {"CACHE_MAX_MB", "total cache capacity in MiB", uint_setter(&Config::cache_max_mb, 1, 1u << 20),
       uint_getter(&Config::cache_max_mb)},
      {"CACHE_SHARDS", "independent LRU shards", uint_setter(&Config::cache_shards, 1, 4096),
       uint_getter(&Config::cache_shards)},
      {"CACHE_TTL_S", "entry TTL in seconds (0 = none)", uint_setter(&Config::cache_ttl_s),
       uint_getter(&Config::cache_ttl_s)},
      {"CACHE_MAX_ENTRY_KB", "largest response that will be cached",
       uint_setter(&Config::cache_max_entry_kb, 1, 1u << 20), uint_getter(&Config::cache_max_entry_kb)},
      {"NODE_ID", "this instance's name on the hash ring", string_setter(&Config::node_id),
       string_getter(&Config::node_id)},
      {"RING_NODES", "comma-separated ring members (default: NODE_ID)",
       [](Config& c, std::string_view v, std::string&) {
         c.ring_nodes = split_csv(v);
         return true;
       },
       [](const Config& c) {
         std::string s;
         for (const auto& n : c.ring_nodes) s += (s.empty() ? "" : ",") + n;
         return s.empty() ? c.node_id : s;
       }},
      {"RING_VNODES", "virtual nodes per ring member", uint_setter(&Config::ring_vnodes, 1, 100000),
       uint_getter(&Config::ring_vnodes)},
      {"CLIENT_IDLE_TIMEOUT_S", "close idle client connections after N seconds",
       uint_setter(&Config::client_idle_timeout_s, 1, 86400), uint_getter(&Config::client_idle_timeout_s)},
      {"MAX_BODY_KB", "largest accepted request body", uint_setter(&Config::max_body_kb, 1, 1u << 20),
       uint_getter(&Config::max_body_kb)},
  };
  return kOptions;
}

std::string flag_name(const char* env) {
  std::string s = "--";
  for (const char* p = env; *p != '\0'; ++p) s.push_back(*p == '_' ? '-' : ascii_lower(*p));
  return s;
}

}  // namespace

std::optional<UpstreamUrl> parse_upstream_url(std::string_view url, std::string& error) {
  constexpr std::string_view kScheme = "http://";
  if (url.substr(0, 8) == "https://") {
    error = "https upstreams are not supported yet (Phase 1 speaks plain HTTP; terminate TLS in a sidecar)";
    return std::nullopt;
  }
  if (url.substr(0, kScheme.size()) != kScheme) {
    error = "UPSTREAM_URL must start with http://";
    return std::nullopt;
  }
  url.remove_prefix(kScheme.size());
  const auto slash = url.find('/');
  const auto authority = url.substr(0, slash);
  std::string_view path = slash == std::string_view::npos ? std::string_view{} : url.substr(slash);
  while (!path.empty() && path.back() == '/') path.remove_suffix(1);

  UpstreamUrl out;
  const auto colon = authority.rfind(':');
  if (colon != std::string_view::npos && authority.find(']') == std::string_view::npos) {
    std::uint64_t port = 0;
    if (!parse_u64(authority.substr(colon + 1), port) || port == 0 || port > 65535) {
      error = "invalid port in UPSTREAM_URL";
      return std::nullopt;
    }
    out.host = std::string(authority.substr(0, colon));
    out.port = static_cast<std::uint16_t>(port);
  } else {
    out.host = std::string(authority);
  }
  if (out.host.empty()) {
    error = "missing host in UPSTREAM_URL";
    return std::nullopt;
  }
  out.base_path = std::string(path);
  return out;
}

std::optional<Config> load_config(int argc, const char* const* argv, std::string& error) {
  Config cfg;
  const auto& opts = options();

  // Deprecated alias: EMBEDDER=lexical|remote|off (SEMANTIC_MODE wins if both are set).
  if (const char* legacy = std::getenv("EMBEDDER"); legacy != nullptr && std::getenv("SEMANTIC_MODE") == nullptr) {
    const std::string_view v = legacy;
    cfg.semantic_mode = v == "remote" ? "neural" : std::string(v);
  }

  for (const auto& opt : opts) {
    if (const char* v = std::getenv(opt.env); v != nullptr) {
      std::string err;
      if (!opt.set(cfg, v, err)) {
        error = std::string(opt.env) + ": " + err;
        return std::nullopt;
      }
    }
  }

  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      error = "help";
      return std::nullopt;
    }
    if (arg.substr(0, 2) != "--") {
      error = "unexpected argument: " + std::string(arg);
      return std::nullopt;
    }
    std::string_view name = arg;
    std::string_view value;
    if (const auto eq = arg.find('='); eq != std::string_view::npos) {
      name = arg.substr(0, eq);
      value = arg.substr(eq + 1);
    } else if (i + 1 < argc) {
      value = argv[++i];
    } else {
      error = "missing value for " + std::string(arg);
      return std::nullopt;
    }
    const Option* match = nullptr;
    for (const auto& opt : opts) {
      if (flag_name(opt.env) == name) match = &opt;
    }
    if (match == nullptr) {
      error = "unknown flag: " + std::string(name);
      return std::nullopt;
    }
    std::string err;
    if (!match->set(cfg, value, err)) {
      error = std::string(name) + ": " + err;
      return std::nullopt;
    }
  }

  std::string err;
  if (!parse_upstream_url(cfg.upstream_url, err)) {
    error = err;
    return std::nullopt;
  }
  if (!cfg.fallback_upstream_url.empty() && !parse_upstream_url(cfg.fallback_upstream_url, err)) {
    error = "FALLBACK_UPSTREAM_URL: " + err;
    return std::nullopt;
  }
  if (cfg.semantic_mode != "lexical" && cfg.semantic_mode != "neural" && cfg.semantic_mode != "off") {
    error = "SEMANTIC_MODE must be one of: lexical, neural, off";
    return std::nullopt;
  }
  if (cfg.semantic_mode == "neural" && !parse_upstream_url(cfg.embedder_url, err)) {
    error = "EMBEDDER_URL: " + err;
    return std::nullopt;
  }
  if (cfg.semantic_threshold <= 0) {
    cfg.semantic_threshold = cfg.semantic_mode == "neural" ? kNeuralDefaultThreshold : kLexicalDefaultThreshold;
  }
  if (cfg.semantic_threshold > 1) {
    error = "SEMANTIC_THRESHOLD must be in (0, 1]";
    return std::nullopt;
  }
  if (cfg.breaker_max_open_ms < cfg.breaker_open_ms) {
    error = "BREAKER_MAX_OPEN_MS must be >= BREAKER_OPEN_MS";
    return std::nullopt;
  }
  if (cfg.node_id.empty()) {
    error = "NODE_ID must not be empty";
    return std::nullopt;
  }
  return cfg;
}

std::string config_usage() {
  std::string out = "Usage: vanguard-proxy [--flag=value ...]\n\nEvery flag can also be set via its environment variable.\n\n";
  for (const auto& opt : options()) {
    const std::string flag = flag_name(opt.env);
    out += "  " + flag + std::string(flag.size() < 30 ? 30 - flag.size() : 1, ' ') + opt.help + "  [" + opt.env +
           ", default " + opt.get(Config{}) + "]\n";
  }
  return out;
}

std::string config_summary(const Config& cfg) {
  std::string out;
  for (const auto& opt : options()) out += std::string("  ") + opt.env + "=" + opt.get(cfg) + "\n";
  return out;
}

}  // namespace vanguard
