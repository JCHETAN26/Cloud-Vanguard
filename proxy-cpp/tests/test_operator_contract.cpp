// Cross-language contract checks between the Go operator and vanguard-proxy,
// pinned by golden files that both test suites read:
//
//   operator-go/internal/render/testdata/tracing_env_golden.json
//       the pod environment the operator renders for a CacheCluster with every
//       tracing field set (Go writes it; here the proxy's config loader must
//       accept every variable and arrive at the same tracing settings);
//   proto/vanguard/control/v1/testdata/apply_result_tracing.golden.json
//       the ApplyResult (with TracingStatus) the proxy answers pushes with
//       (written from the proxy's real serializer below; Go parses it strictly
//       as vanguard.control.v1.ApplyResult and must recognise the settings).
//
// Regenerate the second file with VANGUARD_UPDATE_GOLDEN=1.

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vanguard/config.hpp"
#include "vanguard/control.hpp"
#include "vanguard/tracing.hpp"

using namespace vanguard;

namespace {

std::string repo_path(const std::string& rel) { return std::string(VANGUARD_REPO_ROOT) + "/" + rel; }

std::string flag_for(const std::string& env) {
  std::string f = "--";
  for (char c : env) f.push_back(c == '_' ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return f;
}

}  // namespace

TEST(OperatorContract, ProxyAcceptsTheOperatorsRenderedEnvironment) {
  std::ifstream in(repo_path("operator-go/internal/render/testdata/tracing_env_golden.json"));
  ASSERT_TRUE(in.good()) << "golden file missing (go test ./internal/render -update)";
  const auto vars = nlohmann::json::parse(in);
  std::vector<std::string> storage{"vanguard-proxy"};
  for (const auto& v : vars) storage.push_back(flag_for(v["name"]) + "=" + v["value"].get<std::string>());
  std::vector<const char*> argv;
  for (const auto& s : storage) argv.push_back(s.c_str());

  std::string err;
  const auto cfg = load_config(static_cast<int>(argv.size()), argv.data(), err);
  ASSERT_TRUE(cfg) << "proxy rejected the operator's environment: " << err;
  EXPECT_EQ(cfg->listen_port, 8080);
  EXPECT_EQ(cfg->admin_port, 9901);
  EXPECT_EQ(cfg->upstream_url, "http://vllm:8000");
  EXPECT_EQ(cfg->cache_max_mb, 512u);

  const auto t = tracing_options(*cfg, err);
  ASSERT_TRUE(t) << err;
  EXPECT_TRUE(t->enabled);
  EXPECT_EQ(t->traces_url, "http://otel-collector.observability:4318/v1/traces");
  EXPECT_EQ(t->sampler, tracing::Sampler::kParentTraceIdRatio);
  EXPECT_DOUBLE_EQ(t->ratio, 0.25);
  EXPECT_EQ(t->service_name, "vanguard-prod");
  EXPECT_EQ(t->schedule_delay_ms, 2000u);
  EXPECT_EQ(t->max_queue, 8192u);
  EXPECT_EQ(t->max_batch, 1024u);
  EXPECT_EQ(t->export_timeout_ms, 5000u);
}

TEST(OperatorContract, ApplyResultWithTracingMatchesGolden) {
  tracing::TracerOptions o;
  o.enabled = true;
  o.traces_url = "http://otel-collector.observability:4318/v1/traces";
  o.sampler = tracing::Sampler::kParentTraceIdRatio;
  o.ratio = 0.25;
  o.service_name = "vanguard-prod";
  o.service_instance = "vc-0";
  tracing::Tracer tracer(o);
  ApplyOutcome out;
  out.applied = true;
  out.version = 42;
  out.message = "applied";
  const auto got = nlohmann::json::parse(apply_result_json(out, "vc-0", &tracer));

  const std::string path = repo_path("proto/vanguard/control/v1/testdata/apply_result_tracing.golden.json");
  if (const char* u = std::getenv("VANGUARD_UPDATE_GOLDEN"); u != nullptr && std::string(u) == "1") {
    std::ofstream(path) << got.dump(2) << "\n";
  }
  std::ifstream in(path);
  ASSERT_TRUE(in.good()) << "golden file missing (run with VANGUARD_UPDATE_GOLDEN=1)";
  const auto want = nlohmann::json::parse(in);
  EXPECT_EQ(got, want) << "ApplyResult JSON changed; the Go operator parses this file:\n" << got.dump(2);
  EXPECT_EQ(got["tracing"]["tracesEndpoint"], o.traces_url);
  EXPECT_EQ(got["version"], "42");  // proto3 JSON: int64 as string

  // A proxy without tracing reports it disabled, with no endpoint.
  tracing::Tracer off{tracing::TracerOptions{}};
  const auto j = nlohmann::json::parse(apply_result_json(out, "vc-1", &off));
  EXPECT_EQ(j["tracing"]["enabled"], false);
  EXPECT_FALSE(j["tracing"].contains("tracesEndpoint"));
}

TEST(OperatorContract, ApplyResultSurvivesInvalidUtf8InMessages) {
  ApplyOutcome out;
  out.message = "bad node id \xff\xfe";
  const auto s = apply_result_json(out, "n\xc3", nullptr);
  nlohmann::json parsed;
  ASSERT_NO_THROW(parsed = nlohmann::json::parse(s));
  EXPECT_EQ(parsed["nodeId"], "n\xef\xbf\xbd");  // U+FFFD replaces the truncated sequence
}
