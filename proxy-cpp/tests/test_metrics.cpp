// The /metrics exposition must be valid Prometheus text format: a scrape with
// a metric family declared twice is rejected as a whole by Prometheus.

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <sstream>
#include <string>

#include "vanguard/metrics.hpp"
#include "vanguard/tracing.hpp"

using namespace vanguard;

namespace {

std::string full_exposition() {
  Metrics m;
  m.count_status(200);
  m.latency_hit.observe(std::chrono::microseconds(300));
  std::vector<BreakerReport> breakers(3);
  const char* names[] = {"primary", "fallback", "embedder"};
  for (int i = 0; i < 3; ++i) {
    breakers[static_cast<std::size_t>(i)].name = names[i];
    breakers[static_cast<std::size_t>(i)].configured = true;
  }
  ControlReport control;
  control.node_id = "n1";
  tracing::Tracer tracer(tracing::TracerOptions{});
  return m.render(CacheStats{}, control, 0, breakers) + tracer.render_metrics();
}

// "name{labels} value" -> family name (histogram suffixes folded).
std::string family_of(const std::string& sample, const std::map<std::string, std::string>& types) {
  std::string name = sample.substr(0, sample.find_first_of("{ "));
  if (types.count(name)) return name;
  for (const char* suffix : {"_bucket", "_sum", "_count"}) {
    const std::string s(suffix);
    if (name.size() > s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0) {
      const auto base = name.substr(0, name.size() - s.size());
      if (types.count(base) && types.at(base) == "histogram") return base;
    }
  }
  return name;
}

}  // namespace

TEST(MetricsExposition, EveryFamilyIsDeclaredExactlyOnce) {
  std::istringstream in(full_exposition());
  std::map<std::string, std::string> types;
  std::map<std::string, int> type_lines, help_lines;
  std::set<std::string> sample_lines;
  std::string line;
  std::vector<std::string> samples;
  while (std::getline(in, line)) {
    if (line.rfind("# TYPE ", 0) == 0) {
      std::istringstream f(line.substr(7));
      std::string name, type;
      f >> name >> type;
      types[name] = type;
      ++type_lines[name];
    } else if (line.rfind("# HELP ", 0) == 0) {
      std::istringstream f(line.substr(7));
      std::string name;
      f >> name;
      ++help_lines[name];
    } else if (!line.empty()) {
      samples.push_back(line);
      EXPECT_TRUE(sample_lines.insert(line.substr(0, line.rfind(' '))).second) << "duplicate series: " << line;
    }
  }
  ASSERT_FALSE(types.empty());
  for (const auto& [name, n] : type_lines) EXPECT_EQ(n, 1) << "TYPE declared " << n << " times: " << name;
  for (const auto& [name, n] : help_lines) EXPECT_EQ(n, 1) << "HELP declared " << n << " times: " << name;
  for (const auto& s : samples) {
    EXPECT_TRUE(types.count(family_of(s, types))) << "sample without a TYPE: " << s;
  }
  // Per-upstream breakdowns and the aggregates both exist under distinct names.
  EXPECT_EQ(types["vanguard_upstream_requests_total"], "counter");
  EXPECT_EQ(types["vanguard_llm_requests_total"], "counter");
  EXPECT_EQ(types["vanguard_outbound_timeouts_total"], "counter");
  EXPECT_EQ(types["vanguard_request_duration_seconds"], "histogram");
  EXPECT_EQ(types["vanguard_trace_spans_exported_total"], "counter");
}
