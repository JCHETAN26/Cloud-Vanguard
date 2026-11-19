#include "vanguard/tracing.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "vanguard/config.hpp"

using namespace vanguard::tracing;

namespace {

constexpr const char* kSpecExample = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

// A minimal OTLP/HTTP receiver: accepts connections, records each request
// body and answers with a fixed status.
class OtlpSink {
 public:
  explicit OtlpSink(int status = 200) : status_(status) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    EXPECT_EQ(::listen(fd_, 16), 0);
    thread_ = std::thread([this] { serve(); });
  }
  ~OtlpSink() {
    stop_ = true;
    // Wake the blocking accept() portably (shutdown() on a listener does not on macOS).
    const int w = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port_);
    ::connect(w, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::close(w);
    thread_.join();
    ::close(fd_);
  }
  std::string url() const { return "http://127.0.0.1:" + std::to_string(port_) + "/v1/traces"; }
  std::vector<std::string> bodies() {
    std::lock_guard lock(mu_);
    return bodies_;
  }
  std::vector<std::string> heads() {
    std::lock_guard lock(mu_);
    return heads_;
  }
  std::size_t span_count() {
    std::size_t n = 0;
    for (const auto& b : bodies()) {
      const auto j = nlohmann::json::parse(b);
      n += j["resourceSpans"][0]["scopeSpans"][0]["spans"].size();
    }
    return n;
  }

 private:
  void serve() {
    while (!stop_) {
      const int c = ::accept(fd_, nullptr, nullptr);
      if (c < 0) return;
      std::string data;
      char buf[8192];
      std::size_t need = std::string::npos;
      std::size_t head_end = std::string::npos;
      while (true) {
        const ssize_t r = ::recv(c, buf, sizeof(buf), 0);
        if (r <= 0) break;
        data.append(buf, static_cast<std::size_t>(r));
        if (head_end == std::string::npos) {
          head_end = data.find("\r\n\r\n");
          if (head_end != std::string::npos) {
            const auto cl = data.find("Content-Length: ");
            need = head_end + 4 + std::stoul(data.substr(cl + 16));
          }
        }
        if (need != std::string::npos && data.size() >= need) break;
      }
      if (head_end != std::string::npos) {
        std::lock_guard lock(mu_);
        heads_.push_back(data.substr(0, head_end));
        bodies_.push_back(data.substr(head_end + 4));
      }
      const std::string resp = "HTTP/1.1 " + std::to_string(status_) + " X\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
      ::send(c, resp.data(), resp.size(), 0);
      ::close(c);
    }
  }

  int status_;
  int fd_ = -1;
  std::uint16_t port_ = 0;
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::vector<std::string> bodies_, heads_;
  std::thread thread_;
};

TracerOptions enabled_options(const std::string& url) {
  TracerOptions o;
  o.enabled = true;
  o.traces_url = url;
  o.service_instance = "node-a";
  o.schedule_delay_ms = 60000;  // tests flush explicitly
  return o;
}

}  // namespace

TEST(Traceparent, ParsesTheSpecExample) {
  const auto ctx = parse_traceparent(kSpecExample);
  ASSERT_TRUE(ctx);
  EXPECT_EQ(to_hex(ctx->trace_id.data(), 16), "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_EQ(to_hex(ctx->span_id.data(), 8), "00f067aa0ba902b7");
  EXPECT_TRUE(ctx->sampled);
  EXPECT_EQ(format_traceparent(*ctx), kSpecExample);

  const auto unsampled = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00");
  ASSERT_TRUE(unsampled);
  EXPECT_FALSE(unsampled->sampled);
  EXPECT_TRUE(parse_traceparent("  00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01 "));
  // A future version may append fields after another dash.
  EXPECT_TRUE(parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-future"));
}

TEST(Traceparent, RejectsMalformedHeaders) {
  for (const char* bad : {
           "",
           "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",  // uppercase
           "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  // forbidden version
           "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  // zero trace id
           "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  // zero span id
           "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",   // short trace id
           "00_4bf92f3577b34da6a3ce929d0e0e4736_00f067aa0ba902b7_01",  // separators
           "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-x",  // v00 has no suffix
           "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902g7-01",  // non-hex
       }) {
    EXPECT_FALSE(parse_traceparent(bad)) << bad;
  }
}

TEST(Sampling, RatioIsDeterministicAndProportional) {
  TraceId id{};
  EXPECT_FALSE(ratio_sampled(id, 0.0));
  id[15] = 1;
  EXPECT_TRUE(ratio_sampled(id, 1.0));
  EXPECT_FALSE(ratio_sampled(id, 0.0));

  std::mt19937_64 gen(7);
  int sampled = 0;
  constexpr int kN = 20000;
  for (int i = 0; i < kN; ++i) {
    for (auto& b : id) b = static_cast<std::uint8_t>(gen());
    const bool a = ratio_sampled(id, 0.25);
    EXPECT_EQ(a, ratio_sampled(id, 0.25));  // same id, same decision (every node agrees)
    sampled += a ? 1 : 0;
  }
  EXPECT_NEAR(sampled / static_cast<double>(kN), 0.25, 0.015);
}

TEST(Tracer, DisabledTracerIsInert) {
  Tracer t(TracerOptions{});
  auto s = t.start_span("x", SpanKind::kServer, kSpecExample);
  EXPECT_FALSE(s.active());
  EXPECT_FALSE(s.recording());
  EXPECT_EQ(s.traceparent(), "");
  s.set_attribute("k", "v");  // no-op
  t.end(s);
  EXPECT_EQ(t.stats().started, 0u);
  EXPECT_EQ(t.flush(), 0u);
}

TEST(Tracer, ContinuesTheCallersTraceAndPropagatesToChildren) {
  Tracer t(enabled_options("http://127.0.0.1:9/v1/traces"));
  auto root = t.start_span("POST /v1/chat/completions", SpanKind::kServer, kSpecExample);
  ASSERT_TRUE(root.recording());
  EXPECT_EQ(to_hex(root.context().trace_id.data(), 16), "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_NE(to_hex(root.context().span_id.data(), 8), "00f067aa0ba902b7");  // new span, caller is the parent

  auto child = t.start_child(root, "llm.upstream", SpanKind::kClient);
  ASSERT_TRUE(child.recording());
  const auto header = parse_traceparent(child.traceparent());
  ASSERT_TRUE(header);
  EXPECT_EQ(header->trace_id, root.context().trace_id);
  EXPECT_EQ(header->span_id, child.context().span_id);
  EXPECT_TRUE(header->sampled);

  // No incoming header: a fresh trace.
  auto fresh = t.start_span("x", SpanKind::kServer, "garbage");
  EXPECT_TRUE(fresh.recording());
  EXPECT_NE(fresh.context().trace_id, root.context().trace_id);
  EXPECT_EQ(t.stats().started, 3u);
}

TEST(Tracer, UnsampledParentPropagatesButRecordsNothing) {
  Tracer t(enabled_options("http://127.0.0.1:9/v1/traces"));
  auto root = t.start_span("x", SpanKind::kServer, "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00");
  EXPECT_TRUE(root.active());
  EXPECT_FALSE(root.recording());
  auto child = t.start_child(root, "llm.upstream", SpanKind::kClient);
  EXPECT_TRUE(child.active());
  EXPECT_FALSE(child.recording());
  const auto tp = child.traceparent();
  EXPECT_EQ(tp.substr(0, 36), "00-4bf92f3577b34da6a3ce929d0e0e4736-");
  EXPECT_EQ(tp.substr(52), "-00");
  t.end(child);
  t.end(root);
  EXPECT_EQ(t.stats().started, 0u);
  EXPECT_EQ(t.flush(), 0u);

  auto o = enabled_options("http://127.0.0.1:9/v1/traces");
  o.sampler = Sampler::kAlwaysOff;
  Tracer off(o);
  EXPECT_FALSE(off.start_span("x", SpanKind::kServer, "").recording());
  o.sampler = Sampler::kParentTraceIdRatio;
  o.ratio = 0.0;
  Tracer ratio(o);
  EXPECT_FALSE(ratio.start_span("x", SpanKind::kServer, "").recording());
  EXPECT_TRUE(ratio.start_span("x", SpanKind::kServer, kSpecExample).recording());  // parent decides
}

TEST(Tracer, StatusOkIsFinalAndAttributesOverwrite) {
  Tracer t(enabled_options("http://127.0.0.1:9/v1/traces"));
  auto s = t.start_span("x", SpanKind::kServer, "");
  s.set_status(StatusCode::kError, "boom");
  EXPECT_EQ(s.status(), StatusCode::kError);
  s.set_status(StatusCode::kOk);
  EXPECT_EQ(s.status(), StatusCode::kOk);
  s.set_status(StatusCode::kError, "later");
  EXPECT_EQ(s.status(), StatusCode::kOk);
}

TEST(Otlp, JsonFollowsTheOtlpEncoding) {
  Tracer t(enabled_options("http://127.0.0.1:9/v1/traces"));
  auto root = t.start_span("POST /v1/chat/completions", SpanKind::kServer, kSpecExample);
  root.set_attribute("vanguard.tenant", "acme");
  root.set_attribute("http.response.status_code", 200);
  root.set_attribute("vanguard.semantic.similarity", 0.97);
  root.set_attribute("vanguard.stream", true);
  root.set_attribute("vanguard.tenant", "acme-2");  // overwrite, not duplicate
  SpanData data;
  data.context = root.context();
  data.parent_span_id = parse_traceparent(kSpecExample)->span_id;
  data.has_parent = true;
  data.name = "POST /v1/chat/completions";
  data.kind = SpanKind::kServer;
  data.start_unix_ns = 1700000000000000000ull;
  data.end_unix_ns = 1700000000001000000ull;
  data.status = StatusCode::kError;
  data.status_message = "upstream-503";
  data.attributes = {{"vanguard.tenant", std::string("acme")},
                     {"http.response.status_code", std::int64_t{503}},
                     {"vanguard.semantic.similarity", 0.97},
                     {"vanguard.stream", true}};
  const auto j = nlohmann::json::parse(to_otlp_json({data}, t.options()));
  const auto& rs = j["resourceSpans"][0];
  bool has_service = false;
  for (const auto& a : rs["resource"]["attributes"]) {
    if (a["key"] == "service.name") has_service = a["value"]["stringValue"] == "vanguard-proxy";
  }
  EXPECT_TRUE(has_service);
  EXPECT_EQ(rs["scopeSpans"][0]["scope"]["name"], "cloudvanguard.proxy");
  const auto& s = rs["scopeSpans"][0]["spans"][0];
  EXPECT_EQ(s["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_EQ(s["parentSpanId"], "00f067aa0ba902b7");
  EXPECT_EQ(s["spanId"].get<std::string>().size(), 16u);
  EXPECT_EQ(s["kind"], 2);
  EXPECT_EQ(s["startTimeUnixNano"], "1700000000000000000");
  EXPECT_EQ(s["endTimeUnixNano"], "1700000000001000000");
  EXPECT_EQ(s["status"]["code"], 2);
  EXPECT_EQ(s["status"]["message"], "upstream-503");
  EXPECT_EQ(s["attributes"][0]["value"]["stringValue"], "acme");
  EXPECT_EQ(s["attributes"][1]["value"]["intValue"], "503");  // proto3 JSON: int64 as string
  EXPECT_DOUBLE_EQ(s["attributes"][2]["value"]["doubleValue"].get<double>(), 0.97);
  EXPECT_EQ(s["attributes"][3]["value"]["boolValue"], true);
}

TEST(Otlp, InvalidUtf8AndControlCharactersStillProduceValidJson) {
  SpanData data;
  data.context.trace_id[0] = 1;
  data.context.span_id[0] = 1;
  data.name = "POST /v1/chat/completions";
  data.attributes = {
      {"url.path", std::string("/v1/\xff\xfe/\xc3(/\xed\xa0\x80/\xf0\x9f\x98\x80/\"q\"\\\n\x01")},
      {"ok", std::string("caf\xc3\xa9 \xe2\x82\xac")}};
  const std::string json = to_otlp_json({data}, TracerOptions{});
  nlohmann::json j;
  ASSERT_NO_THROW(j = nlohmann::json::parse(json)) << json;  // nlohmann rejects invalid UTF-8
  const auto& attrs = j["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"];
  // Invalid bytes, a truncated sequence and an encoded surrogate become U+FFFD;
  // a valid 4-byte emoji, quotes, backslash and control characters survive.
  EXPECT_EQ(attrs[0]["value"]["stringValue"].get<std::string>(),
            "/v1/\xef\xbf\xbd\xef\xbf\xbd/\xef\xbf\xbd(/\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd/\xf0\x9f\x98\x80/\"q\"\\\n\x01");
  EXPECT_EQ(attrs[1]["value"]["stringValue"].get<std::string>(), "caf\xc3\xa9 \xe2\x82\xac");
}

TEST(Exporter, BatchesSpansToAnOtlpEndpoint) {
  OtlpSink sink;
  auto o = enabled_options(sink.url());
  o.max_batch = 3;
  Tracer t(o);
  for (int i = 0; i < 7; ++i) {
    auto s = t.start_span("span-" + std::to_string(i), SpanKind::kServer, "");
    t.end(s);
    t.end(s);  // idempotent
  }
  EXPECT_EQ(t.flush(), 7u);
  const auto heads = sink.heads();
  ASSERT_EQ(heads.size(), 3u);  // 3 + 3 + 1
  EXPECT_NE(heads[0].find("POST /v1/traces HTTP/1.1"), std::string::npos);
  EXPECT_NE(heads[0].find("Content-Type: application/json"), std::string::npos);
  EXPECT_EQ(sink.span_count(), 7u);
  const auto st = t.stats();
  EXPECT_EQ(st.exported, 7u);
  EXPECT_EQ(st.batches, 3u);
  EXPECT_EQ(st.failed, 0u);
  EXPECT_NE(t.render_metrics().find("vanguard_trace_spans_exported_total 7"), std::string::npos);
}

TEST(Exporter, BackgroundThreadExportsOnScheduleAndOnShutdown) {
  OtlpSink sink;
  auto o = enabled_options(sink.url());
  o.schedule_delay_ms = 50;
  Tracer t(o);
  t.start();
  {
    auto s = t.start_span("scheduled", SpanKind::kServer, "");
    t.end(s);
  }
  for (int i = 0; i < 100 && t.stats().exported < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(t.stats().exported, 1u);
  {
    auto s = t.start_span("at-shutdown", SpanKind::kServer, "");
    t.end(s);
  }
  t.shutdown();
  EXPECT_EQ(t.stats().exported, 2u);
  EXPECT_EQ(sink.span_count(), 2u);
}

TEST(Exporter, FullQueueDropsAndCounts) {
  OtlpSink sink;
  auto o = enabled_options(sink.url());
  o.max_queue = 4;
  o.max_batch = 4;
  Tracer t(o);
  for (int i = 0; i < 10; ++i) {
    auto s = t.start_span("x", SpanKind::kServer, "");
    t.end(s);
  }
  EXPECT_EQ(t.stats().dropped, 6u);
  EXPECT_EQ(t.flush(), 4u);
  EXPECT_EQ(sink.span_count(), 4u);
}

TEST(Exporter, RejectedAndUnreachableEndpointsAreCountedNotFatal) {
  {
    OtlpSink sink(500);
    Tracer t(enabled_options(sink.url()));
    auto s = t.start_span("x", SpanKind::kServer, "");
    t.end(s);
    EXPECT_EQ(t.flush(), 0u);
    EXPECT_EQ(t.stats().failed, 1u);
    EXPECT_EQ(t.stats().export_errors, 1u);
  }
  // Nothing listens on port 9 (discard) in test environments; connect fails fast.
  auto o = enabled_options("http://127.0.0.1:9/v1/traces");
  o.export_timeout_ms = 500;
  Tracer t(o);
  auto s = t.start_span("x", SpanKind::kServer, "");
  t.end(s);
  EXPECT_EQ(t.flush(), 0u);
  EXPECT_EQ(t.stats().failed, 1u);
  const auto r = http_post("http://127.0.0.1:9/v1/traces", "application/json", "{}", 500);
  EXPECT_FALSE(r.ok);
  EXPECT_FALSE(r.error.empty());
}

TEST(TracerOptions, FollowsOtelEnvironmentSemantics) {
  std::string err;
  auto o = tracer_options("", "", "", "n1", "", "", 5000, 2048, 512, 10000, err);
  ASSERT_TRUE(o);
  EXPECT_FALSE(o->enabled);

  o = tracer_options("http://collector:4318/", "", "svc", "n1", "parentbased_traceidratio", "0.1", 100, 2048, 512,
                     10000, err);
  ASSERT_TRUE(o) << err;
  EXPECT_TRUE(o->enabled);
  EXPECT_EQ(o->traces_url, "http://collector:4318/v1/traces");
  EXPECT_EQ(o->service_name, "svc");
  EXPECT_EQ(o->service_instance, "n1");
  EXPECT_EQ(o->sampler, Sampler::kParentTraceIdRatio);
  EXPECT_DOUBLE_EQ(o->ratio, 0.1);

  o = tracer_options("http://a:4318", "http://b:4318/custom", "", "n1", "", "", 5000, 2048, 512, 10000, err);
  ASSERT_TRUE(o);
  EXPECT_EQ(o->traces_url, "http://b:4318/custom");  // signal-specific endpoint wins, used as-is

  EXPECT_FALSE(tracer_options("https://a:4318", "", "", "n1", "", "", 5000, 2048, 512, 10000, err));
  EXPECT_FALSE(tracer_options("http://a:4318", "", "", "n1", "sometimes", "", 5000, 2048, 512, 10000, err));
  EXPECT_FALSE(tracer_options("http://a:4318", "", "", "n1", "traceidratio", "1.5", 5000, 2048, 512, 10000, err));
  EXPECT_FALSE(tracer_options("http://a:4318", "", "", "n1", "", "", 5000, 16, 512, 10000, err));
}

TEST(TracerOptions, LoadConfigValidatesOtelSettings) {
  std::string err;
  const char* ok[] = {"vanguard-proxy", "--otel-exporter-otlp-endpoint=http://127.0.0.1:4318",
                      "--otel-traces-sampler=traceidratio", "--otel-traces-sampler-arg=0.5"};
  auto cfg = vanguard::load_config(4, ok, err);
  ASSERT_TRUE(cfg) << err;
  const auto o = vanguard::tracing_options(*cfg, err);
  ASSERT_TRUE(o);
  EXPECT_TRUE(o->enabled);
  EXPECT_DOUBLE_EQ(o->ratio, 0.5);

  const char* bad[] = {"vanguard-proxy", "--otel-exporter-otlp-endpoint=https://collector:4318"};
  EXPECT_FALSE(vanguard::load_config(2, bad, err));
  EXPECT_NE(err.find("OTLP"), std::string::npos);
}
