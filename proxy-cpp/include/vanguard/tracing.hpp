#pragma once

// OpenTelemetry-compatible distributed tracing for the data plane, without
// third-party dependencies:
//
//   * W3C Trace Context (`traceparent`) parsing and propagation: an incoming
//     header makes the gateway's span a child of the caller's; every outgoing
//     call (LLM upstream, embedding sidecar, peer node) carries a new
//     `traceparent`, so a request forwarded across the ring is one trace.
//   * OpenTelemetry sampler semantics (always_on / always_off / traceidratio
//     and their parentbased_* variants, OTEL_TRACES_SAMPLER[_ARG]).
//   * A batch span processor (OTEL_BSP_*) with a bounded queue: ending a span
//     never blocks the event loop and never allocates a socket; when the queue
//     is full spans are dropped and counted.
//   * An OTLP/HTTP exporter using the protocol's JSON encoding
//     (POST <endpoint>/v1/traces, Content-Type: application/json), accepted by
//     the OpenTelemetry Collector, Jaeger, Tempo and other OTLP receivers.
//
// Spans are recorded only when sampled; unsampled requests still propagate
// their context (flags 00) so downstream services make the same decision.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace vanguard::tracing {

using TraceId = std::array<std::uint8_t, 16>;
using SpanId = std::array<std::uint8_t, 8>;

struct SpanContext {
  TraceId trace_id{};
  SpanId span_id{};
  bool sampled = false;

  [[nodiscard]] bool valid() const noexcept;
};

// W3C Trace Context level 1. Returns nullopt for anything malformed
// (wrong lengths, uppercase hex, version ff, all-zero ids), in which case the
// caller starts a new trace, as the spec requires.
std::optional<SpanContext> parse_traceparent(std::string_view header);
// "00-<trace-id>-<span-id>-<01|00>"
std::string format_traceparent(const SpanContext& ctx);
std::string to_hex(const std::uint8_t* data, std::size_t n);

enum class SpanKind : int { kInternal = 1, kServer = 2, kClient = 3 };
enum class StatusCode : int { kUnset = 0, kOk = 1, kError = 2 };

using AttrValue = std::variant<std::string, std::int64_t, double, bool>;

struct SpanData {
  SpanContext context;
  SpanId parent_span_id{};
  bool has_parent = false;
  std::string name;
  SpanKind kind = SpanKind::kInternal;
  std::uint64_t start_unix_ns = 0;
  std::uint64_t end_unix_ns = 0;
  StatusCode status = StatusCode::kUnset;
  std::string status_message;
  std::vector<std::pair<std::string, AttrValue>> attributes;
};

// A span handle owned by one thread (the worker that owns the connection).
// Inactive handles (tracing disabled) make every call a no-op; unsampled
// handles carry a context for propagation but record nothing.
class Span {
 public:
  Span() = default;
  Span(Span&&) noexcept = default;
  Span& operator=(Span&&) noexcept = default;
  Span(const Span&) = delete;
  Span& operator=(const Span&) = delete;

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] bool recording() const noexcept { return data_ != nullptr; }
  [[nodiscard]] const SpanContext& context() const noexcept { return ctx_; }
  [[nodiscard]] std::string traceparent() const { return active_ ? format_traceparent(ctx_) : std::string(); }

  void set_attribute(std::string_view key, std::string_view value);
  void set_attribute(std::string_view key, const char* value) { set_attribute(key, std::string_view(value)); }
  void set_attribute(std::string_view key, std::int64_t value);
  void set_attribute(std::string_view key, int value) { set_attribute(key, static_cast<std::int64_t>(value)); }
  void set_attribute(std::string_view key, std::uint64_t value) {
    set_attribute(key, static_cast<std::int64_t>(value));
  }
  void set_attribute(std::string_view key, double value);
  void set_attribute(std::string_view key, bool value);
  // OK/ERROR are final; a later kUnset is ignored.
  void set_status(StatusCode code, std::string_view message = {});
  [[nodiscard]] StatusCode status() const noexcept { return data_ ? data_->status : StatusCode::kUnset; }

 private:
  friend class Tracer;
  void set(std::string_view key, AttrValue value);

  bool active_ = false;
  SpanContext ctx_;
  std::unique_ptr<SpanData> data_;
};

enum class Sampler { kAlwaysOn, kAlwaysOff, kTraceIdRatio, kParentAlwaysOn, kParentAlwaysOff, kParentTraceIdRatio };
std::optional<Sampler> parse_sampler(std::string_view name);
// TraceIdRatioBased: the lower 8 bytes of the trace id (big-endian) compared
// against ratio * 2^64, so every service with the same ratio agrees.
bool ratio_sampled(const TraceId& id, double ratio) noexcept;

struct TracerOptions {
  bool enabled = false;
  std::string traces_url;  // full URL, e.g. http://otel-collector:4318/v1/traces
  std::string service_name = "vanguard-proxy";
  std::string service_version = "0.5.0";
  std::string service_instance;  // node id
  Sampler sampler = Sampler::kParentAlwaysOn;
  double ratio = 1.0;
  unsigned schedule_delay_ms = 5000;
  std::size_t max_queue = 2048;
  std::size_t max_batch = 512;
  unsigned export_timeout_ms = 10000;
};

// Serializes spans as an OTLP ExportTraceServiceRequest (JSON encoding).
std::string to_otlp_json(const std::vector<SpanData>& spans, const TracerOptions& resource);

struct ExportResult {
  bool ok = false;
  int status = 0;  // HTTP status, 0 = transport error
  std::string error;
};
// Minimal blocking HTTP/1.1 POST (exporter thread only).
ExportResult http_post(std::string_view url, std::string_view content_type, std::string_view body,
                       unsigned timeout_ms);

class Tracer {
 public:
  explicit Tracer(TracerOptions options);
  ~Tracer();
  Tracer(const Tracer&) = delete;
  Tracer& operator=(const Tracer&) = delete;

  [[nodiscard]] bool enabled() const noexcept { return opts_.enabled; }
  [[nodiscard]] const TracerOptions& options() const noexcept { return opts_; }

  // Starts the background exporter (no-op when disabled or already started).
  void start();
  // Exports everything queued, then stops the exporter. Idempotent.
  void shutdown();
  // Synchronously exports queued spans (tests, shutdown). Returns spans sent.
  std::size_t flush();

  // Server span for an incoming request; `traceparent` may be empty/invalid.
  Span start_span(std::string_view name, SpanKind kind, std::string_view traceparent);
  // Child span in the same trace with the parent's sampling decision.
  Span start_child(const Span& parent, std::string_view name, SpanKind kind);
  // Ends the span (records end time, enqueues if sampled). Idempotent.
  void end(Span& span);

  struct Stats {
    std::uint64_t started = 0;   // sampled spans
    std::uint64_t exported = 0;
    std::uint64_t dropped = 0;   // queue full
    std::uint64_t failed = 0;    // spans in batches the endpoint rejected
    std::uint64_t batches = 0;
    std::uint64_t export_errors = 0;
  };
  [[nodiscard]] Stats stats() const noexcept;
  // Prometheus exposition lines for /metrics.
  [[nodiscard]] std::string render_metrics() const;

 private:
  bool sample_root(const TraceId& id) const noexcept;
  void run();
  std::size_t export_batch(std::vector<SpanData>& batch);

  const TracerOptions opts_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<SpanData> queue_;
  bool stop_ = false;
  bool started_ = false;
  std::mutex export_mu_;  // serializes flush() with the exporter thread
  std::thread thread_;

  std::atomic<std::uint64_t> started_spans_{0};
  std::atomic<std::uint64_t> exported_{0};
  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<std::uint64_t> failed_{0};
  std::atomic<std::uint64_t> batches_{0};
  std::atomic<std::uint64_t> export_errors_{0};
};

// Builds TracerOptions from the standard OTEL_* settings carried in Config.
// OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is used as-is; otherwise
// OTEL_EXPORTER_OTLP_ENDPOINT + "/v1/traces". No endpoint = tracing disabled.
std::optional<TracerOptions> tracer_options(std::string_view otlp_endpoint, std::string_view traces_endpoint,
                                            std::string_view service_name, std::string_view node_id,
                                            std::string_view sampler, std::string_view sampler_arg,
                                            unsigned schedule_delay_ms, std::size_t max_queue,
                                            std::size_t max_batch, unsigned export_timeout_ms, std::string& error);

}  // namespace vanguard::tracing
