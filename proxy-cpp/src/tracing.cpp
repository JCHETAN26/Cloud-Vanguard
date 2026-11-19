#include "vanguard/tracing.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>

#include "vanguard/config.hpp"

namespace vanguard::tracing {
namespace {

constexpr char kHex[] = "0123456789abcdef";

int hex_value(char c) noexcept {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;  // uppercase is invalid in traceparent
}

template <std::size_t N>
bool parse_hex(std::string_view s, std::array<std::uint8_t, N>& out) noexcept {
  if (s.size() != 2 * N) return false;
  for (std::size_t i = 0; i < N; ++i) {
    const int hi = hex_value(s[2 * i]);
    const int lo = hex_value(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
  }
  return true;
}

template <std::size_t N>
bool all_zero(const std::array<std::uint8_t, N>& a) noexcept {
  for (auto b : a) {
    if (b != 0) return false;
  }
  return true;
}

std::uint64_t now_unix_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}

std::mt19937_64& rng() {
  thread_local std::mt19937_64 gen([] {
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), rd(), static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
                      static_cast<unsigned>(now_unix_ns())};
    return std::mt19937_64(seq);
  }());
  return gen;
}

template <std::size_t N>
void random_id(std::array<std::uint8_t, N>& out) {
  do {
    for (std::size_t i = 0; i < N; i += 8) {
      std::uint64_t v = rng()();
      for (std::size_t j = 0; j < 8 && i + j < N; ++j) {
        out[i + j] = static_cast<std::uint8_t>(v & 0xff);
        v >>= 8;
      }
    }
  } while (all_zero(out));
}

// JSON string with UTF-8 validation: span attributes carry client-controlled
// bytes (url.path), and invalid UTF-8 must neither break the collector's
// parser nor throw. Invalid sequences become U+FFFD.
void append_json_string(std::string& out, std::string_view v) {
  out.push_back('"');
  const auto* p = reinterpret_cast<const unsigned char*>(v.data());
  const std::size_t n = v.size();
  for (std::size_t i = 0; i < n;) {
    const unsigned char c = p[i];
    if (c < 0x80) {
      switch (c) {
        case '"': out.append("\\\""); break;
        case '\\': out.append("\\\\"); break;
        case '\n': out.append("\\n"); break;
        case '\r': out.append("\\r"); break;
        case '\t': out.append("\\t"); break;
        default:
          if (c < 0x20) {
            out.append("\\u00");
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 0x0f]);
          } else {
            out.push_back(static_cast<char>(c));
          }
      }
      ++i;
      continue;
    }
    // Multi-byte sequence: validate length, continuation bytes, overlongs,
    // surrogates and the U+10FFFF limit.
    std::size_t len = 0;
    std::uint32_t cp = 0;
    if (c >= 0xc2 && c <= 0xdf) {
      len = 2;
      cp = c & 0x1f;
    } else if (c >= 0xe0 && c <= 0xef) {
      len = 3;
      cp = c & 0x0f;
    } else if (c >= 0xf0 && c <= 0xf4) {
      len = 4;
      cp = c & 0x07;
    }
    bool ok = len != 0 && i + len <= n;
    for (std::size_t k = 1; ok && k < len; ++k) {
      ok = (p[i + k] & 0xc0) == 0x80;
      cp = (cp << 6) | (p[i + k] & 0x3f);
    }
    if (ok) {
      ok = !(len == 3 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) && !(len == 4 && (cp < 0x10000 || cp > 0x10ffff));
    }
    if (ok) {
      out.append(v.data() + i, len);
      i += len;
    } else {
      out.append("\xef\xbf\xbd");  // U+FFFD
      ++i;
    }
  }
  out.push_back('"');
}

void append_attr(std::string& out, std::string_view key, const AttrValue& v) {
  out.append("{\"key\":");
  append_json_string(out, key);
  out.append(",\"value\":{");
  if (const auto* s = std::get_if<std::string>(&v)) {
    out.append("\"stringValue\":");
    append_json_string(out, *s);
  } else if (const auto* i = std::get_if<std::int64_t>(&v)) {
    out.append("\"intValue\":\"").append(std::to_string(*i)).append("\"");  // proto3 JSON: int64 as string
  } else if (const auto* d = std::get_if<double>(&v)) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", std::isfinite(*d) ? *d : 0.0);
    out.append("\"doubleValue\":").append(buf);
  } else {
    out.append("\"boolValue\":").append(std::get<bool>(v) ? "true" : "false");
  }
  out.append("}}");
}

void counter(std::string& out, const char* name, const char* help, std::uint64_t v) {
  out.append("# HELP ").append(name).append(" ").append(help).append("\n# TYPE ").append(name).append(" counter\n");
  out.append(name).append(" ").append(std::to_string(v)).append("\n");
}

void gauge(std::string& out, const char* name, const char* help, std::uint64_t v) {
  out.append("# HELP ").append(name).append(" ").append(help).append("\n# TYPE ").append(name).append(" gauge\n");
  out.append(name).append(" ").append(std::to_string(v)).append("\n");
}

// Waits for `fd` to become ready for `events` before the deadline.
bool wait_fd(int fd, short events, std::chrono::steady_clock::time_point deadline) {
  for (;;) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (left.count() <= 0) return false;
    pollfd p{fd, events, 0};
    const int rc = ::poll(&p, 1, static_cast<int>(left.count()));
    if (rc > 0) return true;
    if (rc == 0) return false;
    if (errno != EINTR) return false;
  }
}

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

}  // namespace

bool SpanContext::valid() const noexcept { return !all_zero(trace_id) && !all_zero(span_id); }

std::string to_hex(const std::uint8_t* data, std::size_t n) {
  std::string out(2 * n, '0');
  for (std::size_t i = 0; i < n; ++i) {
    out[2 * i] = kHex[data[i] >> 4];
    out[2 * i + 1] = kHex[data[i] & 0x0f];
  }
  return out;
}

std::optional<SpanContext> parse_traceparent(std::string_view h) {
  // Optional whitespace from header folding.
  while (!h.empty() && (h.front() == ' ' || h.front() == '\t')) h.remove_prefix(1);
  while (!h.empty() && (h.back() == ' ' || h.back() == '\t')) h.remove_suffix(1);
  // version(2) - trace-id(32) - parent-id(16) - flags(2) = 55; future versions may append "-..."
  if (h.size() < 55 || h[2] != '-' || h[35] != '-' || h[52] != '-') return std::nullopt;
  std::array<std::uint8_t, 1> version{};
  std::array<std::uint8_t, 1> flags{};
  SpanContext ctx;
  if (!parse_hex(h.substr(0, 2), version) || version[0] == 0xff) return std::nullopt;
  if (version[0] == 0x00 && h.size() != 55) return std::nullopt;
  if (version[0] != 0x00 && h.size() > 55 && h[55] != '-') return std::nullopt;
  if (!parse_hex(h.substr(3, 32), ctx.trace_id) || !parse_hex(h.substr(36, 16), ctx.span_id) ||
      !parse_hex(h.substr(53, 2), flags)) {
    return std::nullopt;
  }
  if (!ctx.valid()) return std::nullopt;
  ctx.sampled = (flags[0] & 0x01) != 0;
  return ctx;
}

std::string format_traceparent(const SpanContext& ctx) {
  std::string out;
  out.reserve(55);
  out.append("00-").append(to_hex(ctx.trace_id.data(), ctx.trace_id.size())).append("-");
  out.append(to_hex(ctx.span_id.data(), ctx.span_id.size())).append(ctx.sampled ? "-01" : "-00");
  return out;
}

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

void Span::set(std::string_view key, AttrValue value) {
  if (!data_) return;
  for (auto& [k, v] : data_->attributes) {
    if (k == key) {
      v = std::move(value);
      return;
    }
  }
  data_->attributes.emplace_back(std::string(key), std::move(value));
}

void Span::set_attribute(std::string_view key, std::string_view value) {
  if (data_) set(key, std::string(value));
}
void Span::set_attribute(std::string_view key, std::int64_t value) { set(key, value); }
void Span::set_attribute(std::string_view key, double value) { set(key, value); }
void Span::set_attribute(std::string_view key, bool value) { set(key, value); }

void Span::set_status(StatusCode code, std::string_view message) {
  if (!data_ || code == StatusCode::kUnset || data_->status == StatusCode::kOk) return;
  data_->status = code;
  data_->status_message = code == StatusCode::kError ? std::string(message) : std::string();
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

std::optional<Sampler> parse_sampler(std::string_view name) {
  if (name == "always_on") return Sampler::kAlwaysOn;
  if (name == "always_off") return Sampler::kAlwaysOff;
  if (name == "traceidratio") return Sampler::kTraceIdRatio;
  if (name == "parentbased_always_on" || name.empty()) return Sampler::kParentAlwaysOn;
  if (name == "parentbased_always_off") return Sampler::kParentAlwaysOff;
  if (name == "parentbased_traceidratio") return Sampler::kParentTraceIdRatio;
  return std::nullopt;
}

bool ratio_sampled(const TraceId& id, double ratio) noexcept {
  if (!(ratio > 0.0)) return false;
  if (ratio >= 1.0) return true;
  std::uint64_t v = 0;
  for (std::size_t i = 8; i < 16; ++i) v = (v << 8) | id[i];
  // ratio * 2^64 without overflow: compare in long double.
  return static_cast<long double>(v) < static_cast<long double>(ratio) * 18446744073709551616.0L;
}

// ---------------------------------------------------------------------------
// OTLP/JSON + HTTP
// ---------------------------------------------------------------------------

// Serialized directly into one buffer (no JSON DOM): at 100% sampling the
// exporter thread handles tens of thousands of spans per second.
std::string to_otlp_json(const std::vector<SpanData>& spans, const TracerOptions& r) {
  std::string out;
  out.reserve(512 + spans.size() * 900);
  out.append("{\"resourceSpans\":[{\"resource\":{\"attributes\":[");
  append_attr(out, "service.name", r.service_name);
  out.push_back(',');
  append_attr(out, "service.version", r.service_version);
  if (!r.service_instance.empty()) {
    out.push_back(',');
    append_attr(out, "service.instance.id", r.service_instance);
  }
  out.push_back(',');
  append_attr(out, "telemetry.sdk.name", std::string("cloudvanguard-otlp"));
  out.push_back(',');
  append_attr(out, "telemetry.sdk.language", std::string("cpp"));
  out.append("]},\"scopeSpans\":[{\"scope\":{\"name\":\"cloudvanguard.proxy\",\"version\":");
  append_json_string(out, r.service_version);
  out.append("},\"spans\":[");
  bool first = true;
  for (const auto& s : spans) {
    if (!first) out.push_back(',');
    first = false;
    out.append("{\"traceId\":\"").append(to_hex(s.context.trace_id.data(), s.context.trace_id.size()));
    out.append("\",\"spanId\":\"").append(to_hex(s.context.span_id.data(), s.context.span_id.size())).append("\"");
    if (s.has_parent) {
      out.append(",\"parentSpanId\":\"").append(to_hex(s.parent_span_id.data(), s.parent_span_id.size())).append("\"");
    }
    out.append(",\"name\":");
    append_json_string(out, s.name);
    out.append(",\"kind\":").append(std::to_string(static_cast<int>(s.kind)));
    out.append(",\"startTimeUnixNano\":\"").append(std::to_string(s.start_unix_ns));
    out.append("\",\"endTimeUnixNano\":\"").append(std::to_string(s.end_unix_ns)).append("\",\"attributes\":[");
    for (std::size_t a = 0; a < s.attributes.size(); ++a) {
      if (a > 0) out.push_back(',');
      append_attr(out, s.attributes[a].first, s.attributes[a].second);
    }
    out.append("],\"status\":{\"code\":").append(std::to_string(static_cast<int>(s.status)));
    if (!s.status_message.empty()) {
      out.append(",\"message\":");
      append_json_string(out, s.status_message);
    }
    out.append("}}");
  }
  out.append("]}]}]}");
  return out;
}

ExportResult http_post(std::string_view url, std::string_view content_type, std::string_view body,
                       unsigned timeout_ms) {
  ExportResult res;
  std::string err;
  const auto u = parse_upstream_url(url, err);
  if (!u) {
    res.error = err;
    return res;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* ai = nullptr;
  const std::string port = std::to_string(u->port);
  if (const int rc = ::getaddrinfo(u->host.c_str(), port.c_str(), &hints, &ai); rc != 0) {
    res.error = std::string("resolve ") + u->host + ": " + ::gai_strerror(rc);
    return res;
  }
  int fd = -1;
  for (addrinfo* p = ai; p != nullptr && fd < 0; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    ::fcntl(fd, F_SETFD, FD_CLOEXEC);
#ifdef SO_NOSIGPIPE
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = ::connect(fd, p->ai_addr, p->ai_addrlen);
    if (rc < 0 && errno == EINPROGRESS) {
      int so_err = 0;
      socklen_t len = sizeof(so_err);
      rc = wait_fd(fd, POLLOUT, deadline) && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) == 0 && so_err == 0
               ? 0
               : -1;
      if (rc < 0) errno = so_err != 0 ? so_err : ETIMEDOUT;
    }
    if (rc < 0) {
      res.error = std::string("connect: ") + std::strerror(errno);
      ::close(fd);
      fd = -1;
    }
  }
  ::freeaddrinfo(ai);
  if (fd < 0) return res;

  std::string req;
  req.reserve(256 + body.size());
  req.append("POST ").append(u->base_path.empty() ? "/" : u->base_path).append(" HTTP/1.1\r\nHost: ");
  req.append(u->host).append(":").append(port).append("\r\nContent-Type: ").append(content_type);
  req.append("\r\nContent-Length: ").append(std::to_string(body.size()));
  req.append("\r\nUser-Agent: cloudvanguard-otlp/0.5\r\nConnection: close\r\n\r\n").append(body);

  std::size_t sent = 0;
  while (sent < req.size()) {
    const ssize_t w = ::send(fd, req.data() + sent, req.size() - sent, kSendFlags);
    if (w > 0) {
      sent += static_cast<std::size_t>(w);
      continue;
    }
    if (w < 0 && errno == EINTR) continue;
    if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) && wait_fd(fd, POLLOUT, deadline)) continue;
    res.error = w < 0 ? std::string("send: ") + std::strerror(errno) : "send: timeout";
    ::close(fd);
    return res;
  }

  std::string resp;
  char buf[4096];
  for (;;) {
    const ssize_t r = ::recv(fd, buf, sizeof(buf), 0);
    if (r > 0) {
      if (resp.size() < 65536) resp.append(buf, static_cast<std::size_t>(r));
      continue;
    }
    if (r == 0) break;
    if (errno == EINTR) continue;
    if ((errno == EAGAIN || errno == EWOULDBLOCK) && wait_fd(fd, POLLIN, deadline)) continue;
    if (resp.find("\r\n") == std::string::npos) {
      res.error = "recv: timeout or error before status line";
      ::close(fd);
      return res;
    }
    break;
  }
  ::close(fd);
  // "HTTP/1.1 200 OK"
  if (resp.size() < 12 || resp.compare(0, 5, "HTTP/") != 0) {
    res.error = "malformed response";
    return res;
  }
  const auto sp = resp.find(' ');
  res.status = std::atoi(resp.c_str() + sp + 1);
  res.ok = res.status >= 200 && res.status < 300;
  if (!res.ok) res.error = "HTTP " + std::to_string(res.status);
  return res;
}

// ---------------------------------------------------------------------------
// Tracer
// ---------------------------------------------------------------------------

Tracer::Tracer(TracerOptions options) : opts_(std::move(options)) {}

Tracer::~Tracer() { shutdown(); }

void Tracer::start() {
  if (!opts_.enabled) return;
  std::lock_guard lock(mu_);
  if (started_ || stop_) return;
  started_ = true;
  thread_ = std::thread([this] { run(); });
}

void Tracer::shutdown() {
  {
    std::lock_guard lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  if (opts_.enabled) flush();
}

bool Tracer::sample_root(const TraceId& id) const noexcept {
  switch (opts_.sampler) {
    case Sampler::kAlwaysOn:
    case Sampler::kParentAlwaysOn:
      return true;
    case Sampler::kAlwaysOff:
    case Sampler::kParentAlwaysOff:
      return false;
    case Sampler::kTraceIdRatio:
    case Sampler::kParentTraceIdRatio:
      return ratio_sampled(id, opts_.ratio);
  }
  return false;
}

Span Tracer::start_span(std::string_view name, SpanKind kind, std::string_view traceparent) {
  Span s;
  if (!opts_.enabled) return s;
  s.active_ = true;
  const auto parent = traceparent.empty() ? std::nullopt : parse_traceparent(traceparent);
  const bool parent_based = opts_.sampler == Sampler::kParentAlwaysOn || opts_.sampler == Sampler::kParentAlwaysOff ||
                            opts_.sampler == Sampler::kParentTraceIdRatio;
  if (parent) {
    s.ctx_.trace_id = parent->trace_id;
    s.ctx_.sampled = parent_based ? parent->sampled : sample_root(parent->trace_id);
  } else {
    random_id(s.ctx_.trace_id);
    s.ctx_.sampled = sample_root(s.ctx_.trace_id);
  }
  random_id(s.ctx_.span_id);
  if (!s.ctx_.sampled) return s;
  s.data_ = std::make_unique<SpanData>();
  s.data_->context = s.ctx_;
  if (parent) {
    s.data_->parent_span_id = parent->span_id;
    s.data_->has_parent = true;
  }
  s.data_->name = std::string(name);
  s.data_->kind = kind;
  s.data_->start_unix_ns = now_unix_ns();
  s.data_->attributes.reserve(16);
  started_spans_.fetch_add(1, std::memory_order_relaxed);
  return s;
}

Span Tracer::start_child(const Span& parent, std::string_view name, SpanKind kind) {
  Span s;
  if (!opts_.enabled || !parent.active_) return s;
  s.active_ = true;
  s.ctx_.trace_id = parent.ctx_.trace_id;
  s.ctx_.sampled = parent.ctx_.sampled;
  random_id(s.ctx_.span_id);
  if (!s.ctx_.sampled) return s;
  s.data_ = std::make_unique<SpanData>();
  s.data_->context = s.ctx_;
  s.data_->parent_span_id = parent.ctx_.span_id;
  s.data_->has_parent = true;
  s.data_->name = std::string(name);
  s.data_->kind = kind;
  s.data_->start_unix_ns = now_unix_ns();
  s.data_->attributes.reserve(8);
  started_spans_.fetch_add(1, std::memory_order_relaxed);
  return s;
}

void Tracer::end(Span& span) {
  if (!span.active_) return;
  span.active_ = false;
  if (!span.data_) return;
  span.data_->end_unix_ns = now_unix_ns();
  bool notify = false;
  {
    std::lock_guard lock(mu_);
    if (queue_.size() >= opts_.max_queue) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
    } else {
      queue_.push_back(std::move(*span.data_));
      notify = queue_.size() >= opts_.max_batch;
    }
  }
  span.data_.reset();
  if (notify) cv_.notify_one();
}

std::size_t Tracer::export_batch(std::vector<SpanData>& spans) {
  std::size_t sent = 0;
  for (std::size_t i = 0; i < spans.size(); i += opts_.max_batch) {
    const std::size_t n = std::min(opts_.max_batch, spans.size() - i);
    std::vector<SpanData> chunk(std::make_move_iterator(spans.begin() + static_cast<std::ptrdiff_t>(i)),
                                std::make_move_iterator(spans.begin() + static_cast<std::ptrdiff_t>(i + n)));
    const auto res = http_post(opts_.traces_url, "application/json", to_otlp_json(chunk, opts_), opts_.export_timeout_ms);
    batches_.fetch_add(1, std::memory_order_relaxed);
    if (res.ok) {
      exported_.fetch_add(n, std::memory_order_relaxed);
      sent += n;
    } else {
      failed_.fetch_add(n, std::memory_order_relaxed);
      export_errors_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return sent;
}

std::size_t Tracer::flush() {
  if (!opts_.enabled) return 0;
  std::lock_guard export_lock(export_mu_);
  std::vector<SpanData> batch;
  {
    std::lock_guard lock(mu_);
    batch.swap(queue_);
  }
  return batch.empty() ? 0 : export_batch(batch);
}

void Tracer::run() {
  const auto delay = std::chrono::milliseconds(opts_.schedule_delay_ms);
  std::unique_lock lock(mu_);
  for (;;) {
    cv_.wait_for(lock, delay, [this] { return stop_ || queue_.size() >= opts_.max_batch; });
    if (queue_.empty()) {
      if (stop_) return;
      continue;
    }
    std::vector<SpanData> batch;
    batch.swap(queue_);
    lock.unlock();
    {
      std::lock_guard export_lock(export_mu_);
      export_batch(batch);
    }
    lock.lock();
    if (stop_ && queue_.empty()) return;
  }
}

Tracer::Stats Tracer::stats() const noexcept {
  Stats s;
  s.started = started_spans_.load(std::memory_order_relaxed);
  s.exported = exported_.load(std::memory_order_relaxed);
  s.dropped = dropped_.load(std::memory_order_relaxed);
  s.failed = failed_.load(std::memory_order_relaxed);
  s.batches = batches_.load(std::memory_order_relaxed);
  s.export_errors = export_errors_.load(std::memory_order_relaxed);
  return s;
}

std::string Tracer::render_metrics() const {
  const auto s = stats();
  std::size_t queued = 0;
  {
    std::lock_guard lock(mu_);
    queued = queue_.size();
  }
  std::string out;
  gauge(out, "vanguard_tracing_enabled", "1 when spans are exported over OTLP.", opts_.enabled ? 1 : 0);
  counter(out, "vanguard_trace_spans_started_total", "Sampled spans started.", s.started);
  counter(out, "vanguard_trace_spans_exported_total", "Spans accepted by the OTLP endpoint.", s.exported);
  counter(out, "vanguard_trace_spans_dropped_total", "Spans dropped because the export queue was full.", s.dropped);
  counter(out, "vanguard_trace_spans_failed_total", "Spans in batches the OTLP endpoint rejected or never received.",
          s.failed);
  counter(out, "vanguard_trace_export_batches_total", "OTLP export requests sent.", s.batches);
  counter(out, "vanguard_trace_export_errors_total", "OTLP export requests that failed.", s.export_errors);
  gauge(out, "vanguard_trace_queue_spans", "Ended spans waiting for export.", queued);
  return out;
}

std::optional<TracerOptions> tracer_options(std::string_view otlp_endpoint, std::string_view traces_endpoint,
                                            std::string_view service_name, std::string_view node_id,
                                            std::string_view sampler, std::string_view sampler_arg,
                                            unsigned schedule_delay_ms, std::size_t max_queue,
                                            std::size_t max_batch, unsigned export_timeout_ms, std::string& error) {
  TracerOptions o;
  if (!traces_endpoint.empty()) {
    o.traces_url = std::string(traces_endpoint);
  } else if (!otlp_endpoint.empty()) {
    std::string base(otlp_endpoint);
    while (!base.empty() && base.back() == '/') base.pop_back();
    o.traces_url = base + "/v1/traces";
  }
  o.enabled = !o.traces_url.empty();
  if (o.enabled) {
    std::string err;
    if (!parse_upstream_url(o.traces_url, err)) {
      error = "OTLP traces endpoint: " + err + " (use an http:// collector; terminate TLS in the collector)";
      return std::nullopt;
    }
  }
  if (!service_name.empty()) o.service_name = std::string(service_name);
  o.service_instance = std::string(node_id);
  const auto s = parse_sampler(sampler);
  if (!s) {
    error = "OTEL_TRACES_SAMPLER must be one of always_on, always_off, traceidratio, parentbased_always_on, "
            "parentbased_always_off, parentbased_traceidratio";
    return std::nullopt;
  }
  o.sampler = *s;
  if (!sampler_arg.empty() && (o.sampler == Sampler::kTraceIdRatio || o.sampler == Sampler::kParentTraceIdRatio)) {
    char* end = nullptr;
    const std::string arg(sampler_arg);
    const double r = std::strtod(arg.c_str(), &end);
    if (end == arg.c_str() || *end != '\0' || !(r >= 0.0 && r <= 1.0)) {
      error = "OTEL_TRACES_SAMPLER_ARG must be a ratio in [0, 1]";
      return std::nullopt;
    }
    o.ratio = r;
  }
  if (max_batch == 0 || max_queue == 0 || max_batch > max_queue) {
    error = "OTEL_BSP_MAX_EXPORT_BATCH_SIZE must be in [1, OTEL_BSP_MAX_QUEUE_SIZE]";
    return std::nullopt;
  }
  o.schedule_delay_ms = schedule_delay_ms;
  o.max_queue = max_queue;
  o.max_batch = max_batch;
  o.export_timeout_ms = export_timeout_ms;
  return o;
}

}  // namespace vanguard::tracing
