// Linux-only data plane: N independent epoll event loops, one per worker
// thread, each with its own SO_REUSEPORT listener and upstream connection pool.
//
// Threading model
//   - A connection (client or upstream) is owned by exactly one worker for its
//     whole life, so per-connection state needs no locks.
//   - Shared state (cache, rate limiter, ring, metrics) is internally
//     synchronized: sharded mutexes for the cache and limiter, a shared_mutex
//     for the ring and relaxed atomics for metrics.
//
// I/O model
//   - All sockets are non-blocking and registered once with
//     EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET. With edge-triggering, every
//     readiness notification is drained until EAGAIN, and output is queued in
//     a per-connection Buffer and flushed until EAGAIN. Registering EPOLLOUT up
//     front (instead of toggling it with epoll_ctl MOD) costs nothing under ET
//     and saves a syscall per partial write.
//   - Connections are addressed through epoll_event.data.u64 = connection id,
//     never raw pointers, and closed connections are reaped only after the
//     current batch of events has been dispatched. A stale event for a
//     connection closed earlier in the same batch is therefore harmless.

#include "vanguard/server.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vanguard/buffer.hpp"
#include "vanguard/http_parser.hpp"
#include "vanguard/llm_request.hpp"
#include "vanguard/sse.hpp"
#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kListenerId = 1;
constexpr std::size_t kReadChunk = 64 * 1024;
constexpr std::size_t kMaxEvents = 512;
constexpr int kEpollTimeoutMs = 250;
// Streaming backpressure: stop reading the upstream while a slow client has
// more than kHighWater bytes queued; resume below kLowWater.
constexpr std::size_t kClientHighWater = 4 * 1024 * 1024;
constexpr std::size_t kClientLowWater = 1024 * 1024;
constexpr std::size_t kMaxUpstreamBody = 32 * 1024 * 1024;
constexpr std::string_view kJson = "application/json";

std::atomic<bool> g_stop{false};
extern "C" void handle_stop_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

std::string json_error(std::string_view message, std::string_view type, std::string_view code) {
  nlohmann::json j;
  j["error"] = {{"message", message}, {"type", type}, {"code", code}};
  return j.dump();
}

void set_nodelay(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// ---------------------------------------------------------------------------
// Connection state
// ---------------------------------------------------------------------------

enum class ConnKind { kClient, kUpstream };

struct Conn {
  explicit Conn(ConnKind k) : kind(k) {}
  virtual ~Conn() = default;
  Conn(const Conn&) = delete;
  Conn& operator=(const Conn&) = delete;

  const ConnKind kind;
  std::uint64_t id = 0;
  int fd = -1;
  bool closed = false;
  Clock::time_point last_active{};
};

// Everything about the chat request currently in flight on a client
// connection (one at a time; pipelined requests wait in the input buffer).
struct InFlight {
  std::string tenant;
  std::string key;      // raw 32-byte cache key
  std::string key_hex;  // short hex prefix for X-Vanguard-Cache-Key
  std::string owner;    // ring owner of the key
  bool stream = false;
  bool store = true;    // false for Cache-Control: no-store
  bool bypass = false;  // Cache-Control: no-cache (skipped lookup)
  std::string upstream_request;
  bool retried = false;
  bool response_started = false;  // SSE headers already sent to the client
  Clock::time_point start{};

  // Stream capture for the cache fill: complete SSE events re-framed as
  // HTTP chunks, ready to be replayed verbatim.
  SseFrameSplitter splitter;
  std::string wire;
  std::size_t frames = 0;
  bool saw_done = false;
  bool cacheable = true;
};

struct ClientConn final : Conn {
  explicit ClientConn(RequestParser::Limits limits) : Conn(ConnKind::kClient), parser(limits) {}

  enum class State { kReading, kUpstream, kClosing };
  State state = State::kReading;
  Buffer in;
  Buffer out;
  RequestParser parser;
  bool keep_alive = true;
  bool peer_closed = false;
  std::uint64_t upstream_id = 0;
  std::unique_ptr<InFlight> req;
};

struct UpstreamConn final : Conn {
  UpstreamConn() : Conn(ConnKind::kUpstream) {}

  enum class State { kConnecting, kActive, kIdle };
  State state = State::kConnecting;
  Buffer out;
  ResponseParser parser;
  std::string decoded;  // decoded body bytes not yet consumed
  std::uint64_t client_id = 0;
  bool reused = false;
  bool headers_handled = false;
  bool paused = false;
  std::size_t bytes_received = 0;
};

// ---------------------------------------------------------------------------
// Worker: one epoll loop
// ---------------------------------------------------------------------------

class Worker {
 public:
  Worker(SharedState& s, int index) : s_(s), index_(index) {
    limits_.max_header_bytes = 16 * 1024;
    limits_.max_body_bytes = s.cfg.max_body_kb * 1024;
  }
  ~Worker() {
    for (auto& [id, c] : conns_) {
      if (!c->closed && c->fd >= 0) ::close(c->fd);
    }
    if (listen_fd_ >= 0) ::close(listen_fd_);
    if (epfd_ >= 0) ::close(epfd_);
  }
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  bool init(std::string& err);
  void run();

 private:
  // Event dispatch
  void on_accept();
  void on_client_event(ClientConn* c, std::uint32_t ev);
  void on_upstream_event(UpstreamConn* u, std::uint32_t ev);

  // Client side
  void process_client(ClientConn* c);
  void handle_request(ClientConn* c, HttpRequest& req);
  void handle_chat(ClientConn* c, HttpRequest& req, Clock::time_point start);
  void serve_hit(ClientConn* c, const CachedResponse& hit, const InFlight& r);
  void respond(ClientConn* c, int status, std::string_view content_type, std::string_view body,
               std::string_view extra_headers = {});
  void respond_error(ClientConn* c, int status, std::string_view message, std::string_view type,
                     std::string_view code, std::string_view extra_headers = {});
  void reject_rate_limited(ClientConn* c, const RateDecision& d, std::string_view which);
  void flush_client(ClientConn* c);
  void finish_exchange(ClientConn* c);

  // Upstream side
  void dispatch_upstream(ClientConn* c);
  UpstreamConn* acquire_upstream();
  void flush_upstream(UpstreamConn* u);
  void read_upstream(UpstreamConn* u);
  void on_upstream_progress(UpstreamConn* u);
  void complete_upstream(UpstreamConn* u, ClientConn* c);
  void upstream_failed(UpstreamConn* u, int status, std::string_view message, bool timeout = false);
  void release_upstream(UpstreamConn* u);

  // Bookkeeping
  template <typename T>
  T* add_conn(std::unique_ptr<T> conn);
  bool watch(Conn* c);
  void close_conn(Conn* c);
  ClientConn* client(std::uint64_t id);
  UpstreamConn* upstream(std::uint64_t id);
  void reap();
  void sweep_timeouts(Clock::time_point now);

  SharedState& s_;
  const int index_;
  RequestParser::Limits limits_;
  int epfd_ = -1;
  int listen_fd_ = -1;
  std::uint64_t next_id_ = kListenerId + 1;
  std::unordered_map<std::uint64_t, std::unique_ptr<Conn>> conns_;
  std::vector<std::uint64_t> idle_upstreams_;
  std::vector<std::uint64_t> graveyard_;
  std::array<char, kReadChunk> scratch_{};
};

bool Worker::init(std::string& err) {
  epfd_ = ::epoll_create1(EPOLL_CLOEXEC);
  if (epfd_ < 0) {
    err = std::string("epoll_create1: ") + std::strerror(errno);
    return false;
  }
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) {
    err = std::string("socket: ") + std::strerror(errno);
    return false;
  }
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  // SO_REUSEPORT: every worker binds its own listener on the same port and the
  // kernel load-balances new connections across them (no accept() thundering
  // herd, no shared accept lock).
  if (::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one)) < 0) {
    err = std::string("SO_REUSEPORT: ") + std::strerror(errno);
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(s_.cfg.listen_port);
  if (::inet_pton(AF_INET, s_.cfg.listen_addr.c_str(), &addr.sin_addr) != 1) {
    err = "LISTEN_ADDR must be an IPv4 address: " + s_.cfg.listen_addr;
    return false;
  }
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    err = "bind " + s_.cfg.listen_addr + ":" + std::to_string(s_.cfg.listen_port) + ": " + std::strerror(errno);
    return false;
  }
  if (::listen(listen_fd_, 4096) < 0) {
    err = std::string("listen: ") + std::strerror(errno);
    return false;
  }
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ev.data.u64 = kListenerId;
  if (::epoll_ctl(epfd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0) {
    err = std::string("epoll_ctl(listener): ") + std::strerror(errno);
    return false;
  }
  return true;
}

void Worker::run() {
  std::array<epoll_event, kMaxEvents> events{};
  auto last_sweep = Clock::now();
  while (!g_stop.load(std::memory_order_relaxed)) {
    const int n = ::epoll_wait(epfd_, events.data(), static_cast<int>(events.size()), kEpollTimeoutMs);
    if (n < 0) {
      if (errno == EINTR) continue;
      std::fprintf(stderr, "[worker %d] epoll_wait: %s\n", index_, std::strerror(errno));
      break;
    }
    const auto now = Clock::now();
    for (int i = 0; i < n; ++i) {
      const std::uint64_t id = events[i].data.u64;
      const std::uint32_t ev = events[i].events;
      if (id == kListenerId) {
        on_accept();
        continue;
      }
      const auto it = conns_.find(id);
      if (it == conns_.end() || it->second->closed) continue;  // closed earlier in this batch
      Conn* c = it->second.get();
      c->last_active = now;
      if (c->kind == ConnKind::kClient) {
        on_client_event(static_cast<ClientConn*>(c), ev);
      } else {
        on_upstream_event(static_cast<UpstreamConn*>(c), ev);
      }
    }
    reap();
    if (now - last_sweep >= std::chrono::seconds(1)) {
      sweep_timeouts(now);
      reap();
      last_sweep = now;
    }
  }
}

// ---------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------

template <typename T>
T* Worker::add_conn(std::unique_ptr<T> conn) {
  conn->id = next_id_++;
  conn->last_active = Clock::now();
  T* raw = conn.get();
  conns_.emplace(raw->id, std::move(conn));
  return raw;
}

bool Worker::watch(Conn* c) {
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET;
  ev.data.u64 = c->id;
  return ::epoll_ctl(epfd_, EPOLL_CTL_ADD, c->fd, &ev) == 0;
}

void Worker::close_conn(Conn* c) {
  if (c->closed) return;
  c->closed = true;
  ::close(c->fd);  // also removes it from the epoll set
  graveyard_.push_back(c->id);
  if (c->kind == ConnKind::kClient) {
    auto* cc = static_cast<ClientConn*>(c);
    s_.metrics.connections_active.fetch_sub(1, std::memory_order_relaxed);
    // A client going away aborts its upstream request: for streams this stops
    // token generation (and spend) as soon as nobody is listening.
    if (UpstreamConn* u = upstream(cc->upstream_id)) {
      u->client_id = 0;
      close_conn(u);
    }
    cc->upstream_id = 0;
  } else {
    auto* u = static_cast<UpstreamConn*>(c);
    if (ClientConn* cc = client(u->client_id)) cc->upstream_id = 0;
    u->client_id = 0;
    if (u->state == UpstreamConn::State::kIdle) {
      idle_upstreams_.erase(std::remove(idle_upstreams_.begin(), idle_upstreams_.end(), u->id),
                            idle_upstreams_.end());
    }
  }
}

ClientConn* Worker::client(std::uint64_t id) {
  if (id == 0) return nullptr;
  const auto it = conns_.find(id);
  if (it == conns_.end() || it->second->closed || it->second->kind != ConnKind::kClient) return nullptr;
  return static_cast<ClientConn*>(it->second.get());
}

UpstreamConn* Worker::upstream(std::uint64_t id) {
  if (id == 0) return nullptr;
  const auto it = conns_.find(id);
  if (it == conns_.end() || it->second->closed || it->second->kind != ConnKind::kUpstream) return nullptr;
  return static_cast<UpstreamConn*>(it->second.get());
}

void Worker::reap() {
  for (const auto id : graveyard_) conns_.erase(id);
  graveyard_.clear();
}

void Worker::sweep_timeouts(Clock::time_point now) {
  const auto client_idle = std::chrono::seconds(s_.cfg.client_idle_timeout_s);
  const auto upstream_idle = std::chrono::milliseconds(s_.cfg.upstream_timeout_ms);
  std::vector<std::uint64_t> idle_clients;
  std::vector<std::uint64_t> stuck_upstreams;
  for (const auto& [id, c] : conns_) {
    if (c->closed) continue;
    if (c->kind == ConnKind::kClient) {
      const auto* cc = static_cast<const ClientConn*>(c.get());
      if (cc->state != ClientConn::State::kUpstream && now - c->last_active > client_idle) {
        idle_clients.push_back(id);
      }
    } else {
      const auto* u = static_cast<const UpstreamConn*>(c.get());
      if (u->state != UpstreamConn::State::kIdle && now - c->last_active > upstream_idle) {
        stuck_upstreams.push_back(id);
      }
    }
  }
  // Act after iterating: handlers may insert into conns_.
  for (const auto id : idle_clients) {
    if (ClientConn* c = client(id)) close_conn(c);
  }
  for (const auto id : stuck_upstreams) {
    if (UpstreamConn* u = upstream(id)) upstream_failed(u, 504, "upstream timed out", /*timeout=*/true);
  }
}

// ---------------------------------------------------------------------------
// Client side
// ---------------------------------------------------------------------------

void Worker::on_accept() {
  for (;;) {
    const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EINTR || errno == ECONNABORTED) continue;
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        // EMFILE/ENFILE: leave the backlog queued; retried on the next edge.
        std::fprintf(stderr, "[worker %d] accept4: %s\n", index_, std::strerror(errno));
      }
      return;
    }
    set_nodelay(fd);
    auto conn = std::make_unique<ClientConn>(limits_);
    conn->fd = fd;
    ClientConn* c = add_conn(std::move(conn));
    if (!watch(c)) {
      ::close(fd);
      c->closed = true;
      graveyard_.push_back(c->id);
      continue;
    }
    inc(s_.metrics.connections_accepted);
    s_.metrics.connections_active.fetch_add(1, std::memory_order_relaxed);
  }
}

void Worker::on_client_event(ClientConn* c, std::uint32_t ev) {
  if (ev & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
    // Edge-triggered: drain the socket until EAGAIN or we'd never hear again.
    for (;;) {
      const ssize_t r = ::recv(c->fd, scratch_.data(), scratch_.size(), 0);
      if (r > 0) {
        c->in.append(std::string_view(scratch_.data(), static_cast<std::size_t>(r)));
        // Bound memory for clients that pipeline unboundedly while we wait.
        if (c->in.size() > limits_.max_header_bytes + limits_.max_body_bytes + kReadChunk) {
          close_conn(c);
          return;
        }
        continue;
      }
      if (r == 0) {
        c->peer_closed = true;
        break;
      }
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      if (c->state == ClientConn::State::kUpstream) inc(s_.metrics.client_aborts);
      close_conn(c);
      return;
    }
  }
  if (ev & EPOLLOUT) {
    flush_client(c);
    if (c->closed) return;
  }
  process_client(c);
}

void Worker::process_client(ClientConn* c) {
  while (!c->closed && c->state == ClientConn::State::kReading && !c->in.empty()) {
    HttpRequest req;
    std::size_t consumed = 0;
    const auto r = c->parser.parse(c->in.view(), req, consumed);
    if (r == ParseResult::kIncomplete) break;
    if (r == ParseResult::kError) {
      inc(s_.metrics.bad_requests);
      c->keep_alive = false;
      respond_error(c, c->parser.error().status, c->parser.error().message, "invalid_request_error",
                    "malformed_http");
      c->state = ClientConn::State::kClosing;
      break;
    }
    c->in.consume(consumed);
    c->keep_alive = req.keep_alive;
    handle_request(c, req);
    if (c->state == ClientConn::State::kReading && !c->keep_alive) c->state = ClientConn::State::kClosing;
  }
  if (c->closed) return;
  flush_client(c);
  if (c->closed) return;

  if (c->out.empty()) {
    if (c->state == ClientConn::State::kClosing ||
        (c->state == ClientConn::State::kReading && c->peer_closed)) {
      close_conn(c);
      return;
    }
  }
  // Client hung up mid-stream: abort the upstream generation right away. For
  // a non-streaming miss we let the upstream finish so the cache still fills.
  if (c->peer_closed && c->state == ClientConn::State::kUpstream && c->req && c->req->stream) {
    inc(s_.metrics.client_aborts);
    close_conn(c);
  }
}

void Worker::flush_client(ClientConn* c) {
  while (!c->out.empty()) {
    const ssize_t w = ::send(c->fd, c->out.data(), c->out.size(), MSG_NOSIGNAL);
    if (w > 0) {
      c->out.consume(static_cast<std::size_t>(w));
      continue;
    }
    if (w < 0 && errno == EINTR) continue;
    if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
    if (c->state == ClientConn::State::kUpstream) inc(s_.metrics.client_aborts);
    close_conn(c);
    return;
  }
  // Resume a paused upstream once the client has drained enough.
  if (c->out.size() < kClientLowWater) {
    if (UpstreamConn* u = upstream(c->upstream_id); u != nullptr && u->paused) {
      u->paused = false;
      read_upstream(u);  // ET: data may already be waiting with no new edge
    }
  }
}

void Worker::respond(ClientConn* c, int status, std::string_view content_type, std::string_view body,
                     std::string_view extra_headers) {
  std::string head;
  head.reserve(192 + extra_headers.size());
  head.append("HTTP/1.1 ").append(std::to_string(status)).append(" ").append(status_reason(status));
  head.append("\r\nServer: cloudvanguard\r\nContent-Type: ").append(content_type);
  head.append("\r\nContent-Length: ").append(std::to_string(body.size())).append("\r\n");
  head.append(extra_headers);
  head.append(c->keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n");
  c->out.append(head);
  c->out.append(body);
  s_.metrics.count_status(status);
}

void Worker::respond_error(ClientConn* c, int status, std::string_view message, std::string_view type,
                           std::string_view code, std::string_view extra_headers) {
  respond(c, status, kJson, json_error(message, type, code), extra_headers);
}

void Worker::reject_rate_limited(ClientConn* c, const RateDecision& d, std::string_view which) {
  const bool never = !std::isfinite(d.retry_after_s);
  const long secs = never ? 60 : std::max(1L, static_cast<long>(std::ceil(d.retry_after_s)));
  std::string extra = "Retry-After: " + std::to_string(secs) + "\r\nX-RateLimit-Reason: " + std::string(which) + "\r\n";
  const std::string msg =
      never ? "request's estimated token cost exceeds the tenant's per-minute token limit"
            : "rate limit exceeded (" + std::string(which) + "); retry after " + std::to_string(secs) + "s";
  respond_error(c, 429, msg, "rate_limit_error", which == "rpm" ? "rpm_exceeded" : "tpm_exceeded", extra);
}

void Worker::handle_request(ClientConn* c, HttpRequest& req) {
  inc(s_.metrics.requests_total);
  const auto start = Clock::now();
  const bool get = req.method == "GET";

  if (req.path == "/v1/chat/completions") {
    if (req.method != "POST") {
      respond_error(c, 405, "use POST", "invalid_request_error", "method_not_allowed", "Allow: POST\r\n");
      return;
    }
    handle_chat(c, req, start);
    return;
  }
  if (req.path == "/healthz") {
    if (!get) {
      respond_error(c, 405, "use GET", "invalid_request_error", "method_not_allowed", "Allow: GET\r\n");
      return;
    }
    respond(c, 200, "text/plain", "ok\n");
    return;
  }
  if (req.path == "/metrics") {
    if (!get) {
      respond_error(c, 405, "use GET", "invalid_request_error", "method_not_allowed", "Allow: GET\r\n");
      return;
    }
    respond(c, 200, "text/plain; version=0.0.4",
            s_.metrics.render(s_.cache.stats(), s_.cfg.node_id, s_.ring.node_count(), s_.ring.point_count()));
    return;
  }
  respond_error(c, 404, "no route for " + req.path, "invalid_request_error", "not_found");
}

void Worker::handle_chat(ClientConn* c, HttpRequest& req, Clock::time_point start) {
  // 1. Authenticate.
  const auto token = bearer_token(req.header("authorization"));
  if (!token) {
    inc(s_.metrics.auth_failures);
    respond_error(c, 401, "missing 'Authorization: Bearer <tenant key>' header", "invalid_request_error",
                  "missing_api_key");
    return;
  }
  const auto tenant = s_.tenants.resolve(*token);
  if (!tenant) {
    inc(s_.metrics.auth_failures);
    respond_error(c, 401, "unknown API key", "invalid_request_error", "invalid_api_key");
    return;
  }

  // 2. RPM applies to every request, hits included: it protects the gateway.
  const auto now = Clock::now();
  if (const auto d = s_.limiter.check_request(tenant->id, tenant->limits, now); !d.allowed) {
    inc(s_.metrics.rate_limited_rpm);
    reject_rate_limited(c, d, "rpm");
    return;
  }

  // 3. Validate the body and derive the cache key.
  std::string error;
  auto chat = parse_chat_request(req.body, error);
  if (!chat) {
    inc(s_.metrics.bad_requests);
    respond_error(c, 400, error, "invalid_request_error", "invalid_body");
    return;
  }
  auto r = std::make_unique<InFlight>();
  r->start = start;
  r->tenant = tenant->id;
  r->stream = chat->stream;
  r->key = cache_key(tenant->id, *chat);
  r->key_hex = to_hex(r->key.data(), 8);
  r->owner = s_.ring.node_for(r->key).value_or(s_.cfg.node_id);
  const auto cache_control = req.header("cache-control");
  r->bypass = contains_token(cache_control, "no-cache");
  r->store = !contains_token(cache_control, "no-store");

  // 4. Cache lookup.
  if (r->bypass) {
    inc(s_.metrics.cache_bypass);
  } else if (const auto hit = s_.cache.get(r->key, now)) {
    serve_hit(c, *hit, *r);
    return;
  }
  inc(s_.metrics.cache_misses);

  // 5. TPM applies only to misses: it caps upstream spend, and hits are free.
  const std::uint64_t estimate = estimate_tokens(*chat, s_.cfg.default_completion_tokens);
  if (const auto d = s_.limiter.check_tokens(tenant->id, tenant->limits, static_cast<double>(estimate), now);
      !d.allowed) {
    inc(s_.metrics.rate_limited_tpm);
    reject_rate_limited(c, d, "tpm");
    return;
  }
  inc(s_.metrics.estimated_tokens_forwarded, estimate);

  // 6. Forward. The client body is sent upstream byte-for-byte.
  const auto& up = s_.upstream;
  std::string& out = r->upstream_request;
  out.reserve(512 + req.body.size());
  out.append("POST ").append(up.url.base_path).append(req.target).append(" HTTP/1.1\r\nHost: ");
  out.append(up.host_header).append("\r\nContent-Type: application/json\r\nAccept: ");
  out.append(r->stream ? "text/event-stream" : "application/json");
  out.append("\r\nContent-Length: ").append(std::to_string(req.body.size())).append("\r\n");
  if (!s_.cfg.upstream_api_key.empty()) {
    out.append("Authorization: Bearer ").append(s_.cfg.upstream_api_key).append("\r\n");
  }
  out.append("X-Vanguard-Tenant: ").append(r->tenant).append("\r\nConnection: keep-alive\r\n\r\n");
  out.append(req.body);

  c->req = std::move(r);
  c->state = ClientConn::State::kUpstream;
  dispatch_upstream(c);
}

void Worker::serve_hit(ClientConn* c, const CachedResponse& hit, const InFlight& r) {
  std::string head;
  head.reserve(256);
  head.append("HTTP/1.1 200 OK\r\nServer: cloudvanguard\r\nContent-Type: ").append(hit.content_type);
  if (hit.stream) {
    head.append("\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n");
  } else {
    head.append("\r\nContent-Length: ").append(std::to_string(hit.body.size())).append("\r\n");
  }
  head.append("X-Cache: HIT\r\nX-Vanguard-Cache-Key: ").append(r.key_hex);
  head.append("\r\nX-Vanguard-Owner: ").append(r.owner).append("\r\n");
  head.append(c->keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n");
  c->out.append(head);
  c->out.append(hit.body);  // JSON body, or every SSE event pre-framed as chunks

  s_.metrics.count_status(200);
  inc(s_.metrics.cache_hits);
  const auto elapsed = Clock::now() - r.start;
  if (hit.stream) {
    inc(s_.metrics.cache_hits_stream);
    s_.metrics.latency_hit_stream.observe(elapsed);
  } else {
    s_.metrics.latency_hit.observe(elapsed);
  }
}

// Called when the in-flight exchange on `c` is over (response fully queued).
void Worker::finish_exchange(ClientConn* c) {
  c->req.reset();
  c->upstream_id = 0;
  c->state = c->keep_alive ? ClientConn::State::kReading : ClientConn::State::kClosing;
  process_client(c);  // flush, then serve any pipelined request
}

// ---------------------------------------------------------------------------
// Upstream side
// ---------------------------------------------------------------------------

UpstreamConn* Worker::acquire_upstream() {
  while (!idle_upstreams_.empty()) {
    const auto id = idle_upstreams_.back();
    idle_upstreams_.pop_back();
    if (UpstreamConn* u = upstream(id)) {
      u->state = UpstreamConn::State::kActive;
      u->reused = true;
      return u;
    }
  }
  const auto& up = s_.upstream;
  const int fd = ::socket(up.addr.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) return nullptr;
  set_nodelay(fd);
  const int rc = ::connect(fd, reinterpret_cast<const sockaddr*>(&up.addr), up.addr_len);
  if (rc < 0 && errno != EINPROGRESS) {
    ::close(fd);
    return nullptr;
  }
  auto conn = std::make_unique<UpstreamConn>();
  conn->fd = fd;
  conn->state = rc == 0 ? UpstreamConn::State::kActive : UpstreamConn::State::kConnecting;
  UpstreamConn* u = add_conn(std::move(conn));
  if (!watch(u)) {
    close_conn(u);
    return nullptr;
  }
  inc(s_.metrics.upstream_connects);
  return u;
}

void Worker::dispatch_upstream(ClientConn* c) {
  UpstreamConn* u = acquire_upstream();
  if (u == nullptr) {
    inc(s_.metrics.upstream_errors);
    respond_error(c, 502, "cannot connect to upstream", "upstream_error", "upstream_unavailable");
    finish_exchange(c);
    return;
  }
  u->client_id = c->id;
  c->upstream_id = u->id;
  u->parser.reset();
  u->decoded.clear();
  u->headers_handled = false;
  u->paused = false;
  u->bytes_received = 0;
  u->last_active = Clock::now();
  u->out.clear();
  u->out.append(c->req->upstream_request);
  inc(s_.metrics.upstream_requests);
  if (u->state == UpstreamConn::State::kActive) flush_upstream(u);
}

void Worker::on_upstream_event(UpstreamConn* u, std::uint32_t ev) {
  if (u->state == UpstreamConn::State::kIdle) {
    // Pooled connection: any readability means the server closed it (or sent
    // garbage). Either way it is unusable.
    if (ev & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR)) close_conn(u);
    return;
  }
  if (u->state == UpstreamConn::State::kConnecting) {
    if (!(ev & (EPOLLOUT | EPOLLERR | EPOLLHUP))) return;
    int err = 0;
    socklen_t len = sizeof(err);
    ::getsockopt(u->fd, SOL_SOCKET, SO_ERROR, &err, &len);
    if (err != 0) {
      upstream_failed(u, 502, std::string("upstream connect failed: ") + std::strerror(err));
      return;
    }
    u->state = UpstreamConn::State::kActive;
  }
  if (ev & EPOLLOUT) {
    flush_upstream(u);
    if (u->closed) return;
  }
  if (ev & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR)) read_upstream(u);
}

void Worker::flush_upstream(UpstreamConn* u) {
  while (!u->out.empty()) {
    const ssize_t w = ::send(u->fd, u->out.data(), u->out.size(), MSG_NOSIGNAL);
    if (w > 0) {
      u->out.consume(static_cast<std::size_t>(w));
      continue;
    }
    if (w < 0 && errno == EINTR) continue;
    if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
    upstream_failed(u, 502, "upstream write failed");
    return;
  }
}

void Worker::read_upstream(UpstreamConn* u) {
  while (!u->closed && !u->paused && u->state == UpstreamConn::State::kActive) {
    const ssize_t r = ::recv(u->fd, scratch_.data(), scratch_.size(), 0);
    if (r > 0) {
      u->bytes_received += static_cast<std::size_t>(r);
      if (!u->parser.feed(std::string_view(scratch_.data(), static_cast<std::size_t>(r)), u->decoded)) {
        upstream_failed(u, 502, "malformed upstream response: " + u->parser.error());
        return;
      }
      on_upstream_progress(u);
      continue;
    }
    if (r < 0 && errno == EINTR) continue;
    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
    // EOF or connection error.
    if (r == 0 && u->parser.finish_on_eof()) {
      on_upstream_progress(u);  // read-until-close body is now complete
      return;
    }
    if (u->reused && u->bytes_received == 0) {
      // The server closed a pooled keep-alive connection just as we reused it.
      // Nothing was processed, so retrying once on a fresh connection is safe.
      ClientConn* c = client(u->client_id);
      close_conn(u);
      if (c != nullptr && c->req && !c->req->retried) {
        c->req->retried = true;
        inc(s_.metrics.upstream_retries);
        dispatch_upstream(c);
      } else if (c != nullptr) {
        inc(s_.metrics.upstream_errors);
        respond_error(c, 502, "upstream closed the connection", "upstream_error", "upstream_closed");
        finish_exchange(c);
      }
      return;
    }
    upstream_failed(u, 502, "upstream closed the connection mid-response");
    return;
  }
}

void Worker::on_upstream_progress(UpstreamConn* u) {
  ClientConn* c = client(u->client_id);
  if (c == nullptr || !c->req) {
    close_conn(u);
    return;
  }
  InFlight& r = *c->req;

  if (u->parser.headers_complete() && !u->headers_handled) {
    u->headers_handled = true;
    if (r.stream && u->parser.status() == 200) {
      // Start relaying SSE immediately: time-to-first-token on a miss is the
      // upstream's TTFT plus one hop, not the full generation time.
      const auto ct = u->parser.header("content-type");
      std::string head;
      head.reserve(256);
      head.append("HTTP/1.1 200 OK\r\nServer: cloudvanguard\r\nContent-Type: ");
      head.append(ct.empty() ? std::string_view("text/event-stream") : ct);
      head.append("\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nX-Cache: ");
      head.append(r.bypass ? "BYPASS" : "MISS").append("\r\nX-Vanguard-Cache-Key: ").append(r.key_hex);
      head.append("\r\nX-Vanguard-Owner: ").append(r.owner).append("\r\n");
      head.append(c->keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n");
      c->out.append(head);
      r.response_started = true;
    }
  }

  if (r.response_started) {
    if (!u->decoded.empty()) {
      std::string chunk;
      append_http_chunk(chunk, u->decoded);
      c->out.append(chunk);
      if (r.cacheable && r.store) {
        std::vector<std::string> frames;
        r.splitter.feed(u->decoded, frames);
        for (const auto& f : frames) {
          r.saw_done = r.saw_done || is_sse_done_frame(f);
          append_http_chunk(r.wire, f);
          ++r.frames;
        }
        if (r.wire.size() > s_.cfg.cache_max_entry_kb * 1024) {
          r.cacheable = false;  // too big to cache; keep relaying
          r.wire.clear();
          r.wire.shrink_to_fit();
        }
      }
      u->decoded.clear();
      flush_client(c);
      if (c->closed || u->closed) return;
      if (c->out.size() > kClientHighWater) u->paused = true;
    }
  } else if (u->decoded.size() > kMaxUpstreamBody) {
    upstream_failed(u, 502, "upstream response too large");
    return;
  }

  if (u->parser.done()) complete_upstream(u, c);
}

void Worker::complete_upstream(UpstreamConn* u, ClientConn* c) {
  InFlight& r = *c->req;
  const int status = u->parser.status();
  const auto elapsed = Clock::now() - r.start;
  bool filled = false;

  if (r.response_started) {
    c->out.append(kLastHttpChunk);
    s_.metrics.count_status(200);
    if (r.store && r.cacheable && r.saw_done) {
      if (!r.splitter.pending().empty()) append_http_chunk(r.wire, r.splitter.pending());
      r.wire.append(kLastHttpChunk);
      auto entry = std::make_shared<CachedResponse>();
      entry->stream = true;
      entry->status = 200;
      const auto ct = u->parser.header("content-type");
      entry->content_type = ct.empty() ? "text/event-stream" : std::string(ct);
      entry->sse_frames = r.frames;
      entry->body = std::move(r.wire);
      const auto charge = entry->charge(r.key.size());
      filled = s_.cache.put(r.key, std::move(entry), charge);
    }
    s_.metrics.latency_miss_stream.observe(elapsed);
  } else {
    const auto ct = u->parser.header("content-type");
    std::string extra = std::string("X-Cache: ") + (r.bypass ? "BYPASS" : "MISS") +
                        "\r\nX-Vanguard-Cache-Key: " + r.key_hex + "\r\nX-Vanguard-Owner: " + r.owner + "\r\n";
    if (const auto ra = u->parser.header("retry-after"); !ra.empty()) {
      extra.append("Retry-After: ").append(ra).append("\r\n");
    }
    respond(c, status, ct.empty() ? kJson : ct, u->decoded, extra);
    // Only successful, complete JSON completions are cached; errors and
    // upstream throttling must never be replayed.
    if (r.store && status == 200 && !r.stream && u->decoded.size() <= s_.cfg.cache_max_entry_kb * 1024) {
      auto entry = std::make_shared<CachedResponse>();
      entry->stream = false;
      entry->status = 200;
      entry->content_type = ct.empty() ? std::string(kJson) : std::string(ct);
      entry->body = std::move(u->decoded);
      const auto charge = entry->charge(r.key.size());
      filled = s_.cache.put(r.key, std::move(entry), charge);
    }
    s_.metrics.latency_miss.observe(elapsed);
  }
  inc(filled ? s_.metrics.cache_fills : s_.metrics.cache_fill_skipped);

  release_upstream(u);
  finish_exchange(c);
}

void Worker::release_upstream(UpstreamConn* u) {
  u->client_id = 0;
  u->decoded.clear();
  const bool reusable = u->parser.keep_alive() && u->parser.leftover() == 0 && u->out.empty();
  if (reusable && idle_upstreams_.size() < s_.cfg.upstream_max_idle) {
    u->state = UpstreamConn::State::kIdle;
    idle_upstreams_.push_back(u->id);
  } else {
    close_conn(u);
  }
}

void Worker::upstream_failed(UpstreamConn* u, int status, std::string_view message, bool timeout) {
  ClientConn* c = client(u->client_id);
  inc(timeout ? s_.metrics.upstream_timeouts : s_.metrics.upstream_errors);
  close_conn(u);
  if (c == nullptr || !c->req) return;
  if (c->req->response_started) {
    // Headers are already on the wire; the only honest signal left is to
    // terminate the connection without the final chunk.
    close_conn(c);
    return;
  }
  respond_error(c, status, message, "upstream_error", timeout ? "upstream_timeout" : "upstream_failure");
  finish_exchange(c);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::optional<UpstreamTarget> resolve_upstream(const Config& cfg, std::string& error) {
  auto url = parse_upstream_url(cfg.upstream_url, error);
  if (!url) return std::nullopt;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port = std::to_string(url->port);
  if (const int rc = ::getaddrinfo(url->host.c_str(), port.c_str(), &hints, &res); rc != 0 || res == nullptr) {
    error = "cannot resolve upstream host '" + url->host + "': " + ::gai_strerror(rc);
    return std::nullopt;
  }
  UpstreamTarget t;
  std::memcpy(&t.addr, res->ai_addr, res->ai_addrlen);
  t.addr_len = static_cast<socklen_t>(res->ai_addrlen);
  ::freeaddrinfo(res);
  t.host_header = url->host + ":" + port;
  t.url = std::move(*url);
  return t;
}

SharedState::SharedState(Config c, TenantRegistry t, UpstreamTarget u)
    : cfg(std::move(c)),
      tenants(std::move(t)),
      upstream(std::move(u)),
      limiter(64),
      cache(cfg.cache_max_mb * 1024 * 1024, cfg.cache_shards, std::chrono::seconds(cfg.cache_ttl_s)),
      ring(cfg.ring_vnodes) {
  if (cfg.ring_nodes.empty()) {
    ring.add_node(cfg.node_id);
  } else {
    for (const auto& n : cfg.ring_nodes) ring.add_node(n);
  }
}

int run_server(SharedState& state) {
  struct sigaction sa{};
  sa.sa_handler = handle_stop_signal;
  sigemptyset(&sa.sa_mask);
  ::sigaction(SIGINT, &sa, nullptr);
  ::sigaction(SIGTERM, &sa, nullptr);
  ::signal(SIGPIPE, SIG_IGN);

  unsigned n = state.cfg.workers;
  if (n == 0) n = std::max(1u, std::thread::hardware_concurrency());

  // Bind every listener before starting any thread so configuration errors
  // (port in use, bad address) fail fast and cleanly.
  std::vector<std::unique_ptr<Worker>> workers;
  for (unsigned i = 0; i < n; ++i) {
    auto w = std::make_unique<Worker>(state, static_cast<int>(i));
    std::string err;
    if (!w->init(err)) {
      std::fprintf(stderr, "vanguard-proxy: %s\n", err.c_str());
      return 1;
    }
    workers.push_back(std::move(w));
  }
  std::fprintf(stderr, "vanguard-proxy listening on %s:%u with %u epoll workers -> upstream %s\n",
               state.cfg.listen_addr.c_str(), state.cfg.listen_port, n, state.cfg.upstream_url.c_str());

  std::vector<std::thread> threads;
  threads.reserve(n);
  for (auto& w : workers) threads.emplace_back([&w] { w->run(); });
  for (auto& t : threads) t.join();
  std::fprintf(stderr, "vanguard-proxy: shutdown complete\n");
  return 0;
}

}  // namespace vanguard
