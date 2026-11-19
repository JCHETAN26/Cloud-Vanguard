// Linux-only data plane: N independent epoll event loops, one per worker
// thread, each with its own SO_REUSEPORT listener, upstream connection pools
// and timer heap.
//
// Threading model
//   - A connection (client or upstream) is owned by exactly one worker for its
//     whole life, so per-connection state needs no locks.
//   - Shared state (response cache, semantic index, rate limiter, circuit
//     breakers, ring, metrics) is internally synchronized.
//
// I/O model
//   - Non-blocking sockets registered once with EPOLLIN | EPOLLOUT |
//     EPOLLRDHUP | EPOLLET; every notification is drained until EAGAIN.
//   - Connections are addressed by id (epoll_event.data.u64), never by raw
//     pointer, and are reaped only after the current batch of events, so a
//     stale event for a connection closed earlier in the batch is harmless.
//   - Deadlines (upstream time-to-first-byte, embedding calls) live in a
//     per-worker min-heap; epoll_wait sleeps until the earliest one.
//
// Request pipeline for POST /v1/chat/completions
//   auth -> RPM -> validate -> exact cache -> [embed] -> semantic cache
//        -> TPM -> route (breaker / failover) -> upstream -> cache fill

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
#include <functional>
#include <memory>
#include <queue>
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
using Millis = std::chrono::milliseconds;

constexpr std::uint64_t kListenerId = 1;
constexpr std::size_t kReadChunk = 64 * 1024;
constexpr std::size_t kMaxEvents = 512;
constexpr int kEpollTimeoutMs = 250;
// Streaming backpressure: stop reading the upstream while a slow client has
// more than kHighWater bytes queued; resume below kLowWater.
constexpr std::size_t kClientHighWater = 4 * 1024 * 1024;
constexpr std::size_t kClientLowWater = 1024 * 1024;
constexpr std::size_t kMaxUpstreamBody = 32 * 1024 * 1024;
constexpr std::size_t kMaxEmbeddingBody = 4 * 1024 * 1024;
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

enum class SemanticMode { kOff, kLexical, kRemote };

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

// The chat request in flight on a client connection (one at a time;
// pipelined requests wait in the input buffer).
struct InFlight {
  std::string tenant;
  TenantLimits limits;
  std::string key;      // raw 32-byte exact-cache key
  std::string key_hex;  // short hex prefix for X-Vanguard-Cache-Key
  std::string owner;    // ring owner of the key
  bool stream = false;
  bool store = true;    // false for Cache-Control: no-store
  bool bypass = false;  // Cache-Control: no-cache
  std::uint64_t estimate = 0;
  std::string body;  // original request body
  std::string path;  // original request target
  Clock::time_point start{};

  // Semantic cache.
  bool semantic = false;  // eligible for semantic lookup + insert
  std::string scope;
  std::uint64_t guard = 0;
  std::string semantic_text;  // for the remote embedder
  std::vector<float> embedding;
  Clock::time_point embed_start{};

  // Routing.
  int target = -1;
  bool primary_failed = false;
  bool failed_over = false;
  std::string failover_reason;
  bool retried = false;  // stale pooled connection already retried once
  bool response_started = false;

  // Stream capture for the cache fill: complete SSE events re-framed as HTTP
  // chunks, ready to replay verbatim.
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
  bool in_process = false;  // re-entrancy guard for process_client
  std::uint64_t upstream_id = 0;
  std::unique_ptr<InFlight> req;
};

struct UpstreamConn final : Conn {
  explicit UpstreamConn(int t) : Conn(ConnKind::kUpstream), target(t) {}

  enum class State { kConnecting, kActive, kIdle };
  const int target;
  State state = State::kConnecting;
  Buffer out;
  ResponseParser parser;
  std::string decoded;  // decoded body bytes not yet consumed
  std::uint64_t client_id = 0;
  bool probe = false;  // holds a half-open breaker probe slot
  bool reused = false;
  bool headers_handled = false;
  bool paused = false;
  std::size_t bytes_received = 0;
  std::uint64_t timer_seq = 0;
  bool timer_armed = false;
};

struct Timer {
  Clock::time_point at;
  std::uint64_t conn_id;
  std::uint64_t seq;
  bool operator>(const Timer& o) const noexcept { return at > o.at; }
};

// ---------------------------------------------------------------------------
// Worker: one epoll loop
// ---------------------------------------------------------------------------

class Worker {
 public:
  Worker(SharedState& s, int index) : s_(s), index_(index) {
    limits_.max_header_bytes = 16 * 1024;
    limits_.max_body_bytes = s.cfg.max_body_kb * 1024;
    mode_ = s.cfg.embedder == "off"      ? SemanticMode::kOff
            : s.cfg.embedder == "remote" ? SemanticMode::kRemote
                                         : SemanticMode::kLexical;
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
  void serve_hit(ClientConn* c, const CachedResponse& hit, const InFlight& r, std::optional<float> similarity);
  void respond(ClientConn* c, int status, std::string_view content_type, std::string_view body,
               std::string_view extra_headers = {});
  void respond_error(ClientConn* c, int status, std::string_view message, std::string_view type,
                     std::string_view code, std::string_view extra_headers = {});
  void reject_rate_limited(ClientConn* c, const RateDecision& d, std::string_view which);
  void flush_client(ClientConn* c);
  void finish_exchange(ClientConn* c);

  // Miss pipeline
  bool try_semantic_hit(ClientConn* c);
  void start_embedding(ClientConn* c);
  void dispatch_embedding(ClientConn* c, bool probe);
  void complete_embedding(UpstreamConn* u, ClientConn* c);
  void embedding_failed(ClientConn* c, bool probe, bool timeout);
  void continue_miss(ClientConn* c);
  void route_llm(ClientConn* c);
  void dispatch_llm(ClientConn* c, int target, bool probe);
  std::string build_llm_request(const InFlight& r, int target) const;
  std::string routing_headers(const InFlight& r) const;
  bool on_llm_headers(UpstreamConn* u, ClientConn* c);
  void complete_llm(UpstreamConn* u, ClientConn* c);
  void llm_attempt_failed(ClientConn* c, int target, bool probe, int status, std::string_view message,
                          std::string_view reason, bool timeout);

  // Upstream side
  UpstreamConn* acquire_upstream(int target);
  void flush_upstream(UpstreamConn* u);
  void read_upstream(UpstreamConn* u);
  void on_upstream_progress(UpstreamConn* u);
  void attempt_failed(UpstreamConn* u, int status, std::string_view message, std::string_view reason,
                      bool timeout);
  void release_upstream(UpstreamConn* u);

  // Timers
  void arm_timer(UpstreamConn* u, Clock::time_point at);
  void disarm_timer(UpstreamConn* u);
  int next_timeout_ms(Clock::time_point now);
  void fire_timers(Clock::time_point now);

  // Bookkeeping
  template <typename T>
  T* add_conn(std::unique_ptr<T> conn);
  bool watch(Conn* c);
  void close_conn(Conn* c);
  ClientConn* client(std::uint64_t id);
  UpstreamConn* upstream(std::uint64_t id);
  void reap();
  void sweep_timeouts(Clock::time_point now);

  CircuitBreaker& breaker(int target) const { return *s_.targets[static_cast<std::size_t>(target)].breaker; }
  UpstreamTargetMetrics& tmetrics(int target) { return s_.metrics.upstream[static_cast<std::size_t>(target)]; }
  const UpstreamTarget& target_of(int target) const { return s_.targets[static_cast<std::size_t>(target)]; }

  SharedState& s_;
  const int index_;
  SemanticMode mode_;
  RequestParser::Limits limits_;
  int epfd_ = -1;
  int listen_fd_ = -1;
  std::uint64_t next_id_ = kListenerId + 1;
  std::unordered_map<std::uint64_t, std::unique_ptr<Conn>> conns_;
  std::array<std::vector<std::uint64_t>, 3> idle_;
  std::vector<std::uint64_t> graveyard_;
  std::priority_queue<Timer, std::vector<Timer>, std::greater<>> timers_;
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
  // kernel load-balances new connections across them.
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
    const int timeout = next_timeout_ms(Clock::now());
    const int n = ::epoll_wait(epfd_, events.data(), static_cast<int>(events.size()), timeout);
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
    fire_timers(Clock::now());
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
    // A client going away aborts its upstream call: for streams this stops
    // token generation (and spend) as soon as nobody is listening.
    if (UpstreamConn* u = upstream(cc->upstream_id)) {
      u->client_id = 0;
      close_conn(u);
    }
    cc->upstream_id = 0;
  } else {
    auto* u = static_cast<UpstreamConn*>(c);
    if (u->probe) {
      breaker(u->target).release_probe();  // ended without an outcome
      u->probe = false;
    }
    u->timer_armed = false;
    if (ClientConn* cc = client(u->client_id)) cc->upstream_id = 0;
    u->client_id = 0;
    if (u->state == UpstreamConn::State::kIdle) {
      auto& pool = idle_[static_cast<std::size_t>(u->target)];
      pool.erase(std::remove(pool.begin(), pool.end(), u->id), pool.end());
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
  const auto upstream_idle = Millis(s_.cfg.upstream_timeout_ms);
  std::vector<std::uint64_t> idle_clients;
  std::vector<std::uint64_t> stalled_upstreams;
  for (const auto& [id, c] : conns_) {
    if (c->closed) continue;
    if (c->kind == ConnKind::kClient) {
      const auto* cc = static_cast<const ClientConn*>(c.get());
      if (cc->state != ClientConn::State::kUpstream && now - c->last_active > client_idle) {
        idle_clients.push_back(id);
      }
    } else {
      // Before headers the time-to-first-byte timer is in charge; this catches
      // a response that started and then went silent.
      const auto* u = static_cast<const UpstreamConn*>(c.get());
      if (u->state == UpstreamConn::State::kActive && u->headers_handled && now - c->last_active > upstream_idle) {
        stalled_upstreams.push_back(id);
      }
    }
  }
  // Act after iterating: handlers may insert into conns_.
  for (const auto id : idle_clients) {
    if (ClientConn* c = client(id)) close_conn(c);
  }
  for (const auto id : stalled_upstreams) {
    if (UpstreamConn* u = upstream(id)) attempt_failed(u, 504, "upstream stalled", "upstream-stalled", true);
  }
}

// ---------------------------------------------------------------------------
// Timers
// ---------------------------------------------------------------------------

void Worker::arm_timer(UpstreamConn* u, Clock::time_point at) {
  u->timer_armed = true;
  timers_.push(Timer{at, u->id, ++u->timer_seq});
}

void Worker::disarm_timer(UpstreamConn* u) {
  u->timer_armed = false;
  ++u->timer_seq;  // invalidates the heap entry lazily
}

int Worker::next_timeout_ms(Clock::time_point now) {
  while (!timers_.empty()) {
    const Timer& t = timers_.top();
    const UpstreamConn* u = upstream(t.conn_id);
    if (u != nullptr && u->timer_armed && u->timer_seq == t.seq) break;
    timers_.pop();  // stale
  }
  if (timers_.empty()) return kEpollTimeoutMs;
  const auto wait = std::chrono::duration_cast<Millis>(timers_.top().at - now).count() + 1;
  return static_cast<int>(std::clamp<long long>(wait, 0, kEpollTimeoutMs));
}

void Worker::fire_timers(Clock::time_point now) {
  while (!timers_.empty() && timers_.top().at <= now) {
    const Timer t = timers_.top();
    timers_.pop();
    UpstreamConn* u = upstream(t.conn_id);
    if (u == nullptr || !u->timer_armed || u->timer_seq != t.seq) continue;
    u->timer_armed = false;
    if (u->target == kEmbedderTarget) {
      attempt_failed(u, 504, "embedding sidecar timed out", "embedder-timeout", true);
    } else if (!u->headers_handled) {
      attempt_failed(u, 504,
                     "upstream sent no response within " + std::to_string(target_of(u->target).ttfb_timeout_ms) + " ms",
                     "upstream-timeout", true);
    }
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
    // Edge-triggered: drain the socket until EAGAIN.
    for (;;) {
      const ssize_t r = ::recv(c->fd, scratch_.data(), scratch_.size(), 0);
      if (r > 0) {
        c->in.append(std::string_view(scratch_.data(), static_cast<std::size_t>(r)));
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
  // Handlers that finish synchronously call finish_exchange(), which calls
  // back into here; the guard turns that into "the outer loop continues"
  // instead of recursing once per pipelined request.
  if (c->closed || c->in_process) return;
  c->in_process = true;
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
  c->in_process = false;
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
  // Client hung up mid-stream: abort the upstream generation right away. A
  // non-streaming miss is allowed to finish so the cache still fills.
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
            s_.metrics.render(s_.cache.stats(), s_.cfg.node_id, s_.ring.node_count(), s_.ring.point_count(),
                              s_.semantic.size(), s_.breaker_reports()));
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

  // 3. Validate and derive the exact-cache key.
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
  r->limits = tenant->limits;
  r->stream = chat->stream;
  r->key = cache_key(tenant->id, *chat);
  r->key_hex = to_hex(r->key.data(), 8);
  r->owner = s_.ring.node_for(r->key).value_or(s_.cfg.node_id);
  const auto cache_control = req.header("cache-control");
  r->bypass = contains_token(cache_control, "no-cache");
  r->store = !contains_token(cache_control, "no-store");

  // 4. Exact cache.
  if (r->bypass) {
    inc(s_.metrics.cache_bypass);
  } else if (const auto hit = s_.cache.get(r->key, now)) {
    serve_hit(c, *hit, *r, std::nullopt);
    return;
  }

  // From here the request is a small state machine that may wait on the
  // embedding sidecar and/or an upstream; it always ends in finish_exchange().
  r->estimate = estimate_tokens(*chat, s_.cfg.default_completion_tokens);
  r->body = std::move(req.body);
  r->path = req.target;
  if (mode_ != SemanticMode::kOff && !r->bypass) {
    std::string text = semantic_text(*chat);
    if (!text.empty() && text.size() <= s_.cfg.semantic_max_chars) {
      const auto features = analyze_prompt(text);
      if (!features.tokens.empty()) {
        r->semantic = true;
        r->scope = semantic_scope(tenant->id, *chat);
        r->guard = features.guard;
        if (mode_ == SemanticMode::kLexical) {
          r->embedding = s_.lexical.embed_tokens(features.tokens);
          r->semantic = !r->embedding.empty();
        } else {
          r->semantic_text = std::move(text);
        }
      }
    }
  }
  c->req = std::move(r);
  c->state = ClientConn::State::kUpstream;

  if (c->req->semantic && mode_ == SemanticMode::kLexical) {
    if (!try_semantic_hit(c)) continue_miss(c);
  } else if (c->req->semantic && mode_ == SemanticMode::kRemote) {
    start_embedding(c);
  } else {
    continue_miss(c);
  }
}

void Worker::serve_hit(ClientConn* c, const CachedResponse& hit, const InFlight& r, std::optional<float> similarity) {
  std::string head;
  head.reserve(320);
  head.append("HTTP/1.1 200 OK\r\nServer: cloudvanguard\r\nContent-Type: ").append(hit.content_type);
  if (hit.stream) {
    head.append("\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n");
  } else {
    head.append("\r\nContent-Length: ").append(std::to_string(hit.body.size())).append("\r\n");
  }
  if (similarity) {
    char sim[32];
    std::snprintf(sim, sizeof(sim), "%.4f", static_cast<double>(*similarity));
    head.append("X-Cache: SEMANTIC-HIT\r\nX-Vanguard-Similarity: ").append(sim).append("\r\n");
  } else {
    head.append("X-Cache: HIT\r\n");
  }
  head.append("X-Vanguard-Cache-Key: ").append(r.key_hex);
  head.append("\r\nX-Vanguard-Owner: ").append(r.owner).append("\r\n");
  head.append(c->keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n");
  c->out.append(head);
  c->out.append(hit.body);  // JSON body, or every SSE event pre-framed as chunks

  s_.metrics.count_status(200);
  const auto elapsed = Clock::now() - r.start;
  if (similarity) {
    inc(s_.metrics.semantic_hits);
    s_.metrics.latency_semantic_hit.observe(elapsed);
    return;
  }
  inc(s_.metrics.cache_hits);
  if (hit.stream) {
    inc(s_.metrics.cache_hits_stream);
    s_.metrics.latency_hit_stream.observe(elapsed);
  } else {
    s_.metrics.latency_hit.observe(elapsed);
  }
}

// The exchange on `c` is over (response fully queued): serve whatever the
// client pipelined next.
void Worker::finish_exchange(ClientConn* c) {
  c->req.reset();
  c->upstream_id = 0;
  c->state = c->keep_alive ? ClientConn::State::kReading : ClientConn::State::kClosing;
  process_client(c);
}

// ---------------------------------------------------------------------------
// Miss pipeline: semantic cache
// ---------------------------------------------------------------------------

bool Worker::try_semantic_hit(ClientConn* c) {
  InFlight& r = *c->req;
  inc(s_.metrics.semantic_lookups);
  const auto match =
      s_.semantic.lookup(r.scope, r.guard, r.embedding, static_cast<float>(s_.cfg.semantic_threshold));
  if (!match) return false;
  auto hit = s_.cache.get(match->cache_key);
  if (!hit) {
    inc(s_.metrics.semantic_stale);  // response evicted since the vector was indexed
    s_.semantic.erase(r.scope, match->cache_key);
    return false;
  }
  // Alias the exact key too, so a repeat of this exact phrasing is an exact hit.
  if (r.store) s_.cache.put(r.key, hit, hit->charge(r.key.size()));
  serve_hit(c, *hit, r, match->similarity);
  finish_exchange(c);
  return true;
}

void Worker::start_embedding(ClientConn* c) {
  const auto permit = breaker(kEmbedderTarget).try_acquire(Clock::now());
  if (!permit.allowed) {
    // Sidecar circuit open: degrade to exact-match-only instead of waiting.
    inc(tmetrics(kEmbedderTarget).circuit_rejections);
    inc(s_.metrics.semantic_skipped);
    c->req->semantic = false;
    continue_miss(c);
    return;
  }
  dispatch_embedding(c, permit.probe);
}

void Worker::dispatch_embedding(ClientConn* c, bool probe) {
  InFlight& r = *c->req;
  UpstreamConn* u = acquire_upstream(kEmbedderTarget);
  if (u == nullptr) {
    embedding_failed(c, probe, false);
    return;
  }
  const auto& t = target_of(kEmbedderTarget);
  const std::string body = make_embedding_request(s_.cfg.embedder_model, r.semantic_text);
  std::string wire;
  wire.reserve(256 + body.size());
  wire.append("POST ").append(t.url.base_path).append("/v1/embeddings HTTP/1.1\r\nHost: ").append(t.host_header);
  wire.append("\r\nContent-Type: application/json\r\nContent-Length: ").append(std::to_string(body.size()));
  wire.append("\r\nConnection: keep-alive\r\n\r\n").append(body);

  u->client_id = c->id;
  c->upstream_id = u->id;
  u->probe = probe;
  u->parser.reset();
  u->decoded.clear();
  u->headers_handled = false;
  u->paused = false;
  u->bytes_received = 0;
  u->out.clear();
  u->out.append(wire);
  const auto now = Clock::now();
  u->last_active = now;
  r.embed_start = now;
  inc(tmetrics(kEmbedderTarget).requests);
  arm_timer(u, now + Millis(t.ttfb_timeout_ms));
  if (u->state == UpstreamConn::State::kActive) flush_upstream(u);
}

void Worker::complete_embedding(UpstreamConn* u, ClientConn* c) {
  disarm_timer(u);
  const bool probe = u->probe;
  u->probe = false;
  auto vec = u->parser.status() == 200 ? parse_embedding_response(u->decoded) : std::vector<float>{};
  release_upstream(u);
  if (vec.empty()) {
    embedding_failed(c, probe, false);
    return;
  }
  const auto now = Clock::now();
  breaker(kEmbedderTarget).record_success(probe, now);
  inc(tmetrics(kEmbedderTarget).successes);
  InFlight& r = *c->req;
  s_.metrics.latency_embedding.observe(now - r.embed_start);
  r.embedding = std::move(vec);
  if (!try_semantic_hit(c)) continue_miss(c);
}

void Worker::embedding_failed(ClientConn* c, bool probe, bool timeout) {
  breaker(kEmbedderTarget).record_failure(probe, Clock::now());
  inc(timeout ? tmetrics(kEmbedderTarget).timeouts : tmetrics(kEmbedderTarget).failures);
  inc(s_.metrics.semantic_skipped);
  if (c == nullptr || !c->req) return;
  c->req->semantic = false;  // no vector: exact-match-only for this request
  continue_miss(c);
}

// ---------------------------------------------------------------------------
// Miss pipeline: upstream routing and failover
// ---------------------------------------------------------------------------

void Worker::continue_miss(ClientConn* c) {
  InFlight& r = *c->req;
  inc(s_.metrics.cache_misses);
  // TPM applies only when we will actually call an upstream: exact and
  // semantic hits are free.
  const auto d =
      s_.limiter.check_tokens(r.tenant, r.limits, static_cast<double>(r.estimate), Clock::now());
  if (!d.allowed) {
    inc(s_.metrics.rate_limited_tpm);
    reject_rate_limited(c, d, "tpm");
    finish_exchange(c);
    return;
  }
  inc(s_.metrics.estimated_tokens_forwarded, r.estimate);
  route_llm(c);
}

void Worker::route_llm(ClientConn* c) {
  InFlight& r = *c->req;
  const auto now = Clock::now();
  const auto d = s_.router.route(r.primary_failed, now);
  if (d.target == FailoverRouter::kNone) {
    inc(s_.metrics.no_upstream_available);
    if (!r.primary_failed) inc(tmetrics(kPrimaryTarget).circuit_rejections);
    if (s_.router.has_fallback()) inc(tmetrics(kFallbackTarget).circuit_rejections);
    auto wait = breaker(kPrimaryTarget).snapshot(now).retry_after;
    if (s_.router.has_fallback()) wait = std::min(wait, breaker(kFallbackTarget).snapshot(now).retry_after);
    const long secs = std::max(1L, static_cast<long>((wait.count() + 999) / 1000));
    respond_error(c, 503, "no upstream available (" + std::string(d.reason) + ")", "upstream_error", "circuit_open",
                  "Retry-After: " + std::to_string(secs) + "\r\n");
    finish_exchange(c);
    return;
  }
  if (d.failover) {
    if (!r.primary_failed) inc(tmetrics(kPrimaryTarget).circuit_rejections);
    if (!r.failed_over) {
      r.failed_over = true;
      inc(s_.metrics.failovers);
    }
    if (r.failover_reason.empty()) r.failover_reason = std::string(d.reason);
  }
  dispatch_llm(c, d.target, d.probe);
}

std::string Worker::build_llm_request(const InFlight& r, int target) const {
  const auto& up = target_of(target);
  std::string rewritten;
  std::string_view body = r.body;
  if (!up.model_override.empty()) {
    rewritten = rewrite_model(r.body, up.model_override);
    body = rewritten;
  }
  std::string out;
  out.reserve(512 + body.size());
  out.append("POST ").append(up.url.base_path).append(r.path).append(" HTTP/1.1\r\nHost: ");
  out.append(up.host_header).append("\r\nContent-Type: application/json\r\nAccept: ");
  out.append(r.stream ? "text/event-stream" : "application/json");
  out.append("\r\nContent-Length: ").append(std::to_string(body.size())).append("\r\n");
  if (!up.api_key.empty()) out.append("Authorization: Bearer ").append(up.api_key).append("\r\n");
  out.append("X-Vanguard-Tenant: ").append(r.tenant).append("\r\nConnection: keep-alive\r\n\r\n");
  out.append(body);
  return out;
}

std::string Worker::routing_headers(const InFlight& r) const {
  std::string h;
  if (r.target >= 0) h.append("X-Vanguard-Upstream: ").append(target_of(r.target).name).append("\r\n");
  if (r.failed_over) h.append("X-Vanguard-Failover: ").append(r.failover_reason).append("\r\n");
  return h;
}

void Worker::dispatch_llm(ClientConn* c, int target, bool probe) {
  InFlight& r = *c->req;
  r.target = target;
  UpstreamConn* u = acquire_upstream(target);
  if (u == nullptr) {
    llm_attempt_failed(c, target, probe, 502, "cannot connect to " + target_of(target).name + " upstream",
                       "connect-error", false);
    return;
  }
  u->client_id = c->id;
  c->upstream_id = u->id;
  u->probe = probe;
  u->parser.reset();
  u->decoded.clear();
  u->headers_handled = false;
  u->paused = false;
  u->bytes_received = 0;
  u->out.clear();
  u->out.append(build_llm_request(r, target));
  const auto now = Clock::now();
  u->last_active = now;
  inc(s_.metrics.upstream_requests);
  inc(tmetrics(target).requests);
  arm_timer(u, now + Millis(target_of(target).ttfb_timeout_ms));
  if (u->state == UpstreamConn::State::kActive) flush_upstream(u);
}

void Worker::llm_attempt_failed(ClientConn* c, int target, bool probe, int status, std::string_view message,
                                std::string_view reason, bool timeout) {
  breaker(target).record_failure(probe, Clock::now());
  inc(timeout ? tmetrics(target).timeouts : tmetrics(target).failures);
  inc(timeout ? s_.metrics.upstream_timeouts : s_.metrics.upstream_errors);
  if (c == nullptr || !c->req) return;
  InFlight& r = *c->req;
  if (s_.router.can_fail_over(static_cast<FailoverRouter::Target>(target), r.failed_over, r.response_started)) {
    r.primary_failed = true;
    r.failover_reason = std::string(reason);
    route_llm(c);
    return;
  }
  if (r.response_started) {
    // Headers already on the wire: the only honest signal left is to
    // terminate the connection without the final chunk.
    close_conn(c);
    return;
  }
  respond_error(c, status, message, "upstream_error", timeout ? "upstream_timeout" : "upstream_failure",
                routing_headers(r));
  finish_exchange(c);
}

// Returns false if `u` was handed off (closed / failed over).
bool Worker::on_llm_headers(UpstreamConn* u, ClientConn* c) {
  InFlight& r = *c->req;
  u->headers_handled = true;
  disarm_timer(u);
  const int status = u->parser.status();
  const int t = u->target;
  const auto now = Clock::now();
  if (FailoverRouter::is_failure_status(status)) {
    if (s_.router.can_fail_over(static_cast<FailoverRouter::Target>(t), r.failed_over, r.response_started)) {
      attempt_failed(u, 502, "upstream returned " + std::to_string(status), "upstream-" + std::to_string(status),
                     false);
      return false;
    }
    // No failover possible: record it and relay the error as-is.
    breaker(t).record_failure(u->probe, now);
    inc(tmetrics(t).failures);
  } else {
    breaker(t).record_success(u->probe, now);
    inc(tmetrics(t).successes);
  }
  u->probe = false;

  if (r.stream && status == 200) {
    // Relay SSE immediately: time-to-first-token on a miss is the upstream's
    // TTFT plus one hop, not the full generation time.
    const auto ct = u->parser.header("content-type");
    std::string head;
    head.reserve(384);
    head.append("HTTP/1.1 200 OK\r\nServer: cloudvanguard\r\nContent-Type: ");
    head.append(ct.empty() ? std::string_view("text/event-stream") : ct);
    head.append("\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nX-Cache: ");
    head.append(r.bypass ? "BYPASS" : "MISS").append("\r\nX-Vanguard-Cache-Key: ").append(r.key_hex);
    head.append("\r\nX-Vanguard-Owner: ").append(r.owner).append("\r\n").append(routing_headers(r));
    head.append(c->keep_alive ? "Connection: keep-alive\r\n\r\n" : "Connection: close\r\n\r\n");
    c->out.append(head);
    r.response_started = true;
  }
  return true;
}

void Worker::complete_llm(UpstreamConn* u, ClientConn* c) {
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
                        "\r\nX-Vanguard-Cache-Key: " + r.key_hex + "\r\nX-Vanguard-Owner: " + r.owner + "\r\n" +
                        routing_headers(r);
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
  if (filled && r.semantic && !r.embedding.empty()) {
    s_.semantic.insert(r.scope, r.guard, std::move(r.embedding), r.key);
    inc(s_.metrics.semantic_inserts);
  }

  release_upstream(u);
  finish_exchange(c);
}

// ---------------------------------------------------------------------------
// Upstream side
// ---------------------------------------------------------------------------

UpstreamConn* Worker::acquire_upstream(int target) {
  auto& pool = idle_[static_cast<std::size_t>(target)];
  while (!pool.empty()) {
    const auto id = pool.back();
    pool.pop_back();
    if (UpstreamConn* u = upstream(id)) {
      u->state = UpstreamConn::State::kActive;
      u->reused = true;
      return u;
    }
  }
  const auto& up = target_of(target);
  const int fd = ::socket(up.addr.ss_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) return nullptr;
  set_nodelay(fd);
  const int rc = ::connect(fd, reinterpret_cast<const sockaddr*>(&up.addr), up.addr_len);
  if (rc < 0 && errno != EINPROGRESS) {
    ::close(fd);
    return nullptr;
  }
  auto conn = std::make_unique<UpstreamConn>(target);
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
      attempt_failed(u, 502, std::string("connect to ") + target_of(u->target).name + " failed: " + std::strerror(err),
                     "connect-error", false);
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
    attempt_failed(u, 502, "upstream write failed", "upstream-write-error", false);
    return;
  }
}

void Worker::read_upstream(UpstreamConn* u) {
  while (!u->closed && !u->paused && u->state == UpstreamConn::State::kActive) {
    const ssize_t r = ::recv(u->fd, scratch_.data(), scratch_.size(), 0);
    if (r > 0) {
      u->bytes_received += static_cast<std::size_t>(r);
      if (!u->parser.feed(std::string_view(scratch_.data(), static_cast<std::size_t>(r)), u->decoded)) {
        attempt_failed(u, 502, "malformed upstream response: " + u->parser.error(), "upstream-protocol-error", false);
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
      // The server closed a pooled keep-alive connection just as we reused
      // it. Nothing was processed, so retrying once on a fresh connection to
      // the same upstream is safe, and it is not the upstream's fault.
      ClientConn* c = client(u->client_id);
      const int t = u->target;
      const bool probe = u->probe;
      u->probe = false;
      close_conn(u);
      if (c == nullptr || !c->req) {
        if (probe) breaker(t).release_probe();
        return;
      }
      if (!c->req->retried) {
        c->req->retried = true;
        inc(s_.metrics.upstream_retries);
        if (t == kEmbedderTarget) {
          dispatch_embedding(c, probe);
        } else {
          dispatch_llm(c, t, probe);
        }
      } else if (t == kEmbedderTarget) {
        embedding_failed(c, probe, false);
      } else {
        llm_attempt_failed(c, t, probe, 502, "upstream closed the connection", "upstream-closed", false);
      }
      return;
    }
    attempt_failed(u, 502, "upstream closed the connection mid-response", "upstream-dropped", false);
    return;
  }
}

void Worker::on_upstream_progress(UpstreamConn* u) {
  ClientConn* c = client(u->client_id);
  if (c == nullptr || !c->req) {
    close_conn(u);
    return;
  }
  if (u->target == kEmbedderTarget) {
    if (u->decoded.size() > kMaxEmbeddingBody) {
      attempt_failed(u, 502, "embedding response too large", "embedder-protocol-error", false);
    } else if (u->parser.done()) {
      complete_embedding(u, c);
    }
    return;
  }

  InFlight& r = *c->req;
  if (u->parser.headers_complete() && !u->headers_handled) {
    if (!on_llm_headers(u, c)) return;
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
    attempt_failed(u, 502, "upstream response too large", "upstream-protocol-error", false);
    return;
  }

  if (u->parser.done()) complete_llm(u, c);
}

void Worker::attempt_failed(UpstreamConn* u, int status, std::string_view message, std::string_view reason,
                            bool timeout) {
  ClientConn* c = client(u->client_id);
  const int t = u->target;
  const bool probe = u->probe;
  u->probe = false;  // the outcome is recorded below, not released on close
  close_conn(u);
  if (t == kEmbedderTarget) {
    embedding_failed(c, probe, timeout);
  } else {
    llm_attempt_failed(c, t, probe, status, message, reason, timeout);
  }
}

void Worker::release_upstream(UpstreamConn* u) {
  if (ClientConn* c = client(u->client_id)) c->upstream_id = 0;
  u->client_id = 0;
  u->decoded.clear();
  disarm_timer(u);
  if (u->probe) {
    breaker(u->target).release_probe();
    u->probe = false;
  }
  const bool reusable = u->parser.keep_alive() && u->parser.leftover() == 0 && u->out.empty();
  auto& pool = idle_[static_cast<std::size_t>(u->target)];
  if (reusable && pool.size() < s_.cfg.upstream_max_idle) {
    u->state = UpstreamConn::State::kIdle;
    pool.push_back(u->id);
  } else {
    close_conn(u);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::optional<UpstreamTarget> resolve_target(std::string name, const std::string& url, std::string api_key,
                                             unsigned ttfb_timeout_ms, const CircuitBreaker::Options& breaker,
                                             std::string& error) {
  auto parsed = parse_upstream_url(url, error);
  if (!parsed) {
    error = name + " upstream: " + error;
    return std::nullopt;
  }
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port = std::to_string(parsed->port);
  if (const int rc = ::getaddrinfo(parsed->host.c_str(), port.c_str(), &hints, &res); rc != 0 || res == nullptr) {
    error = "cannot resolve " + name + " upstream host '" + parsed->host + "': " + ::gai_strerror(rc);
    return std::nullopt;
  }
  UpstreamTarget t;
  t.name = std::move(name);
  t.configured = true;
  std::memcpy(&t.addr, res->ai_addr, res->ai_addrlen);
  t.addr_len = static_cast<socklen_t>(res->ai_addrlen);
  ::freeaddrinfo(res);
  t.host_header = parsed->host + ":" + port;
  t.url = std::move(*parsed);
  t.api_key = std::move(api_key);
  t.ttfb_timeout_ms = ttfb_timeout_ms;
  t.breaker = std::make_unique<CircuitBreaker>(breaker);
  return t;
}

SharedState::SharedState(Config c, TenantRegistry t, std::array<UpstreamTarget, 3> upstreams)
    : cfg(std::move(c)),
      tenants(std::move(t)),
      targets(std::move(upstreams)),
      router(targets[kPrimaryTarget].breaker.get(),
             targets[kFallbackTarget].configured ? targets[kFallbackTarget].breaker.get() : nullptr),
      lexical(cfg.semantic_dim),
      limiter(64),
      cache(cfg.cache_max_mb * 1024 * 1024, cfg.cache_shards, std::chrono::seconds(cfg.cache_ttl_s)),
      semantic(SemanticIndex::Options{16, cfg.semantic_max_entries, 2048}),
      ring(cfg.ring_vnodes) {
  if (cfg.ring_nodes.empty()) {
    ring.add_node(cfg.node_id);
  } else {
    for (const auto& n : cfg.ring_nodes) ring.add_node(n);
  }
}

std::vector<BreakerReport> SharedState::breaker_reports() const {
  const auto now = Clock::now();
  std::vector<BreakerReport> out;
  for (const auto& t : targets) {
    BreakerReport b;
    b.name = t.name;
    b.configured = t.configured;
    const auto snap = t.breaker->snapshot(now);
    b.state = static_cast<int>(snap.state);
    b.consecutive_failures = snap.consecutive_failures;
    b.open_seconds = static_cast<double>(snap.open_for.count()) / 1000.0;
    b.trips = snap.trips;
    out.push_back(std::move(b));
  }
  return out;
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
  const auto& fb = state.targets[kFallbackTarget];
  std::fprintf(stderr,
               "vanguard-proxy listening on %s:%u with %u epoll workers -> primary %s%s%s, semantic=%s (threshold %.2f)\n",
               state.cfg.listen_addr.c_str(), state.cfg.listen_port, n, state.cfg.upstream_url.c_str(),
               fb.configured ? ", fallback " : "", fb.configured ? state.cfg.fallback_upstream_url.c_str() : "",
               state.cfg.embedder.c_str(), state.cfg.semantic_threshold);

  std::vector<std::thread> threads;
  threads.reserve(n);
  for (auto& w : workers) threads.emplace_back([&w] { w->run(); });
  for (auto& t : threads) t.join();
  std::fprintf(stderr, "vanguard-proxy: shutdown complete\n");
  return 0;
}

}  // namespace vanguard
