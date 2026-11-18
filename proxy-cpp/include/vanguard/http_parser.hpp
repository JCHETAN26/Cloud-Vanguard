#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vanguard {

struct Header {
  std::string name;
  std::string value;
};

// Case-insensitive lookup; returns "" if absent.
std::string_view find_header(const std::vector<Header>& headers, std::string_view name);

std::string_view status_reason(int status) noexcept;

struct HttpRequest {
  std::string method;
  std::string target;  // as sent, e.g. "/v1/chat/completions?x=1"
  std::string path;    // target without the query string
  int version_minor = 1;
  std::vector<Header> headers;
  std::string body;
  bool keep_alive = true;

  [[nodiscard]] std::string_view header(std::string_view name) const {
    return find_header(headers, name);
  }
};

struct HttpError {
  int status = 400;
  std::string message;
};

enum class ParseResult { kIncomplete, kComplete, kError };

// Incremental HTTP/1.1 request parser. Call parse() with everything buffered
// so far; it remembers how far it has scanned for the end of the header block,
// so feeding a request one byte at a time stays linear. On kComplete,
// `consumed` bytes belong to the request and anything after them is the next
// pipelined request. Request bodies must use Content-Length; chunked request
// bodies are rejected with 411.
class RequestParser {
 public:
  struct Limits {
    std::size_t max_header_bytes = 16 * 1024;
    std::size_t max_body_bytes = 4 * 1024 * 1024;
  };

  RequestParser() = default;
  explicit RequestParser(Limits limits) : limits_(limits) {}

  ParseResult parse(std::string_view in, HttpRequest& out, std::size_t& consumed);
  [[nodiscard]] const HttpError& error() const noexcept { return error_; }
  void reset();

 private:
  ParseResult fail(int status, std::string message);
  bool parse_head(std::string_view head);

  Limits limits_{};
  std::size_t scan_from_ = 0;
  std::size_t head_len_ = 0;  // 0 until "\r\n\r\n" has been found
  std::size_t body_len_ = 0;
  bool failed_ = false;
  HttpRequest pending_;
  HttpError error_;
};

// Incremental HTTP/1.1 response parser for upstream responses. Decodes
// Content-Length, chunked and read-until-close bodies; decoded body bytes are
// appended to the caller's buffer as they arrive, which is what lets the proxy
// forward SSE tokens without waiting for the full response.
class ResponseParser {
 public:
  explicit ResponseParser(std::size_t max_head_bytes = 64 * 1024) : max_head_(max_head_bytes) {}

  // Returns false on a protocol error (see error()).
  bool feed(std::string_view in, std::string& body);
  // Call when the peer closes. Returns true if the message is now complete.
  bool finish_on_eof();
  void reset();

  [[nodiscard]] bool headers_complete() const noexcept { return headers_done_; }
  [[nodiscard]] bool done() const noexcept { return state_ == State::kDone; }
  [[nodiscard]] bool failed() const noexcept { return state_ == State::kError; }
  [[nodiscard]] int status() const noexcept { return status_; }
  [[nodiscard]] std::string_view header(std::string_view name) const { return find_header(headers_, name); }
  [[nodiscard]] const std::vector<Header>& headers() const noexcept { return headers_; }
  [[nodiscard]] bool keep_alive() const noexcept { return keep_alive_; }
  [[nodiscard]] bool chunked() const noexcept { return chunked_; }
  // Bytes received after the end of the message (makes the connection unusable).
  [[nodiscard]] std::size_t leftover() const noexcept { return leftover_; }
  [[nodiscard]] const std::string& error() const noexcept { return error_; }

 private:
  enum class State {
    kHead,
    kBodyLength,
    kChunkSize,
    kChunkData,
    kChunkCrlf,
    kTrailers,
    kUntilClose,
    kDone,
    kError
  };

  bool fail(std::string message);
  bool parse_head(std::string_view head);

  std::size_t max_head_;
  State state_ = State::kHead;
  bool headers_done_ = false;
  int status_ = 0;
  bool keep_alive_ = false;
  bool chunked_ = false;
  std::vector<Header> headers_;
  std::string head_;
  std::string line_;
  std::uint64_t remaining_ = 0;
  int crlf_seen_ = 0;
  std::size_t leftover_ = 0;
  std::string error_;
};

}  // namespace vanguard
