#include "vanguard/http_parser.hpp"

#include <algorithm>

#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

constexpr bool is_tchar(char c) noexcept {
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return true;
  switch (c) {
    case '!': case '#': case '$': case '%': case '&': case '\'': case '*': case '+':
    case '-': case '.': case '^': case '_': case '`': case '|': case '~':
      return true;
    default:
      return false;
  }
}

constexpr bool is_token(std::string_view s) noexcept {
  if (s.empty()) return false;
  for (char c : s) {
    if (!is_tchar(c)) return false;
  }
  return true;
}

// Parses "Name: value\r\n..." lines (the block after the start line, without
// the final blank line). Rejects obsolete line folding and malformed names.
bool parse_header_lines(std::string_view block, std::vector<Header>& out) {
  while (!block.empty()) {
    const auto eol = block.find("\r\n");
    const auto line = block.substr(0, eol);
    block = eol == std::string_view::npos ? std::string_view{} : block.substr(eol + 2);
    if (line.empty()) continue;
    if (line.front() == ' ' || line.front() == '\t') return false;  // obs-fold
    const auto colon = line.find(':');
    if (colon == std::string_view::npos) return false;
    const auto name = line.substr(0, colon);
    if (!is_token(name)) return false;
    out.push_back(Header{std::string(name), std::string(trim(line.substr(colon + 1)))});
  }
  return true;
}

}  // namespace

std::string_view find_header(const std::vector<Header>& headers, std::string_view name) {
  for (const auto& h : headers) {
    if (iequals(h.name, name)) return h.value;
  }
  return {};
}

std::string_view status_reason(int status) noexcept {
  switch (status) {
    case 100: return "Continue";
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 304: return "Not Modified";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    case 505: return "HTTP Version Not Supported";
    default: return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// RequestParser
// ---------------------------------------------------------------------------

void RequestParser::reset() {
  scan_from_ = 0;
  head_len_ = 0;
  body_len_ = 0;
  failed_ = false;
  pending_ = HttpRequest{};
  error_ = HttpError{};
}

ParseResult RequestParser::fail(int status, std::string message) {
  failed_ = true;
  error_ = HttpError{status, std::move(message)};
  return ParseResult::kError;
}

ParseResult RequestParser::parse(std::string_view in, HttpRequest& out, std::size_t& consumed) {
  if (failed_) return ParseResult::kError;

  if (head_len_ == 0) {
    const auto pos = in.find("\r\n\r\n", scan_from_);
    if (pos == std::string_view::npos) {
      if (in.size() > limits_.max_header_bytes) return fail(431, "request header too large");
      scan_from_ = in.size() >= 3 ? in.size() - 3 : 0;
      return ParseResult::kIncomplete;
    }
    if (pos + 4 > limits_.max_header_bytes) return fail(431, "request header too large");
    head_len_ = pos + 4;
    if (!parse_head(in.substr(0, pos))) return ParseResult::kError;
  }

  if (in.size() < head_len_ + body_len_) return ParseResult::kIncomplete;

  pending_.body.assign(in.data() + head_len_, body_len_);
  consumed = head_len_ + body_len_;
  out = std::move(pending_);
  reset();
  return ParseResult::kComplete;
}

bool RequestParser::parse_head(std::string_view head) {
  const auto eol = head.find("\r\n");
  const auto request_line = head.substr(0, eol);
  const auto header_block = eol == std::string_view::npos ? std::string_view{} : head.substr(eol + 2);

  // METHOD SP request-target SP HTTP-version
  const auto sp1 = request_line.find(' ');
  const auto sp2 = sp1 == std::string_view::npos ? sp1 : request_line.find(' ', sp1 + 1);
  if (sp1 == std::string_view::npos || sp2 == std::string_view::npos) {
    fail(400, "malformed request line");
    return false;
  }
  const auto method = request_line.substr(0, sp1);
  const auto target = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
  const auto version = request_line.substr(sp2 + 1);
  if (!is_token(method) || target.empty() || target.find(' ') != std::string_view::npos) {
    fail(400, "malformed request line");
    return false;
  }
  if (version == "HTTP/1.1") {
    pending_.version_minor = 1;
  } else if (version == "HTTP/1.0") {
    pending_.version_minor = 0;
  } else if (version.substr(0, 5) == "HTTP/") {
    fail(505, "unsupported HTTP version");
    return false;
  } else {
    fail(400, "malformed request line");
    return false;
  }
  pending_.method.assign(method);
  pending_.target.assign(target);
  pending_.path.assign(target.substr(0, target.find('?')));

  if (!parse_header_lines(header_block, pending_.headers)) {
    fail(400, "malformed header");
    return false;
  }

  // Body framing: Content-Length only. Multiple differing Content-Length
  // headers or CL + Transfer-Encoding are smuggling vectors -> 400.
  bool have_length = false;
  std::uint64_t length = 0;
  bool have_te = false;
  for (const auto& h : pending_.headers) {
    if (iequals(h.name, "content-length")) {
      std::uint64_t v = 0;
      if (!parse_u64(h.value, v) || (have_length && v != length)) {
        fail(400, "invalid Content-Length");
        return false;
      }
      have_length = true;
      length = v;
    } else if (iequals(h.name, "transfer-encoding")) {
      have_te = true;
    }
  }
  if (have_te) {
    if (have_length) {
      fail(400, "both Content-Length and Transfer-Encoding present");
    } else {
      fail(411, "chunked request bodies are not supported; send Content-Length");
    }
    return false;
  }
  if (length > limits_.max_body_bytes) {
    fail(413, "request body too large");
    return false;
  }
  body_len_ = static_cast<std::size_t>(length);

  const auto connection = pending_.header("connection");
  pending_.keep_alive = pending_.version_minor == 1 ? !contains_token(connection, "close")
                                                    : contains_token(connection, "keep-alive");
  return true;
}

// ---------------------------------------------------------------------------
// ResponseParser
// ---------------------------------------------------------------------------

void ResponseParser::reset() {
  state_ = State::kHead;
  headers_done_ = false;
  status_ = 0;
  keep_alive_ = false;
  chunked_ = false;
  headers_.clear();
  head_.clear();
  line_.clear();
  remaining_ = 0;
  crlf_seen_ = 0;
  leftover_ = 0;
  error_.clear();
}

bool ResponseParser::fail(std::string message) {
  state_ = State::kError;
  error_ = std::move(message);
  return false;
}

bool ResponseParser::parse_head(std::string_view head) {
  const auto eol = head.find("\r\n");
  const auto status_line = head.substr(0, eol);
  const auto header_block = eol == std::string_view::npos ? std::string_view{} : head.substr(eol + 2);

  // HTTP/1.x SP 3DIGIT [SP reason]
  if (status_line.size() < 12 || status_line.substr(0, 7) != "HTTP/1." || status_line[8] != ' ') {
    return fail("malformed status line");
  }
  const bool http11 = status_line[7] == '1';
  std::uint64_t code = 0;
  if (!parse_u64(status_line.substr(9, 3), code) || code < 100 || code > 999) {
    return fail("malformed status code");
  }
  headers_.clear();
  if (!parse_header_lines(header_block, headers_)) return fail("malformed header");
  status_ = static_cast<int>(code);

  if (status_ >= 100 && status_ < 200) {
    // Interim response (e.g. 100 Continue): discard and wait for the real one.
    headers_.clear();
    status_ = 0;
    state_ = State::kHead;
    return true;
  }

  headers_done_ = true;
  const auto connection = header("connection");
  keep_alive_ = http11 ? !contains_token(connection, "close") : contains_token(connection, "keep-alive");

  const auto te = header("transfer-encoding");
  const auto cl = header("content-length");
  if (status_ == 204 || status_ == 304) {
    state_ = State::kDone;
  } else if (!te.empty()) {
    if (!contains_token(te, "chunked")) return fail("unsupported transfer-encoding");
    chunked_ = true;
    state_ = State::kChunkSize;
  } else if (!cl.empty()) {
    if (!parse_u64(cl, remaining_)) return fail("invalid Content-Length");
    state_ = remaining_ == 0 ? State::kDone : State::kBodyLength;
  } else {
    state_ = State::kUntilClose;
    keep_alive_ = false;
  }
  return true;
}

bool ResponseParser::feed(std::string_view in, std::string& body) {
  std::size_t i = 0;
  while (i < in.size()) {
    switch (state_) {
      case State::kHead: {
        const std::size_t old = head_.size();
        head_.append(in.data() + i, in.size() - i);
        const auto pos = head_.find("\r\n\r\n", old >= 3 ? old - 3 : 0);
        if (pos == std::string::npos) {
          if (head_.size() > max_head_) return fail("response header too large");
          return true;
        }
        // The terminator always ends inside the newly appended bytes.
        i += pos + 4 - old;
        head_.resize(pos);
        const std::string head = std::move(head_);
        head_.clear();
        if (!parse_head(head)) return false;
        break;
      }
      case State::kBodyLength:
      case State::kChunkData: {
        const auto n = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining_, static_cast<std::uint64_t>(in.size() - i)));
        body.append(in.data() + i, n);
        i += n;
        remaining_ -= n;
        if (remaining_ == 0) {
          if (state_ == State::kBodyLength) {
            state_ = State::kDone;
          } else {
            state_ = State::kChunkCrlf;
            crlf_seen_ = 0;
          }
        }
        break;
      }
      case State::kChunkCrlf: {
        const char c = in[i++];
        if (crlf_seen_ == 0 && c == '\r') {
          crlf_seen_ = 1;
        } else if (c == '\n') {
          state_ = State::kChunkSize;
        } else {
          return fail("missing CRLF after chunk data");
        }
        break;
      }
      case State::kChunkSize:
      case State::kTrailers: {
        const auto nl = in.find('\n', i);
        if (nl == std::string_view::npos) {
          line_.append(in.data() + i, in.size() - i);
          i = in.size();
          if (line_.size() > 4096) return fail("chunk header line too long");
          break;
        }
        line_.append(in.data() + i, nl - i);
        i = nl + 1;
        if (!line_.empty() && line_.back() == '\r') line_.pop_back();
        const std::string line = std::move(line_);
        line_.clear();
        if (state_ == State::kTrailers) {
          if (line.empty()) state_ = State::kDone;  // trailer fields are ignored
          break;
        }
        // chunk-size [; chunk-ext]
        std::string_view size_str = trim(std::string_view(line).substr(0, line.find(';')));
        if (size_str.empty() || size_str.size() > 15) return fail("invalid chunk size");
        std::uint64_t size = 0;
        for (char c : size_str) {
          int d;
          if (c >= '0' && c <= '9') d = c - '0';
          else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
          else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
          else return fail("invalid chunk size");
          size = size * 16 + static_cast<std::uint64_t>(d);
        }
        if (size == 0) {
          state_ = State::kTrailers;
        } else {
          remaining_ = size;
          state_ = State::kChunkData;
        }
        break;
      }
      case State::kUntilClose:
        body.append(in.data() + i, in.size() - i);
        i = in.size();
        break;
      case State::kDone:
        leftover_ += in.size() - i;
        i = in.size();
        break;
      case State::kError:
        return false;
    }
  }
  return state_ != State::kError;
}

bool ResponseParser::finish_on_eof() {
  if (state_ == State::kUntilClose) state_ = State::kDone;
  return state_ == State::kDone;
}

}  // namespace vanguard
