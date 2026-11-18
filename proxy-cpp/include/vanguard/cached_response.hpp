#pragma once

#include <cstddef>
#include <string>

namespace vanguard {

// An immutable cached completion.
//  - JSON responses: `body` is the upstream response body.
//  - SSE responses: `body` is the stream already framed as HTTP/1.1 chunks,
//    one chunk per SSE event, ending with the terminal "0\r\n\r\n". Replaying
//    a stream from cache is a single memcpy into the socket buffer.
struct CachedResponse {
  bool stream = false;
  int status = 200;
  std::string content_type;
  std::string body;
  std::size_t sse_frames = 0;

  [[nodiscard]] std::size_t charge(std::size_t key_bytes) const noexcept {
    // Payload plus a rough allocation/bookkeeping overhead per entry.
    return body.size() + content_type.size() + key_bytes + 160;
  }
};

}  // namespace vanguard
