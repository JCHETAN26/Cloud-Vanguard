#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace vanguard {

// Splits a Server-Sent Events byte stream into complete events. An event ends
// at a blank line ("\n\n" or "\r\n\r\n"); each emitted frame includes its
// terminator, so concatenating frames reproduces the original stream exactly.
// Upstream TCP segmentation is arbitrary, so partial events are buffered
// until their terminator arrives.
class SseFrameSplitter {
 public:
  void feed(std::string_view data, std::vector<std::string>& frames_out);
  [[nodiscard]] const std::string& pending() const noexcept { return buf_; }
  void reset() {
    buf_.clear();
    scan_from_ = 0;
  }

 private:
  std::string buf_;
  std::size_t scan_from_ = 0;
};

// True for the OpenAI end-of-stream sentinel event ("data: [DONE]").
bool is_sse_done_frame(std::string_view frame) noexcept;

// Appends `data` as one HTTP/1.1 chunk ("<hex>\r\n<data>\r\n"). Empty input is
// ignored, because a zero-length chunk would terminate the body.
void append_http_chunk(std::string& out, std::string_view data);

inline constexpr std::string_view kLastHttpChunk = "0\r\n\r\n";

}  // namespace vanguard
