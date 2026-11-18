#include "vanguard/sse.hpp"

#include <algorithm>
#include <cstdio>

#include "vanguard/strutil.hpp"

namespace vanguard {

void SseFrameSplitter::feed(std::string_view data, std::vector<std::string>& frames_out) {
  buf_.append(data);
  std::size_t start = 0;
  for (;;) {
    const std::size_t from = std::max(start, scan_from_);
    const auto lf = buf_.find("\n\n", from);
    const auto crlf = buf_.find("\r\n\r\n", from);
    if (lf == std::string::npos && crlf == std::string::npos) break;
    const std::size_t end = (crlf != std::string::npos && (lf == std::string::npos || crlf < lf))
                                ? crlf + 4
                                : lf + 2;
    frames_out.emplace_back(buf_, start, end - start);
    start = end;
  }
  if (start > 0) buf_.erase(0, start);
  // Next search may start 3 bytes back to catch a terminator split across feeds.
  scan_from_ = buf_.size() >= 3 ? buf_.size() - 3 : 0;
}

bool is_sse_done_frame(std::string_view frame) noexcept {
  while (!frame.empty()) {
    const auto nl = frame.find('\n');
    auto line = frame.substr(0, nl);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.substr(0, 5) == "data:" && trim(line.substr(5)) == "[DONE]") return true;
    if (nl == std::string_view::npos) break;
    frame.remove_prefix(nl + 1);
  }
  return false;
}

void append_http_chunk(std::string& out, std::string_view data) {
  if (data.empty()) return;
  char size_line[32];
  const int n = std::snprintf(size_line, sizeof(size_line), "%zx\r\n", data.size());
  out.append(size_line, static_cast<std::size_t>(n));
  out.append(data);
  out.append("\r\n");
}

}  // namespace vanguard
