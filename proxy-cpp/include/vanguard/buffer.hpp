#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace vanguard {

// Growable byte buffer with a read cursor, used for per-connection input and
// output queues. Producers append at the back, consumers advance the front.
// Storage is compacted lazily, so repeated partial consumes (e.g. after a
// short write() under EAGAIN) are amortized O(1) instead of O(n) memmoves.
class Buffer {
 public:
  [[nodiscard]] const char* data() const noexcept { return buf_.data() + rpos_; }
  [[nodiscard]] std::size_t size() const noexcept { return buf_.size() - rpos_; }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] std::string_view view() const noexcept { return {data(), size()}; }

  void append(std::string_view s) {
    if (rpos_ >= kCompactThreshold && rpos_ * 2 >= buf_.size()) compact();
    buf_.append(s.data(), s.size());
  }

  void consume(std::size_t n) noexcept {
    rpos_ += n;
    if (rpos_ >= buf_.size()) {
      buf_.clear();
      rpos_ = 0;
    }
  }

  void clear() noexcept {
    buf_.clear();
    rpos_ = 0;
  }

 private:
  void compact() {
    buf_.erase(0, rpos_);
    rpos_ = 0;
  }

  static constexpr std::size_t kCompactThreshold = 4096;
  std::string buf_;
  std::size_t rpos_ = 0;
};

}  // namespace vanguard
