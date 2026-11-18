#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vanguard {

// Self-contained FIPS 180-4 SHA-256 (no OpenSSL dependency). Used to derive
// cache keys and anonymous tenant ids; not used for anything secret.
class Sha256 {
 public:
  using Digest = std::array<std::uint8_t, 32>;

  Sha256() noexcept { reset(); }
  void reset() noexcept;
  void update(const void* data, std::size_t len) noexcept;
  void update(std::string_view s) noexcept { update(s.data(), s.size()); }
  Digest finish() noexcept;

  static Digest hash(std::string_view s) noexcept;
  static std::string hex(std::string_view s);

 private:
  void transform(const std::uint8_t* block) noexcept;

  std::array<std::uint32_t, 8> h_{};
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
  std::uint64_t total_len_ = 0;
};

}  // namespace vanguard
