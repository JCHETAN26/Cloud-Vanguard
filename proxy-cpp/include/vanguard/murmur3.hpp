#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vanguard {

// MurmurHash3_x64_128 (Austin Appleby, public domain). Returns the full
// 128-bit result as {h1, h2}; murmur3_64 returns h1, which is what the hash
// ring and cache-shard selection use.
struct Hash128 {
  std::uint64_t h1;
  std::uint64_t h2;
};

Hash128 murmur3_128(const void* key, std::size_t len, std::uint32_t seed = 0) noexcept;

inline std::uint64_t murmur3_64(std::string_view s, std::uint32_t seed = 0) noexcept {
  return murmur3_128(s.data(), s.size(), seed).h1;
}

}  // namespace vanguard
