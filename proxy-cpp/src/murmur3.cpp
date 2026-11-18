#include "vanguard/murmur3.hpp"

#include <cstring>

namespace vanguard {
namespace {

constexpr std::uint64_t rotl64(std::uint64_t x, int r) noexcept {
  return (x << r) | (x >> (64 - r));
}

constexpr std::uint64_t fmix64(std::uint64_t k) noexcept {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

// Little-endian load. Every supported target (x86-64, aarch64) is LE; memcpy
// keeps this free of alignment/aliasing UB.
inline std::uint64_t load64(const std::uint8_t* p) noexcept {
  std::uint64_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

}  // namespace

Hash128 murmur3_128(const void* key, std::size_t len, std::uint32_t seed) noexcept {
  const auto* data = static_cast<const std::uint8_t*>(key);
  const std::size_t nblocks = len / 16;
  std::uint64_t h1 = seed;
  std::uint64_t h2 = seed;
  constexpr std::uint64_t c1 = 0x87c37b91114253d5ULL;
  constexpr std::uint64_t c2 = 0x4cf5ad432745937fULL;

  for (std::size_t i = 0; i < nblocks; ++i) {
    std::uint64_t k1 = load64(data + i * 16);
    std::uint64_t k2 = load64(data + i * 16 + 8);

    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;
    h2 = rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  const std::uint8_t* tail = data + nblocks * 16;
  std::uint64_t k1 = 0;
  std::uint64_t k2 = 0;
  switch (len & 15) {
    case 15: k2 ^= std::uint64_t{tail[14]} << 48; [[fallthrough]];
    case 14: k2 ^= std::uint64_t{tail[13]} << 40; [[fallthrough]];
    case 13: k2 ^= std::uint64_t{tail[12]} << 32; [[fallthrough]];
    case 12: k2 ^= std::uint64_t{tail[11]} << 24; [[fallthrough]];
    case 11: k2 ^= std::uint64_t{tail[10]} << 16; [[fallthrough]];
    case 10: k2 ^= std::uint64_t{tail[9]} << 8; [[fallthrough]];
    case 9:
      k2 ^= std::uint64_t{tail[8]};
      k2 *= c2;
      k2 = rotl64(k2, 33);
      k2 *= c1;
      h2 ^= k2;
      [[fallthrough]];
    case 8: k1 ^= std::uint64_t{tail[7]} << 56; [[fallthrough]];
    case 7: k1 ^= std::uint64_t{tail[6]} << 48; [[fallthrough]];
    case 6: k1 ^= std::uint64_t{tail[5]} << 40; [[fallthrough]];
    case 5: k1 ^= std::uint64_t{tail[4]} << 32; [[fallthrough]];
    case 4: k1 ^= std::uint64_t{tail[3]} << 24; [[fallthrough]];
    case 3: k1 ^= std::uint64_t{tail[2]} << 16; [[fallthrough]];
    case 2: k1 ^= std::uint64_t{tail[1]} << 8; [[fallthrough]];
    case 1:
      k1 ^= std::uint64_t{tail[0]};
      k1 *= c1;
      k1 = rotl64(k1, 31);
      k1 *= c2;
      h1 ^= k1;
      break;
    default:
      break;
  }

  h1 ^= static_cast<std::uint64_t>(len);
  h2 ^= static_cast<std::uint64_t>(len);
  h1 += h2;
  h2 += h1;
  h1 = fmix64(h1);
  h2 = fmix64(h2);
  h1 += h2;
  h2 += h1;
  return {h1, h2};
}

}  // namespace vanguard
