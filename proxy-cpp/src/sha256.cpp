#include "vanguard/sha256.hpp"

#include <algorithm>
#include <cstring>

#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

constexpr std::array<std::uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) noexcept {
  return (x >> n) | (x << (32 - n));
}

}  // namespace

std::string to_hex(const void* data, std::size_t n) {
  static constexpr char kDigits[] = "0123456789abcdef";
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::string out(n * 2, '0');
  for (std::size_t i = 0; i < n; ++i) {
    out[2 * i] = kDigits[p[i] >> 4];
    out[2 * i + 1] = kDigits[p[i] & 0x0f];
  }
  return out;
}

void Sha256::reset() noexcept {
  h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  buf_len_ = 0;
  total_len_ = 0;
}

void Sha256::transform(const std::uint8_t* block) noexcept {
  std::array<std::uint32_t, 64> w{};
  for (int i = 0; i < 16; ++i) {
    w[i] = (std::uint32_t{block[4 * i]} << 24) | (std::uint32_t{block[4 * i + 1]} << 16) |
           (std::uint32_t{block[4 * i + 2]} << 8) | std::uint32_t{block[4 * i + 3]};
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
  std::uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t big_s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch = (e & f) ^ (~e & g);
    const std::uint32_t t1 = h + big_s1 + ch + kK[i] + w[i];
    const std::uint32_t big_s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = big_s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

void Sha256::update(const void* data, std::size_t len) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  total_len_ += len;
  if (buf_len_ > 0) {
    const std::size_t take = std::min(len, buf_.size() - buf_len_);
    std::memcpy(buf_.data() + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    len -= take;
    if (buf_len_ == buf_.size()) {
      transform(buf_.data());
      buf_len_ = 0;
    }
  }
  while (len >= 64) {
    transform(p);
    p += 64;
    len -= 64;
  }
  if (len > 0) {
    std::memcpy(buf_.data(), p, len);
    buf_len_ = len;
  }
}

Sha256::Digest Sha256::finish() noexcept {
  const std::uint64_t bit_len = total_len_ * 8;
  buf_[buf_len_++] = 0x80;
  if (buf_len_ > 56) {
    std::memset(buf_.data() + buf_len_, 0, 64 - buf_len_);
    transform(buf_.data());
    buf_len_ = 0;
  }
  std::memset(buf_.data() + buf_len_, 0, 56 - buf_len_);
  for (int i = 0; i < 8; ++i) {
    buf_[56 + i] = static_cast<std::uint8_t>(bit_len >> (56 - 8 * i));
  }
  transform(buf_.data());

  Digest out{};
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<std::uint8_t>(h_[i] >> 24);
    out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
    out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
    out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
  }
  reset();
  return out;
}

Sha256::Digest Sha256::hash(std::string_view s) noexcept {
  Sha256 h;
  h.update(s);
  return h.finish();
}

std::string Sha256::hex(std::string_view s) {
  const auto d = hash(s);
  return to_hex(d.data(), d.size());
}

}  // namespace vanguard
