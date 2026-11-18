#include <gtest/gtest.h>

#include <set>
#include <string>

#include "vanguard/murmur3.hpp"

// Reference values from the canonical MurmurHash3_x64_128 (seed 0), as also
// produced by Python's mmh3.hash128.
TEST(Murmur3, KnownVectors) {
  const auto empty = vanguard::murmur3_128("", 0);
  EXPECT_EQ(empty.h1, 0u);
  EXPECT_EQ(empty.h2, 0u);
  const auto hello = vanguard::murmur3_128("hello", 5);
  EXPECT_EQ(hello.h1, 0xcbd8a7b341bd9b02ULL);
  EXPECT_EQ(hello.h2, 0x5b1e906a48ae1d19ULL);
}

TEST(Murmur3, AllTailLengthsAreStableAndDistinct) {
  const std::string s = "The quick brown fox jumps over the lazy dog!!";
  std::set<std::uint64_t> seen;
  for (std::size_t n = 0; n <= s.size(); ++n) {
    const auto h = vanguard::murmur3_64(std::string_view(s).substr(0, n));
    EXPECT_EQ(h, vanguard::murmur3_64(std::string_view(s).substr(0, n)));
    seen.insert(h);
  }
  EXPECT_EQ(seen.size(), s.size() + 1);
}
