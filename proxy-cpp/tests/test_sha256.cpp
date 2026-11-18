#include <gtest/gtest.h>

#include <string>

#include "vanguard/sha256.hpp"

using vanguard::Sha256;

// NIST FIPS 180-4 / RFC 6234 test vectors.
TEST(Sha256, KnownVectors) {
  EXPECT_EQ(Sha256::hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Sha256::hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Sha256::hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(Sha256::hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, IncrementalMatchesOneShot) {
  const std::string msg(1000, 'z');
  for (std::size_t step : {1u, 7u, 63u, 64u, 65u, 333u}) {
    Sha256 h;
    for (std::size_t i = 0; i < msg.size(); i += step) h.update(std::string_view(msg).substr(i, step));
    EXPECT_EQ(h.finish(), Sha256::hash(msg)) << "step=" << step;
  }
}
