#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "vanguard/snapshot.hpp"

using namespace std::chrono_literals;
using vanguard::CachedResponse;
using vanguard::ResponseCache;
using vanguard::SnapshotMode;

namespace fs = std::filesystem;

namespace {

fs::path temp_path(const std::string& name) {
  auto dir = fs::temp_directory_path() / ("vanguard-snap-" + std::to_string(::getpid()));
  fs::create_directories(dir);
  return dir / name;
}

// A raw "volume": a zero-filled file standing in for a CSI block device.
fs::path raw_volume(const std::string& name, std::uintmax_t bytes) {
  const auto p = temp_path(name);
  fs::remove(p);
  std::ofstream(p, std::ios::binary).close();
  fs::resize_file(p, bytes);
  return p;
}

std::shared_ptr<const CachedResponse> response(const std::string& body, bool stream = false) {
  auto r = std::make_shared<CachedResponse>();
  r->stream = stream;
  r->content_type = stream ? "text/event-stream" : "application/json";
  r->body = body;
  r->sse_frames = stream ? 3 : 0;
  return r;
}

void fill(ResponseCache& c, int n, const std::string& tag = "") {
  for (int i = 0; i < n; ++i) {
    const std::string key = "key-" + std::to_string(i);
    c.put(key, response(tag + "body-" + std::to_string(i) + std::string(200, 'x'), i % 5 == 0), 512);
  }
}

}  // namespace

TEST(Crc32, KnownVector) {
  EXPECT_EQ(vanguard::crc32("123456789", 9), 0xCBF43926u);
  // Incremental == one-shot.
  EXPECT_EQ(vanguard::crc32("6789", 4, vanguard::crc32("12345", 5)), 0xCBF43926u);
}

TEST(Snapshot, FileModeRoundTripPreservesEverything) {
  ResponseCache src(64 << 20, 8, 3600s);
  fill(src, 500);
  const auto path = temp_path("cache.snap").string();
  const auto saved = vanguard::save_snapshot(src, path, SnapshotMode::kFile);
  ASSERT_TRUE(saved.ok) << saved.error;
  EXPECT_EQ(saved.entries, 500u);
  EXPECT_FALSE(fs::exists(path + ".tmp"));

  ResponseCache dst(64 << 20, 8, 3600s);
  const auto loaded = vanguard::load_snapshot(dst, path, SnapshotMode::kFile);
  ASSERT_TRUE(loaded.ok) << loaded.error;
  EXPECT_EQ(loaded.loaded, 500u);
  EXPECT_FALSE(loaded.truncated);
  for (int i = 0; i < 500; ++i) {
    const auto v = dst.get("key-" + std::to_string(i));
    ASSERT_NE(v, nullptr) << i;
    EXPECT_EQ(v->body, src.get("key-" + std::to_string(i))->body);
    EXPECT_EQ(v->stream, i % 5 == 0);
    EXPECT_EQ(v->content_type, i % 5 == 0 ? "text/event-stream" : "application/json");
  }
}

TEST(Snapshot, RestorePreservesRecencyAndTtl) {
  ResponseCache src(1 << 20, 1, 0s);
  src.put("old", response("o"), 100, vanguard::SteadyClock::now(), 3600s);
  src.put("new", response("n"), 100, vanguard::SteadyClock::now(), 2s);
  src.put("forever", response("f"), 100);  // cache-wide TTL 0 = no expiry
  ASSERT_NE(src.get("old"), nullptr);       // "old" becomes most recent
  const auto path = temp_path("order.snap").string();
  ASSERT_TRUE(vanguard::save_snapshot(src, path, SnapshotMode::kFile).ok);

  ResponseCache dst(1 << 20, 1, 0s);
  ASSERT_EQ(vanguard::load_snapshot(dst, path, SnapshotMode::kFile).loaded, 3u);
  std::vector<std::string> order;
  dst.for_each_lru_to_mru(vanguard::SteadyClock::now(),
                          [&](const std::string& k, const auto&, auto) { order.push_back(k); });
  EXPECT_EQ(order, (std::vector<std::string>{"new", "forever", "old"}));
  // Remaining TTLs survive: "new" expires within ~3 s, "forever" never does.
  EXPECT_EQ(dst.get("new", vanguard::SteadyClock::now() + 4s), nullptr);
  EXPECT_NE(dst.get("old", vanguard::SteadyClock::now() + 4s), nullptr);
  EXPECT_NE(dst.get("forever", vanguard::SteadyClock::now() + 100000h), nullptr);
}

TEST(Snapshot, BlockModeAlternatesSlotsAndLoadsNewest) {
  const auto vol = raw_volume("vol-ab.raw", 8 << 20).string();
  ResponseCache c(64 << 20, 4, 3600s);
  fill(c, 100, "v1-");
  const auto s1 = vanguard::save_snapshot(c, vol, SnapshotMode::kBlock);
  ASSERT_TRUE(s1.ok) << s1.error;
  EXPECT_EQ(s1.slot, 0);
  EXPECT_EQ(s1.generation, 1u);
  fill(c, 150, "v2-");
  const auto s2 = vanguard::save_snapshot(c, vol, SnapshotMode::kBlock);
  ASSERT_TRUE(s2.ok) << s2.error;
  EXPECT_EQ(s2.slot, 1);
  EXPECT_EQ(s2.generation, 2u);
  EXPECT_EQ(fs::file_size(vol), 8u << 20) << "block mode never resizes the device";

  ResponseCache dst(64 << 20, 4, 3600s);
  const auto l = vanguard::load_snapshot(dst, vol, SnapshotMode::kBlock);
  ASSERT_TRUE(l.ok) << l.error;
  EXPECT_EQ(l.generation, 2u);
  EXPECT_EQ(l.loaded, 150u);
  EXPECT_EQ(dst.get("key-7")->body.substr(0, 3), "v2-");
}

// Crash consistency: a write torn mid-way (data written, header never
// committed) must leave the previous snapshot as the one that loads.
TEST(Snapshot, TornWriteFallsBackToPreviousCommittedSnapshot) {
  const auto vol = raw_volume("vol-torn.raw", 8 << 20).string();
  ResponseCache c(64 << 20, 4, 3600s);
  fill(c, 50, "good-");
  ASSERT_TRUE(vanguard::save_snapshot(c, vol, SnapshotMode::kBlock).ok);  // gen 1, slot A
  fill(c, 50, "next-");
  ASSERT_TRUE(vanguard::save_snapshot(c, vol, SnapshotMode::kBlock).ok);  // gen 2, slot B
  // Corrupt slot B's data region as if power failed while rewriting it.
  {
    std::fstream f(vol, std::ios::in | std::ios::out | std::ios::binary);
    const auto cap = vanguard::block_slot_capacity(8 << 20);
    f.seekp(static_cast<std::streamoff>(vanguard::kBlockDataOffset + cap + 100));
    f.write("GARBAGE!", 8);
  }
  ResponseCache dst(64 << 20, 4, 3600s);
  const auto l = vanguard::load_snapshot(dst, vol, SnapshotMode::kBlock);
  ASSERT_TRUE(l.ok) << l.error;
  EXPECT_EQ(l.generation, 1u) << "fell back to the older committed slot";
  EXPECT_EQ(l.loaded, 50u);
  EXPECT_NE(l.error.find("checksum"), std::string::npos);
  EXPECT_EQ(dst.get("key-3")->body.substr(0, 5), "good-");
}

TEST(Snapshot, EmptyVolumeAndCapacityErrors) {
  const auto vol = raw_volume("vol-empty.raw", 1 << 20).string();
  ResponseCache dst(1 << 20, 1, 0s);
  const auto l = vanguard::load_snapshot(dst, vol, SnapshotMode::kBlock);
  EXPECT_FALSE(l.ok);
  EXPECT_EQ(l.loaded, 0u);

  ResponseCache big(512 << 20, 4, 0s);
  for (int i = 0; i < 40; ++i) big.put("k" + std::to_string(i), response(std::string(64 << 10, 'z')), 70000);
  const auto s = vanguard::save_snapshot(big, vol, SnapshotMode::kBlock);
  EXPECT_FALSE(s.ok);
  EXPECT_NE(s.error.find("too small"), std::string::npos);
}

TEST(Snapshot, CorruptFileKeepsValidPrefix) {
  ResponseCache src(64 << 20, 1, 0s);
  fill(src, 20);
  const auto path = temp_path("corrupt.snap").string();
  ASSERT_TRUE(vanguard::save_snapshot(src, path, SnapshotMode::kFile).ok);
  const auto size = fs::file_size(path);
  fs::resize_file(path, size - 400);  // truncated (e.g. copied while being written)
  ResponseCache dst(64 << 20, 1, 0s);
  const auto l = vanguard::load_snapshot(dst, path, SnapshotMode::kFile);
  EXPECT_TRUE(l.ok);
  EXPECT_TRUE(l.truncated);
  EXPECT_GT(l.loaded, 0u);
  EXPECT_LT(l.loaded, 20u);

  std::ofstream(path, std::ios::binary | std::ios::trunc) << "not a snapshot at all";
  EXPECT_FALSE(vanguard::load_snapshot(dst, path, SnapshotMode::kFile).ok);
}
