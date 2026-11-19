#pragma once

// Warm-cache snapshots: persist the response cache so a restarted gateway
// (same pod, or a new pod that re-attaches the same CSI volume) serves hits
// immediately instead of re-paying upstream LLM calls.
//
// Stream format (shared by both storage modes):
//   "CVSNAP01" u32 version u64 created_unix
//   repeated record: u32 payload_len, u32 crc32(payload), payload
//     payload = u8 stream, u16 status, u32 ttl_s (0xFFFFFFFF = no expiry),
//               u32 sse_frames, u16 key_len, key, u16 ctype_len, ctype,
//               u32 body_len, body
//   trailer: "CVEND001" u64 record_count
// All integers little-endian. A record with a bad CRC stops the load; records
// before it are kept.
//
// Storage modes:
//   file   SNAPSHOT_PATH is a file on a filesystem volume: written to
//          "<path>.tmp", fsync'd, renamed over <path>, directory fsync'd.
//   block  SNAPSHOT_PATH is a raw block device (CSI volumeMode: Block) or a
//          raw image file. No filesystem: two slots alternate (A/B). The data
//          goes to the older slot, is fsync'd, and only then that slot's
//          header (generation, length, data CRC, header CRC) is written and
//          fsync'd. Loading picks the highest-generation slot whose header and
//          data CRCs verify, falling back to the other slot. A crash at any
//          point leaves the previously committed snapshot readable.

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vanguard/cached_response.hpp"
#include "vanguard/lru_cache.hpp"

namespace vanguard {

std::uint32_t crc32(const void* data, std::size_t len, std::uint32_t seed = 0) noexcept;

enum class SnapshotMode { kAuto, kFile, kBlock };

std::optional<SnapshotMode> parse_snapshot_mode(std::string_view s);

struct SnapshotSaveStats {
  bool ok = false;
  std::string error;
  std::string mode;            // "file" | "block"
  std::uint64_t entries = 0;
  std::uint64_t bytes = 0;
  std::uint64_t generation = 0;  // block mode slot generation
  int slot = -1;                 // block mode slot written (0 = A, 1 = B)
  double duration_ms = 0;
};

struct SnapshotLoadStats {
  bool ok = false;               // a snapshot was found and parsed (possibly partially)
  std::string error;
  std::string mode;
  std::uint64_t loaded = 0;
  std::uint64_t expired = 0;     // records whose TTL ran out while persisted
  bool truncated = false;        // stopped early on a bad record / missing trailer
  std::uint64_t generation = 0;
  int slot = -1;
  double duration_ms = 0;
};

using ResponseCache = ShardedLruCache<CachedResponse>;

// Writes every live cache entry (LRU -> MRU per shard, so a restore rebuilds
// recency) to `path`.
SnapshotSaveStats save_snapshot(const ResponseCache& cache, const std::string& path, SnapshotMode mode);

// Inserts the snapshot's unexpired entries into `cache` with their remaining TTL.
SnapshotLoadStats load_snapshot(ResponseCache& cache, const std::string& path, SnapshotMode mode);

// Block-mode layout helpers (exposed for tests and tooling).
inline constexpr std::uint64_t kBlockHeaderSize = 4096;
inline constexpr std::uint64_t kBlockDataOffset = 2 * kBlockHeaderSize;
// Capacity of one slot for a device/file of `device_bytes`.
std::uint64_t block_slot_capacity(std::uint64_t device_bytes) noexcept;

}  // namespace vanguard
