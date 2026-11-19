#include "vanguard/snapshot.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/fs.h>
#include <sys/ioctl.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <string_view>

namespace vanguard {
namespace {

using Clock = std::chrono::steady_clock;

constexpr char kStreamMagic[8] = {'C', 'V', 'S', 'N', 'A', 'P', '0', '1'};
constexpr char kTrailerMagic[8] = {'C', 'V', 'E', 'N', 'D', '0', '0', '1'};
constexpr char kSlotMagic[8] = {'C', 'V', 'B', 'L', 'K', 'S', 'L', '1'};
constexpr std::uint32_t kStreamVersion = 1;
constexpr std::uint32_t kNoExpiry = 0xFFFFFFFFu;
constexpr std::size_t kSlotHeaderBytes = 8 + 8 + 8 + 4 + 8 + 4;  // through header_crc

std::array<std::uint32_t, 256> make_crc_table() {
  std::array<std::uint32_t, 256> t{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    t[i] = c;
  }
  return t;
}

// --- little-endian encoding -------------------------------------------------

void put_u8(std::string& b, std::uint8_t v) { b.push_back(static_cast<char>(v)); }
void put_u16(std::string& b, std::uint16_t v) {
  for (int i = 0; i < 2; ++i) b.push_back(static_cast<char>(v >> (8 * i)));
}
void put_u32(std::string& b, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<char>(v >> (8 * i)));
}
void put_u64(std::string& b, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(static_cast<char>(v >> (8 * i)));
}

struct Cursor {
  std::string_view data;
  std::size_t pos = 0;

  bool has(std::size_t n) const { return data.size() - pos >= n; }
  template <typename T>
  bool get(T& out) {
    if (!has(sizeof(T))) return false;
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      v |= static_cast<std::uint64_t>(static_cast<unsigned char>(data[pos + i])) << (8 * i);
    }
    out = static_cast<T>(v);
    pos += sizeof(T);
    return true;
  }
  bool bytes(std::size_t n, std::string_view& out) {
    if (!has(n)) return false;
    out = data.substr(pos, n);
    pos += n;
    return true;
  }
};

// --- collecting & encoding ---------------------------------------------------

struct Item {
  std::string key;
  std::shared_ptr<const CachedResponse> value;
  std::uint32_t ttl_s;
};

std::vector<Item> collect(const ResponseCache& cache, Clock::time_point now) {
  std::vector<Item> items;
  cache.for_each_lru_to_mru(now, [&](const std::string& key, const std::shared_ptr<const CachedResponse>& value,
                                     Clock::time_point expires) {
    std::uint32_t ttl = kNoExpiry;
    if (expires != Clock::time_point::max()) {
      const auto left = std::chrono::duration_cast<std::chrono::seconds>(expires - now).count() + 1;
      ttl = static_cast<std::uint32_t>(std::max<long long>(1, std::min<long long>(left, kNoExpiry - 1)));
    }
    items.push_back(Item{key, value, ttl});
  });
  return items;
}

std::string encode_payload(const Item& it) {
  const auto& v = *it.value;
  std::string p;
  p.reserve(32 + it.key.size() + v.content_type.size() + v.body.size());
  put_u8(p, v.stream ? 1 : 0);
  put_u16(p, static_cast<std::uint16_t>(v.status));
  put_u32(p, it.ttl_s);
  put_u32(p, static_cast<std::uint32_t>(v.sse_frames));
  put_u16(p, static_cast<std::uint16_t>(it.key.size()));
  p.append(it.key);
  put_u16(p, static_cast<std::uint16_t>(v.content_type.size()));
  p.append(v.content_type);
  put_u32(p, static_cast<std::uint32_t>(v.body.size()));
  p.append(v.body);
  return p;
}

std::uint64_t payload_size(const Item& it) {
  return 1 + 2 + 4 + 4 + 2 + it.key.size() + 2 + it.value->content_type.size() + 4 + it.value->body.size();
}

std::uint64_t stream_size(const std::vector<Item>& items) {
  std::uint64_t n = 8 + 4 + 8 + 8 + 8;
  for (const auto& it : items) n += 8 + payload_size(it);
  return n;
}

// Sequential writer at an offset; tracks a running CRC of everything written.
struct Sink {
  Sink(int f, std::uint64_t off) : fd(f), offset(off) {}
  int fd;
  std::uint64_t offset;
  std::uint64_t written = 0;
  std::uint32_t crc = 0;
  std::string error;

  bool write(std::string_view b) {
    crc = crc32(b.data(), b.size(), crc);
    std::size_t done = 0;
    while (done < b.size()) {
      const ssize_t w = ::pwrite(fd, b.data() + done, b.size() - done, static_cast<off_t>(offset + written + done));
      if (w < 0) {
        if (errno == EINTR) continue;
        error = std::string("write: ") + std::strerror(errno);
        return false;
      }
      done += static_cast<std::size_t>(w);
    }
    written += b.size();
    return true;
  }
};

bool write_stream(Sink& sink, const std::vector<Item>& items) {
  std::string head(kStreamMagic, 8);
  put_u32(head, kStreamVersion);
  put_u64(head, static_cast<std::uint64_t>(std::time(nullptr)));
  if (!sink.write(head)) return false;
  std::string frame;
  for (const auto& it : items) {
    const std::string payload = encode_payload(it);
    frame.clear();
    put_u32(frame, static_cast<std::uint32_t>(payload.size()));
    put_u32(frame, crc32(payload.data(), payload.size()));
    if (!sink.write(frame) || !sink.write(payload)) return false;
  }
  std::string tail(kTrailerMagic, 8);
  put_u64(tail, items.size());
  return sink.write(tail);
}

// --- parsing -----------------------------------------------------------------

void parse_stream(std::string_view data, ResponseCache& cache, SnapshotLoadStats& st) {
  Cursor c{data};
  std::string_view magic;
  std::uint32_t version = 0;
  std::uint64_t created = 0;
  if (!c.bytes(8, magic) || magic != std::string_view(kStreamMagic, 8) || !c.get(version) || !c.get(created) ||
      version != kStreamVersion) {
    st.error = "not a vanguard snapshot (bad magic/version)";
    return;
  }
  st.ok = true;
  const auto now = Clock::now();
  for (;;) {
    if (c.has(8) && data.substr(c.pos, 8) == std::string_view(kTrailerMagic, 8)) {
      std::uint64_t count = 0;
      c.pos += 8;
      if (!c.get(count) || count != st.loaded + st.expired) {
        st.truncated = true;
        st.error = "trailer count mismatch";
      }
      return;
    }
    std::uint32_t len = 0, crc = 0;
    std::string_view payload;
    if (!c.get(len) || !c.get(crc) || !c.bytes(len, payload) || crc32(payload.data(), payload.size()) != crc) {
      st.truncated = true;
      st.error = "corrupt or truncated record after " + std::to_string(st.loaded + st.expired) + " entries";
      return;
    }
    Cursor p{payload};
    std::uint8_t stream = 0;
    std::uint16_t status = 0, klen = 0, clen = 0;
    std::uint32_t ttl = 0, frames = 0, blen = 0;
    std::string_view key, ctype, body;
    if (!p.get(stream) || !p.get(status) || !p.get(ttl) || !p.get(frames) || !p.get(klen) || !p.bytes(klen, key) ||
        !p.get(clen) || !p.bytes(clen, ctype) || !p.get(blen) || !p.bytes(blen, body) || p.pos != payload.size()) {
      st.truncated = true;
      st.error = "malformed record";
      return;
    }
    // TTLs are relative to save time; downtime counts against them.
    const auto age = static_cast<std::int64_t>(std::time(nullptr)) - static_cast<std::int64_t>(created);
    std::optional<std::chrono::seconds> ttl_left = std::chrono::seconds(0);  // 0 = no expiry
    if (ttl != kNoExpiry) {
      const std::int64_t left = static_cast<std::int64_t>(ttl) - std::max<std::int64_t>(0, age);
      if (left <= 0) {
        ++st.expired;
        continue;
      }
      ttl_left = std::chrono::seconds(left);
    }
    auto entry = std::make_shared<CachedResponse>();
    entry->stream = stream != 0;
    entry->status = status;
    entry->sse_frames = frames;
    entry->content_type.assign(ctype);
    entry->body.assign(body);
    const auto charge = entry->charge(key.size());
    cache.put(std::string(key), std::move(entry), charge, now, ttl_left);
    ++st.loaded;
  }
}

bool read_at(int fd, std::uint64_t offset, std::size_t n, std::string& out) {
  out.resize(n);
  std::size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pread(fd, out.data() + done, n - done, static_cast<off_t>(offset + done));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    done += static_cast<std::size_t>(r);
  }
  return true;
}

// --- block mode --------------------------------------------------------------

struct SlotHeader {
  bool valid = false;
  std::uint64_t generation = 0;
  std::uint64_t data_len = 0;
  std::uint32_t data_crc = 0;
  std::uint64_t created = 0;
};

SlotHeader read_slot_header(int fd, int slot) {
  SlotHeader h;
  std::string buf;
  if (!read_at(fd, static_cast<std::uint64_t>(slot) * kBlockHeaderSize, kSlotHeaderBytes, buf)) return h;
  Cursor c{buf};
  std::string_view magic;
  std::uint32_t header_crc = 0;
  if (!c.bytes(8, magic) || magic != std::string_view(kSlotMagic, 8)) return h;
  c.get(h.generation);
  c.get(h.data_len);
  c.get(h.data_crc);
  c.get(h.created);
  c.get(header_crc);
  h.valid = crc32(buf.data(), kSlotHeaderBytes - 4) == header_crc && h.generation > 0;
  return h;
}

std::string encode_slot_header(std::uint64_t gen, std::uint64_t len, std::uint32_t data_crc) {
  std::string b(kSlotMagic, 8);
  put_u64(b, gen);
  put_u64(b, len);
  put_u32(b, data_crc);
  put_u64(b, static_cast<std::uint64_t>(std::time(nullptr)));
  put_u32(b, crc32(b.data(), b.size()));
  b.resize(kBlockHeaderSize, '\0');
  return b;
}

std::uint64_t device_size(int fd) {
  struct stat st {};
  if (::fstat(fd, &st) != 0) return 0;
#if defined(__linux__)
  if (S_ISBLK(st.st_mode)) {
    std::uint64_t bytes = 0;
    if (::ioctl(fd, BLKGETSIZE64, &bytes) == 0) return bytes;
    return 0;
  }
#endif
  return static_cast<std::uint64_t>(st.st_size);
}

std::uint64_t slot_offset(int slot, std::uint64_t capacity) {
  return kBlockDataOffset + static_cast<std::uint64_t>(slot) * capacity;
}

SnapshotMode resolve_mode(const std::string& path, SnapshotMode mode) {
  if (mode != SnapshotMode::kAuto) return mode;
  struct stat st {};
  return (::stat(path.c_str(), &st) == 0 && S_ISBLK(st.st_mode)) ? SnapshotMode::kBlock : SnapshotMode::kFile;
}

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

std::uint32_t crc32(const void* data, std::size_t len, std::uint32_t seed) noexcept {
  static const auto kTable = make_crc_table();
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint32_t c = seed ^ 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) c = kTable[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

std::optional<SnapshotMode> parse_snapshot_mode(std::string_view s) {
  if (s == "auto" || s.empty()) return SnapshotMode::kAuto;
  if (s == "file") return SnapshotMode::kFile;
  if (s == "block") return SnapshotMode::kBlock;
  return std::nullopt;
}

std::uint64_t block_slot_capacity(std::uint64_t device_bytes) noexcept {
  if (device_bytes <= kBlockDataOffset) return 0;
  return ((device_bytes - kBlockDataOffset) / 2) / kBlockHeaderSize * kBlockHeaderSize;
}

SnapshotSaveStats save_snapshot(const ResponseCache& cache, const std::string& path, SnapshotMode requested) {
  const auto t0 = Clock::now();
  SnapshotSaveStats st;
  const auto mode = resolve_mode(path, requested);
  st.mode = mode == SnapshotMode::kBlock ? "block" : "file";
  const auto items = collect(cache, Clock::now());
  st.entries = items.size();

  if (mode == SnapshotMode::kFile) {
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
      st.error = "open " + tmp + ": " + std::strerror(errno);
      return st;
    }
    Sink sink(fd, 0);
    const bool ok = write_stream(sink, items) && ::fsync(fd) == 0;
    ::close(fd);
    if (!ok) {
      st.error = sink.error.empty() ? "fsync failed" : sink.error;
      ::unlink(tmp.c_str());
      return st;
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
      st.error = "rename: " + std::string(std::strerror(errno));
      return st;
    }
    // Make the rename itself durable.
    const auto slash = path.rfind('/');
    const std::string dir = slash == std::string::npos ? "." : (slash == 0 ? "/" : path.substr(0, slash));
    if (const int dfd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC); dfd >= 0) {
      ::fsync(dfd);
      ::close(dfd);
    }
    st.bytes = sink.written;
    st.ok = true;
    st.duration_ms = ms_since(t0);
    return st;
  }

  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    st.error = "open " + path + ": " + std::strerror(errno);
    return st;
  }
  const std::uint64_t capacity = block_slot_capacity(device_size(fd));
  const std::uint64_t needed = stream_size(items);
  if (needed > capacity) {
    ::close(fd);
    st.error = "snapshot needs " + std::to_string(needed) + " bytes but a slot holds " + std::to_string(capacity) +
               " (volume too small)";
    return st;
  }
  const SlotHeader a = read_slot_header(fd, 0);
  const SlotHeader b = read_slot_header(fd, 1);
  // Overwrite the older (or invalid) slot; the newer one stays intact until
  // this write is fully durable.
  const int slot = !a.valid ? 0 : !b.valid ? 1 : (a.generation <= b.generation ? 0 : 1);
  const std::uint64_t generation = std::max(a.valid ? a.generation : 0, b.valid ? b.generation : 0) + 1;
  Sink sink(fd, slot_offset(slot, capacity));
  bool ok = write_stream(sink, items) && ::fsync(fd) == 0;
  if (ok) {
    const std::string header = encode_slot_header(generation, sink.written, sink.crc);
    Sink hs(fd, static_cast<std::uint64_t>(slot) * kBlockHeaderSize);
    ok = hs.write(header) && ::fsync(fd) == 0;
    if (!ok) sink.error = hs.error.empty() ? "header fsync failed" : hs.error;
  }
  ::close(fd);
  if (!ok) {
    st.error = sink.error.empty() ? "fsync failed" : sink.error;
    return st;
  }
  st.bytes = sink.written;
  st.slot = slot;
  st.generation = generation;
  st.ok = true;
  st.duration_ms = ms_since(t0);
  return st;
}

SnapshotLoadStats load_snapshot(ResponseCache& cache, const std::string& path, SnapshotMode requested) {
  const auto t0 = Clock::now();
  SnapshotLoadStats st;
  const auto mode = resolve_mode(path, requested);
  st.mode = mode == SnapshotMode::kBlock ? "block" : "file";
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    st.error = "open " + path + ": " + std::strerror(errno);
    return st;
  }

  if (mode == SnapshotMode::kFile) {
    std::string data;
    const auto size = device_size(fd);
    if (!read_at(fd, 0, static_cast<std::size_t>(size), data)) {
      st.error = "read failed";
    } else {
      parse_stream(data, cache, st);
    }
    ::close(fd);
    st.duration_ms = ms_since(t0);
    return st;
  }

  const std::uint64_t capacity = block_slot_capacity(device_size(fd));
  SlotHeader headers[2] = {read_slot_header(fd, 0), read_slot_header(fd, 1)};
  int order[2] = {0, 1};
  if (headers[1].valid && (!headers[0].valid || headers[1].generation > headers[0].generation)) {
    order[0] = 1;
    order[1] = 0;
  }
  st.error = "no committed snapshot on volume";
  for (const int slot : order) {
    const auto& h = headers[slot];
    if (!h.valid || h.data_len > capacity) continue;
    std::string data;
    if (!read_at(fd, slot_offset(slot, capacity), static_cast<std::size_t>(h.data_len), data) ||
        crc32(data.data(), data.size()) != h.data_crc) {
      st.error = "slot " + std::to_string(slot) + " (generation " + std::to_string(h.generation) +
                 ") failed its data checksum; trying the other slot";
      continue;
    }
    SnapshotLoadStats parsed;
    parsed.mode = st.mode;
    parse_stream(data, cache, parsed);
    parsed.slot = slot;
    parsed.generation = h.generation;
    if (!st.error.empty() && parsed.error.empty() && slot == order[1]) parsed.error = st.error;  // fell back
    st = parsed;
    break;
  }
  ::close(fd);
  st.duration_ms = ms_since(t0);
  return st;
}

}  // namespace vanguard
