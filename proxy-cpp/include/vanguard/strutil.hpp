#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vanguard {

constexpr char ascii_lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

constexpr bool iequals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (ascii_lower(a[i]) != ascii_lower(b[i])) return false;
  }
  return true;
}

constexpr std::string_view trim(std::string_view s) noexcept {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

// True if the comma-separated header list contains `token` (case-insensitive),
// e.g. contains_token("keep-alive, Upgrade", "upgrade").
constexpr bool contains_token(std::string_view list, std::string_view token) noexcept {
  while (!list.empty()) {
    const auto comma = list.find(',');
    const auto item = trim(list.substr(0, comma));
    if (iequals(item, token)) return true;
    if (comma == std::string_view::npos) break;
    list.remove_prefix(comma + 1);
  }
  return false;
}

// Strict unsigned decimal parse (no sign, no whitespace, overflow-checked).
constexpr bool parse_u64(std::string_view s, std::uint64_t& out) noexcept {
  if (s.empty() || s.size() > 20) return false;
  std::uint64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const auto d = static_cast<std::uint64_t>(c - '0');
    if (v > (UINT64_MAX - d) / 10) return false;
    v = v * 10 + d;
  }
  out = v;
  return true;
}

std::string to_hex(const void* data, std::size_t n);
inline std::string to_hex(std::string_view s) { return to_hex(s.data(), s.size()); }

}  // namespace vanguard
