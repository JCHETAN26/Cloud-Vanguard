#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace vanguard {

// The fields of an OpenAI-compatible /v1/chat/completions body that matter to
// the gateway. The original body is forwarded upstream byte-for-byte; this is
// only used for validation, cache keying and cost estimation.
struct ChatRequest {
  std::string model;
  nlohmann::json messages;                          // validated non-empty array
  nlohmann::json params = nlohmann::json::object();  // other generation params
  bool stream = false;
  std::optional<std::uint64_t> max_tokens;  // max_completion_tokens or max_tokens
};

std::optional<ChatRequest> parse_chat_request(std::string_view body, std::string& error);

// 32-byte raw SHA-256 over a length-prefixed encoding of
//   (tenant_id, model, stream, canonical(messages), canonical(params)).
// nlohmann::json orders object keys, so dump() is canonical: requests that
// differ only in JSON key order or whitespace share a key. Generation params
// (temperature, max_tokens, tools, ...) are part of the key because they change
// the completion; "user" is excluded because it only tags the end user.
std::string cache_key(std::string_view tenant_id, const ChatRequest& req);

// Rough cost estimate used for TPM limiting before the upstream call:
// ceil(prompt chars / 4) + 4 per message + the requested completion budget
// (or `default_completion_tokens` when the client sets none).
std::uint64_t estimate_tokens(const ChatRequest& req, std::uint64_t default_completion_tokens);

}  // namespace vanguard
