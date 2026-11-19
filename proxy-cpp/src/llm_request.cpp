#include "vanguard/llm_request.hpp"

#include "vanguard/sha256.hpp"

namespace vanguard {
namespace {

using nlohmann::json;

void put_field(Sha256& h, std::string_view field) {
  const std::uint64_t len = field.size();
  std::uint8_t prefix[8];
  for (int i = 0; i < 8; ++i) prefix[i] = static_cast<std::uint8_t>(len >> (8 * i));
  h.update(prefix, sizeof(prefix));
  h.update(field);
}

bool read_token_limit(const json& j, const char* name, std::optional<std::uint64_t>& out,
                      std::string& error) {
  const auto it = j.find(name);
  if (it == j.end() || it->is_null()) return true;
  if (!it->is_number_integer() || it->get<std::int64_t>() < 0) {
    error = std::string("'") + name + "' must be a non-negative integer";
    return false;
  }
  out = it->get<std::uint64_t>();
  return true;
}

}  // namespace

std::optional<ChatRequest> parse_chat_request(std::string_view body, std::string& error) {
  json j = json::parse(body.begin(), body.end(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) {
    error = "request body is not valid JSON";
    return std::nullopt;
  }
  if (!j.is_object()) {
    error = "request body must be a JSON object";
    return std::nullopt;
  }

  ChatRequest req;
  const auto model = j.find("model");
  if (model == j.end() || !model->is_string() || model->get_ref<const std::string&>().empty()) {
    error = "'model' is required and must be a non-empty string";
    return std::nullopt;
  }
  req.model = model->get<std::string>();

  const auto messages = j.find("messages");
  if (messages == j.end() || !messages->is_array() || messages->empty()) {
    error = "'messages' is required and must be a non-empty array";
    return std::nullopt;
  }
  for (const auto& m : *messages) {
    const auto role = m.is_object() ? m.find("role") : m.end();
    if (!m.is_object() || role == m.end() || !role->is_string()) {
      error = "each message must be an object with a string 'role'";
      return std::nullopt;
    }
  }

  if (const auto stream = j.find("stream"); stream != j.end() && !stream->is_null()) {
    if (!stream->is_boolean()) {
      error = "'stream' must be a boolean";
      return std::nullopt;
    }
    req.stream = stream->get<bool>();
  }

  std::optional<std::uint64_t> max_completion;
  if (!read_token_limit(j, "max_completion_tokens", max_completion, error)) return std::nullopt;
  if (!read_token_limit(j, "max_tokens", req.max_tokens, error)) return std::nullopt;
  if (max_completion) req.max_tokens = max_completion;

  req.messages = std::move(*messages);
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& k = it.key();
    if (k == "model" || k == "messages" || k == "stream" || k == "user") continue;
    req.params[k] = it.value();
  }
  return req;
}

std::string cache_key(std::string_view tenant_id, const ChatRequest& req) {
  Sha256 h;
  put_field(h, "cloudvanguard/v1");
  put_field(h, tenant_id);
  put_field(h, req.model);
  put_field(h, req.stream ? "stream" : "json");
  put_field(h, req.messages.dump());
  put_field(h, req.params.dump());
  const auto digest = h.finish();
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::uint64_t estimate_tokens(const ChatRequest& req, std::uint64_t default_completion_tokens) {
  std::uint64_t chars = 0;
  for (const auto& m : req.messages) {
    const auto content = m.find("content");
    if (content == m.end()) continue;
    if (content->is_string()) {
      chars += content->get_ref<const std::string&>().size();
    } else if (content->is_array()) {
      for (const auto& part : *content) {
        if (!part.is_object()) continue;
        const auto text = part.find("text");
        if (text != part.end() && text->is_string()) chars += text->get_ref<const std::string&>().size();
      }
    }
  }
  const std::uint64_t prompt = (chars + 3) / 4 + 4 * static_cast<std::uint64_t>(req.messages.size());
  return prompt + req.max_tokens.value_or(default_completion_tokens);
}

std::string semantic_text(const ChatRequest& req) {
  if (req.messages.empty()) return {};
  const auto& last = req.messages.back();
  const auto role = last.find("role");
  if (role == last.end() || *role != "user") return {};
  const auto content = last.find("content");
  if (content == last.end()) return {};
  if (content->is_string()) return content->get<std::string>();
  std::string out;
  if (content->is_array()) {
    for (const auto& part : *content) {
      if (!part.is_object()) continue;
      const auto type = part.find("type");
      const auto text = part.find("text");
      if (type != part.end() && *type != "text") return {};  // images etc.: not eligible
      if (text == part.end() || !text->is_string()) continue;
      if (!out.empty()) out.push_back('\n');
      out += text->get_ref<const std::string&>();
    }
  }
  return out;
}

std::string semantic_scope(std::string_view tenant_id, const ChatRequest& req) {
  Sha256 h;
  put_field(h, "cloudvanguard/semantic/v1");
  put_field(h, tenant_id);
  put_field(h, req.model);
  put_field(h, req.stream ? "stream" : "json");
  json prefix = json::array();
  for (std::size_t i = 0; i + 1 < req.messages.size(); ++i) prefix.push_back(req.messages[i]);
  put_field(h, prefix.dump());
  put_field(h, req.params.dump());
  const auto digest = h.finish();
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::string rewrite_model(std::string_view body, std::string_view model) {
  json j = json::parse(body.begin(), body.end(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) return std::string(body);
  j["model"] = model;
  return j.dump();
}

}  // namespace vanguard
