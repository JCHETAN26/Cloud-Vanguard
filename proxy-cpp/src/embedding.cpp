#include "vanguard/embedding.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "vanguard/murmur3.hpp"
#include "vanguard/strutil.hpp"

namespace vanguard {
namespace {

// Function words plus question/instruction framing that rarely changes what
// is being asked ("what is X" ~ "explain X" ~ "tell me about X"). Interrogatives
// that change intent (how, why, when, where, who, which) are deliberately kept.
const std::unordered_set<std::string_view>& stop_words() {
  static const std::unordered_set<std::string_view> kWords = {
      "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
      "of", "to", "in", "on", "at", "for", "from", "by", "with", "about", "as", "into", "and", "or",
      "it", "its", "this", "that", "these", "those", "there", "here", "i", "me", "my", "we", "us",
      "our", "you", "your", "he", "she", "they", "them", "their", "what", "whats", "please", "kindly",
      "can", "could", "would", "should", "will", "shall", "may", "might", "must", "tell", "explain",
      "describe", "define", "give", "show", "help", "understand", "know", "want", "need", "like",
      "just", "really", "hey", "hi", "hello", "thanks", "thank", "so", "some", "any", "also", "very"};
  return kWords;
}

const std::unordered_set<std::string_view>& negation_words() {
  static const std::unordered_set<std::string_view> kWords = {
      "not", "no", "never", "without", "none", "nor", "cannot", "neither", "nobody", "nothing", "nowhere"};
  return kWords;
}

bool ends_with(const std::string& s, std::string_view suffix) {
  return s.size() >= suffix.size() && std::string_view(s).substr(s.size() - suffix.size()) == suffix;
}

// Deliberately tiny suffix stripper: it only needs to be consistent, not
// linguistically correct ("vaccines"/"vaccine" -> "vaccine", "computing" ->
// "comput", "computers" -> "computer").
std::string stem(std::string t) {
  if (t.size() > 4 && ends_with(t, "ies")) {
    t.resize(t.size() - 3);
    t.push_back('y');
  } else if (t.size() > 5 && ends_with(t, "ing")) {
    t.resize(t.size() - 3);
  } else if (t.size() > 4 && ends_with(t, "ed")) {
    t.resize(t.size() - 2);
  } else if (t.size() > 4 && (ends_with(t, "ches") || ends_with(t, "shes") || ends_with(t, "xes") ||
                              ends_with(t, "sses") || ends_with(t, "zes"))) {
    t.resize(t.size() - 2);
  } else if (t.size() > 3 && t.back() == 's' && !ends_with(t, "ss") && !ends_with(t, "us") &&
             !ends_with(t, "is")) {
    t.pop_back();
  }
  return t;
}

bool has_digit(std::string_view s) {
  return std::any_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

}  // namespace

PromptFeatures analyze_prompt(std::string_view text) {
  std::vector<std::string> raw;
  std::string cur;
  auto flush = [&] {
    if (!cur.empty()) raw.push_back(std::move(cur));
    cur.clear();
  };
  for (const char ch : text) {
    const auto u = static_cast<unsigned char>(ch);
    const bool word = (u >= '0' && u <= '9') || (u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
                      u >= 0x80 || ch == '\'';
    if (word) {
      cur.push_back(ascii_lower(ch));
    } else {
      flush();
    }
  }
  flush();

  PromptFeatures out;
  std::vector<std::string> numeric;
  int negations = 0;
  for (auto& t : raw) {
    while (!t.empty() && t.front() == '\'') t.erase(t.begin());
    while (!t.empty() && t.back() == '\'') t.pop_back();
    if (t.empty()) continue;
    if (ends_with(t, "n't")) {  // don't, isn't, can't, won't ...
      ++negations;
      out.tokens.emplace_back("not");
      continue;
    }
    if (ends_with(t, "'s")) t.resize(t.size() - 2);
    t.erase(std::remove(t.begin(), t.end(), '\''), t.end());
    if (t.empty()) continue;
    if (negation_words().count(t) != 0) {
      ++negations;
      out.tokens.emplace_back("not");
      continue;
    }
    if (has_digit(t)) {
      numeric.push_back(t);
      out.tokens.push_back(std::move(t));
      continue;
    }
    if (stop_words().count(t) != 0) continue;
    out.tokens.push_back(stem(std::move(t)));
  }

  std::sort(numeric.begin(), numeric.end());
  std::string guard_material;
  for (const auto& n : numeric) guard_material.append(n).push_back('\x1f');
  guard_material.append("|neg=").append(std::to_string(negations));
  out.guard = murmur3_64(guard_material, 0x5eed);
  return out;
}

std::vector<float> LexicalEmbedder::embed_tokens(const std::vector<std::string>& tokens) const {
  if (tokens.empty()) return {};
  std::vector<float> v(dim_, 0.0f);
  auto add = [&](std::string_view feature, std::uint32_t seed, float weight) {
    const std::uint64_t h = murmur3_64(feature, seed);
    v[h % dim_] += (h >> 63) != 0 ? -weight : weight;  // signed hashing de-biases collisions
  };

  std::unordered_map<std::string, int> tf;
  for (const auto& t : tokens) ++tf[t];
  std::string tri;
  for (const auto& [token, count] : tf) {
    const float w = 1.0f + std::log(static_cast<float>(count));
    add(token, 1, w);
    // Character trigrams of "#token#" make near-identical spellings overlap.
    const std::string padded = "#" + token + "#";
    const std::size_t n = padded.size() - 2;
    const float tri_w = 0.4f * w / std::sqrt(static_cast<float>(n));
    for (std::size_t i = 0; i < n; ++i) add(std::string_view(padded).substr(i, 3), 2, tri_w);
  }
  std::string bigram;
  for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
    bigram.assign(tokens[i]).push_back(' ');
    bigram.append(tokens[i + 1]);
    add(bigram, 3, 0.5f);
  }
  if (!l2_normalize(v)) return {};
  return v;
}

bool l2_normalize(std::vector<float>& v) noexcept {
  double sum = 0;
  for (const float x : v) sum += static_cast<double>(x) * x;
  if (sum <= 0) return false;
  const auto inv = static_cast<float>(1.0 / std::sqrt(sum));
  for (float& x : v) x *= inv;
  return true;
}

float dot(const float* a, const float* b, std::size_t n) noexcept {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;  // independent accumulators vectorize well
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) noexcept {
  if (a.size() != b.size() || a.empty()) return 0.0f;
  const float ab = dot(a.data(), b.data(), a.size());
  const float aa = dot(a.data(), a.data(), a.size());
  const float bb = dot(b.data(), b.data(), b.size());
  if (aa <= 0 || bb <= 0) return 0.0f;
  return ab / std::sqrt(aa * bb);
}

std::vector<float> parse_embedding_response(std::string_view json) {
  const auto j = nlohmann::json::parse(json.begin(), json.end(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) return {};
  const auto data = j.find("data");
  if (data == j.end() || !data->is_array() || data->empty()) return {};
  const auto& first = (*data)[0];
  const auto emb = first.is_object() ? first.find("embedding") : first.end();
  if (!first.is_object() || emb == first.end() || !emb->is_array() || emb->empty()) return {};
  std::vector<float> v;
  v.reserve(emb->size());
  for (const auto& x : *emb) {
    if (!x.is_number()) return {};
    v.push_back(x.get<float>());
  }
  if (!l2_normalize(v)) return {};
  return v;
}

std::string make_embedding_request(std::string_view model, std::string_view input) {
  nlohmann::json j;
  j["model"] = model;
  j["input"] = input;
  return j.dump();
}

}  // namespace vanguard
