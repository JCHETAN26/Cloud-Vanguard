#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vanguard {

// Normalized view of a prompt used by the semantic cache.
struct PromptFeatures {
  // Lower-cased, stemmed content tokens: stop words and question framing
  // ("what is", "explain", "please", "can you") removed, every negation
  // normalized to "not".
  std::vector<std::string> tokens;
  // Hash of the facts a paraphrase must preserve exactly: the multiset of
  // tokens containing digits ("2", "gpt4", "2024") and the number of
  // negations. Two prompts can only match semantically if their guards are
  // equal, so "what is 2+2" never matches "what is 2+3" and "is X safe" never
  // matches "is X not safe", however close their embeddings are.
  std::uint64_t guard = 0;
};

PromptFeatures analyze_prompt(std::string_view text);

// In-process lexical encoder: signed feature hashing of unigrams (log tf),
// adjacent bigrams and per-token character trigrams into a fixed-size,
// L2-normalized vector. ~10-30 us per prompt and no model files.
//
// What it is: a near-duplicate detector. It is robust to casing,
// punctuation, whitespace, filler words, question framing and inflection.
// What it is not: a semantic model. Paraphrases that share few words
// ("Explain quantum computing" vs "How do qubits work?") score low. Use the
// neural sidecar (EMBEDDER=remote) for that.
class LexicalEmbedder {
 public:
  explicit LexicalEmbedder(std::size_t dim = 512) : dim_(dim == 0 ? 1 : dim) {}

  // Empty result means "nothing to embed" (no content tokens).
  [[nodiscard]] std::vector<float> embed(std::string_view text) const {
    return embed_tokens(analyze_prompt(text).tokens);
  }
  [[nodiscard]] std::vector<float> embed_tokens(const std::vector<std::string>& tokens) const;
  [[nodiscard]] std::size_t dim() const noexcept { return dim_; }

 private:
  std::size_t dim_;
};

// Scales `v` to unit length; returns false (leaving v unchanged) if it is all zeros.
bool l2_normalize(std::vector<float>& v) noexcept;

// Dot product of two equal-length vectors (cosine similarity for unit vectors).
float dot(const float* a, const float* b, std::size_t n) noexcept;
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) noexcept;

// Parses an OpenAI-compatible /v1/embeddings response ({"data":[{"embedding":[...]}]})
// into a unit vector. Empty on any error.
std::vector<float> parse_embedding_response(std::string_view json);

// Request body for an OpenAI-compatible /v1/embeddings call.
std::string make_embedding_request(std::string_view model, std::string_view input);

}  // namespace vanguard
