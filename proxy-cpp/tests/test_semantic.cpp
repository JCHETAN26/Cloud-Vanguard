#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "vanguard/embedding.hpp"
#include "vanguard/llm_request.hpp"
#include "vanguard/semantic_index.hpp"

using vanguard::analyze_prompt;
using vanguard::cosine_similarity;
using vanguard::LexicalEmbedder;
using vanguard::SemanticIndex;

namespace {

constexpr float kThreshold = 0.94f;  // the gateway default (SEMANTIC_THRESHOLD)

struct Pair {
  const char* a;
  const char* b;
};

// Would `b` be served from the cache entry for `a` at the default threshold?
bool semantic_hit(const Pair& p, float* sim_out = nullptr) {
  const LexicalEmbedder emb;
  const auto fa = analyze_prompt(p.a);
  const auto fb = analyze_prompt(p.b);
  const float sim = cosine_similarity(emb.embed_tokens(fa.tokens), emb.embed_tokens(fb.tokens));
  if (sim_out != nullptr) *sim_out = sim;
  return fa.guard == fb.guard && sim >= kThreshold;
}

}  // namespace

TEST(PromptFeatures, NormalizesFramingCaseAndInflection) {
  const auto f = analyze_prompt("Can you PLEASE explain how Vaccines work?!");
  EXPECT_EQ(f.tokens, (std::vector<std::string>{"how", "vaccine", "work"}));
  EXPECT_EQ(analyze_prompt("Can you PLEASE explain how Vaccines work?!").guard,
            analyze_prompt("how does a vaccine work").guard);
}

TEST(PromptFeatures, GuardCapturesNumbersAndNegation) {
  EXPECT_NE(analyze_prompt("what is 2+2").guard, analyze_prompt("what is 2+3").guard);
  EXPECT_EQ(analyze_prompt("what is 2+2").guard, analyze_prompt("2 + 2 = ?").guard);
  EXPECT_NE(analyze_prompt("is it safe").guard, analyze_prompt("is it not safe").guard);
  EXPECT_EQ(analyze_prompt("isn't it safe").guard, analyze_prompt("is it not safe").guard);
  EXPECT_NE(analyze_prompt("summarize the 2023 report").guard, analyze_prompt("summarize the 2024 report").guard);
}

TEST(LexicalEmbedder, ProducesUnitVectorsDeterministically) {
  const LexicalEmbedder emb(256);
  const auto v = emb.embed("Explain quantum computing");
  ASSERT_EQ(v.size(), 256u);
  EXPECT_NEAR(cosine_similarity(v, v), 1.0f, 1e-5f);
  EXPECT_EQ(v, emb.embed("Explain quantum computing"));
  EXPECT_TRUE(emb.embed("what is the").empty());  // only stop words
}

// Phrasing variations that should be served from the same cache entry.
TEST(SemanticMatching, PhrasingVariationsHit) {
  const std::vector<Pair> hits = {
      {"Explain quantum computing", "explain quantum computing"},
      {"Explain quantum computing", "  EXPLAIN   quantum computing!!! "},
      {"Explain quantum computing", "Can you please explain quantum computing?"},
      {"Explain quantum computing", "What is quantum computing?"},
      {"Explain quantum computing", "Tell me about quantum computing"},
      {"How do vaccines work?", "how does a vaccine work"},
      {"What are the benefits of unit testing?", "Explain the benefits of unit tests"},
      {"Why is the sky blue?", "why's the sky blue"},
      {"What is 2+2?", "what's 2 + 2"},
  };
  for (const auto& p : hits) {
    float sim = 0;
    EXPECT_TRUE(semantic_hit(p, &sim)) << "'" << p.a << "' vs '" << p.b << "' sim=" << sim;
    std::printf("[semantic] HIT  %.3f  %-40s | %s\n", sim, p.a, p.b);
  }
}

// Different questions that must never share an answer.
TEST(SemanticMatching, DistinctQuestionsMiss) {
  const std::vector<Pair> misses = {
      {"Explain quantum computing", "Explain classical computing"},
      {"Explain quantum computing", "Explain quantum entanglement"},
      {"How do vaccines work?", "Why do vaccines work?"},
      {"How do vaccines work?", "How do vaccines fail?"},
      {"What is 2+2?", "What is 2+3?"},
      {"Is it safe to mix bleach and ammonia?", "Is it not safe to mix bleach and ammonia?"},
      {"Summarize the 2023 annual report", "Summarize the 2024 annual report"},
      {"Translate 'good morning' to French", "Translate 'good morning' to German"},
      {"Write a haiku about the ocean", "Write a limerick about the ocean"},
      {"What is the capital of France?", "What is the population of France?"},
  };
  for (const auto& p : misses) {
    float sim = 0;
    EXPECT_FALSE(semantic_hit(p, &sim)) << "'" << p.a << "' vs '" << p.b << "' sim=" << sim;
    std::printf("[semantic] MISS %.3f  %-40s | %s\n", sim, p.a, p.b);
  }
}

// The documented limit of the lexical encoder: real paraphrases with little
// word overlap need the neural sidecar (EMBEDDER=remote).
TEST(SemanticMatching, LexicalEncoderDoesNotClaimTrueParaphrase) {
  float sim = 0;
  EXPECT_FALSE(semantic_hit({"Explain quantum computing", "What is quantum computing in simple terms?"}, &sim));
  std::printf("[semantic] lexical limit: 'Explain quantum computing' vs '...in simple terms?' sim=%.3f\n", sim);
}

// ---------------------------------------------------------------------------
// SemanticIndex
// ---------------------------------------------------------------------------

namespace {
std::vector<float> unit(std::initializer_list<float> xs) {
  std::vector<float> v(xs);
  vanguard::l2_normalize(v);
  return v;
}
}  // namespace

TEST(SemanticIndex, ThresholdIsInclusiveAndBestMatchWins) {
  SemanticIndex idx;
  idx.insert("s", 1, unit({1, 0, 0}), "k-x");
  idx.insert("s", 1, unit({0.9f, 0.1f, 0}), "k-near");
  const auto q = unit({0.9f, 0.1f, 0});
  const auto m = idx.lookup("s", 1, q, 0.99f);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(m->cache_key, "k-near");
  EXPECT_NEAR(m->similarity, 1.0f, 1e-5f);
  // cos(q, x) ~ 0.9939: a hit at 0.99, a miss at 0.995.
  idx.erase("s", "k-near");
  EXPECT_TRUE(idx.lookup("s", 1, q, 0.99f).has_value());
  EXPECT_FALSE(idx.lookup("s", 1, q, 0.995f).has_value());
}

TEST(SemanticIndex, ScopesAndGuardsIsolate) {
  SemanticIndex idx;
  idx.insert("tenant-a", 7, unit({1, 0}), "a");
  EXPECT_FALSE(idx.lookup("tenant-b", 7, unit({1, 0}), 0.5f).has_value());
  EXPECT_FALSE(idx.lookup("tenant-a", 8, unit({1, 0}), 0.5f).has_value());
  EXPECT_FALSE(idx.lookup("tenant-a", 7, unit({1, 0, 0}), 0.5f).has_value());  // dim mismatch
  EXPECT_TRUE(idx.lookup("tenant-a", 7, unit({1, 0}), 0.5f).has_value());
}

TEST(SemanticIndex, PerScopeCapOverwritesOldest) {
  SemanticIndex idx({1, 1000, 3});
  for (int i = 0; i < 5; ++i) idx.insert("s", 0, unit({1, static_cast<float>(i)}), "k" + std::to_string(i));
  EXPECT_EQ(idx.size(), 3u);
  EXPECT_FALSE(idx.lookup("s", 0, unit({1, 0}), 0.9999f).has_value());  // k0 overwritten
  EXPECT_TRUE(idx.lookup("s", 0, unit({1, 4}), 0.9999f).has_value());
}

TEST(SemanticIndex, ReinsertingSameKeyReplaces) {
  SemanticIndex idx;
  idx.insert("s", 0, unit({1, 0}), "k");
  idx.insert("s", 0, unit({0, 1}), "k");
  EXPECT_EQ(idx.size(), 1u);
  EXPECT_FALSE(idx.lookup("s", 0, unit({1, 0}), 0.9f).has_value());
}

TEST(SemanticIndex, GlobalCapEvictsLeastRecentlyUsedScopes) {
  SemanticIndex idx({1, 4, 10});
  idx.insert("old", 0, unit({1, 0}), "o1");
  idx.insert("old", 0, unit({1, 1}), "o2");
  idx.insert("warm", 0, unit({1, 0}), "w1");
  ASSERT_TRUE(idx.lookup("old", 0, unit({1, 0}), 0.9f).has_value());  // touch "old"
  idx.insert("new", 0, unit({1, 0}), "n1");
  idx.insert("new", 0, unit({0, 1}), "n2");  // 5 > 4: evict LRU scope ("warm")
  EXPECT_FALSE(idx.lookup("warm", 0, unit({1, 0}), 0.9f).has_value());
  EXPECT_TRUE(idx.lookup("old", 0, unit({1, 0}), 0.9f).has_value());
  EXPECT_LE(idx.size(), 4u);
}

TEST(SemanticIndex, ConcurrentInsertAndLookup) {
  SemanticIndex idx({8, 5000, 256});
  const LexicalEmbedder emb(64);
  std::atomic<int> found{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 500; ++i) {
        const std::string scope = "scope-" + std::to_string(i % 20);
        const auto v = emb.embed("question number " + std::to_string(t) + " topic " + std::to_string(i));
        idx.insert(scope, 0, v, "key-" + std::to_string(t) + "-" + std::to_string(i));
        if (idx.lookup(scope, 0, v, 0.99f)) found.fetch_add(1);
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_GT(found.load(), 0);
  EXPECT_LE(idx.size(), 5000u);
}

// ---------------------------------------------------------------------------
// Request scoping
// ---------------------------------------------------------------------------

namespace {
vanguard::ChatRequest parse(const std::string& body) {
  std::string err;
  auto r = vanguard::parse_chat_request(body, err);
  EXPECT_TRUE(r) << err;
  return r.value_or(vanguard::ChatRequest{});
}
}  // namespace

TEST(SemanticScope, OnlyTheFinalUserMessageMayDiffer) {
  const auto a = parse(R"({"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"Explain X"}]})");
  const auto b = parse(R"({"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"What is X?"}]})");
  const auto other_system = parse(R"({"model":"m","messages":[{"role":"system","content":"be verbose"},{"role":"user","content":"Explain X"}]})");
  const auto other_params = parse(R"({"model":"m","temperature":1,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"Explain X"}]})");
  EXPECT_EQ(vanguard::semantic_scope("t", a), vanguard::semantic_scope("t", b));
  EXPECT_NE(vanguard::semantic_scope("t", a), vanguard::semantic_scope("t2", a));
  EXPECT_NE(vanguard::semantic_scope("t", a), vanguard::semantic_scope("t", other_system));
  EXPECT_NE(vanguard::semantic_scope("t", a), vanguard::semantic_scope("t", other_params));
  EXPECT_EQ(vanguard::semantic_text(a), "Explain X");
}

TEST(SemanticScope, IneligibleShapes) {
  EXPECT_EQ(vanguard::semantic_text(parse(R"({"model":"m","messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]})")), "");
  EXPECT_EQ(vanguard::semantic_text(parse(R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})")), "");
  EXPECT_EQ(vanguard::semantic_text(parse(R"({"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]})")), "a\nb");
}

TEST(Embeddings, ParsesOpenAiResponseAndRequest) {
  const auto v = vanguard::parse_embedding_response(R"({"object":"list","data":[{"embedding":[3,4],"index":0}]})");
  ASSERT_EQ(v.size(), 2u);
  EXPECT_NEAR(v[0], 0.6f, 1e-6f);
  EXPECT_TRUE(vanguard::parse_embedding_response(R"({"data":[]})").empty());
  EXPECT_TRUE(vanguard::parse_embedding_response("nope").empty());
  EXPECT_EQ(vanguard::make_embedding_request("bge", "hi \"there\""), R"({"input":"hi \"there\"","model":"bge"})");
  EXPECT_EQ(vanguard::rewrite_model(R"({"model":"llama","messages":[]})", "gpt-x"), R"({"messages":[],"model":"gpt-x"})");
}

// ---------------------------------------------------------------------------
// Salient-token guard: the exact failures bge-small produced at 0.94
// ---------------------------------------------------------------------------

namespace {
bool salient_ok(const char* a, const char* b) {
  return vanguard::salient_compatible(vanguard::salient_tokens(analyze_prompt(a).tokens),
                                      vanguard::salient_tokens(analyze_prompt(b).tokens));
}
}  // namespace

TEST(SalientGuard, RejectsTheNeuralFalsePositives) {
  // Both scored >= 0.954 with BAAI/bge-small-en-v1.5.
  EXPECT_FALSE(salient_ok("How do I reverse a list in Python?", "How do I reverse a list in JavaScript?"));
  EXPECT_FALSE(salient_ok("How do I center a div in CSS?", "How do I center a div vertically in CSS?"));
  // Other entity / qualifier swaps.
  EXPECT_FALSE(salient_ok("Translate 'thank you' to Spanish", "Translate 'thank you' to Japanese"));
  EXPECT_FALSE(salient_ok("What are the health benefits of green tea?", "What are the health risks of green tea?"));
  EXPECT_FALSE(salient_ok("Explain Java generics", "Explain JavaScript generics"));
  EXPECT_FALSE(salient_ok("What is the capital of Australia?", "What is the population of Australia?"));
}

TEST(SalientGuard, AcceptsTheNeuralTruePositives) {
  // The paraphrases bge-small scored >= 0.96: the guard must not block them.
  EXPECT_TRUE(salient_ok("How do I reverse a list in Python?", "What's the way to reverse a Python list?"));
  EXPECT_TRUE(salient_ok("What is the difference between TCP and UDP?", "How does TCP differ from UDP?"));
  EXPECT_TRUE(salient_ok("How do I center a div in CSS?", "What's the CSS to center a div?"));
  // Framing / inflection variations.
  EXPECT_TRUE(salient_ok("Explain quantum computing", "What is quantum computing in simple terms?"));
  EXPECT_TRUE(salient_ok("Center it vertical", "center it vertically"));
}

TEST(SalientGuard, IndexAppliesItAfterTheThreshold) {
  SemanticIndex idx;
  const auto stored = vanguard::salient_tokens(analyze_prompt("How do I reverse a list in Python?").tokens);
  const auto query = vanguard::salient_tokens(analyze_prompt("How do I reverse a list in JavaScript?").tokens);
  idx.insert("s", 0, unit({1, 0}), "python-answer", stored);
  std::size_t rejected = 0;
  // Identical vectors (similarity 1.0) are still refused when a salient token differs.
  EXPECT_FALSE(idx.lookup("s", 0, unit({1, 0}), 0.96f, &query, &rejected).has_value());
  EXPECT_EQ(rejected, 1u);
  EXPECT_TRUE(idx.lookup("s", 0, unit({1, 0}), 0.96f, &stored, &rejected).has_value());
  EXPECT_TRUE(idx.lookup("s", 0, unit({1, 0}), 0.96f).has_value());  // guard is opt-in per call
}
