#include <gtest/gtest.h>

#include <string>

#include "vanguard/llm_request.hpp"

using vanguard::cache_key;
using vanguard::ChatRequest;
using vanguard::estimate_tokens;
using vanguard::parse_chat_request;

namespace {

ChatRequest must_parse(const std::string& body) {
  std::string err;
  auto r = parse_chat_request(body, err);
  EXPECT_TRUE(r.has_value()) << err << " for " << body;
  return r.value_or(ChatRequest{});
}

}  // namespace

TEST(ChatRequest, ParsesCoreFields) {
  const auto r = must_parse(
      R"({"model":"gpt-x","stream":true,"max_tokens":64,"temperature":0.2,"user":"u1",)"
      R"("messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]})");
  EXPECT_EQ(r.model, "gpt-x");
  EXPECT_TRUE(r.stream);
  EXPECT_EQ(r.max_tokens.value_or(0), 64u);
  EXPECT_EQ(r.messages.size(), 2u);
  EXPECT_TRUE(r.params.contains("temperature"));
  EXPECT_TRUE(r.params.contains("max_tokens"));
  EXPECT_FALSE(r.params.contains("user"));
  EXPECT_FALSE(r.params.contains("stream"));
}

TEST(ChatRequest, MaxCompletionTokensWins) {
  const auto r = must_parse(
      R"({"model":"m","max_tokens":10,"max_completion_tokens":20,"messages":[{"role":"user","content":"x"}]})");
  EXPECT_EQ(r.max_tokens.value_or(0), 20u);
}

TEST(ChatRequest, ValidationErrors) {
  const std::vector<std::string> bad = {
      "not json",
      "[]",
      R"({"messages":[{"role":"user","content":"x"}]})",
      R"({"model":"","messages":[{"role":"user","content":"x"}]})",
      R"({"model":"m"})",
      R"({"model":"m","messages":[]})",
      R"({"model":"m","messages":["hi"]})",
      R"({"model":"m","messages":[{"content":"no role"}]})",
      R"({"model":"m","stream":"yes","messages":[{"role":"user","content":"x"}]})",
      R"({"model":"m","max_tokens":-1,"messages":[{"role":"user","content":"x"}]})",
  };
  for (const auto& b : bad) {
    std::string err;
    EXPECT_FALSE(parse_chat_request(b, err).has_value()) << b;
    EXPECT_FALSE(err.empty());
  }
}

TEST(CacheKey, CanonicalUnderKeyOrderAndWhitespace) {
  const auto a = must_parse(R"({"model":"m","temperature":0,"messages":[{"role":"user","content":"hi"}]})");
  const auto b = must_parse(R"({ "messages" : [ { "content" : "hi", "role" : "user" } ],
                                 "temperature": 0, "model": "m", "user": "someone" })");
  EXPECT_EQ(cache_key("t1", a), cache_key("t1", b));
  EXPECT_EQ(cache_key("t1", a).size(), 32u);
}

TEST(CacheKey, DistinguishesEverythingThatChangesTheAnswer) {
  const auto base = must_parse(R"({"model":"m","messages":[{"role":"user","content":"hi"}]})");
  const auto k = cache_key("t1", base);
  EXPECT_NE(k, cache_key("t2", base)) << "tenants must never share entries";
  EXPECT_NE(k, cache_key("t1", must_parse(R"({"model":"m2","messages":[{"role":"user","content":"hi"}]})")));
  EXPECT_NE(k, cache_key("t1", must_parse(R"({"model":"m","messages":[{"role":"user","content":"hi!"}]})")));
  EXPECT_NE(k, cache_key("t1", must_parse(R"({"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]})")));
  EXPECT_NE(k, cache_key("t1", must_parse(R"({"model":"m","max_tokens":5,"messages":[{"role":"user","content":"hi"}]})")));
  // Length-prefixing prevents field-boundary collisions.
  EXPECT_NE(cache_key("ab", base), cache_key("a", must_parse(R"({"model":"bm","messages":[{"role":"user","content":"hi"}]})")));
}

TEST(EstimateTokens, CountsPromptAndCompletionBudget) {
  const auto r = must_parse(
      R"({"model":"m","max_tokens":100,"messages":[{"role":"user","content":"12345678"},)"
      R"({"role":"user","content":[{"type":"text","text":"abcd"},{"type":"image_url","image_url":{}}]}]})");
  // 12 chars -> 3 tokens, + 4 per message * 2, + 100 completion.
  EXPECT_EQ(estimate_tokens(r, 256), 3u + 8u + 100u);
  const auto no_max = must_parse(R"({"model":"m","messages":[{"role":"user","content":""}]})");
  EXPECT_EQ(estimate_tokens(no_max, 256), 4u + 256u);
}
