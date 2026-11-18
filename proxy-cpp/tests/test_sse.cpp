#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "vanguard/http_parser.hpp"
#include "vanguard/sse.hpp"

using vanguard::SseFrameSplitter;

TEST(SseFrameSplitter, SplitsAcrossArbitraryBoundaries) {
  const std::string stream =
      "data: {\"a\":1}\n\n"
      "event: x\ndata: {\"b\":2}\n\n"
      "data: [DONE]\n\n";
  // Feed with every possible split point; result must always be identical.
  for (std::size_t cut = 0; cut <= stream.size(); ++cut) {
    SseFrameSplitter s;
    std::vector<std::string> frames;
    s.feed(std::string_view(stream).substr(0, cut), frames);
    s.feed(std::string_view(stream).substr(cut), frames);
    ASSERT_EQ(frames.size(), 3u) << "cut=" << cut;
    EXPECT_EQ(frames[0], "data: {\"a\":1}\n\n");
    EXPECT_EQ(frames[1], "event: x\ndata: {\"b\":2}\n\n");
    EXPECT_TRUE(vanguard::is_sse_done_frame(frames[2]));
    EXPECT_TRUE(s.pending().empty());
  }
}

TEST(SseFrameSplitter, HandlesCrlfFramesByteByByte) {
  const std::string stream = "data: one\r\n\r\ndata: two\r\n\r\ndata: partial";
  SseFrameSplitter s;
  std::vector<std::string> frames;
  for (char c : stream) s.feed(std::string_view(&c, 1), frames);
  ASSERT_EQ(frames.size(), 2u);
  EXPECT_EQ(frames[0], "data: one\r\n\r\n");
  EXPECT_EQ(frames[1], "data: two\r\n\r\n");
  EXPECT_EQ(s.pending(), "data: partial");
}

TEST(Sse, DoneDetection) {
  EXPECT_TRUE(vanguard::is_sse_done_frame("data: [DONE]\n\n"));
  EXPECT_TRUE(vanguard::is_sse_done_frame("data:[DONE]\r\n\r\n"));
  EXPECT_FALSE(vanguard::is_sse_done_frame("data: {\"content\":\"[DONE]\"}\n\n"));
  EXPECT_FALSE(vanguard::is_sse_done_frame(": keep-alive\n\n"));
}

TEST(Sse, ChunkEncodingRoundTripsThroughResponseParser) {
  std::string wire = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
  const std::vector<std::string> frames = {"data: " + std::string(300, 'x') + "\n\n", "data: [DONE]\n\n"};
  for (const auto& f : frames) vanguard::append_http_chunk(wire, f);
  vanguard::append_http_chunk(wire, "");  // must be a no-op, not a terminator
  wire += vanguard::kLastHttpChunk;
  vanguard::ResponseParser p;
  std::string body;
  ASSERT_TRUE(p.feed(wire, body));
  EXPECT_TRUE(p.done());
  EXPECT_EQ(body, frames[0] + frames[1]);
}
