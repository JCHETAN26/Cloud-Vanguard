#include <gtest/gtest.h>

#include <string>

#include "vanguard/http_parser.hpp"

using vanguard::HttpRequest;
using vanguard::ParseResult;
using vanguard::RequestParser;
using vanguard::ResponseParser;

TEST(RequestParser, ParsesGet) {
  RequestParser p;
  HttpRequest req;
  std::size_t consumed = 0;
  const std::string raw = "GET /healthz?verbose=1 HTTP/1.1\r\nHost: x\r\nUser-Agent: t\r\n\r\n";
  ASSERT_EQ(p.parse(raw, req, consumed), ParseResult::kComplete);
  EXPECT_EQ(consumed, raw.size());
  EXPECT_EQ(req.method, "GET");
  EXPECT_EQ(req.path, "/healthz");
  EXPECT_EQ(req.target, "/healthz?verbose=1");
  EXPECT_EQ(req.header("HOST"), "x");
  EXPECT_TRUE(req.keep_alive);
}

TEST(RequestParser, PostBodyArrivingByteByByte) {
  const std::string body = R"({"model":"m","messages":[{"role":"user","content":"hi"}]})";
  const std::string raw = "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n"
                          "Authorization: Bearer sk-1\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  RequestParser p;
  HttpRequest req;
  std::size_t consumed = 0;
  for (std::size_t n = 1; n < raw.size(); ++n) {
    ASSERT_EQ(p.parse(std::string_view(raw).substr(0, n), req, consumed), ParseResult::kIncomplete) << n;
  }
  ASSERT_EQ(p.parse(raw, req, consumed), ParseResult::kComplete);
  EXPECT_EQ(req.body, body);
  EXPECT_EQ(req.header("authorization"), "Bearer sk-1");
}

TEST(RequestParser, PipelinedRequests) {
  const std::string raw =
      "POST /a HTTP/1.1\r\nContent-Length: 3\r\n\r\nabcGET /b HTTP/1.1\r\n\r\n";
  RequestParser p;
  HttpRequest req;
  std::size_t consumed = 0;
  ASSERT_EQ(p.parse(raw, req, consumed), ParseResult::kComplete);
  EXPECT_EQ(req.body, "abc");
  const auto rest = std::string_view(raw).substr(consumed);
  ASSERT_EQ(p.parse(rest, req, consumed), ParseResult::kComplete);
  EXPECT_EQ(req.path, "/b");
  EXPECT_EQ(consumed, rest.size());
}

TEST(RequestParser, ConnectionSemantics) {
  RequestParser p;
  HttpRequest req;
  std::size_t consumed = 0;
  ASSERT_EQ(p.parse("GET / HTTP/1.1\r\nConnection: close\r\n\r\n", req, consumed), ParseResult::kComplete);
  EXPECT_FALSE(req.keep_alive);
  ASSERT_EQ(p.parse("GET / HTTP/1.0\r\n\r\n", req, consumed), ParseResult::kComplete);
  EXPECT_FALSE(req.keep_alive);
  ASSERT_EQ(p.parse("GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n", req, consumed), ParseResult::kComplete);
  EXPECT_TRUE(req.keep_alive);
}

TEST(RequestParser, Errors) {
  struct Case {
    std::string raw;
    int status;
  };
  const std::vector<Case> cases = {
      {"GARBAGE\r\n\r\n", 400},
      {"GET / HTTP/2.0\r\n\r\n", 505},
      {"GET / HTTP/1.1\r\nBad Header\r\n\r\n", 400},
      {"GET / HTTP/1.1\r\n folded: x\r\n\r\n", 400},
      {"POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n", 400},
      {"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n", 400},
      {"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 411},
      {"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\n", 400},
      {"POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n", 413},
  };
  for (const auto& c : cases) {
    RequestParser p;
    HttpRequest req;
    std::size_t consumed = 0;
    EXPECT_EQ(p.parse(c.raw, req, consumed), ParseResult::kError) << c.raw;
    EXPECT_EQ(p.error().status, c.status) << c.raw;
  }
}

TEST(RequestParser, HeaderTooLarge) {
  RequestParser p(RequestParser::Limits{64, 1024});
  HttpRequest req;
  std::size_t consumed = 0;
  EXPECT_EQ(p.parse("GET / HTTP/1.1\r\nX-Pad: " + std::string(100, 'a'), req, consumed), ParseResult::kError);
  EXPECT_EQ(p.error().status, 431);
}

TEST(ResponseParser, ContentLength) {
  ResponseParser p;
  std::string body;
  ASSERT_TRUE(p.feed("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 5\r\n\r\nhel", body));
  EXPECT_TRUE(p.headers_complete());
  EXPECT_FALSE(p.done());
  ASSERT_TRUE(p.feed("lo", body));
  EXPECT_TRUE(p.done());
  EXPECT_EQ(body, "hello");
  EXPECT_EQ(p.status(), 200);
  EXPECT_TRUE(p.keep_alive());
  EXPECT_EQ(p.header("content-type"), "application/json");
}

TEST(ResponseParser, ChunkedByteByByte) {
  const std::string raw =
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
      "6\r\ndata: \r\n"
      "A;ext=1\r\n0123456789\r\n"
      "0\r\nX-Trailer: t\r\n\r\n";
  ResponseParser p;
  std::string body;
  for (char c : raw) ASSERT_TRUE(p.feed(std::string_view(&c, 1), body));
  EXPECT_TRUE(p.done());
  EXPECT_TRUE(p.chunked());
  EXPECT_EQ(body, "data: 0123456789");
  EXPECT_EQ(p.leftover(), 0u);
}

TEST(ResponseParser, ReadUntilClose) {
  ResponseParser p;
  std::string body;
  ASSERT_TRUE(p.feed("HTTP/1.1 200 OK\r\n\r\nstreamed", body));
  EXPECT_FALSE(p.done());
  EXPECT_FALSE(p.keep_alive());
  EXPECT_TRUE(p.finish_on_eof());
  EXPECT_EQ(body, "streamed");
}

TEST(ResponseParser, IncompleteBodyAtEofIsAnError) {
  ResponseParser p;
  std::string body;
  ASSERT_TRUE(p.feed("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc", body));
  EXPECT_FALSE(p.finish_on_eof());
}

TEST(ResponseParser, SkipsInterimResponses) {
  ResponseParser p;
  std::string body;
  ASSERT_TRUE(p.feed("HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok", body));
  EXPECT_TRUE(p.done());
  EXPECT_EQ(p.status(), 201);
  EXPECT_EQ(body, "ok");
}

TEST(ResponseParser, RejectsGarbage) {
  ResponseParser p;
  std::string body;
  EXPECT_FALSE(p.feed("SSH-2.0-OpenSSH\r\n\r\n", body));
  EXPECT_TRUE(p.failed());
  ResponseParser q;
  EXPECT_FALSE(q.feed("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nZZ\r\n", body));
}

TEST(ResponseParser, CountsLeftoverBytes) {
  ResponseParser p;
  std::string body;
  ASSERT_TRUE(p.feed("HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nxEXTRA", body));
  EXPECT_TRUE(p.done());
  EXPECT_EQ(p.leftover(), 5u);
}
