cmake_minimum_required(VERSION 3.20)
project(cloudvanguard_proxy VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Some fetched dependencies declare very old cmake_minimum_required values.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VANGUARD_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(VANGUARD_WERROR "Treat compiler warnings as errors" OFF)
option(VANGUARD_SANITIZE "Build with AddressSanitizer + UBSan" OFF)

include(FetchContent)
find_package(Threads REQUIRED)

# ---- nlohmann/json: system package first, otherwise fetch -------------------
find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found; fetching v3.11.3")
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---- Warnings (applied to our targets only) ---------------------------------
add_library(vanguard_warnings INTERFACE)
target_compile_options(vanguard_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
  -Wcast-align -Wformat=2 -Wimplicit-fallthrough -Wnull-dereference)
if(VANGUARD_WERROR)
  target_compile_options(vanguard_warnings INTERFACE -Werror)
endif()

if(VANGUARD_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- Portable core (builds on Linux and macOS) ------------------------------
add_library(vanguard_core STATIC
  src/sha256.cpp
  src/murmur3.cpp
  src/hash_ring.cpp
  src/rate_limiter.cpp
  src/http_parser.cpp
  src/sse.cpp
  src/llm_request.cpp
  src/tenants.cpp
  src/metrics.cpp
  src/config.cpp
  src/circuit_breaker.cpp
  src/embedding.cpp
  src/semantic_index.cpp
  src/control.cpp
  src/snapshot.cpp
  src/tracing.cpp)
target_include_directories(vanguard_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vanguard_core
  PUBLIC nlohmann_json::nlohmann_json Threads::Threads
  PRIVATE vanguard_warnings)

# ---- epoll server (Linux only) ----------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(vanguard-proxy src/server.cpp src/main.cpp)
  target_link_libraries(vanguard-proxy PRIVATE vanguard_core vanguard_warnings)
else()
  message(STATUS "Host is ${CMAKE_SYSTEM_NAME}: building vanguard_core + tests only "
                 "(vanguard-proxy uses epoll and requires Linux)")
endif()

# ---- Tests -------------------------------------------------------------------
if(VANGUARD_BUILD_TESTS)
  enable_testing()
  find_package(GTest 1.11 QUIET)
  if(NOT GTest_FOUND)
    message(STATUS "GTest not found; fetching googletest v1.15.2")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SYSTEM)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(vanguard_tests
    tests/test_sha256.cpp
    tests/test_murmur3.cpp
    tests/test_hash_ring.cpp
    tests/test_lru_cache.cpp
    tests/test_rate_limiter.cpp
    tests/test_http_parser.cpp
    tests/test_sse.cpp
    tests/test_llm_request.cpp
    tests/test_tenants.cpp
    tests/test_config.cpp
    tests/test_circuit_breaker.cpp
    tests/test_semantic.cpp
    tests/test_control.cpp
    tests/test_snapshot.cpp
    tests/test_tracing.cpp
    tests/test_metrics.cpp)
  target_link_libraries(vanguard_tests PRIVATE vanguard_core vanguard_warnings GTest::gtest_main)
  target_compile_definitions(vanguard_tests PRIVATE VANGUARD_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
  include(GoogleTest)
  gtest_discover_tests(vanguard_tests DISCOVERY_TIMEOUT 60)

  # End-to-end: 3 real proxy processes + mock upstream (Linux only, needs python3).
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(TARGET vanguard-proxy AND Python3_Interpreter_FOUND)
    add_test(NAME multinode_forwarding
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/multinode_test.py
              --binary $<TARGET_FILE:vanguard-proxy> --base-port 18100)
    set_tests_properties(multinode_forwarding PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
    # OpenTelemetry trace propagation across a 2-node ring (Milestone 1).
    add_test(NAME tracing_e2e
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tests/integration/test_tracing_e2e.py
              --binary $<TARGET_FILE:vanguard-proxy> --base-port 18300)
    set_tests_properties(tracing_e2e PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
  endif()
endif()
