# CloudVanguard monorepo entry points.
#   proxy-cpp/     C++20 data plane         (make build | test | test-asan)
#   operator-go/   Go operator + CSI driver (make test-go)
#   tests/         cross-component tests    (run by ctest: make test)
#   observability/ OTel Collector + Prometheus configs (make verify-observability)
#   benchmarks/    load, chaos and persistence suites (make bench | bench-m1 | ...)
#   training/      Milestone 3 (scaffold only)
BUILD_DIR ?= proxy-cpp/build
BUILD_TYPE ?= RelWithDebInfo
RELEASE_DIR ?= proxy-cpp/build-release

.PHONY: help build test test-asan test-go test-all bench bench-m1 bench-phase2 bench-phase4 \
        verify-observability clean

help:
	@echo "make test                C++ unit tests + integration tests (multinode, tracing) via ctest"
	@echo "make test-asan           same under AddressSanitizer + UBSan"
	@echo "make test-go             operator + CSI driver tests (envtest, csi-sanity, QEMU when present)"
	@echo "make test-all            test + test-asan + test-go"
	@echo "make bench               Phase 1 MISS/HIT benchmark"
	@echo "make bench-m1            Milestone 1: latency/throughput baseline + tracing overhead"
	@echo "make bench-phase2        semantic cache + failover benchmark"
	@echo "make bench-phase4        CSI/QEMU persistence + chaos benchmark"
	@echo "make verify-observability  real otelcol + promtool checks (downloads pinned binaries)"

build:
	cmake -S proxy-cpp -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-asan:
	cmake -S proxy-cpp -B proxy-cpp/build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVANGUARD_SANITIZE=ON
	cmake --build proxy-cpp/build-asan
	ctest --test-dir proxy-cpp/build-asan --output-on-failure

test-go:
	$(MAKE) -C operator-go test

test-all: test test-asan test-go

bench:
	./benchmarks/bench_phase1.sh

bench-m1:
	./benchmarks/bench_milestone1.sh

bench-phase2:
	./benchmarks/bench_phase2.sh

bench-phase4:
	./benchmarks/bench_phase4.sh

verify-observability:
	cmake -S proxy-cpp -B $(RELEASE_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release -DVANGUARD_BUILD_TESTS=OFF
	cmake --build $(RELEASE_DIR) --target vanguard-proxy
	./observability/scripts/verify-collector.sh $(RELEASE_DIR)/vanguard-proxy
	./observability/scripts/verify-prometheus.sh $(RELEASE_DIR)/vanguard-proxy

clean:
	rm -rf proxy-cpp/build proxy-cpp/build-asan proxy-cpp/build-release benchmarks/results
