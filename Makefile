# CloudVanguard developer entry points.
BUILD_DIR ?= proxy-cpp/build
BUILD_TYPE ?= RelWithDebInfo

.PHONY: build test test-asan bench clean

build:
	cmake -S proxy-cpp -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-asan:
	cmake -S proxy-cpp -B proxy-cpp/build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVANGUARD_SANITIZE=ON
	cmake --build proxy-cpp/build-asan
	ctest --test-dir proxy-cpp/build-asan --output-on-failure

bench:
	./benchmarks/bench_phase1.sh

clean:
	rm -rf proxy-cpp/build proxy-cpp/build-asan proxy-cpp/build-release benchmarks/results
