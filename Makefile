# CloudVanguard monorepo entry points.
#   proxy-cpp/     C++20 data plane                    (make build | test | test-asan)
#   operator-go/   Go operator + CSI driver            (make test-go)
#   tests/         cross-component tests               (run by ctest: make test)
#   observability/ OTel Collector + Prometheus configs (make verify-observability)
#   training/      PyTorch LoRA/QLoRA pipeline         (make test-training | bench-m3)
#   deploy/        images, manifests, kind e2e         (make verify-deploy | images | e2e-kind)
#   benchmarks/    load, chaos and persistence suites  (make bench | bench-m1 | bench-m3 | ...)
BUILD_DIR ?= proxy-cpp/build
BUILD_TYPE ?= RelWithDebInfo
RELEASE_DIR ?= proxy-cpp/build-release
PYTHON ?= python3
TAG ?= 0.4.0
KUBECTL ?= kubectl
SCRIPTS := deploy/scripts/*.sh benchmarks/*.sh observability/scripts/*.sh

.PHONY: help build test test-asan test-go test-training test-all bench bench-m1 bench-m3 bench-gpu bench-phase2 bench-phase4 \
        verify-observability verify-deploy images e2e-kind env-check clean

help:
	@echo "Tests"
	@echo "  make test                C++ unit tests + integration tests via ctest (multinode ring, tracing,"
	@echo "                           hostile-input robustness, training eval through the gateway)"
	@echo "  make test-asan           same under AddressSanitizer + UBSan"
	@echo "  make test-go             operator + CSI driver (envtest incl. deploy/ bundle under its RBAC, csi-sanity,"
	@echo "                           QEMU hot-plug when present; root-only loop-device tests skip otherwise)"
	@echo "  make test-training       training/ pytest suite (needs torch, pyarrow, mlflow; PYTHON=...)"
	@echo "  make test-all            test + test-asan + test-go + test-training"
	@echo "Verification"
	@echo "  make verify-observability  real otelcol + promtool checks (downloads pinned binaries)"
	@echo "  make verify-deploy       render every kustomization + check the shell scripts (no cluster needed)"
	@echo "  make images              docker build the proxy, operator and CSI images (TAG=$(TAG); needs docker)"
	@echo "  make e2e-kind            real cluster: images, kind, operator + CSI + collector, warm restart, tracing"
	@echo "  make env-check           which container / cluster / hypervisor tools this machine has"
	@echo "Benchmarks"
	@echo "  make bench               Phase 1 MISS/HIT benchmark"
	@echo "  make bench-m1            Milestone 1: latency/throughput baseline + tracing overhead"
	@echo "  make bench-m3            Milestone 3: data loading, LoRA/QLoRA/full/DDP training, evaluation"
	@echo "  make bench-gpu           GPU box: Qwen2.5-14B LoRA vs QLoRA-NF4, 2-GPU DDP, vLLM eval, GPU telemetry"
	@echo "  make bench-phase2        semantic cache + failover benchmark"
	@echo "  make bench-phase4        CSI/QEMU persistence + chaos benchmark"

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

test-training:
	$(PYTHON) -m pytest training/tests -q

test-all: test test-asan test-go test-training

bench-gpu:
	./benchmarks/gpu/run_gpu_bench.sh

bench:
	./benchmarks/bench_phase1.sh

bench-m1:
	./benchmarks/bench_milestone1.sh

bench-m3:
	PYTHON=$(PYTHON) ./benchmarks/bench_milestone3.sh

bench-phase2:
	./benchmarks/bench_phase2.sh

bench-phase4:
	./benchmarks/bench_phase4.sh

verify-observability:
	cmake -S proxy-cpp -B $(RELEASE_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release -DVANGUARD_BUILD_TESTS=OFF
	cmake --build $(RELEASE_DIR) --target vanguard-proxy
	./observability/scripts/verify-collector.sh $(RELEASE_DIR)/vanguard-proxy
	./observability/scripts/verify-prometheus.sh $(RELEASE_DIR)/vanguard-proxy

verify-deploy:
	@for k in deploy operator-go/config/default observability/otel-collector; do \
	  n=$$($(KUBECTL) kustomize $$k | grep -c '^kind:') || exit 1; echo "kustomize $$k: $$n objects"; done
	@for f in $(SCRIPTS); do bash -n $$f || exit 1; done; echo "bash -n: $(words $(wildcard $(SCRIPTS))) scripts parse"
	@if command -v shellcheck >/dev/null; then shellcheck -S warning $(SCRIPTS) && echo "shellcheck: no warnings"; \
	  else echo "shellcheck not installed: syntax check only"; fi

images:
	docker build -f deploy/docker/Dockerfile.proxy -t ghcr.io/jchetan26/vanguard-proxy:$(TAG) .
	docker build -f deploy/docker/Dockerfile.operator -t ghcr.io/jchetan26/cloudvanguard-operator:$(TAG) .
	docker build -f deploy/docker/Dockerfile.csi -t ghcr.io/jchetan26/cloudvanguard-csi:$(TAG) .

e2e-kind:
	TAG=$(TAG) ./deploy/scripts/e2e-kind.sh

env-check:
	./deploy/scripts/env-check.sh

clean:
	rm -rf proxy-cpp/build proxy-cpp/build-asan proxy-cpp/build-release benchmarks/results
