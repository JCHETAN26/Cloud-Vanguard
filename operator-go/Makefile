# CloudVanguard operator (Kubebuilder v4 conventions).
LOCALBIN ?= $(shell pwd)/bin
CONTROLLER_TOOLS_VERSION ?= v0.17.2
PROTOC_GEN_GO_VERSION ?= v1.36.5
ENVTEST_VERSION ?= release-0.20
ENVTEST_K8S_VERSION ?= 1.32.0

CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
SETUP_ENVTEST ?= $(LOCALBIN)/setup-envtest
PROTOC_GEN_GO ?= $(LOCALBIN)/protoc-gen-go

.PHONY: all
all: generate manifests build

$(LOCALBIN):
	mkdir -p $(LOCALBIN)

$(CONTROLLER_GEN): | $(LOCALBIN)
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION)
$(SETUP_ENVTEST): | $(LOCALBIN)
	GOBIN=$(LOCALBIN) go install sigs.k8s.io/controller-runtime/tools/setup-envtest@$(ENVTEST_VERSION)
$(PROTOC_GEN_GO): | $(LOCALBIN)
	GOBIN=$(LOCALBIN) go install google.golang.org/protobuf/cmd/protoc-gen-go@$(PROTOC_GEN_GO_VERSION)

.PHONY: generate
generate: $(CONTROLLER_GEN) ## DeepCopy methods
	$(CONTROLLER_GEN) object paths="./api/..."

.PHONY: manifests
manifests: $(CONTROLLER_GEN) ## CRDs + RBAC
	$(CONTROLLER_GEN) rbac:roleName=manager-role crd paths="./..." output:crd:artifacts:config=config/crd/bases

.PHONY: proto
proto: $(PROTOC_GEN_GO) ## Go types for proto/vanguard/control/v1 (needs protoc)
	PATH=$(LOCALBIN):$$PATH protoc -I ../proto --go_out=. \
	  --go_opt=module=github.com/JCHETAN26/Cloud-Vanguard/operator-go ../proto/vanguard/control/v1/control.proto

.PHONY: envtest
envtest: $(SETUP_ENVTEST) ## kube-apiserver + etcd binaries for controller tests
	$(SETUP_ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN)/k8s -p path

.PHONY: test
test: generate manifests envtest ## go vet + go test ./... against a real API server
	go vet ./...
	KUBEBUILDER_ASSETS="$$($(SETUP_ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN)/k8s -p path)" \
	  go test ./... -count=1

.PHONY: build
build: ## manager binary
	go build -o $(LOCALBIN)/manager ./cmd

.PHONY: install
install: manifests ## install CRDs into the current kubectl context
	kubectl apply -f config/crd/bases

.PHONY: run
run: ## run the operator against the current kubectl context
	go run ./cmd
