#!/usr/bin/env bash
# Real-cluster end-to-end: build the three images, create a kind cluster,
# install the operator + CSI driver, deploy a persistent CacheCluster, and
# verify a warm restart (pod deleted -> new pod re-attaches its block volume
# -> cache restored -> repeated prompts are hits with no upstream calls).
#
# Requires: docker (daemon running), kind, kubectl, curl. Exits 3 with the
# list of missing tools otherwise; it never pretends to have run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${TAG:-0.4.0}"
CLUSTER="${CLUSTER:-cloudvanguard}"

missing=()
command -v docker >/dev/null && docker info >/dev/null 2>&1 || missing+=("docker (running daemon)")
command -v kind >/dev/null || missing+=("kind")
command -v kubectl >/dev/null || missing+=("kubectl")
command -v curl >/dev/null || missing+=("curl")
if ((${#missing[@]})); then
  echo "e2e-kind: cannot run, missing: ${missing[*]}" >&2
  echo "e2e-kind: run deploy/scripts/env-check.sh for details; benchmarks/bench_phase4.sh covers the same flows without a cluster" >&2
  exit 3
fi

echo "==> building images"
for c in proxy operator csi; do
  name=$([[ $c == proxy ]] && echo vanguard-proxy || echo "cloudvanguard-$c")
  docker build -q -f "$ROOT/deploy/docker/Dockerfile.$c" -t "ghcr.io/jchetan26/$name:$TAG" "$ROOT"
done

echo "==> kind cluster"
sudo mkdir -p /var/lib/cloudvanguard-csi
kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --config "$ROOT/deploy/kind/kind-config.yaml"
# Raw-block volumes need loop devices twice: the CSI node plugin attaches each image file to one, and
# kubelet attaches another to hold the published device. kind nodes are containers without a live
# devtmpfs, so /dev/loopN for N beyond what existed at node creation never appears and losetup fails
# ("failed to set up loop device: No such file or directory"). Pre-create the nodes; real nodes do
# not need this.
for node in $(kind get nodes --name "$CLUSTER"); do
  docker exec "$node" sh -c 'for i in $(seq 0 63); do [ -e /dev/loop$i ] || mknod -m 0660 /dev/loop$i b 7 $i; done
                             chgrp disk /dev/loop[0-9]* 2>/dev/null || true'
done
for img in vanguard-proxy cloudvanguard-operator cloudvanguard-csi; do
  kind load docker-image --name "$CLUSTER" "ghcr.io/jchetan26/$img:$TAG"
done

echo "==> snapshot CRDs + controller (external-snapshotter v8.0.1)"
SNAP=https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.0.1
for f in client/config/crd/snapshot.storage.k8s.io_volumesnapshotclasses.yaml \
         client/config/crd/snapshot.storage.k8s.io_volumesnapshotcontents.yaml \
         client/config/crd/snapshot.storage.k8s.io_volumesnapshots.yaml \
         deploy/kubernetes/snapshot-controller/rbac-snapshot-controller.yaml \
         deploy/kubernetes/snapshot-controller/setup-snapshot-controller.yaml; do
  kubectl apply -f "$SNAP/$f"
done

echo "==> operator + CSI driver"
kubectl apply -k "$ROOT/deploy"
kubectl -n cloudvanguard-system rollout status deploy/cloudvanguard-operator --timeout=180s
kubectl -n cloudvanguard-csi rollout status deploy/csi-controller --timeout=180s
kubectl -n cloudvanguard-csi rollout status ds/csi-node --timeout=180s

echo "==> mock upstream + persistent CacheCluster"
kubectl create configmap mock-upstream --from-file="$ROOT/benchmarks/mock_upstream.py" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata: {name: mock-llm}
spec:
  selector: {matchLabels: {app: mock-llm}}
  template:
    metadata: {labels: {app: mock-llm}}
    spec:
      containers:
        - name: mock
          image: python:3.12-slim
          command: ["python3", "/app/mock_upstream.py", "--host", "0.0.0.0", "--port", "9000"]
          volumeMounts: [{name: app, mountPath: /app}]
      volumes: [{name: app, configMap: {name: mock-upstream}}]
---
apiVersion: v1
kind: Service
metadata: {name: mock-llm}
spec: {selector: {app: mock-llm}, ports: [{port: 9000}]}
---
apiVersion: cache.cloudvanguard.io/v1alpha1
kind: CacheCluster
metadata: {name: vc}
spec:
  replicas: 1
  upstreamPrimary: http://mock-llm.default.svc.cluster.local:9000
  image: ghcr.io/jchetan26/vanguard-proxy:0.4.0
  persistence: {storageClassName: cloudvanguard-block, size: 256Mi, snapshotIntervalSeconds: 10}
  tracing: {enabled: true, endpoint: "http://otel-collector.observability.svc.cluster.local:4318", samplerRatio: "1"}
EOF
kubectl wait --for=condition=Available cachecluster/vc --timeout=300s

echo "==> warm restart"
kubectl port-forward svc/vc 18080:8080 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null' EXIT
sleep 3
ask() { curl -s -o /dev/null -w '%header{x-cache}\n' localhost:18080/v1/chat/completions \
  -H 'Authorization: Bearer sk-e2e' -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"kind e2e $1\"}]}"; }
for i in $(seq 1 50); do ask "$i" >/dev/null; done
sleep 12                                   # one periodic snapshot
kubectl delete pod vc-0 --wait=true        # SIGTERM -> final snapshot; PVC is retained
kubectl wait --for=condition=Ready pod/vc-0 --timeout=180s
kill $PF; kubectl port-forward svc/vc 18080:8080 >/dev/null 2>&1 & PF=$!; sleep 3
hits=0; for i in $(seq 1 50); do [[ "$(ask "$i")" == HIT ]] && hits=$((hits+1)); done
echo "warm restart: $hits/50 hits after pod replacement"
[[ $hits -eq 50 ]]

stats() { kubectl exec deploy/mock-llm -- python3 -c \
  'import json,urllib.request;s=json.load(urllib.request.urlopen("http://127.0.0.1:9000/stats"));print(s["requests"], sum(1 for t in s.get("traceparents",[]) if t))'; }
read -r upstream traced < <(stats)
echo "upstream: $upstream chat requests in total (50 misses before the restart, none after); $traced carried a traceparent"
[[ $upstream -eq 50 && $traced -eq 50 ]]

echo "==> tracing through the operator and the in-cluster collector"
# TracingReady=True needs the operator's config push to reach each pod's admin API and the pod's
# tracing report to match spec.tracing.
kubectl wait --for=condition=TracingReady cachecluster/vc --timeout=120s
kubectl get cachecluster vc -o jsonpath='{.status.tracing}{"\n"}'
spans=0
for _ in $(seq 1 20); do   # the proxy batches every 5 s, the collector every 2 s
  spans=$(kubectl -n observability logs deploy/otel-collector | grep -o '"spans": [0-9]*' | awk '{s+=$2} END {print s+0}')
  [[ $spans -gt 0 ]] && break
  sleep 3
done
echo "collector: $spans spans received from the gateway"
[[ $spans -gt 0 ]]
echo "e2e-kind: PASS"
