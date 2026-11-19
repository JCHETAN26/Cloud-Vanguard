# cloudvanguard-operator

A Kubernetes operator (Go 1.24, controller-runtime v0.20, Kubebuilder v4 layout) that runs fleets of
`vanguard-proxy` gateways and distributes tenant policies to them without restarts.

## APIs (`cache.cloudvanguard.io/v1alpha1`)

| Kind | Spec | Status |
|---|---|---|
| `CacheCluster` (`cc`) | `replicas`, `upstreamPrimary`, `upstreamFallback`, `vnodes` (default 1000), `maxCacheBytes` (256Mi), `defaultTtlSeconds` (3600), `image` | `readyReplicas`, `endpoints`, `configVersion`, `configHash`, `syncedReplicas`, conditions `Available` / `ConfigSynced` |
| `TenantCachePolicy` (`tcp`) | `tenantId`, `clusterRef` (empty = all clusters in the namespace), `apiKeysSecretRef`, `rpmLimit`, `tpmLimit`, `semanticMode` (lexical/neural/off), `semanticThreshold` ("0.96"), `ttlSeconds` | `applied`, `observedGeneration`, `appliedClusters`, conditions `Applied` / `Conflict` |

See `config/samples/cachecluster.yaml`.

## What the CacheCluster controller manages

```
CacheCluster vc
├── Secret      vc-admin     random token: operator -> pod admin API, and pod <-> pod (X-Vanguard-Hop)
├── ConfigMap   vc-control   control.json = ControlDocument (ring + tenant policies), mounted at CONTROL_FILE
├── Service     vc-peers     headless, publishNotReadyAddresses: stable DNS per ring member
├── Service     vc           ClusterIP for tenants (port 8080)
└── StatefulSet vc           NODE_ID = pod name; reads the ConfigMap; admin API on :9901
```

All of them carry a controller `ownerReference`, so deleting the `CacheCluster` garbage-collects everything.

## How configuration reaches the proxies

1. **Render.** One writer per cluster: the CacheCluster controller renders the ring (ordinals `0..replicas-1`,
   addresses `<pod>.<name>-peers.<ns>.svc.cluster.local:8080`) and the winning `TenantCachePolicy` objects
   into a `ControlDocument` ([`proto/vanguard/control/v1/control.proto`](../proto/vanguard/control/v1/control.proto)).
   API keys are read from Secrets and only their **SHA-256** is rendered.
2. **Version.** The document's version increases only when its content hash changes. The hash is stored on the
   ConfigMap, so cache lag cannot double-bump it. Proxies apply only newer versions, so an old push can never
   overwrite a newer document (409).
3. **Durable path.** The document is written to the `vc-control` ConfigMap. Pods mount it, load it at startup and
   poll it every second, so a restarted pod converges on its own.
4. **Fast path.** The same JSON is `PUT` to `http://<podIP>:9901/admin/v1/config` on every ready pod, using the
   bearer token from `vc-admin`. Changes go live in milliseconds instead of waiting for the kubelet's ConfigMap
   sync (up to about a minute). Failures set `ConfigSynced=False` and retry every 5 s; a periodic resync
   (1 min) repairs drift.
5. **Status.** The TenantCachePolicy controller reads the rendered document back and sets `Applied=True` once
   the policy's current `metadata.generation` is present in every target cluster.

## Development

```bash
make generate manifests   # controller-gen: deepcopy, CRDs, RBAC
make proto                # Go types from ../proto (needs protoc)
make test                 # go vet + go test ./... with envtest (real kube-apiserver + etcd 1.32)
make build                # bin/manager
kubectl apply -k config/default   # CRDs, RBAC, operator Deployment (needs an image; Phase 4)
```

Tools are pinned and installed into `bin/`: controller-gen v0.17.2, setup-envtest (release-0.20),
protoc-gen-go v1.36.5.

### Test coverage (`go test ./...`)

- **Unit tests (render):** pod naming and addresses, threshold parsing, key hashing, oldest-wins conflict
  resolution, and a document hash that ignores version and input order. Also a byte-stable **golden** document
  that the C++ proxy test suite parses, which pins the cross-language contract.
- **Controller tests (envtest):**
  - API defaults and validation.
  - Workload rendering: env, headless Service, admin Secret, ring.
  - Scaling 3→5 with exactly one config-version bump and no churn.
  - Status and push to simulated ready pods, including a failing pod that recovers.
  - Policy lifecycle: create, update, Secret key rotation, delete.
  - Conflicts, schema rejection and a missing `clusterRef`.
  - Deletion: owner references, and policies that report they are no longer applied.
  - The published samples.

### Known limits

- **Not run on a real cluster yet.** Neither kind nor k3s is available, and there's no container image, so
  StatefulSet pods, kubelet ConfigMap sync and garbage collection are simulated or asserted through owner
  references. Phase 4 adds a kind-based end-to-end test.
- **Ring membership comes from `spec.replicas`, not readiness.** This keeps ownership stable while pods restart;
  requests for an unready owner are served by the ingress pod (local fallback) instead of reshuffling keys.
  Scaling down moves about 1/N of the keys.
- **Limits are per pod.** Each of N pods enforces 1/N of a tenant's limit, which assumes the client Service
  spreads traffic evenly. There's no global token bucket.
