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

## Persistence and the CSI driver (Phase 4)

`spec.persistence` gives every gateway pod a retained, raw block PVC (`volumeMode: Block`, mounted at
`/dev/vanguard-cache`), which vanguard-proxy snapshots its cache to. A restarted or rescheduled pod re-attaches its own
volume and starts warm.

The `csi.cloudvanguard.io` driver (`cmd/csi-driver`, `internal/csi`, `internal/storage`) provides those volumes:
- **Controller:** create, delete and expand volumes as `qemu-img` raw/qcow2 images; attach them to a node (the image
  path for single-host clusters, or QMP `blockdev-add` + `scsi-hd` hot-plug into a node VM); take snapshots
  (`qemu-img convert -c -O qcow2`), restore from snapshots, and list volumes and snapshots with pagination.
- **Node:** stage and publish. `HostMounter` (privileged) attaches image files (`--attacher=local`) to loop devices,
  runs mkfs and mount, and bind-mounts device nodes or directories into pods; `LinkMounter` (no privileges) exposes
  block volumes via symlinks and simulates filesystem volumes.

Pods with `spec.persistence` get `supplementalGroups: [6]` (`disk`): block devices reach the container with the node's
`root:disk 0660` ownership and fsGroup does not apply to them, so this is what lets the non-root proxy open its volume.

`cmd/csictl` issues the same RPCs as the Kubernetes sidecars and kubelet, for scripts. Manifests are in `../deploy/csi`.

Tests:
- **CSI conformance:** the upstream `csi-test` sanity suite, for block and filesystem volumes.
- **QEMU data integrity:** volume bytes survive a qcow2 snapshot and restore.
- **QMP:** the command protocol against a fake QMP server, plus real hot-plug into a QEMU VM running under TCG when
  `qemu-system-x86_64` is installed.
- **Failure paths (Milestone 2):**
  - A half-attached volume (block node without its device) is repaired on retry.
  - A failed `device_add` rolls back its block node.
  - Detaching from a VM that is gone succeeds, and attaching to one returns `NotFound`.
  - Failed image operations leave no partial images.
  - An attachment that can't be recorded is rolled back.
  - A restore from a snapshot larger than `limit_bytes` returns `OUT_OF_RANGE`.
  - Metadata writes are fsynced (file and directory).
- **Node data path (Milestone 4, root only):** a block volume from an image file is published as a real device node
  backed by a loop device, writes reach the image, retries reuse the loop device and unstage detaches it; ext4
  volumes survive unstage/restage. Run with `sudo` (see `internal/csi/hostmounter_test.go`).

## Tracing (Milestone 2)

`spec.tracing` turns on OpenTelemetry span export in every gateway pod:

```yaml
tracing:
  enabled: true
  endpoint: http://otel-collector.observability.svc.cluster.local:4318  # base URL; /v1/traces is appended
  samplerRatio: "0.1"        # decimal string in [0, 1]; default "1"; parent-based
  serviceName: vanguard-proxy
  batch: {scheduleDelayMillis: 2000, maxQueueSize: 8192, maxExportBatchSize: 1024, exportTimeoutMillis: 5000}
```

- **Validation.** The API server rejects what the proxy would refuse, using OpenAPI patterns and CEL: an endpoint
  required when enabled, `http://` only, no `/v1/traces` suffix, a ratio in [0, 1], batch ≤ queue (with the
  proxy's defaults when either is unset). The operator re-validates; if an invalid spec gets through anyway, it
  leaves the StatefulSet untouched and reports `TracingReady=False, reason InvalidConfig`.
- **Rendering.** `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_TRACES_SAMPLER=parentbased_traceidratio`,
  `OTEL_TRACES_SAMPLER_ARG`, `OTEL_SERVICE_NAME`, and `OTEL_BSP_*` / `OTEL_EXPORTER_OTLP_TIMEOUT` when set, in a fixed
  order. The ratio is canonicalised (`"0.10"` and `"0.1"` render the same), and disabling removes every `OTEL_*`
  variable.
- **No rollout churn.** The StatefulSet carries `cache.cloudvanguard.io/pod-template-hash`, and the pod template is
  rewritten only when the rendered template's hash changes. A steady-state reconcile sends **no writes at all**
  (no StatefulSet update, no status patch). Trade-off: a manual edit of the pod template is not reverted until the
  rendered template changes.
- **Status.** Every config push returns the pod's `TracingStatus` (`vanguard.control.v1.ApplyResult.tracing`). The
  operator reports `status.tracing` (`configuredReplicas`, `exportFailingReplicas`, `lastExportError`) and the
  `TracingReady` condition:

  | Reason | Meaning |
  |---|---|
  | `Exporting` (True) | Every ready pod exports with the current settings and its last export succeeded |
  | `RolloutPending` | Some ready pods still run old settings (or a proxy that predates tracing) |
  | `ExportFailing` | Some pods' last OTLP export failed (e.g. collector down) |
  | `NoReadyPods` | Nothing has reported yet |
  | `Disabled` | `spec.tracing.enabled` is false |
  | `InvalidConfig` | The spec failed the operator's validation; pods were left as they were |

  Export health has no Kubernetes event, so while tracing is enabled but not ready the cluster is re-checked every
  30 s.
- **Cross-language contract.** Two golden files are read by both test suites:
  `internal/render/testdata/tracing_env_golden.json` (Go writes it; the C++ config loader must accept it) and
  `../proto/vanguard/control/v1/testdata/apply_result_tracing.golden.json` (the C++ serializer writes it; Go parses
  it strictly as `ApplyResult`).

## Development

```bash
make generate manifests   # controller-gen: deepcopy, CRDs, RBAC
make proto                # Go types from ../proto (needs protoc)
make test                 # go vet + go test ./... with envtest (real kube-apiserver + etcd 1.32)
make build                # bin/manager
kubectl apply -k config/default   # CRDs, RBAC, operator Deployment (image: make -C .. images)
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
  - The published samples, including `../deploy/samples/cachecluster-persistent.yaml` with tracing.
  - Tracing: env propagation, exactly one template change per real change and none for unrelated or cosmetic
    changes, `status.tracing`/`TracingReady` from pod reports, API rejection of invalid specs, and (fake client)
    invalid specs leaving the StatefulSet untouched.
  - A steady-state reconcile issues zero API writes.
  - **The deploy bundle under its own RBAC (Milestone 4):** `kubectl kustomize ../deploy` is server-side-applied to
    the API server, then both reconcilers run on the persistent sample while impersonating the operator's
    ServiceAccount (read from the rendered Deployment): cluster-wide list/watch of every watched type, leader-election
    Lease and events in its namespace only, StatefulSet/Services/ConfigMap/Secret/status writes, delete and recreate.
    Removing any needed verb from `config/rbac/role.yaml` fails it with Forbidden.

### Known limits

- **envtest has no kubelet, scheduler or garbage collector**, so these tests simulate pods and assert owner
  references. The real-cluster path (pods, CSI attach, ConfigMap sync, warm restart, tracing) is covered by
  `make e2e-kind` (`../deploy/scripts/e2e-kind.sh`), which passes on kind (Milestone 4).
- **Ring membership comes from `spec.replicas`, not readiness.** This keeps ownership stable while pods restart;
  requests for an unready owner are served by the ingress pod (local fallback) instead of reshuffling keys.
  Scaling down moves about 1/N of the keys.
- **Limits are per pod.** Each of N pods enforces 1/N of a tenant's limit, which assumes the client Service
  spreads traffic evenly. There's no global token bucket.
