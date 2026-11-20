# deploy

| Path | What |
|---|---|
| `docker/Dockerfile.proxy` | vanguard-proxy: Ubuntu build stage (C++ unit tests run with `-Werror`) → slim runtime, non-root (uid 10001) |
| `docker/Dockerfile.operator` | operator manager → distroless static, non-root |
| `docker/Dockerfile.csi` | `csi-driver` + `csictl` with `qemu-img`, e2fsprogs, util-linux, mount (`losetup`) |
| `csi/driver.yaml` | `CSIDriver`, RBAC, controller Deployment (driver + provisioner, attacher, snapshotter and resizer sidecars), node DaemonSet (privileged, node-driver-registrar) |
| `csi/storageclass.yaml` | `cloudvanguard-block` StorageClass (raw images, Retain, expandable) + `VolumeSnapshotClass` |
| `kustomization.yaml` | `kubectl apply -k deploy/`: CRDs + operator + CSI driver + OpenTelemetry Collector (`../observability/otel-collector`, namespace `observability`) |
| `samples/cachecluster-persistent.yaml` | 3-replica CacheCluster with per-pod 1Gi block volumes and tracing to the in-cluster collector (10% sampled), a tenant policy, a VolumeSnapshot |
| `kind/kind-config.yaml` | single-host kind cluster sharing the image directory with the node plugin |
| `scripts/env-check.sh` | which tools are present (kvm, loop, mount caps, docker, kind, k3s, kubectl, qemu-img, qemu-system, go) |
| `scripts/e2e-kind.sh` | builds the images, creates kind, installs everything, verifies a warm pod restart (50/50 hits, zero upstream calls), tracing through the operator and spans at the collector; **exits 3 when docker/kind are missing** |

## How warm caches persist

```
CacheCluster.spec.persistence ──► StatefulSet volumeClaimTemplate "warm-cache" (volumeMode: Block, RWO)
   PVC ──► csi.cloudvanguard.io CreateVolume ──► qemu-img create -f raw  (images on the storage host)
   pod scheduled ──► ControllerPublishVolume ──► local: image path | qmp: blockdev-add + scsi-hd into the node VM
                 ──► NodeStage/NodePublish ──► /dev/vanguard-cache in the container
   vanguard-proxy: SNAPSHOT_PATH=/dev/vanguard-cache SNAPSHOT_MODE=block
       every snapshotIntervalSeconds + on SIGTERM: write the older A/B slot, fsync, commit its header
       on start: load the newest slot whose header + data CRCs verify (else the other slot)
   VolumeSnapshot ──► CreateSnapshot ──► qemu-img convert -c -O qcow2 (compressed point-in-time copy)
   PVC dataSource: VolumeSnapshot ──► CreateVolume from snapshot ──► qemu-img convert -O raw
```

PVCs are retained when pods are deleted or scaled away (`persistentVolumeClaimRetentionPolicy: Retain`), so a pod that is
rescheduled re-attaches its own volume and starts warm.

## Verification

| Layer | How | Status |
|---|---|---|
| Image builds | `make images` (Docker 29, in the devcontainer) | **verified**: proxy 119 MB (135 unit tests pass in the build stage), operator 62 MB, CSI 170 MB |
| Images run | proxy `/healthz` 200 as uid 10001 and exits 0 on SIGTERM; operator and CSI binaries start | **verified** |
| Manifests | `make verify-deploy` (every kustomization renders) and `operator-go/.../deploy_test.go` (bundle server-side-applied to kube-apiserver 1.32; operator reconciles the sample under its own RBAC) | **verified** |
| Real cluster | `make e2e-kind`: kind v0.24 (Kubernetes 1.31 node image), operator ×2 with leader election, CSI controller + node, external-snapshotter v8.0.1, OTel Collector, persistent traced CacheCluster | **verified** |
| ↳ warm restart | pod deleted → replacement re-attaches its CSI block volume → `restore ok: 50 entries` → 50/50 HIT, upstream count unchanged | **verified** |
| ↳ tracing | `TracingReady=True` (operator push → pod report), 50/50 upstream calls carry `traceparent`, spans arrive at the collector | **verified** |
| QMP attacher in a cluster | VM-based nodes with `--attacher=qmp` | not run in a cluster (QMP hot-plug is tested against a real TCG VM in `make test-go`) |
| Multi-node storage | `--attacher=local` assumes one storage host | single-worker kind only |
| VolumeSnapshot round trip in a cluster | CRDs + controller installed by e2e-kind; snapshot/restore exercised by `bench_phase4.sh` via CSI gRPC | not exercised through the Kubernetes API |

Bugs the first real builds and cluster runs found (all fixed, each with a regression test or check):

1. `Dockerfile.proxy` could not build: its test stage lacked two contract goldens added in Milestone 2.
2. Raw-block pods could not start (`not a device node`): with `--attacher=local` the node plugin published the image
   *file*; it now attaches it to a loop device (`internal/csi/hostmounter_test.go`, root-only).
3. The non-root proxy could not open its block device (`Permission denied`): block devices keep the node's
   `root:disk 0660` ownership and fsGroup does not apply, so persistent pods now get `supplementalGroups: [6]`.

`benchmarks/bench_phase4.sh` still covers the storage lifecycle without a cluster (real CSI gRPC via `csictl`, real
`qemu-img`, real QMP hot-plug into a TCG VM when `qemu-system-x86_64` is present).

### Running e2e-kind in a Codespace

The devcontainer installs Docker-in-Docker, kubectl and kind. kind nodes are containers without a live devtmpfs, so
`e2e-kind.sh` pre-creates `/dev/loop0-63` in each node (loop devices allocated later otherwise have no device node and
kubelet's `losetup` fails). The Codespace root filesystem has ~8 GB free; if you move Docker's `data-root` to the larger
`/tmp` disk, strip `/tmp`'s default ACL from the new directory first (`setfacl -b`): inherited `other::rw-` entries make
every image directory untraversable for non-root users and the proxy fails with `libc.so.6: Permission denied`.

## Known limits

- **Disk group id.** Persistent pods get supplemental group 6 (`disk` on Debian, Ubuntu, RHEL, Fedora). A node OS that
  gives block devices a different group needs that gid instead (or containerd's
  `device_ownership_from_security_context`).
- **`--attacher=local` requires shared images.** The image directory must be visible to the node plugin (single host). For
  VM-based nodes use `--attacher=qmp` on the hypervisor host; node VMs need a QMP socket and a `virtio-scsi-pci,id=scsi0`
  controller.
- **Snapshots are crash-consistent with respect to the proxy's committed slots.** A VolumeSnapshot taken mid-write captures
  the previous committed slot.
