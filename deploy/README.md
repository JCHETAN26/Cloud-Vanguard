# deploy

| Path | What |
|---|---|
| `docker/Dockerfile.proxy` | vanguard-proxy: Ubuntu build stage (unit tests run with `-Werror`) → slim runtime, non-root |
| `docker/Dockerfile.operator` | operator manager → distroless static, non-root |
| `docker/Dockerfile.csi` | `csi-driver` + `csictl` with `qemu-img`, e2fsprogs, util-linux |
| `csi/driver.yaml` | `CSIDriver`, RBAC, controller Deployment (driver + provisioner, attacher, snapshotter and resizer sidecars), node DaemonSet (privileged, node-driver-registrar) |
| `csi/storageclass.yaml` | `cloudvanguard-block` StorageClass (raw images, Retain, expandable) + `VolumeSnapshotClass` |
| `kustomization.yaml` | `kubectl apply -k deploy/`: CRDs + operator + CSI driver + OpenTelemetry Collector (`../observability/otel-collector`, namespace `observability`) |
| `samples/cachecluster-persistent.yaml` | 3-replica CacheCluster with per-pod 1Gi block volumes and tracing to the in-cluster collector (10% sampled), a tenant policy, a VolumeSnapshot |
| `kind/kind-config.yaml` | single-host kind cluster sharing the image directory with the node plugin |
| `scripts/env-check.sh` | which tools are present (kvm, loop, mount caps, docker, kind, k3s, kubectl, qemu-img, qemu-system, go) |
| `scripts/e2e-kind.sh` | builds the images, creates kind, installs everything, verifies a warm pod restart; **exits 3 when docker/kind are missing** |

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

## Verification without a cluster

`benchmarks/bench_phase4.sh` drives the same lifecycle with real CSI gRPC calls (`csictl`), real `qemu-img` images and, if
`qemu-system-x86_64` is present, real QMP hot-plug into a TCG virtual machine. The Kubernetes control plane, kubelet mounts
and image builds are the parts that need `e2e-kind.sh` and a docker/kind host.

## Known limits

- **Raw block devices and ownership.** Raw block devices in pods are owned by root unless the runtime applies `fsGroup` to
  devices. The proxy image runs as uid 10001, so a cluster may need `securityContext.runAsUser: 0` or a device-ownership
  policy for `/dev/vanguard-cache`.
- **`--attacher=local` requires shared images.** The image directory must be visible to the node plugin (single host). For
  VM-based nodes use `--attacher=qmp` on the hypervisor host; node VMs need a QMP socket and a `virtio-scsi-pci,id=scsi0`
  controller.
- **Snapshots are crash-consistent with respect to the proxy's committed slots.** A VolumeSnapshot taken mid-write captures
  the previous committed slot.
