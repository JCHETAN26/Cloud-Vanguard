package csi

import (
	"context"
	"path/filepath"
	"testing"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/storage"
)

func blockCap() []*csipb.VolumeCapability {
	return []*csipb.VolumeCapability{{
		AccessType: &csipb.VolumeCapability_Block{Block: &csipb.VolumeCapability_BlockVolume{}},
		AccessMode: &csipb.VolumeCapability_AccessMode{Mode: csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER},
	}}
}

// A volume restored from a snapshot is at least the snapshot's size; when
// that exceeds the caller's limit_bytes the driver must answer OUT_OF_RANGE
// instead of silently provisioning more than allowed.
func TestCreateFromSnapshotRespectsLimitBytes(t *testing.T) {
	ctx := context.Background()
	store, err := storage.NewStore(filepath.Join(t.TempDir(), "images"), storage.RawFileTool{}, storage.LocalAttacher{})
	if err != nil {
		t.Fatal(err)
	}
	cs := &controllerServer{d: &Driver{Name: "csi.cloudvanguard.io", NodeID: "node-1", Store: store}}
	src, err := cs.CreateVolume(ctx, &csipb.CreateVolumeRequest{Name: "src", VolumeCapabilities: blockCap(),
		CapacityRange: &csipb.CapacityRange{RequiredBytes: 8 << 20}})
	if err != nil {
		t.Fatal(err)
	}
	snap, err := cs.CreateSnapshot(ctx, &csipb.CreateSnapshotRequest{Name: "s", SourceVolumeId: src.GetVolume().GetVolumeId()})
	if err != nil {
		t.Fatal(err)
	}
	from := &csipb.VolumeContentSource{Type: &csipb.VolumeContentSource_Snapshot{
		Snapshot: &csipb.VolumeContentSource_SnapshotSource{SnapshotId: snap.GetSnapshot().GetSnapshotId()}}}

	_, err = cs.CreateVolume(ctx, &csipb.CreateVolumeRequest{Name: "too-small", VolumeCapabilities: blockCap(),
		VolumeContentSource: from, CapacityRange: &csipb.CapacityRange{RequiredBytes: 1 << 20, LimitBytes: 4 << 20}})
	if status.Code(err) != codes.OutOfRange {
		t.Fatalf("restore above limit_bytes: %v, want OutOfRange", err)
	}
	// Without a limit (or a large enough one) the volume grows to the snapshot size.
	got, err := cs.CreateVolume(ctx, &csipb.CreateVolumeRequest{Name: "ok", VolumeCapabilities: blockCap(),
		VolumeContentSource: from, CapacityRange: &csipb.CapacityRange{RequiredBytes: 1 << 20, LimitBytes: 8 << 20}})
	if err != nil || got.GetVolume().GetCapacityBytes() != 8<<20 {
		t.Fatalf("restore within limit: %v %v", got, err)
	}
}
