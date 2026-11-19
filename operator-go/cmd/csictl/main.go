// Command csictl is a minimal CSI client: it issues the same RPCs the
// Kubernetes sidecars and kubelet issue (CreateVolume, ControllerPublish,
// NodeStage, NodePublish, CreateSnapshot, ...) so the full volume lifecycle
// can be driven from scripts without a cluster. Output is JSON.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func usage() {
	fmt.Fprintln(os.Stderr, `usage: csictl --endpoint unix:///path/csi.sock <command> [flags]
commands: info, create-volume, delete-volume, expand-volume, publish, unpublish,
          stage, unstage, node-publish, node-unpublish, snapshot, delete-snapshot,
          list-volumes, list-snapshots`)
	os.Exit(2)
}

func blockCap(block bool) *csipb.VolumeCapability {
	c := &csipb.VolumeCapability{AccessMode: &csipb.VolumeCapability_AccessMode{
		Mode: csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER}}
	if block {
		c.AccessType = &csipb.VolumeCapability_Block{Block: &csipb.VolumeCapability_BlockVolume{}}
	} else {
		c.AccessType = &csipb.VolumeCapability_Mount{Mount: &csipb.VolumeCapability_MountVolume{FsType: "ext4"}}
	}
	return c
}

func main() {
	endpoint := flag.String("endpoint", "unix:///csi/csi.sock", "CSI endpoint")
	flag.Parse()
	if flag.NArg() < 1 {
		usage()
	}
	cmd, args := flag.Arg(0), flag.Args()[1:]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	name := fs.String("name", "", "volume / snapshot name")
	id := fs.String("id", "", "volume / snapshot id")
	size := fs.Int64("size", 64<<20, "bytes")
	format := fs.String("format", "raw", "raw | qcow2")
	fromSnap := fs.String("from-snapshot", "", "snapshot id to restore from")
	node := fs.String("node", "", "node id")
	device := fs.String("device", "", "device path from publish")
	staging := fs.String("staging", "", "staging directory")
	target := fs.String("target", "", "target path")
	source := fs.String("source", "", "source volume id")
	block := fs.Bool("block", true, "block volume (false = filesystem)")
	_ = fs.Parse(args)

	target_ := strings.TrimPrefix(*endpoint, "unix://")
	conn, err := grpc.NewClient("unix://"+target_, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if strings.HasPrefix(*endpoint, "tcp://") {
		conn, err = grpc.NewClient(strings.TrimPrefix(*endpoint, "tcp://"), grpc.WithTransportCredentials(insecure.NewCredentials()))
	}
	if err != nil {
		fail(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	ctrl, node_, ident := csipb.NewControllerClient(conn), csipb.NewNodeClient(conn), csipb.NewIdentityClient(conn)

	var out any
	switch cmd {
	case "info":
		info, err := ident.GetPluginInfo(ctx, &csipb.GetPluginInfoRequest{})
		check(err)
		ni, err := node_.NodeGetInfo(ctx, &csipb.NodeGetInfoRequest{})
		check(err)
		out = map[string]any{"name": info.GetName(), "version": info.GetVendorVersion(), "nodeId": ni.GetNodeId()}
	case "create-volume":
		req := &csipb.CreateVolumeRequest{Name: *name, CapacityRange: &csipb.CapacityRange{RequiredBytes: *size},
			VolumeCapabilities: []*csipb.VolumeCapability{blockCap(*block)}, Parameters: map[string]string{"format": *format}}
		if *fromSnap != "" {
			req.VolumeContentSource = &csipb.VolumeContentSource{Type: &csipb.VolumeContentSource_Snapshot{
				Snapshot: &csipb.VolumeContentSource_SnapshotSource{SnapshotId: *fromSnap}}}
		}
		r, err := ctrl.CreateVolume(ctx, req)
		check(err)
		out = map[string]any{"volumeId": r.GetVolume().GetVolumeId(), "capacityBytes": r.GetVolume().GetCapacityBytes(),
			"context": r.GetVolume().GetVolumeContext()}
	case "delete-volume":
		_, err := ctrl.DeleteVolume(ctx, &csipb.DeleteVolumeRequest{VolumeId: *id})
		check(err)
		out = map[string]any{"deleted": *id}
	case "expand-volume":
		r, err := ctrl.ControllerExpandVolume(ctx, &csipb.ControllerExpandVolumeRequest{VolumeId: *id,
			CapacityRange: &csipb.CapacityRange{RequiredBytes: *size}, VolumeCapability: blockCap(*block)})
		check(err)
		out = map[string]any{"capacityBytes": r.GetCapacityBytes()}
	case "publish":
		r, err := ctrl.ControllerPublishVolume(ctx, &csipb.ControllerPublishVolumeRequest{VolumeId: *id, NodeId: *node,
			VolumeCapability: blockCap(*block)})
		check(err)
		out = r.GetPublishContext()
	case "unpublish":
		_, err := ctrl.ControllerUnpublishVolume(ctx, &csipb.ControllerUnpublishVolumeRequest{VolumeId: *id, NodeId: *node})
		check(err)
		out = map[string]any{"unpublished": *id}
	case "stage":
		_, err := node_.NodeStageVolume(ctx, &csipb.NodeStageVolumeRequest{VolumeId: *id, StagingTargetPath: *staging,
			VolumeCapability: blockCap(*block), PublishContext: map[string]string{"devicePath": *device}})
		check(err)
		out = map[string]any{"staged": *staging}
	case "unstage":
		_, err := node_.NodeUnstageVolume(ctx, &csipb.NodeUnstageVolumeRequest{VolumeId: *id, StagingTargetPath: *staging})
		check(err)
		out = map[string]any{"unstaged": *staging}
	case "node-publish":
		_, err := node_.NodePublishVolume(ctx, &csipb.NodePublishVolumeRequest{VolumeId: *id, StagingTargetPath: *staging,
			TargetPath: *target, VolumeCapability: blockCap(*block)})
		check(err)
		out = map[string]any{"published": *target}
	case "node-unpublish":
		_, err := node_.NodeUnpublishVolume(ctx, &csipb.NodeUnpublishVolumeRequest{VolumeId: *id, TargetPath: *target})
		check(err)
		out = map[string]any{"unpublished": *target}
	case "snapshot":
		r, err := ctrl.CreateSnapshot(ctx, &csipb.CreateSnapshotRequest{Name: *name, SourceVolumeId: *source})
		check(err)
		out = map[string]any{"snapshotId": r.GetSnapshot().GetSnapshotId(), "sizeBytes": r.GetSnapshot().GetSizeBytes(),
			"readyToUse": r.GetSnapshot().GetReadyToUse()}
	case "delete-snapshot":
		_, err := ctrl.DeleteSnapshot(ctx, &csipb.DeleteSnapshotRequest{SnapshotId: *id})
		check(err)
		out = map[string]any{"deleted": *id}
	case "list-volumes":
		r, err := ctrl.ListVolumes(ctx, &csipb.ListVolumesRequest{})
		check(err)
		var ids []string
		for _, e := range r.GetEntries() {
			ids = append(ids, e.GetVolume().GetVolumeId())
		}
		out = map[string]any{"volumes": ids}
	case "list-snapshots":
		r, err := ctrl.ListSnapshots(ctx, &csipb.ListSnapshotsRequest{SourceVolumeId: *source})
		check(err)
		var ids []string
		for _, e := range r.GetEntries() {
			ids = append(ids, e.GetSnapshot().GetSnapshotId())
		}
		out = map[string]any{"snapshots": ids}
	default:
		usage()
	}
	b, _ := json.Marshal(out)
	fmt.Println(string(b))
}

func check(err error) {
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	b, _ := json.Marshal(map[string]string{"error": err.Error()})
	fmt.Println(string(b))
	os.Exit(1)
}
