package csi

import (
	"context"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/storage"
)

type controllerServer struct {
	csipb.UnimplementedControllerServer
	d *Driver
}

func (s *controllerServer) ControllerGetCapabilities(context.Context, *csipb.ControllerGetCapabilitiesRequest) (*csipb.ControllerGetCapabilitiesResponse, error) {
	var caps []*csipb.ControllerServiceCapability
	for _, t := range []csipb.ControllerServiceCapability_RPC_Type{
		csipb.ControllerServiceCapability_RPC_CREATE_DELETE_VOLUME,
		csipb.ControllerServiceCapability_RPC_PUBLISH_UNPUBLISH_VOLUME,
		csipb.ControllerServiceCapability_RPC_LIST_VOLUMES,
		csipb.ControllerServiceCapability_RPC_CREATE_DELETE_SNAPSHOT,
		csipb.ControllerServiceCapability_RPC_LIST_SNAPSHOTS,
		csipb.ControllerServiceCapability_RPC_EXPAND_VOLUME,
	} {
		caps = append(caps, &csipb.ControllerServiceCapability{Type: &csipb.ControllerServiceCapability_Rpc{
			Rpc: &csipb.ControllerServiceCapability_RPC{Type: t}}})
	}
	return &csipb.ControllerGetCapabilitiesResponse{Capabilities: caps}, nil
}

func requestedSize(r *csipb.CapacityRange) (int64, error) {
	size := r.GetRequiredBytes()
	if size == 0 {
		size = defaultSize
		if l := r.GetLimitBytes(); l > 0 && l < size {
			size = l
		}
	}
	size = (size + mib - 1) / mib * mib // whole MiB
	if l := r.GetLimitBytes(); l > 0 && size > l {
		return 0, status.Errorf(codes.OutOfRange, "required %d bytes exceeds limit %d", size, l)
	}
	return size, nil
}

func volumeProto(v *storage.Volume) *csipb.Volume {
	out := &csipb.Volume{
		VolumeId:      v.ID,
		CapacityBytes: v.Capacity,
		VolumeContext: map[string]string{ParamFormat: v.Format},
	}
	if v.SourceSnap != "" {
		out.ContentSource = &csipb.VolumeContentSource{Type: &csipb.VolumeContentSource_Snapshot{
			Snapshot: &csipb.VolumeContentSource_SnapshotSource{SnapshotId: v.SourceSnap}}}
	}
	return out
}

func (s *controllerServer) CreateVolume(ctx context.Context, req *csipb.CreateVolumeRequest) (*csipb.CreateVolumeResponse, error) {
	if req.GetName() == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}
	if err := validateCaps(req.GetVolumeCapabilities()); err != nil {
		return nil, err
	}
	size, err := requestedSize(req.GetCapacityRange())
	if err != nil {
		return nil, err
	}
	fromSnap := ""
	if src := req.GetVolumeContentSource(); src != nil {
		snap := src.GetSnapshot()
		if snap == nil {
			return nil, status.Error(codes.InvalidArgument, "only snapshot content sources are supported")
		}
		source, err := s.d.Store.GetSnapshot(snap.GetSnapshotId())
		if err != nil {
			return nil, status.Errorf(codes.NotFound, "source snapshot %s: %v", snap.GetSnapshotId(), err)
		}
		// The restored volume is at least as large as the snapshot; that must
		// still respect the caller's limit.
		if l := req.GetCapacityRange().GetLimitBytes(); l > 0 && source.Size > l {
			return nil, status.Errorf(codes.OutOfRange, "snapshot %s is %d bytes, above limit_bytes %d",
				source.ID, source.Size, l)
		}
		fromSnap = snap.GetSnapshotId()
	}
	vol, err := s.d.Store.CreateVolume(ctx, req.GetName(), size, req.GetParameters()[ParamFormat], fromSnap)
	if err != nil {
		return nil, toStatus(err)
	}
	return &csipb.CreateVolumeResponse{Volume: volumeProto(vol)}, nil
}

func (s *controllerServer) DeleteVolume(_ context.Context, req *csipb.DeleteVolumeRequest) (*csipb.DeleteVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	}
	return &csipb.DeleteVolumeResponse{}, toStatus(s.d.Store.DeleteVolume(req.GetVolumeId()))
}

func (s *controllerServer) ControllerPublishVolume(ctx context.Context, req *csipb.ControllerPublishVolumeRequest) (*csipb.ControllerPublishVolumeResponse, error) {
	switch {
	case req.GetVolumeId() == "":
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	case req.GetNodeId() == "":
		return nil, status.Error(codes.InvalidArgument, "node_id is required")
	case req.GetVolumeCapability() == nil:
		return nil, status.Error(codes.InvalidArgument, "volume_capability is required")
	}
	if !supported(req.GetVolumeCapability()) {
		return nil, status.Error(codes.InvalidArgument, "unsupported volume capability")
	}
	if _, err := s.d.Store.GetVolume(req.GetVolumeId()); err != nil {
		return nil, toStatus(err)
	}
	if s.d.NodeExists != nil && !s.d.NodeExists(req.GetNodeId()) {
		return nil, status.Errorf(codes.NotFound, "node %q not found", req.GetNodeId())
	}
	multi := req.GetVolumeCapability().GetAccessMode().GetMode() == csipb.VolumeCapability_AccessMode_SINGLE_NODE_MULTI_WRITER
	dev, err := s.d.Store.Attach(ctx, req.GetVolumeId(), req.GetNodeId(), multi)
	if err != nil {
		return nil, toStatus(err)
	}
	return &csipb.ControllerPublishVolumeResponse{PublishContext: map[string]string{ContextDevicePath: dev}}, nil
}

func (s *controllerServer) ControllerUnpublishVolume(ctx context.Context, req *csipb.ControllerUnpublishVolumeRequest) (*csipb.ControllerUnpublishVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	}
	if req.GetNodeId() == "" {
		// Unpublish from every node.
		vol, err := s.d.Store.GetVolume(req.GetVolumeId())
		if err != nil {
			return &csipb.ControllerUnpublishVolumeResponse{}, nil
		}
		for node := range vol.Attached {
			if err := s.d.Store.Detach(ctx, vol.ID, node); err != nil {
				return nil, toStatus(err)
			}
		}
		return &csipb.ControllerUnpublishVolumeResponse{}, nil
	}
	return &csipb.ControllerUnpublishVolumeResponse{}, toStatus(s.d.Store.Detach(ctx, req.GetVolumeId(), req.GetNodeId()))
}

func (s *controllerServer) ValidateVolumeCapabilities(_ context.Context, req *csipb.ValidateVolumeCapabilitiesRequest) (*csipb.ValidateVolumeCapabilitiesResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	}
	if len(req.GetVolumeCapabilities()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "volume_capabilities are required")
	}
	if _, err := s.d.Store.GetVolume(req.GetVolumeId()); err != nil {
		return nil, toStatus(err)
	}
	for _, c := range req.GetVolumeCapabilities() {
		if !supported(c) {
			return &csipb.ValidateVolumeCapabilitiesResponse{Message: "unsupported capability"}, nil
		}
	}
	return &csipb.ValidateVolumeCapabilitiesResponse{Confirmed: &csipb.ValidateVolumeCapabilitiesResponse_Confirmed{
		VolumeContext:      req.GetVolumeContext(),
		VolumeCapabilities: req.GetVolumeCapabilities(),
		Parameters:         req.GetParameters(),
	}}, nil
}

func (s *controllerServer) ListVolumes(_ context.Context, req *csipb.ListVolumesRequest) (*csipb.ListVolumesResponse, error) {
	vols, err := s.d.Store.ListVolumes()
	if err != nil {
		return nil, toStatus(err)
	}
	page, next, err := paginate(vols, req.GetStartingToken(), req.GetMaxEntries())
	if err != nil {
		return nil, err
	}
	out := &csipb.ListVolumesResponse{NextToken: next}
	for _, v := range page {
		out.Entries = append(out.Entries, &csipb.ListVolumesResponse_Entry{Volume: volumeProto(v)})
	}
	return out, nil
}

func snapshotProto(s *storage.Snapshot) *csipb.Snapshot {
	return &csipb.Snapshot{
		SnapshotId:     s.ID,
		SourceVolumeId: s.SourceID,
		SizeBytes:      s.Size,
		CreationTime:   timestamppb.New(s.Created),
		ReadyToUse:     true,
	}
}

func (s *controllerServer) CreateSnapshot(ctx context.Context, req *csipb.CreateSnapshotRequest) (*csipb.CreateSnapshotResponse, error) {
	if req.GetName() == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}
	if req.GetSourceVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "source_volume_id is required")
	}
	snap, err := s.d.Store.CreateSnapshot(ctx, req.GetName(), req.GetSourceVolumeId())
	if err != nil {
		return nil, toStatus(err)
	}
	return &csipb.CreateSnapshotResponse{Snapshot: snapshotProto(snap)}, nil
}

func (s *controllerServer) DeleteSnapshot(_ context.Context, req *csipb.DeleteSnapshotRequest) (*csipb.DeleteSnapshotResponse, error) {
	if req.GetSnapshotId() == "" {
		return nil, status.Error(codes.InvalidArgument, "snapshot_id is required")
	}
	return &csipb.DeleteSnapshotResponse{}, toStatus(s.d.Store.DeleteSnapshot(req.GetSnapshotId()))
}

func (s *controllerServer) ListSnapshots(_ context.Context, req *csipb.ListSnapshotsRequest) (*csipb.ListSnapshotsResponse, error) {
	all, err := s.d.Store.ListSnapshots()
	if err != nil {
		return nil, toStatus(err)
	}
	var filtered []*storage.Snapshot
	for _, sn := range all {
		if id := req.GetSnapshotId(); id != "" && sn.ID != id {
			continue
		}
		if src := req.GetSourceVolumeId(); src != "" && sn.SourceID != src {
			continue
		}
		filtered = append(filtered, sn)
	}
	page, next, err := paginate(filtered, req.GetStartingToken(), req.GetMaxEntries())
	if err != nil {
		return nil, err
	}
	out := &csipb.ListSnapshotsResponse{NextToken: next}
	for _, sn := range page {
		out.Entries = append(out.Entries, &csipb.ListSnapshotsResponse_Entry{Snapshot: snapshotProto(sn)})
	}
	return out, nil
}

func (s *controllerServer) ControllerExpandVolume(ctx context.Context, req *csipb.ControllerExpandVolumeRequest) (*csipb.ControllerExpandVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	}
	if req.GetCapacityRange() == nil {
		return nil, status.Error(codes.InvalidArgument, "capacity_range is required")
	}
	size, err := requestedSize(req.GetCapacityRange())
	if err != nil {
		return nil, err
	}
	vol, err := s.d.Store.ExpandVolume(ctx, req.GetVolumeId(), size)
	if err != nil {
		return nil, toStatus(err)
	}
	// Block volumes need nothing on the node; a filesystem would need resize2fs.
	nodeExpansion := req.GetVolumeCapability() != nil && !isBlock(req.GetVolumeCapability())
	return &csipb.ControllerExpandVolumeResponse{CapacityBytes: vol.Capacity, NodeExpansionRequired: nodeExpansion}, nil
}
