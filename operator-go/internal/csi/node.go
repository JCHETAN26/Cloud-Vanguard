package csi

import (
	"context"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type nodeServer struct {
	csipb.UnimplementedNodeServer
	d *Driver
}

func (s *nodeServer) NodeGetInfo(context.Context, *csipb.NodeGetInfoRequest) (*csipb.NodeGetInfoResponse, error) {
	return &csipb.NodeGetInfoResponse{NodeId: s.d.NodeID, MaxVolumesPerNode: 64}, nil
}

func (s *nodeServer) NodeGetCapabilities(context.Context, *csipb.NodeGetCapabilitiesRequest) (*csipb.NodeGetCapabilitiesResponse, error) {
	return &csipb.NodeGetCapabilitiesResponse{Capabilities: []*csipb.NodeServiceCapability{{
		Type: &csipb.NodeServiceCapability_Rpc{Rpc: &csipb.NodeServiceCapability_RPC{
			Type: csipb.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME}},
	}}}, nil
}

func (s *nodeServer) NodeStageVolume(_ context.Context, req *csipb.NodeStageVolumeRequest) (*csipb.NodeStageVolumeResponse, error) {
	switch {
	case req.GetVolumeId() == "":
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	case req.GetStagingTargetPath() == "":
		return nil, status.Error(codes.InvalidArgument, "staging_target_path is required")
	case req.GetVolumeCapability() == nil:
		return nil, status.Error(codes.InvalidArgument, "volume_capability is required")
	}
	if !supported(req.GetVolumeCapability()) {
		return nil, status.Error(codes.InvalidArgument, "unsupported volume capability")
	}
	device := req.GetPublishContext()[ContextDevicePath]
	if device == "" {
		return nil, status.Error(codes.InvalidArgument, "publish_context.devicePath missing (was the volume ControllerPublished?)")
	}
	s.d.nodeMu.Lock()
	defer s.d.nodeMu.Unlock()
	fsType := req.GetVolumeCapability().GetMount().GetFsType()
	if err := s.d.Mounter.Stage(device, req.GetStagingTargetPath(), isBlock(req.GetVolumeCapability()), fsType); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &csipb.NodeStageVolumeResponse{}, nil
}

func (s *nodeServer) NodeUnstageVolume(_ context.Context, req *csipb.NodeUnstageVolumeRequest) (*csipb.NodeUnstageVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	}
	if req.GetStagingTargetPath() == "" {
		return nil, status.Error(codes.InvalidArgument, "staging_target_path is required")
	}
	s.d.nodeMu.Lock()
	defer s.d.nodeMu.Unlock()
	if err := s.d.Mounter.Unstage(req.GetStagingTargetPath()); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &csipb.NodeUnstageVolumeResponse{}, nil
}

func (s *nodeServer) NodePublishVolume(_ context.Context, req *csipb.NodePublishVolumeRequest) (*csipb.NodePublishVolumeResponse, error) {
	switch {
	case req.GetVolumeId() == "":
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	case req.GetTargetPath() == "":
		return nil, status.Error(codes.InvalidArgument, "target_path is required")
	case req.GetVolumeCapability() == nil:
		return nil, status.Error(codes.InvalidArgument, "volume_capability is required")
	case req.GetStagingTargetPath() == "":
		return nil, status.Error(codes.FailedPrecondition, "staging_target_path is required (volume not staged)")
	}
	if !supported(req.GetVolumeCapability()) {
		return nil, status.Error(codes.InvalidArgument, "unsupported volume capability")
	}
	s.d.nodeMu.Lock()
	defer s.d.nodeMu.Unlock()
	if err := s.d.Mounter.Publish(req.GetStagingTargetPath(), req.GetTargetPath(), isBlock(req.GetVolumeCapability()),
		req.GetReadonly()); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &csipb.NodePublishVolumeResponse{}, nil
}

func (s *nodeServer) NodeUnpublishVolume(_ context.Context, req *csipb.NodeUnpublishVolumeRequest) (*csipb.NodeUnpublishVolumeResponse, error) {
	if req.GetVolumeId() == "" {
		return nil, status.Error(codes.InvalidArgument, "volume_id is required")
	}
	if req.GetTargetPath() == "" {
		return nil, status.Error(codes.InvalidArgument, "target_path is required")
	}
	s.d.nodeMu.Lock()
	defer s.d.nodeMu.Unlock()
	if err := s.d.Mounter.Unpublish(req.GetTargetPath()); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &csipb.NodeUnpublishVolumeResponse{}, nil
}
