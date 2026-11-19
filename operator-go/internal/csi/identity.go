package csi

import (
	"context"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

type identityServer struct {
	csipb.UnimplementedIdentityServer
	d *Driver
}

func (s *identityServer) GetPluginInfo(context.Context, *csipb.GetPluginInfoRequest) (*csipb.GetPluginInfoResponse, error) {
	name, version := s.d.Name, s.d.Version
	if name == "" {
		name = DriverName
	}
	if version == "" {
		version = DriverVersion
	}
	return &csipb.GetPluginInfoResponse{Name: name, VendorVersion: version}, nil
}

func (s *identityServer) GetPluginCapabilities(context.Context, *csipb.GetPluginCapabilitiesRequest) (*csipb.GetPluginCapabilitiesResponse, error) {
	caps := []*csipb.PluginCapability{{
		Type: &csipb.PluginCapability_VolumeExpansion_{VolumeExpansion: &csipb.PluginCapability_VolumeExpansion{
			Type: csipb.PluginCapability_VolumeExpansion_OFFLINE}},
	}}
	if s.d.Store != nil {
		caps = append(caps, &csipb.PluginCapability{Type: &csipb.PluginCapability_Service_{
			Service: &csipb.PluginCapability_Service{Type: csipb.PluginCapability_Service_CONTROLLER_SERVICE}}})
	}
	return &csipb.GetPluginCapabilitiesResponse{Capabilities: caps}, nil
}

func (s *identityServer) Probe(context.Context, *csipb.ProbeRequest) (*csipb.ProbeResponse, error) {
	return &csipb.ProbeResponse{Ready: wrapperspb.Bool(true)}, nil
}
