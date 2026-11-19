// Package csi implements the CloudVanguard CSI plugin (csi.cloudvanguard.io):
// Identity, Controller and Node services over gRPC, backed by QEMU disk
// images (internal/storage). Warm-cache volumes are raw block devices
// (volumeMode: Block) that vanguard-proxy writes its A/B snapshot slots to.
package csi

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"
	"sync"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/storage"
)

const (
	DriverName    = "csi.cloudvanguard.io"
	DriverVersion = "0.4.0"
	// Volume parameter (StorageClass.parameters.format): raw | qcow2.
	ParamFormat = "format"
	// Publish context key carrying the device path from controller to node.
	ContextDevicePath = "devicePath"
	defaultSize       = 64 << 20
	mib               = 1 << 20
)

// Driver holds the state shared by the three services. A process can run the
// controller (Store set), the node (Mounter set) or both.
type Driver struct {
	Name    string
	Version string
	NodeID  string
	// Controller side.
	Store *storage.Store
	// NodeExists validates ControllerPublishVolume's node_id.
	NodeExists func(nodeID string) bool
	// Node side.
	Mounter Mounter

	nodeMu sync.Mutex
}

// Register adds the services this driver provides to a gRPC server.
func (d *Driver) Register(s *grpc.Server) {
	csipb.RegisterIdentityServer(s, &identityServer{d: d})
	if d.Store != nil {
		csipb.RegisterControllerServer(s, &controllerServer{d: d})
	}
	if d.Mounter != nil {
		csipb.RegisterNodeServer(s, &nodeServer{d: d})
	}
}

// Listen opens a CSI endpoint: unix:///path/csi.sock or tcp://host:port.
func Listen(endpoint string) (net.Listener, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, err
	}
	switch u.Scheme {
	case "unix":
		path := u.Path
		if u.Host != "" {
			path = u.Host + u.Path
		}
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		return net.Listen("unix", path)
	case "tcp":
		return net.Listen("tcp", u.Host)
	default:
		return nil, fmt.Errorf("unsupported endpoint %q (want unix:// or tcp://)", endpoint)
	}
}

// Serve runs the gRPC server until ctx is cancelled.
func (d *Driver) Serve(ctx context.Context, lis net.Listener) error {
	s := grpc.NewServer(grpc.UnaryInterceptor(logErrors))
	d.Register(s)
	go func() {
		<-ctx.Done()
		s.GracefulStop()
	}()
	return s.Serve(lis)
}

func logErrors(ctx context.Context, req any, info *grpc.UnaryServerInfo, h grpc.UnaryHandler) (any, error) {
	resp, err := h(ctx, req)
	if err != nil && os.Getenv("CSI_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "%s: %v\n", info.FullMethod, err)
	}
	return resp, err
}

// toStatus maps storage errors onto CSI gRPC codes.
func toStatus(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, storage.ErrNotFound):
		return status.Error(codes.NotFound, err.Error())
	case errors.Is(err, storage.ErrAlreadyExists):
		return status.Error(codes.AlreadyExists, err.Error())
	case errors.Is(err, storage.ErrInUse):
		return status.Error(codes.FailedPrecondition, err.Error())
	case errors.Is(err, storage.ErrInvalidRequest):
		return status.Error(codes.InvalidArgument, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}

// supported reports whether a capability is one this driver serves:
// single-node access, block or filesystem.
func supported(c *csipb.VolumeCapability) bool {
	if c == nil || (c.GetBlock() == nil && c.GetMount() == nil) {
		return false
	}
	switch c.GetAccessMode().GetMode() {
	case csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER,
		csipb.VolumeCapability_AccessMode_SINGLE_NODE_READER_ONLY,
		csipb.VolumeCapability_AccessMode_SINGLE_NODE_SINGLE_WRITER,
		csipb.VolumeCapability_AccessMode_SINGLE_NODE_MULTI_WRITER:
		return true
	}
	return false
}

func validateCaps(caps []*csipb.VolumeCapability) error {
	if len(caps) == 0 {
		return status.Error(codes.InvalidArgument, "volume capabilities are required")
	}
	for _, c := range caps {
		if !supported(c) {
			return status.Errorf(codes.InvalidArgument, "unsupported volume capability %v", c)
		}
	}
	return nil
}

func isBlock(c *csipb.VolumeCapability) bool { return c.GetBlock() != nil }

// paginate applies CSI starting_token / max_entries semantics to a slice.
func paginate[T any](items []T, token string, max int32) ([]T, string, error) {
	if max < 0 {
		return nil, "", status.Error(codes.InvalidArgument, "max_entries must not be negative")
	}
	start := 0
	if token != "" {
		n, err := fmt.Sscanf(token, "%d", &start)
		if err != nil || n != 1 || start < 0 || start > len(items) || strings.TrimSpace(token) != fmt.Sprint(start) {
			return nil, "", status.Errorf(codes.Aborted, "invalid starting_token %q", token)
		}
	}
	end := len(items)
	if max > 0 && start+int(max) < end {
		end = start + int(max)
	}
	next := ""
	if end < len(items) {
		next = fmt.Sprint(end)
	}
	return items[start:end], next, nil
}
