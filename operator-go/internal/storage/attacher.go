package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Attacher makes a volume image visible to a node (ControllerPublishVolume)
// and returns the device path the node plugin should stage.
type Attacher interface {
	Name() string
	Attach(ctx context.Context, vol *Volume, nodeID string) (devicePath string, err error)
	Detach(ctx context.Context, vol *Volume, nodeID string) error
}

// LocalAttacher is for nodes that share the host's filesystem (single-host
// dev clusters, the test suite): the "device" is the raw image itself.
type LocalAttacher struct{}

func (LocalAttacher) Name() string { return "local" }

func (LocalAttacher) Attach(_ context.Context, vol *Volume, _ string) (string, error) {
	if vol.Format != FormatRaw {
		return "", fmt.Errorf("local attach needs a raw image (volume %s is %s)", vol.ID, vol.Format)
	}
	return vol.Path, nil
}

func (LocalAttacher) Detach(context.Context, *Volume, string) error { return nil }

// QMPAttacher hot-plugs images into the QEMU/KVM virtual machine that backs
// each Kubernetes node. Every node VM is started with a QMP socket and a
// virtio-scsi controller, e.g.
//
//	qemu-system-x86_64 ... -qmp unix:/run/vanguard/qmp/<node>.sock,server=on,wait=off \
//	    -device virtio-scsi-pci,id=scsi0
//
// Attach = blockdev-add (the image as a block node) + device_add scsi-hd on
// that controller with the volume ID as serial; the guest sees
// /dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_<serial>. SCSI hot-unplug is
// immediate (no guest ACPI handshake), so Detach is synchronous.
type QMPAttacher struct {
	// SocketTemplate maps a node ID to its VM's QMP socket; "{node}" is replaced.
	SocketTemplate string
	// SCSIBus is the controller bus, default "scsi0.0".
	SCSIBus string
	Timeout time.Duration
}

func (a QMPAttacher) Name() string { return "qmp" }

func (a QMPAttacher) socket(node string) string {
	return strings.ReplaceAll(a.SocketTemplate, "{node}", node)
}

// QMP node names / device ids are limited to 31 characters.
func qmpNames(vol *Volume) (nodeName, deviceID, serial string) {
	short := strings.TrimPrefix(vol.ID, "vol-")
	if len(short) > 16 {
		short = short[:16]
	}
	return "vgblk-" + short, "vgdev-" + short, short
}

// GuestDevicePath is where the attached disk appears inside the VM.
func GuestDevicePath(vol *Volume) string {
	_, _, serial := qmpNames(vol)
	return "/dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_" + serial
}

func (a QMPAttacher) dial(ctx context.Context, node string) (*QMPClient, context.CancelFunc, error) {
	timeout := a.Timeout
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	c, err := DialQMP(ctx, a.socket(node))
	if err != nil {
		cancel()
		return nil, nil, err
	}
	return c, cancel, nil
}

// attached reports whether the block node is already present (idempotency).
func attached(c *QMPClient, nodeName string) (bool, error) {
	raw, err := c.Execute("query-named-block-nodes", map[string]any{"flat": true})
	if err != nil {
		return false, err
	}
	var nodes []struct {
		NodeName string `json:"node-name"`
	}
	if err := json.Unmarshal(raw, &nodes); err != nil {
		return false, err
	}
	for _, n := range nodes {
		if n.NodeName == nodeName {
			return true, nil
		}
	}
	return false, nil
}

func (a QMPAttacher) Attach(ctx context.Context, vol *Volume, node string) (string, error) {
	c, cancel, err := a.dial(ctx, node)
	if err != nil {
		return "", err
	}
	defer cancel()
	defer c.Close()
	nodeName, deviceID, serial := qmpNames(vol)
	if ok, err := attached(c, nodeName); err != nil {
		return "", err
	} else if ok {
		return GuestDevicePath(vol), nil
	}
	if _, err := c.Execute("blockdev-add", map[string]any{
		"node-name": nodeName,
		"driver":    vol.Format,
		"file":      map[string]any{"driver": "file", "filename": vol.Path},
	}); err != nil {
		return "", err
	}
	bus := a.SCSIBus
	if bus == "" {
		bus = "scsi0.0"
	}
	if _, err := c.Execute("device_add", map[string]any{
		"driver": "scsi-hd", "bus": bus, "drive": nodeName, "id": deviceID, "serial": serial,
	}); err != nil {
		_, _ = c.Execute("blockdev-del", map[string]any{"node-name": nodeName})
		return "", err
	}
	return GuestDevicePath(vol), nil
}

func (a QMPAttacher) Detach(ctx context.Context, vol *Volume, node string) error {
	c, cancel, err := a.dial(ctx, node)
	if err != nil {
		return err
	}
	defer cancel()
	defer c.Close()
	nodeName, deviceID, _ := qmpNames(vol)
	if ok, err := attached(c, nodeName); err != nil || !ok {
		return err // already detached
	}
	if _, err := c.Execute("device_del", map[string]any{"id": deviceID}); err != nil &&
		!strings.Contains(err.Error(), "not found") {
		return err
	}
	// The block node is released once the device is gone; SCSI unplug is
	// immediate, but allow a short grace period for the DEVICE_DELETED event.
	for i := 0; ; i++ {
		_, err = c.Execute("blockdev-del", map[string]any{"node-name": nodeName})
		if err == nil || i >= 20 || !strings.Contains(err.Error(), "in use") {
			return err
		}
		time.Sleep(100 * time.Millisecond)
	}
}
