package storage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"syscall"
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
//
// QEMU's QMP socket accepts one client at a time: the attacher needs the
// monitor to itself. For VMs managed by libvirt (which holds the monitor),
// attach through libvirt (virsh attach-device) instead.
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

// vmGone reports a dial error meaning the node's VM is not running: its QMP
// socket does not exist, or exists with nobody listening (QEMU exited without
// removing it). A live QEMU always accepts connections on its monitor.
func vmGone(err error) bool {
	return errors.Is(err, syscall.ENOENT) || errors.Is(err, syscall.ECONNREFUSED)
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

// devicePresent reports whether a device with this id is plugged in
// (user-created devices with an id live under /machine/peripheral).
func devicePresent(c *QMPClient, id string) (bool, error) {
	raw, err := c.Execute("qom-list", map[string]any{"path": "/machine/peripheral"})
	if err != nil {
		return false, err
	}
	var props []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &props); err != nil {
		return false, err
	}
	for _, p := range props {
		if p.Name == id {
			return true, nil
		}
	}
	return false, nil
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

// Attach is idempotent and repairs a half-finished earlier attempt: the
// volume counts as attached only when both the block node and the guest
// device exist. A block node left behind by an attempt that died between
// blockdev-add and device_add (or whose rollback failed) gets its device now.
func (a QMPAttacher) Attach(ctx context.Context, vol *Volume, node string) (string, error) {
	c, cancel, err := a.dial(ctx, node)
	if err != nil {
		if vmGone(err) {
			return "", fmt.Errorf("%w: VM for node %s is not running: %v", ErrNotFound, node, err)
		}
		return "", err
	}
	defer cancel()
	defer c.Close()
	nodeName, deviceID, serial := qmpNames(vol)
	haveNode, err := attached(c, nodeName)
	if err != nil {
		return "", err
	}
	haveDevice, err := devicePresent(c, deviceID)
	if err != nil {
		return "", err
	}
	switch {
	case haveNode && haveDevice:
		return GuestDevicePath(vol), nil
	case haveDevice:
		// A device with our id but without our block node is not ours to reuse.
		return "", fmt.Errorf("node %s: device %s exists without block node %s", node, deviceID, nodeName)
	case !haveNode:
		if _, err := c.Execute("blockdev-add", map[string]any{
			"node-name": nodeName,
			"driver":    vol.Format,
			"file":      map[string]any{"driver": "file", "filename": vol.Path},
		}); err != nil {
			return "", err
		}
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
		if vmGone(err) {
			// The VM (and every device hot-plugged into it) is gone: the volume
			// is detached. CSI requires ControllerUnpublish to succeed here, or
			// the volume could never move to a replacement node.
			return nil
		}
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
