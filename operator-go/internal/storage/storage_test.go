package storage

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

var ctx = context.Background()

func tools(t *testing.T) []ImageTool {
	out := []ImageTool{RawFileTool{}}
	if QemuImgAvailable() {
		out = append(out, QemuImg{})
	} else {
		t.Log("qemu-img not installed: QEMU image tests limited to the raw-file fallback")
	}
	return out
}

// writePattern fills [off, off+len(p)) of a raw image, as a workload would.
func writePattern(t *testing.T, path string, off int64, p []byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteAt(p, off); err != nil {
		t.Fatal(err)
	}
}

func readAt(t *testing.T, path string, off int64, n int) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b := make([]byte, n)
	if _, err := f.ReadAt(b, off); err != nil {
		t.Fatal(err)
	}
	return b
}

// Volume data written through the raw device survives snapshot (compressed
// qcow2 with qemu-img) and restore into a new volume, byte for byte.
func TestVolumeSnapshotRestorePreservesData(t *testing.T) {
	for _, tool := range tools(t) {
		t.Run(tool.Name(), func(t *testing.T) {
			s, err := NewStore(t.TempDir(), tool, LocalAttacher{})
			if err != nil {
				t.Fatal(err)
			}
			vol, err := s.CreateVolume(ctx, "pvc-warm-cache", 32<<20, FormatRaw, "")
			if err != nil {
				t.Fatal(err)
			}
			if info, err := tool.Info(ctx, vol.Path); err != nil || info.VirtualSize != 32<<20 || info.Format != FormatRaw {
				t.Fatalf("info %+v %v", info, err)
			}
			pattern := make([]byte, 256<<10)
			_, _ = rand.Read(pattern)
			writePattern(t, vol.Path, 3<<20, pattern)

			snap, err := s.CreateSnapshot(ctx, "nightly", vol.ID)
			if err != nil {
				t.Fatal(err)
			}
			wantFormat := FormatRaw
			if tool.Name() == "qemu-img" {
				wantFormat = FormatQCOW2
				if info, _ := tool.Info(ctx, snap.Path); info.Format != FormatQCOW2 || info.ActualSize >= 32<<20 {
					t.Fatalf("snapshot should be a compressed qcow2: %+v", info)
				}
			}
			if snap.Format != wantFormat {
				t.Fatalf("snapshot format %s", snap.Format)
			}
			// Overwrite the source after the snapshot: the snapshot must not change.
			writePattern(t, vol.Path, 3<<20, bytes.Repeat([]byte{0xAB}, len(pattern)))

			restored, err := s.CreateVolume(ctx, "pvc-restored", 64<<20, FormatRaw, snap.ID)
			if err != nil {
				t.Fatal(err)
			}
			if got := readAt(t, restored.Path, 3<<20, len(pattern)); !bytes.Equal(got, pattern) {
				t.Fatal("restored volume does not contain the snapshotted bytes")
			}
			if info, _ := tool.Info(ctx, restored.Path); info.VirtualSize != 64<<20 {
				t.Fatalf("restore should grow to the requested size: %+v", info)
			}
		})
	}
}

func TestStoreIdempotencyAndGuards(t *testing.T) {
	s, err := NewStore(t.TempDir(), RawFileTool{}, LocalAttacher{})
	if err != nil {
		t.Fatal(err)
	}
	a, err := s.CreateVolume(ctx, "v", 8<<20, FormatRaw, "")
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.CreateVolume(ctx, "v", 8<<20, FormatRaw, "")
	if err != nil || b.ID != a.ID {
		t.Fatalf("CreateVolume not idempotent: %v %v", b, err)
	}
	if _, err := s.CreateVolume(ctx, "v", 16<<20, FormatRaw, ""); !errors.Is(err, ErrAlreadyExists) {
		t.Fatalf("larger re-create should conflict: %v", err)
	}
	dev, err := s.Attach(ctx, a.ID, "node-a", false)
	if err != nil || dev != a.Path {
		t.Fatalf("attach: %s %v", dev, err)
	}
	if _, err := s.Attach(ctx, a.ID, "node-b", false); !errors.Is(err, ErrInUse) {
		t.Fatalf("second single-writer attach should fail: %v", err)
	}
	if err := s.DeleteVolume(a.ID); !errors.Is(err, ErrInUse) {
		t.Fatalf("deleting an attached volume should fail: %v", err)
	}
	if err := s.Detach(ctx, a.ID, "node-a"); err != nil {
		t.Fatal(err)
	}
	if err := s.Detach(ctx, a.ID, "node-a"); err != nil {
		t.Fatal("detach must be idempotent")
	}
	if _, err := s.ExpandVolume(ctx, a.ID, 12<<20); err != nil {
		t.Fatal(err)
	}
	if st, _ := os.Stat(a.Path); st.Size() != 12<<20 {
		t.Fatalf("expanded size %d", st.Size())
	}
	if err := s.DeleteVolume(a.ID); err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteVolume(a.ID); err != nil {
		t.Fatal("delete must be idempotent")
	}
	if _, err := os.Stat(a.Path); !os.IsNotExist(err) {
		t.Fatal("image file left behind")
	}
	// Metadata survives a driver restart.
	c, _ := s.CreateVolume(ctx, "persisted", 4<<20, FormatRaw, "")
	s2, _ := NewStore(s.Dir, RawFileTool{}, LocalAttacher{})
	if got, err := s2.GetVolume(c.ID); err != nil || got.Name != "persisted" {
		t.Fatalf("restart lost volume: %v %v", got, err)
	}
}

// --- QMP -------------------------------------------------------------------

// fakeQMP is a minimal QMP server recording commands and tracking block
// nodes and devices, with failure injection.
type fakeQMPState struct {
	socket        string
	mu            sync.Mutex
	cmds          []string
	nodes         map[string]bool
	devices       map[string]bool
	failDeviceAdd bool
}

func (f *fakeQMPState) commands() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return strings.Join(f.cmds, ",")
}

func (f *fakeQMPState) reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cmds = nil
}

func fakeQMP(t *testing.T) *fakeQMPState {
	t.Helper()
	dir, _ := os.MkdirTemp("", "qmp")
	t.Cleanup(func() { os.RemoveAll(dir) })
	f := &fakeQMPState{socket: filepath.Join(dir, "vm.sock"), nodes: map[string]bool{}, devices: map[string]bool{}}
	lis, err := net.Listen("unix", f.socket)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { lis.Close() })
	go func() {
		for {
			conn, err := lis.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				w := bufio.NewWriter(c)
				fmt.Fprintln(w, `{"QMP":{"version":{},"capabilities":[]}}`)
				w.Flush()
				dec := json.NewDecoder(c)
				for {
					var req struct {
						Execute   string         `json:"execute"`
						Arguments map[string]any `json:"arguments"`
					}
					if dec.Decode(&req) != nil {
						return
					}
					f.mu.Lock()
					f.cmds = append(f.cmds, req.Execute)
					switch req.Execute {
					case "query-named-block-nodes":
						list := []map[string]string{}
						for n := range f.nodes {
							list = append(list, map[string]string{"node-name": n})
						}
						b, _ := json.Marshal(map[string]any{"return": list})
						fmt.Fprintln(w, string(b))
					case "qom-list":
						list := []map[string]string{{"name": "type", "type": "string"}}
						for d := range f.devices {
							list = append(list, map[string]string{"name": d, "type": "child<scsi-hd>"})
						}
						b, _ := json.Marshal(map[string]any{"return": list})
						fmt.Fprintln(w, string(b))
					case "blockdev-add":
						f.nodes[req.Arguments["node-name"].(string)] = true
						fmt.Fprintln(w, `{"return":{}}`)
					case "blockdev-del":
						delete(f.nodes, req.Arguments["node-name"].(string))
						fmt.Fprintln(w, `{"return":{}}`)
					case "device_add":
						if f.failDeviceAdd {
							fmt.Fprintln(w, `{"error":{"class":"GenericError","desc":"Bus 'scsi0.0' not found"}}`)
						} else {
							f.devices[req.Arguments["id"].(string)] = true
							fmt.Fprintln(w, `{"return":{}}`)
						}
					case "device_del":
						delete(f.devices, req.Arguments["id"].(string))
						fmt.Fprintln(w, `{"event":"DEVICE_DELETED","data":{}}`)
						fmt.Fprintln(w, `{"return":{}}`)
					default:
						fmt.Fprintln(w, `{"return":{}}`)
					}
					f.mu.Unlock()
					w.Flush()
				}
			}(conn)
		}
	}()
	return f
}

func TestQMPAttacherProtocol(t *testing.T) {
	f := fakeQMP(t)
	a := QMPAttacher{SocketTemplate: strings.Replace(f.socket, "vm", "{node}", 1)}
	vol := &Volume{ID: "vol-0123456789abcdef0123", Format: FormatQCOW2, Path: "/images/v.qcow2"}
	dev, err := a.Attach(ctx, vol, "vm")
	if err != nil {
		t.Fatal(err)
	}
	if dev != "/dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_0123456789abcdef" {
		t.Fatalf("device path %s", dev)
	}
	if _, err := a.Attach(ctx, vol, "vm"); err != nil { // idempotent: no second blockdev-add
		t.Fatal(err)
	}
	if err := a.Detach(ctx, vol, "vm"); err != nil {
		t.Fatal(err)
	}
	if err := a.Detach(ctx, vol, "vm"); err != nil {
		t.Fatal(err)
	}
	want := "qmp_capabilities,query-named-block-nodes,qom-list,blockdev-add,device_add," +
		"qmp_capabilities,query-named-block-nodes,qom-list," +
		"qmp_capabilities,query-named-block-nodes,device_del,blockdev-del," +
		"qmp_capabilities,query-named-block-nodes"
	if got := f.commands(); got != want {
		t.Fatalf("QMP command sequence:\n got  %s\n want %s", got, want)
	}
}

// A block node left by an attempt that died between blockdev-add and
// device_add is not "attached": a retry must add the device. A failed
// device_add rolls back its block node.
func TestQMPAttachRepairsAndRollsBack(t *testing.T) {
	f := fakeQMP(t)
	a := QMPAttacher{SocketTemplate: strings.Replace(f.socket, "vm", "{node}", 1)}
	vol := &Volume{ID: "vol-feedfacecafebeef0000", Format: FormatRaw, Path: "/images/v.raw"}
	nodeName, deviceID, _ := qmpNames(vol)

	f.mu.Lock()
	f.nodes[nodeName] = true // half-attached leftover
	f.mu.Unlock()
	if _, err := a.Attach(ctx, vol, "vm"); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	repaired := f.devices[deviceID]
	f.mu.Unlock()
	if !repaired || strings.Contains(f.commands(), "blockdev-add") {
		t.Fatalf("half-attached volume not repaired (device present=%v, commands %s)", repaired, f.commands())
	}

	if err := a.Detach(ctx, vol, "vm"); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	f.failDeviceAdd = true
	f.mu.Unlock()
	if _, err := a.Attach(ctx, vol, "vm"); err == nil || !strings.Contains(err.Error(), "scsi0.0") {
		t.Fatalf("device_add failure not reported: %v", err)
	}
	f.mu.Lock()
	leftover := f.nodes[nodeName]
	f.mu.Unlock()
	if leftover {
		t.Fatal("failed attach left its block node behind")
	}
}

// A node whose VM is gone: detach succeeds (CSI requires it, or the volume
// could never move), attach reports NotFound.
func TestQMPVanishedVM(t *testing.T) {
	dir := t.TempDir()
	vol := &Volume{ID: "vol-0123456789abcdef0123", Format: FormatRaw, Path: "/images/v.raw"}
	missing := QMPAttacher{SocketTemplate: filepath.Join(dir, "{node}.sock"), Timeout: time.Second}
	if err := missing.Detach(ctx, vol, "gone"); err != nil {
		t.Fatalf("detach from a VM without a socket: %v", err)
	}
	if _, err := missing.Attach(ctx, vol, "gone"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("attach to a missing VM: %v, want ErrNotFound", err)
	}
	// Stale socket file, nobody listening (QEMU killed): ECONNREFUSED.
	stale := filepath.Join(dir, "dead.sock")
	lis, err := net.ListenUnix("unix", &net.UnixAddr{Name: stale, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	lis.SetUnlinkOnClose(false)
	lis.Close()
	if _, err := os.Stat(stale); err != nil {
		t.Fatalf("stale socket not left behind: %v", err)
	}
	if err := missing.Detach(ctx, vol, "dead"); err != nil {
		t.Fatalf("detach from a dead VM: %v", err)
	}
}

// failingTool wraps an ImageTool and fails selected operations after doing
// part of the work, like a qemu-img killed mid-way or a full disk.
type failingTool struct {
	ImageTool
	failResize, failConvert bool
}

func (f failingTool) Resize(ctx context.Context, path, format string, size int64) error {
	if f.failResize {
		return errors.New("injected resize failure")
	}
	return f.ImageTool.Resize(ctx, path, format, size)
}

func (f failingTool) Convert(ctx context.Context, src, srcFormat, dst, dstFormat string, compress bool) error {
	if f.failConvert {
		_ = os.WriteFile(dst, []byte("partial"), 0o600) // half-written output
		return errors.New("injected convert failure")
	}
	return f.ImageTool.Convert(ctx, src, srcFormat, dst, dstFormat, compress)
}

func images(t *testing.T, dir string) []string {
	t.Helper()
	var out []string
	for _, sub := range []string{"volumes", "snapshots"} {
		entries, _ := os.ReadDir(filepath.Join(dir, sub))
		for _, e := range entries {
			if !strings.HasSuffix(e.Name(), ".json") {
				out = append(out, sub+"/"+e.Name())
			}
		}
	}
	return out
}

// Failed provisioning leaves no partial images (nothing would ever delete
// them if the CO gives up), and a retry then succeeds.
func TestFailedImageOperationsLeaveNoPartialFiles(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "images")
	tool := &failingTool{ImageTool: RawFileTool{}}
	store, err := NewStore(dir, tool, LocalAttacher{})
	if err != nil {
		t.Fatal(err)
	}
	src, err := store.CreateVolume(ctx, "src", 1<<20, FormatRaw, "")
	if err != nil {
		t.Fatal(err)
	}
	tool.failConvert = true
	if _, err := store.CreateSnapshot(ctx, "snap", src.ID); err == nil {
		t.Fatal("snapshot succeeded despite convert failure")
	}
	if got := images(t, dir); len(got) != 1 {
		t.Fatalf("partial snapshot left behind: %v", got)
	}
	tool.failConvert = false
	snap, err := store.CreateSnapshot(ctx, "snap", src.ID)
	if err != nil {
		t.Fatal(err)
	}
	tool.failResize = true // restore into a bigger volume: convert works, resize fails
	if _, err := store.CreateVolume(ctx, "restored", 4<<20, FormatRaw, snap.ID); err == nil {
		t.Fatal("restore succeeded despite resize failure")
	}
	if got := images(t, dir); len(got) != 2 {
		t.Fatalf("partial restored volume left behind: %v", got)
	}
	tool.failResize = false
	if v, err := store.CreateVolume(ctx, "restored", 4<<20, FormatRaw, snap.ID); err != nil || v.Capacity != 4<<20 {
		t.Fatalf("retry after failure: %v %v", v, err)
	}
}

// recordingAttacher records attach/detach calls.
type recordingAttacher struct {
	LocalAttacher
	detached []string
}

func (r *recordingAttacher) Detach(_ context.Context, vol *Volume, node string) error {
	r.detached = append(r.detached, vol.ID+"@"+node)
	return nil
}

// An attachment that cannot be recorded is rolled back, so no device is left
// that the driver does not know about.
func TestAttachRollsBackWhenMetadataCannotBeWritten(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	dir := filepath.Join(t.TempDir(), "images")
	att := &recordingAttacher{}
	store, err := NewStore(dir, RawFileTool{}, att)
	if err != nil {
		t.Fatal(err)
	}
	vol, err := store.CreateVolume(ctx, "v", 1<<20, FormatRaw, "")
	if err != nil {
		t.Fatal(err)
	}
	volumes := filepath.Join(dir, "volumes")
	if err := os.Chmod(volumes, 0o500); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(volumes, 0o750) //nolint:errcheck
	if _, err := store.Attach(ctx, vol.ID, "node-1", false); err == nil {
		t.Fatal("attach succeeded without recording it")
	}
	if len(att.detached) != 1 || att.detached[0] != vol.ID+"@node-1" {
		t.Fatalf("attachment not rolled back: %v", att.detached)
	}
	os.Chmod(volumes, 0o750) //nolint:errcheck
	if v, _ := store.GetVolume(vol.ID); len(v.Attached) != 0 {
		t.Fatalf("metadata claims an attachment: %v", v.Attached)
	}
}

func TestQMPAttachRealQEMU(t *testing.T) {
	qemu, err := exec.LookPath("qemu-system-x86_64")
	if err != nil || !QemuImgAvailable() {
		t.Skip("qemu-system-x86_64 / qemu-img not installed")
	}
	dir, _ := os.MkdirTemp("", "vgvm")
	defer os.RemoveAll(dir)
	socket := filepath.Join(dir, "node-1.qmp")
	vm := exec.Command(qemu, "-machine", "pc,accel=tcg", "-S", "-nodefaults", "-display", "none",
		"-device", "virtio-scsi-pci,id=scsi0", "-qmp", "unix:"+socket+",server=on,wait=off")
	var vmErr bytes.Buffer
	vm.Stderr = &vmErr
	if err := vm.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = vm.Process.Kill(); _ = vm.Wait() }()
	for i := 0; i < 100; i++ {
		if _, err := os.Stat(socket); err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	store, err := NewStore(filepath.Join(dir, "images"), QemuImg{}, QMPAttacher{SocketTemplate: filepath.Join(dir, "{node}.qmp")})
	if err != nil {
		t.Fatal(err)
	}
	vol, err := store.CreateVolume(ctx, "vm-disk", 16<<20, FormatQCOW2, "")
	if err != nil {
		t.Fatal(err)
	}
	dev, err := store.Attach(ctx, vol.ID, "node-1", false)
	if err != nil {
		t.Fatalf("attach: %v (qemu: %s)", err, vmErr.String())
	}

	// QEMU's QMP socket serves one client at a time, so inspect the VM over a
	// short-lived connection and close it before the driver detaches (the
	// attacher needs exclusive access to the monitor; see attacher.go).
	query := func(cmd string, args any) []byte {
		c, err := DialQMP(ctx, socket)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		raw, err := c.Execute(cmd, args)
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	nodeName, deviceID, _ := qmpNames(vol)
	if raw := query("query-block", nil); !bytes.Contains(raw, []byte(vol.Path)) || !bytes.Contains(raw, []byte(deviceID)) {
		t.Fatalf("VM does not see the disk after attach: %s", raw)
	}
	t.Logf("attached %s as %s (guest %s)", filepath.Base(vol.Path), deviceID, dev)

	if err := store.Detach(ctx, vol.ID, "node-1"); err != nil {
		t.Fatalf("detach: %v", err)
	}
	if raw := query("query-named-block-nodes", map[string]any{"flat": true}); bytes.Contains(raw, []byte(nodeName)) {
		t.Fatalf("block node still present after detach: %s", raw)
	}

	// Half-attached leftover (blockdev-add done, device_add never ran, as if
	// the driver died in between): the next attach must plug the device in.
	query("blockdev-add", map[string]any{"node-name": nodeName, "driver": vol.Format,
		"file": map[string]any{"driver": "file", "filename": vol.Path}})
	if raw := query("query-block", nil); bytes.Contains(raw, []byte(deviceID)) {
		t.Fatalf("setup: device present before repair: %s", raw)
	}
	if _, err := store.Attach(ctx, vol.ID, "node-1", false); err != nil {
		t.Fatalf("attach over a half-attached leftover: %v", err)
	}
	if raw := query("query-block", nil); !bytes.Contains(raw, []byte(deviceID)) {
		t.Fatalf("half-attached volume not repaired: %s", raw)
	}
	if err := store.Detach(ctx, vol.ID, "node-1"); err != nil {
		t.Fatalf("detach after repair: %v", err)
	}

	// The VM dies with the volume attached: detach must still succeed.
	if _, err := store.Attach(ctx, vol.ID, "node-1", false); err != nil {
		t.Fatal(err)
	}
	_ = vm.Process.Kill()
	_ = vm.Wait()
	if err := store.Detach(ctx, vol.ID, "node-1"); err != nil {
		t.Fatalf("detach from a dead VM: %v", err)
	}
	if err := store.DeleteVolume(vol.ID); err != nil {
		t.Fatal(err)
	}
}
