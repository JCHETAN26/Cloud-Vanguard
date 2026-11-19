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

// fakeQMP is a minimal QMP server recording commands, tracking block nodes.
func fakeQMP(t *testing.T) (socket string, commands *[]string) {
	t.Helper()
	dir, _ := os.MkdirTemp("", "qmp")
	t.Cleanup(func() { os.RemoveAll(dir) })
	socket = filepath.Join(dir, "vm.sock")
	lis, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { lis.Close() })
	var cmds []string
	nodes := map[string]bool{}
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
					cmds = append(cmds, req.Execute)
					switch req.Execute {
					case "query-named-block-nodes":
						var list []map[string]string
						for n := range nodes {
							list = append(list, map[string]string{"node-name": n})
						}
						b, _ := json.Marshal(map[string]any{"return": list})
						fmt.Fprintln(w, string(b))
					case "blockdev-add":
						nodes[req.Arguments["node-name"].(string)] = true
						fmt.Fprintln(w, `{"return":{}}`)
					case "blockdev-del":
						delete(nodes, req.Arguments["node-name"].(string))
						fmt.Fprintln(w, `{"return":{}}`)
					case "device_del":
						fmt.Fprintln(w, `{"event":"DEVICE_DELETED","data":{}}`)
						fmt.Fprintln(w, `{"return":{}}`)
					default:
						fmt.Fprintln(w, `{"return":{}}`)
					}
					w.Flush()
				}
			}(conn)
		}
	}()
	return socket, &cmds
}

func TestQMPAttacherProtocol(t *testing.T) {
	socket, cmds := fakeQMP(t)
	a := QMPAttacher{SocketTemplate: strings.Replace(socket, "vm", "{node}", 1)}
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
	got := strings.Join(*cmds, ",")
	want := "qmp_capabilities,query-named-block-nodes,blockdev-add,device_add," +
		"qmp_capabilities,query-named-block-nodes," +
		"qmp_capabilities,query-named-block-nodes,device_del,blockdev-del," +
		"qmp_capabilities,query-named-block-nodes"
	if got != want {
		t.Fatalf("QMP command sequence:\n got  %s\n want %s", got, want)
	}
}

// Real hypervisor: hot-plug a qcow2 volume into a running QEMU VM over QMP
// and unplug it again. Uses TCG (software CPU), so /dev/kvm is not needed.
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

	c, err := DialQMP(ctx, socket)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	raw, err := c.Execute("query-block", nil)
	if err != nil {
		t.Fatal(err)
	}
	nodeName, deviceID, _ := qmpNames(vol)
	if !bytes.Contains(raw, []byte(vol.Path)) || !bytes.Contains(raw, []byte(deviceID)) {
		t.Fatalf("VM does not see the disk after attach: %s", raw)
	}
	t.Logf("attached %s as %s (guest %s)", filepath.Base(vol.Path), deviceID, dev)

	if err := store.Detach(ctx, vol.ID, "node-1"); err != nil {
		t.Fatalf("detach: %v", err)
	}
	raw, _ = c.Execute("query-named-block-nodes", map[string]any{"flat": true})
	if bytes.Contains(raw, []byte(nodeName)) {
		t.Fatalf("block node still present after detach: %s", raw)
	}
	if err := store.DeleteVolume(vol.ID); err != nil {
		t.Fatal(err)
	}
}
