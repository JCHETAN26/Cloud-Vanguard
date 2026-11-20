package csi

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// HostMounter with --attacher=local: the "device" is a raw image file. Found by
// the first kind run: publishing the file itself made containerd refuse the pod
// ("not a device node"). Needs root and loop devices; run with
//
//	sudo -E go test ./internal/csi -run HostMounter
func requireLoop(t *testing.T) {
	t.Helper()
	if os.Geteuid() != 0 {
		t.Skip("needs root (loop devices, mount)")
	}
	out, err := exec.Command("losetup", "--find").CombinedOutput()
	if err != nil {
		t.Skipf("no usable loop devices: %v %s", err, out)
	}
	// In containers without a live devtmpfs (Codespaces, kind nodes) the next free loop device may have
	// no device node; losetup then fails with ENOENT. That is the environment, not the driver.
	if next := strings.TrimSpace(string(out)); next != "" {
		if _, err := os.Stat(next); err != nil {
			t.Skipf("next free loop device %s has no device node (container without devtmpfs; "+
				"create it with: mknod %s b 7 <minor>)", next, next)
		}
	}
}

func loopsBacking(t *testing.T, image string) string {
	t.Helper()
	out, err := exec.Command("losetup", "--noheadings", "--output", "NAME", "--associated", image).Output()
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(out))
}

func newImage(t *testing.T, dir string) string {
	t.Helper()
	img := filepath.Join(dir, "vol.img")
	if err := os.WriteFile(img, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(img, 16<<20); err != nil {
		t.Fatal(err)
	}
	return img
}

func TestHostMounterBlockVolumeFromImageFile(t *testing.T) {
	requireLoop(t)
	dir := t.TempDir()
	img := newImage(t, dir)
	staging, target := filepath.Join(dir, "staging"), filepath.Join(dir, "pod", "dev")
	m := HostMounter{}

	if err := m.Stage(img, staging, true, ""); err != nil {
		t.Fatal(err)
	}
	loop, err := os.Readlink(filepath.Join(staging, stagedDevice))
	if err != nil || !strings.HasPrefix(loop, "/dev/loop") {
		t.Fatalf("staged device %q %v", loop, err)
	}
	if err := m.Stage(img, staging, true, ""); err != nil { // NodeStage retry: same loop device
		t.Fatal(err)
	}
	if again, _ := os.Readlink(filepath.Join(staging, stagedDevice)); again != loop || loopsBacking(t, img) != loop {
		t.Fatalf("retry attached another loop device: %q vs %q (%q)", again, loop, loopsBacking(t, img))
	}

	if err := m.Publish(staging, target, true, false); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(target)
	if err != nil || info.Mode()&os.ModeDevice == 0 {
		t.Fatalf("published target is not a device node: %v %v", info.Mode(), err)
	}
	// Bytes written through the pod's device land in the volume image.
	payload := []byte("warm-cache-slot-A")
	f, err := os.OpenFile(target, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt(payload, 4096); err != nil {
		t.Fatal(err)
	}
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	if err := m.Unpublish(target); err != nil {
		t.Fatal(err)
	}
	if err := m.Unstage(staging); err != nil {
		t.Fatal(err)
	}
	if l := loopsBacking(t, img); l != "" {
		t.Fatalf("loop device %s still attached after unstage", l)
	}
	if err := m.Unstage(staging); err != nil { // NodeUnstage retry
		t.Fatal(err)
	}
	raw, err := os.ReadFile(img)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw[4096:4096+len(payload)], payload) {
		t.Fatal("data written through the device is not in the image")
	}
}

func TestHostMounterFilesystemVolumeFromImageFile(t *testing.T) {
	requireLoop(t)
	if _, err := exec.LookPath("mkfs.ext4"); err != nil {
		t.Skip("mkfs.ext4 not installed")
	}
	dir := t.TempDir()
	img := newImage(t, dir)
	staging, target := filepath.Join(dir, "staging"), filepath.Join(dir, "pod", "data")
	m := HostMounter{}
	if err := m.Stage(img, staging, false, "ext4"); err != nil {
		t.Fatal(err)
	}
	if err := m.Publish(staging, target, false, false); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(target, "hello"), []byte("hi"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := m.Unpublish(target); err != nil {
		t.Fatal(err)
	}
	if err := m.Unstage(staging); err != nil {
		t.Fatal(err)
	}
	if l := loopsBacking(t, img); l != "" {
		t.Fatalf("loop device %s still attached after unstage", l)
	}
	// Re-stage: the existing filesystem is mounted, not reformatted.
	if err := m.Stage(img, staging, false, "ext4"); err != nil {
		t.Fatal(err)
	}
	defer m.Unstage(staging)
	if b, err := os.ReadFile(filepath.Join(staging, "hello")); err != nil || string(b) != "hi" {
		t.Fatalf("data lost across unstage/stage: %q %v", b, err)
	}
}
