package csi

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Mounter performs the node-side data-path operations.
type Mounter interface {
	Name() string
	// Stage prepares the device under the (CO-created) staging directory.
	Stage(device, stagingDir string, block bool, fsType string) error
	Unstage(stagingDir string) error
	// Publish exposes the staged volume at targetPath (a file for block
	// volumes, a directory for filesystem volumes).
	Publish(stagingDir, targetPath string, block, readonly bool) error
	Unpublish(targetPath string) error
}

const (
	stagedDevice = "device"     // <staging>/device -> real device (block)
	stagedMarker = ".vg-staged" // filesystem staging marker (LinkMounter)
	stagedLoop   = ".vg-loop"   // loop device HostMounter attached for an image file (detached on unstage)
)

// LinkMounter needs no privileges. Block volumes are real: the staged and
// published paths are symlinks to the attached device (or raw image), so the
// workload reads and writes the volume's actual bytes. Filesystem volumes are
// simulated (a marker file instead of mkfs + mount) and exist so the full CSI
// surface can be exercised by csi-sanity in unprivileged environments.
// Production nodes use HostMounter.
type LinkMounter struct{}

func (LinkMounter) Name() string { return "link" }

func replaceSymlink(target, link string) error {
	if err := os.MkdirAll(filepath.Dir(link), 0o750); err != nil {
		return err
	}
	if cur, err := os.Readlink(link); err == nil && cur == target {
		return nil
	}
	if err := os.RemoveAll(link); err != nil {
		return err
	}
	return os.Symlink(target, link)
}

func (LinkMounter) Stage(device, staging string, block bool, _ string) error {
	if err := os.MkdirAll(staging, 0o750); err != nil {
		return err
	}
	if block {
		if _, err := os.Stat(device); err != nil {
			return fmt.Errorf("device %s: %w", device, err)
		}
		return replaceSymlink(device, filepath.Join(staging, stagedDevice))
	}
	return os.WriteFile(filepath.Join(staging, stagedMarker), []byte(device), 0o600)
}

func (LinkMounter) Unstage(staging string) error {
	for _, f := range []string{stagedDevice, stagedMarker} {
		if err := os.Remove(filepath.Join(staging, f)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func (LinkMounter) Publish(staging, target string, block, _ bool) error {
	if block {
		dev, err := os.Readlink(filepath.Join(staging, stagedDevice))
		if err != nil {
			return fmt.Errorf("volume is not staged at %s: %w", staging, err)
		}
		return replaceSymlink(dev, target)
	}
	if _, err := os.Stat(filepath.Join(staging, stagedMarker)); err != nil {
		return fmt.Errorf("volume is not staged at %s: %w", staging, err)
	}
	if err := os.MkdirAll(target, 0o750); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(target, stagedMarker), []byte(staging), 0o600)
}

func (LinkMounter) Unpublish(target string) error {
	info, err := os.Lstat(target)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return os.Remove(target)
	}
	if err := os.Remove(filepath.Join(target, stagedMarker)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Remove(target)
}

// HostMounter is the production node data path (Linux, CAP_SYS_ADMIN,
// running in the privileged node DaemonSet): formats a blank device once,
// mounts it at the staging dir, and bind-mounts into pods; block volumes are
// bind-mounted device nodes. It shells out to mount/umount/blkid/mkfs so it
// behaves exactly like the host tools.
type HostMounter struct{}

func (HostMounter) Name() string { return "host" }

func run(name string, args ...string) error {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return nil
}

func isMountPoint(path string) bool {
	return exec.Command("mountpoint", "-q", path).Run() == nil
}

// loopFor returns a loop device backed by image, attaching one if none is
// (idempotent across NodeStage retries and plugin restarts).
func loopFor(image string) (string, error) {
	out, err := exec.Command("losetup", "--noheadings", "--output", "NAME", "--associated", image).Output()
	if err != nil {
		return "", fmt.Errorf("losetup --associated %s: %w", image, err)
	}
	if fields := strings.Fields(string(out)); len(fields) > 0 {
		return fields[0], nil
	}
	out, err = exec.Command("losetup", "--find", "--show", image).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("losetup --find --show %s: %w: %s", image, err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

// resolveDevice turns what ControllerPublish handed over into a block device
// node. The qmp attacher returns a disk hot-plugged into the node VM (already a
// device); the local attacher returns the volume's image *file*, which a pod
// cannot use as a raw block device (the runtime rejects it: "not a device
// node"), so it is attached to a loop device and the loop device is recorded
// for NodeUnstage to detach.
func resolveDevice(device, staging string) (string, error) {
	info, err := os.Stat(device)
	if err != nil {
		return "", fmt.Errorf("device %s: %w", device, err)
	}
	if !info.Mode().IsRegular() {
		return device, nil
	}
	loop, err := loopFor(device)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(staging, stagedLoop), []byte(loop), 0o600); err != nil {
		return "", err
	}
	return loop, nil
}

func (HostMounter) Stage(device, staging string, block bool, fsType string) error {
	if err := os.MkdirAll(staging, 0o750); err != nil {
		return err
	}
	if block {
		// Block volumes are published straight from the device node.
		dev, err := resolveDevice(device, staging)
		if err != nil {
			return err
		}
		return replaceSymlink(dev, filepath.Join(staging, stagedDevice))
	}
	if isMountPoint(staging) {
		return nil
	}
	if fsType == "" {
		fsType = "ext4"
	}
	dev, err := resolveDevice(device, staging)
	if err != nil {
		return err
	}
	if err := exec.Command("blkid", dev).Run(); err != nil { // no filesystem yet
		if err := run("mkfs."+fsType, "-q", dev); err != nil {
			return err
		}
	}
	return run("mount", "-t", fsType, dev, staging)
}

func (HostMounter) Unstage(staging string) error {
	if isMountPoint(staging) {
		if err := run("umount", staging); err != nil {
			return err
		}
	}
	if b, err := os.ReadFile(filepath.Join(staging, stagedLoop)); err == nil {
		if loop := strings.TrimSpace(string(b)); loop != "" {
			if out, err := exec.Command("losetup", "--detach", loop).CombinedOutput(); err != nil &&
				!strings.Contains(string(out), "No such device") {
				return fmt.Errorf("losetup --detach %s: %w: %s", loop, err, strings.TrimSpace(string(out)))
			}
		}
		if err := os.Remove(filepath.Join(staging, stagedLoop)); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return LinkMounter{}.Unstage(staging)
}

func (HostMounter) Publish(staging, target string, block, readonly bool) error {
	if isMountPoint(target) {
		return nil
	}
	src := staging
	if block {
		dev, err := os.Readlink(filepath.Join(staging, stagedDevice))
		if err != nil {
			return err
		}
		src = dev
		if err := os.MkdirAll(filepath.Dir(target), 0o750); err != nil {
			return err
		}
		if f, err := os.OpenFile(target, os.O_CREATE, 0o600); err == nil {
			f.Close()
		}
	} else if err := os.MkdirAll(target, 0o750); err != nil {
		return err
	}
	if err := run("mount", "--bind", src, target); err != nil {
		return err
	}
	if readonly {
		return run("mount", "-o", "remount,bind,ro", target)
	}
	return nil
}

func (HostMounter) Unpublish(target string) error {
	if isMountPoint(target) {
		if err := run("umount", target); err != nil {
			return err
		}
	}
	return os.RemoveAll(target)
}
