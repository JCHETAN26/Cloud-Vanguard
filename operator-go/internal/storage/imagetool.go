// Package storage implements the QEMU/KVM block-storage backend behind the
// CloudVanguard CSI driver: disk images (qemu-img), point-in-time snapshots,
// and attaching images to virtual machines over QMP.
package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
)

// Image formats supported by the backend.
const (
	FormatRaw   = "raw"   // plain bytes: usable directly by the host (block-mode volumes, loop devices)
	FormatQCOW2 = "qcow2" // thin, compressible: VM attachment (QMP) and snapshot/backup images
)

// ImageTool manipulates disk images. QemuImg is the production
// implementation; RawFileTool is a dependency-free fallback (raw only) used
// when qemu-img is not installed.
type ImageTool interface {
	Name() string
	Create(ctx context.Context, path, format string, sizeBytes int64) error
	Resize(ctx context.Context, path, format string, sizeBytes int64) error
	// Convert copies src into a new image dst of dstFormat (snapshot, restore,
	// clone). compress applies to qcow2 destinations.
	Convert(ctx context.Context, src, srcFormat, dst, dstFormat string, compress bool) error
	Info(ctx context.Context, path string) (ImageInfo, error)
}

// ImageInfo is the subset of `qemu-img info` the driver needs.
type ImageInfo struct {
	Format      string `json:"format"`
	VirtualSize int64  `json:"virtual-size"`
	ActualSize  int64  `json:"actual-size"`
}

// QemuImg drives the qemu-img binary.
type QemuImg struct {
	Binary string // default "qemu-img"
}

func (q QemuImg) bin() string {
	if q.Binary != "" {
		return q.Binary
	}
	return "qemu-img"
}

func (q QemuImg) Name() string { return "qemu-img" }

func (q QemuImg) run(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, q.bin(), args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("qemu-img %v: %w: %s", args, err, bytes.TrimSpace(stderr.Bytes()))
	}
	return out, nil
}

func (q QemuImg) Create(ctx context.Context, path, format string, size int64) error {
	_, err := q.run(ctx, "create", "-q", "-f", format, path, strconv.FormatInt(size, 10))
	return err
}

func (q QemuImg) Resize(ctx context.Context, path, format string, size int64) error {
	_, err := q.run(ctx, "resize", "-q", "-f", format, path, strconv.FormatInt(size, 10))
	return err
}

func (q QemuImg) Convert(ctx context.Context, src, srcFormat, dst, dstFormat string, compress bool) error {
	args := []string{"convert", "-q", "-f", srcFormat, "-O", dstFormat}
	if compress && dstFormat == FormatQCOW2 {
		args = append(args, "-c")
	}
	_, err := q.run(ctx, append(args, src, dst)...)
	return err
}

func (q QemuImg) Info(ctx context.Context, path string) (ImageInfo, error) {
	out, err := q.run(ctx, "info", "--output=json", path)
	if err != nil {
		return ImageInfo{}, err
	}
	var info ImageInfo
	return info, json.Unmarshal(out, &info)
}

// QemuImgAvailable reports whether qemu-img can be executed.
func QemuImgAvailable() bool {
	_, err := exec.LookPath("qemu-img")
	return err == nil
}

// RawFileTool is the fallback when qemu-img is absent: raw images as sparse
// files, snapshots as byte copies. No qcow2.
type RawFileTool struct{}

var errRawOnly = errors.New("qemu-img not available: only raw images are supported")

func (RawFileTool) Name() string { return "raw-file" }

func (RawFileTool) Create(_ context.Context, path, format string, size int64) error {
	if format != FormatRaw {
		return errRawOnly
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Truncate(size)
}

func (RawFileTool) Resize(_ context.Context, path, format string, size int64) error {
	if format != FormatRaw {
		return errRawOnly
	}
	return os.Truncate(path, size)
}

func (RawFileTool) Convert(_ context.Context, src, srcFormat, dst, dstFormat string, _ bool) error {
	if srcFormat != FormatRaw || dstFormat != FormatRaw {
		return errRawOnly
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func (RawFileTool) Info(_ context.Context, path string) (ImageInfo, error) {
	st, err := os.Stat(path)
	if err != nil {
		return ImageInfo{}, err
	}
	return ImageInfo{Format: FormatRaw, VirtualSize: st.Size(), ActualSize: st.Size()}, nil
}

// DefaultImageTool picks qemu-img when installed.
func DefaultImageTool() ImageTool {
	if QemuImgAvailable() {
		return QemuImg{}
	}
	return RawFileTool{}
}
