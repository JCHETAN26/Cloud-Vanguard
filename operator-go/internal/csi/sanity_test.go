package csi

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kubernetes-csi/csi-test/v5/pkg/sanity"

	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/storage"
)

// startDriver runs controller + node services on a unix socket.
func startDriver(t *testing.T, images storage.ImageTool) (endpoint string, store *storage.Store) {
	t.Helper()
	// Unix socket paths are limited to ~104 bytes; keep them short.
	sockDir, err := os.MkdirTemp("", "vgcsi")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(sockDir) })
	store, err = storage.NewStore(filepath.Join(t.TempDir(), "images"), images, storage.LocalAttacher{})
	if err != nil {
		t.Fatal(err)
	}
	d := &Driver{
		NodeID:     "node-1",
		Store:      store,
		NodeExists: func(id string) bool { return id == "node-1" },
		Mounter:    LinkMounter{},
	}
	sock := filepath.Join(sockDir, "csi.sock")
	lis, err := Listen("unix://" + sock)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = d.Serve(ctx, lis) }()
	return "unix://" + sock, store
}

func runSanity(t *testing.T, images storage.ImageTool, accessType string) {
	endpoint, _ := startDriver(t, images)
	cfg := sanity.NewTestConfig()
	cfg.Address = endpoint
	base := t.TempDir()
	cfg.TargetPath = filepath.Join(base, "target")
	cfg.StagingPath = filepath.Join(base, "staging")
	cfg.TestVolumeSize = 16 << 20
	cfg.TestVolumeExpandSize = 32 << 20
	cfg.TestVolumeAccessType = accessType
	sanity.Test(t, cfg)
}

// The upstream CSI conformance suite (kubernetes-csi/csi-test) against the
// driver: identity, controller (volumes, publish, snapshots, list
// pagination, expansion) and node (stage/publish) behaviour, idempotency and
// error codes. Ginkgo suites cannot run twice per process, so block volumes
// (vanguard's production mode) run here and filesystem volumes run in a
// re-executed copy of this test binary.
func TestCSISanity(t *testing.T) {
	access := os.Getenv("VG_SANITY_ACCESS")
	if access == "" {
		access = "block"
	}
	t.Logf("csi-sanity: %s volumes, image tool %s", access, storage.DefaultImageTool().Name())
	runSanity(t, storage.DefaultImageTool(), access)
}

func TestCSISanityFilesystemVolumes(t *testing.T) {
	if os.Getenv("VG_SANITY_ACCESS") != "" {
		t.Skip("running inside the re-executed sanity process")
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestCSISanity$", "-test.count=1")
	cmd.Env = append(os.Environ(), "VG_SANITY_ACCESS=mount")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("filesystem-volume sanity run failed: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "SUCCESS!") {
		t.Fatalf("filesystem-volume sanity run did not report success:\n%s", out)
	}
}
