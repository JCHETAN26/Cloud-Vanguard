package storage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Errors mapped to CSI status codes by the driver.
var (
	ErrNotFound       = errors.New("not found")
	ErrAlreadyExists  = errors.New("already exists with different parameters")
	ErrInUse          = errors.New("in use")
	ErrInvalidRequest = errors.New("invalid request")
)

// Volume is a provisioned disk image.
type Volume struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	Format     string            `json:"format"`
	Capacity   int64             `json:"capacityBytes"`
	Path       string            `json:"path"`
	SourceSnap string            `json:"sourceSnapshot,omitempty"`
	Attached   map[string]string `json:"attached,omitempty"` // node -> device path
	Created    time.Time         `json:"created"`
}

// Snapshot is a point-in-time copy of a volume (a compressed qcow2 image
// when qemu-img is available).
type Snapshot struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	SourceID string    `json:"sourceVolumeId"`
	Format   string    `json:"format"`
	Size     int64     `json:"sizeBytes"`
	Path     string    `json:"path"`
	Created  time.Time `json:"created"`
}

// Store is the controller-side backend: images under Dir/volumes, snapshots
// under Dir/snapshots, metadata as JSON next to them (so the driver is
// restart-safe and every operation is idempotent).
type Store struct {
	Dir      string
	Images   ImageTool
	Attacher Attacher
	mu       sync.Mutex
}

// NewStore creates the directory layout.
func NewStore(dir string, images ImageTool, attacher Attacher) (*Store, error) {
	for _, d := range []string{"volumes", "snapshots"} {
		if err := os.MkdirAll(filepath.Join(dir, d), 0o750); err != nil {
			return nil, err
		}
	}
	return &Store{Dir: dir, Images: images, Attacher: attacher}, nil
}

func stableID(prefix, name string) string {
	sum := sha256.Sum256([]byte(name))
	return prefix + hex.EncodeToString(sum[:10])
}

func (s *Store) volMeta(id string) string  { return filepath.Join(s.Dir, "volumes", id+".json") }
func (s *Store) snapMeta(id string) string { return filepath.Join(s.Dir, "snapshots", id+".json") }

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// GetVolume returns a volume by ID.
func (s *Store) GetVolume(id string) (*Volume, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.getVolumeLocked(id)
}

func (s *Store) getVolumeLocked(id string) (*Volume, error) {
	v := &Volume{}
	if err := readJSON(s.volMeta(id), v); err != nil {
		return nil, err
	}
	return v, nil
}

// CreateVolume provisions (idempotently, keyed by name) an image of at least
// sizeBytes, optionally restored from a snapshot.
func (s *Store) CreateVolume(ctx context.Context, name string, sizeBytes int64, format, fromSnapshot string) (*Volume, error) {
	if format == "" {
		format = FormatRaw
	}
	if format != FormatRaw && format != FormatQCOW2 {
		return nil, fmt.Errorf("%w: unsupported format %q", ErrInvalidRequest, format)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id := stableID("vol-", name)
	if existing, err := s.getVolumeLocked(id); err == nil {
		if existing.Capacity < sizeBytes || existing.Format != format || existing.SourceSnap != fromSnapshot {
			return nil, ErrAlreadyExists
		}
		return existing, nil
	}
	vol := &Volume{ID: id, Name: name, Format: format, Capacity: sizeBytes, SourceSnap: fromSnapshot,
		Path: filepath.Join(s.Dir, "volumes", id+"."+format), Created: time.Now().UTC()}
	_ = os.Remove(vol.Path) // leftover from a crash between image and metadata creation
	if fromSnapshot != "" {
		snap := &Snapshot{}
		if err := readJSON(s.snapMeta(fromSnapshot), snap); err != nil {
			return nil, fmt.Errorf("source snapshot %s: %w", fromSnapshot, err)
		}
		if snap.Size > sizeBytes {
			vol.Capacity = snap.Size
		}
		if err := s.Images.Convert(ctx, snap.Path, snap.Format, vol.Path, format, false); err != nil {
			return nil, err
		}
		if vol.Capacity > snap.Size {
			if err := s.Images.Resize(ctx, vol.Path, format, vol.Capacity); err != nil {
				return nil, err
			}
		}
	} else if err := s.Images.Create(ctx, vol.Path, format, sizeBytes); err != nil {
		return nil, err
	}
	if err := writeJSON(s.volMeta(id), vol); err != nil {
		return nil, err
	}
	return vol, nil
}

// DeleteVolume removes an unattached volume (idempotent).
func (s *Store) DeleteVolume(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	vol, err := s.getVolumeLocked(id)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	if len(vol.Attached) > 0 {
		return fmt.Errorf("%w: volume %s is attached to %v", ErrInUse, id, keys(vol.Attached))
	}
	if err := os.Remove(vol.Path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Remove(s.volMeta(id))
}

// ExpandVolume grows a volume (never shrinks).
func (s *Store) ExpandVolume(ctx context.Context, id string, sizeBytes int64) (*Volume, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	vol, err := s.getVolumeLocked(id)
	if err != nil {
		return nil, err
	}
	if sizeBytes <= vol.Capacity {
		return vol, nil
	}
	if err := s.Images.Resize(ctx, vol.Path, vol.Format, sizeBytes); err != nil {
		return nil, err
	}
	vol.Capacity = sizeBytes
	return vol, writeJSON(s.volMeta(id), vol)
}

// Attach publishes a volume to a node via the configured Attacher.
// Single-node-writer: attaching to a second node while attached elsewhere fails.
func (s *Store) Attach(ctx context.Context, id, node string, multiNode bool) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	vol, err := s.getVolumeLocked(id)
	if err != nil {
		return "", err
	}
	if dev, ok := vol.Attached[node]; ok {
		return dev, nil
	}
	if len(vol.Attached) > 0 && !multiNode {
		return "", fmt.Errorf("%w: volume %s is attached to %v", ErrInUse, id, keys(vol.Attached))
	}
	dev, err := s.Attacher.Attach(ctx, vol, node)
	if err != nil {
		return "", err
	}
	if vol.Attached == nil {
		vol.Attached = map[string]string{}
	}
	vol.Attached[node] = dev
	return dev, writeJSON(s.volMeta(id), vol)
}

// Detach unpublishes a volume from a node (idempotent).
func (s *Store) Detach(ctx context.Context, id, node string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	vol, err := s.getVolumeLocked(id)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	if _, ok := vol.Attached[node]; !ok {
		return nil
	}
	if err := s.Attacher.Detach(ctx, vol, node); err != nil {
		return err
	}
	delete(vol.Attached, node)
	return writeJSON(s.volMeta(id), vol)
}

// ListVolumes returns all volumes sorted by ID.
func (s *Store) ListVolumes() ([]*Volume, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return list[Volume](filepath.Join(s.Dir, "volumes"))
}

// CreateSnapshot copies a volume into a new snapshot image (idempotent by name).
// The copy is crash-consistent: take it while the workload is quiesced or rely
// on the application's own consistency (vanguard-proxy's A/B snapshot slots).
func (s *Store) CreateSnapshot(ctx context.Context, name, sourceID string) (*Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := stableID("snap-", name)
	if existing := (&Snapshot{}); readJSON(s.snapMeta(id), existing) == nil {
		if existing.SourceID != sourceID {
			return nil, ErrAlreadyExists
		}
		return existing, nil
	}
	vol, err := s.getVolumeLocked(sourceID)
	if err != nil {
		return nil, err
	}
	format := FormatQCOW2
	if s.Images.Name() != "qemu-img" {
		format = FormatRaw
	}
	snap := &Snapshot{ID: id, Name: name, SourceID: sourceID, Format: format, Size: vol.Capacity,
		Path: filepath.Join(s.Dir, "snapshots", id+"."+format), Created: time.Now().UTC()}
	_ = os.Remove(snap.Path)
	if err := s.Images.Convert(ctx, vol.Path, vol.Format, snap.Path, format, true); err != nil {
		return nil, err
	}
	return snap, writeJSON(s.snapMeta(id), snap)
}

// GetSnapshot returns a snapshot by ID.
func (s *Store) GetSnapshot(id string) (*Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := &Snapshot{}
	return snap, readJSON(s.snapMeta(id), snap)
}

// DeleteSnapshot removes a snapshot (idempotent).
func (s *Store) DeleteSnapshot(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := &Snapshot{}
	if err := readJSON(s.snapMeta(id), snap); errors.Is(err, ErrNotFound) {
		return nil
	} else if err != nil {
		return err
	}
	if err := os.Remove(snap.Path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Remove(s.snapMeta(id))
}

// ListSnapshots returns all snapshots sorted by ID.
func (s *Store) ListSnapshots() ([]*Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return list[Snapshot](filepath.Join(s.Dir, "snapshots"))
}

func list[T any](dir string) ([]*T, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	out := make([]*T, 0, len(matches))
	for _, m := range matches {
		v := new(T)
		if err := readJSON(m, v); err == nil {
			out = append(out, v)
		}
	}
	return out, nil
}

func keys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
