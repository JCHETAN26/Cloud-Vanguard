package render

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
)

var update = flag.Bool("update", false, "rewrite testdata/control_golden.json")

func cluster() *cachev1alpha1.CacheCluster {
	return &cachev1alpha1.CacheCluster{
		ObjectMeta: metav1.ObjectMeta{Name: "vc", Namespace: "prod"},
		Spec: cachev1alpha1.CacheClusterSpec{
			Replicas: 3, UpstreamPrimary: "http://vllm:8000", Vnodes: 1000,
			MaxCacheBytes: ptr.To(resource.MustParse("512Mi")), DefaultTTLSeconds: 3600,
		},
	}
}

func policy(name, tenant string, created time.Time) cachev1alpha1.TenantCachePolicy {
	return cachev1alpha1.TenantCachePolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "prod", Generation: 2, CreationTimestamp: metav1.NewTime(created)},
		Spec:       cachev1alpha1.TenantCachePolicySpec{TenantID: tenant, RPMLimit: 1200},
	}
}

func TestPodAddressesAndIDs(t *testing.T) {
	cc := cluster()
	if got := NodeID(cc, 2); got != "vc-2" {
		t.Fatalf("NodeID = %q", got)
	}
	if got := PodAddress(cc, 0); got != "vc-0.vc-peers.prod.svc.cluster.local:8080" {
		t.Fatalf("PodAddress = %q", got)
	}
	if CacheMiB(cc) != 512 {
		t.Fatalf("CacheMiB = %d", CacheMiB(cc))
	}
}

func TestParseThreshold(t *testing.T) {
	for _, ok := range []string{"", "0.96", "1", "1.0", "0.5"} {
		if _, err := ParseThreshold(ok); err != nil {
			t.Errorf("%q rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"0", "1.5", "-0.1", "abc"} {
		if _, err := ParseThreshold(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestHashKeys(t *testing.T) {
	got := HashKeys([]byte("sk-b\n# comment\n\n  sk-a  \nsk-b\n"))
	sum := func(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }
	want := []string{sum("sk-a"), sum("sk-b")}
	if want[0] > want[1] {
		want[0], want[1] = want[1], want[0]
	}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("HashKeys = %v, want %v", got, want)
	}
}

func TestSelectPoliciesOldestWinsAndScopes(t *testing.T) {
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	newer := policy("acme-v2", "acme", t0.Add(time.Hour))
	older := policy("acme-v1", "acme", t0)
	other := policy("beta", "beta", t0)
	other.Spec.ClusterRef = "another-cluster"
	deleting := policy("gone", "gamma", t0)
	deleting.DeletionTimestamp = ptr.To(metav1.NewTime(t0))

	winners, conflicts := SelectPolicies("vc", []cachev1alpha1.TenantCachePolicy{newer, other, older, deleting})
	if len(winners) != 1 || winners[0].Name != "acme-v1" {
		t.Fatalf("winners = %v", winners)
	}
	if conflicts["acme-v2"] != "acme-v1" {
		t.Fatalf("conflicts = %v", conflicts)
	}
}

func TestDocumentHashIgnoresVersionAndInputOrder(t *testing.T) {
	t0 := time.Now()
	a := ResolvedPolicy{Policy: policy("a", "alpha", t0)}
	b := ResolvedPolicy{Policy: policy("b", "beta", t0)}
	d1 := ControlDocument(cluster(), []ResolvedPolicy{a, b}, 1)
	d2 := ControlDocument(cluster(), []ResolvedPolicy{b, a}, 7)
	if Hash(d1) != Hash(d2) {
		t.Fatal("hash depends on version or policy order")
	}
	j1, _ := JSON(d2)
	j2, _ := JSON(d2)
	if string(j1) != string(j2) {
		t.Fatal("JSON rendering is not byte-stable")
	}
	b.Policy.Spec.RPMLimit = 1
	if Hash(ControlDocument(cluster(), []ResolvedPolicy{a, b}, 1)) == Hash(d1) {
		t.Fatal("hash did not change with content")
	}
	cc := cluster()
	cc.Spec.Replicas = 4
	if Hash(ControlDocument(cc, []ResolvedPolicy{a, b}, 1)) == Hash(d1) {
		t.Fatal("hash did not change with ring size")
	}
}

// The golden file is the contract with the C++ data plane: proxy-cpp's
// ControlPlaneGolden test parses exactly this file.
func TestGoldenControlDocument(t *testing.T) {
	ttl := int32(600)
	p := policy("acme", "acme", time.Unix(0, 0))
	p.Spec.TPMLimit = 50000
	p.Spec.SemanticMode = "neural"
	p.Spec.SemanticThreshold = "0.97"
	p.Spec.TTLSeconds = &ttl
	doc := ControlDocument(cluster(), []ResolvedPolicy{{Policy: p, KeyHashes: HashKeys([]byte("sk-acme-1\nsk-acme-2"))},
		{Policy: policy("beta", "beta", time.Unix(0, 0))}}, 42)
	got, err := JSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join("testdata", "control_golden.json")
	if *update {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(got, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test ./internal/render -update)", err)
	}
	if string(want) != string(got)+"\n" {
		t.Fatalf("rendered document differs from %s:\n%s", path, got)
	}
	back, err := ParseJSON(got)
	if err != nil || back.GetRing().GetVnodes() != 1000 || len(back.GetTenants().GetTenants()) != 2 {
		t.Fatalf("round trip failed: %v %v", err, back)
	}
}
