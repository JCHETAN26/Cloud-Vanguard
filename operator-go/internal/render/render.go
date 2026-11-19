// Package render turns CacheCluster / TenantCachePolicy objects into the
// control document pushed to vanguard-proxy and the Kubernetes workload that
// runs it. Everything here is pure (no API calls) so it is unit-testable.
package render

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
)

const (
	DataPort         = 8080
	AdminPort        = 9901
	ControlKey       = "control.json"
	ControlMountPath = "/etc/vanguard/control"
	AdminTokenKey    = "token"
	HashAnnotation    = "cache.cloudvanguard.io/config-hash"
	VersionAnnotation = "cache.cloudvanguard.io/config-version"
	ManagedByLabel   = "app.kubernetes.io/managed-by"
	ManagedByValue   = "cloudvanguard-operator"
	InstanceLabel    = "app.kubernetes.io/instance"
	NameLabel        = "app.kubernetes.io/name"
	DefaultVnodes    = 1000
)

func StatefulSetName(cc *cachev1alpha1.CacheCluster) string     { return cc.Name }
func PeerServiceName(cc *cachev1alpha1.CacheCluster) string     { return cc.Name + "-peers" }
func ClientServiceName(cc *cachev1alpha1.CacheCluster) string   { return cc.Name }
func ControlConfigMapName(cc *cachev1alpha1.CacheCluster) string { return cc.Name + "-control" }
func AdminSecretName(cc *cachev1alpha1.CacheCluster) string     { return cc.Name + "-admin" }

// Labels identifies every object (and pod) belonging to a cluster.
func Labels(cc *cachev1alpha1.CacheCluster) map[string]string {
	return map[string]string{
		NameLabel:      "vanguard-proxy",
		InstanceLabel:  cc.Name,
		ManagedByLabel: ManagedByValue,
	}
}

// NodeID is the ring identity of ordinal i: the StatefulSet pod name, which
// the pod receives as NODE_ID through the downward API.
func NodeID(cc *cachev1alpha1.CacheCluster, i int32) string {
	return fmt.Sprintf("%s-%d", StatefulSetName(cc), i)
}

// PodAddress is the stable DNS name of ordinal i behind the headless Service.
func PodAddress(cc *cachev1alpha1.CacheCluster, i int32) string {
	return fmt.Sprintf("%s.%s.%s.svc.cluster.local:%d", NodeID(cc, i), PeerServiceName(cc), cc.Namespace, DataPort)
}

// ---------------------------------------------------------------------------
// Tenant policies
// ---------------------------------------------------------------------------

// ParseThreshold validates spec.semanticThreshold ("" = unset).
func ParseThreshold(s string) (*float64, error) {
	if s == "" {
		return nil, nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v <= 0 || v > 1 {
		return nil, fmt.Errorf("semanticThreshold %q must be a number in (0, 1]", s)
	}
	return &v, nil
}

// HashKeys returns the sorted, de-duplicated lower-case hex SHA-256 of each
// API key in a Secret value (one key per line; blank lines and # comments ignored).
func HashKeys(data []byte) []string {
	seen := map[string]bool{}
	var out []string
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		k := strings.TrimSpace(sc.Text())
		if k == "" || strings.HasPrefix(k, "#") {
			continue
		}
		sum := sha256.Sum256([]byte(k))
		h := hex.EncodeToString(sum[:])
		if !seen[h] {
			seen[h] = true
			out = append(out, h)
		}
	}
	sort.Strings(out)
	return out
}

// Targets reports whether a policy applies to the named cluster.
func Targets(p *cachev1alpha1.TenantCachePolicy, cluster string) bool {
	return p.Spec.ClusterRef == "" || p.Spec.ClusterRef == cluster
}

// SelectPolicies picks, per tenantId, the policy that applies to `cluster`.
// When several policies claim the same tenant the oldest wins (ties broken by
// name) so the choice is stable; the rest are returned in `conflicts`
// (loser name -> winner name). Policies being deleted are ignored.
func SelectPolicies(cluster string, all []cachev1alpha1.TenantCachePolicy) (winners []cachev1alpha1.TenantCachePolicy, conflicts map[string]string) {
	conflicts = map[string]string{}
	var candidates []cachev1alpha1.TenantCachePolicy
	for _, p := range all {
		if p.DeletionTimestamp.IsZero() && Targets(&p, cluster) {
			candidates = append(candidates, p)
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if !a.CreationTimestamp.Equal(&b.CreationTimestamp) {
			return a.CreationTimestamp.Before(&b.CreationTimestamp)
		}
		return a.Name < b.Name
	})
	byTenant := map[string]string{}
	for _, p := range candidates {
		if winner, taken := byTenant[p.Spec.TenantID]; taken {
			conflicts[p.Name] = winner
			continue
		}
		byTenant[p.Spec.TenantID] = p.Name
		winners = append(winners, p)
	}
	sort.Slice(winners, func(i, j int) bool { return winners[i].Spec.TenantID < winners[j].Spec.TenantID })
	return winners, conflicts
}

// ResolvedPolicy is a winning policy plus the hashes of its API keys.
type ResolvedPolicy struct {
	Policy    cachev1alpha1.TenantCachePolicy
	KeyHashes []string
}

// ---------------------------------------------------------------------------
// Control document
// ---------------------------------------------------------------------------

// ControlDocument renders the ring and tenant policies for a cluster.
func ControlDocument(cc *cachev1alpha1.CacheCluster, policies []ResolvedPolicy, version int64) *controlpb.ControlDocument {
	vnodes := uint32(DefaultVnodes)
	if cc.Spec.Vnodes > 0 {
		vnodes = uint32(cc.Spec.Vnodes)
	}
	ring := &controlpb.RingConfig{Version: version, Vnodes: vnodes}
	for i := int32(0); i < cc.Spec.Replicas; i++ {
		ring.Nodes = append(ring.Nodes, &controlpb.Node{Id: NodeID(cc, i), Address: PodAddress(cc, i)})
	}
	tenants := &controlpb.TenantPolicySet{Version: version}
	sorted := append([]ResolvedPolicy(nil), policies...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Policy.Spec.TenantID < sorted[j].Policy.Spec.TenantID })
	for _, rp := range sorted {
		s := rp.Policy.Spec
		tp := &controlpb.TenantPolicy{
			TenantId:         s.TenantID,
			ApiKeySha256:     rp.KeyHashes,
			RpmLimit:         s.RPMLimit,
			TpmLimit:         s.TPMLimit,
			SourceGeneration: rp.Policy.Generation,
		}
		if s.SemanticMode != "" {
			tp.SemanticMode = proto.String(s.SemanticMode)
		}
		if th, err := ParseThreshold(s.SemanticThreshold); err == nil && th != nil {
			tp.SemanticThreshold = th
		}
		if s.TTLSeconds != nil {
			tp.TtlSeconds = proto.Uint32(uint32(*s.TTLSeconds))
		}
		tenants.Tenants = append(tenants.Tenants, tp)
	}
	return &controlpb.ControlDocument{Version: version, Ring: ring, Tenants: tenants}
}

// Hash identifies a document's content independent of its version numbers,
// so the operator bumps the version only when something actually changed.
func Hash(doc *controlpb.ControlDocument) string {
	c := proto.Clone(doc).(*controlpb.ControlDocument)
	c.Version = 0
	if c.Ring != nil {
		c.Ring.Version = 0
	}
	if c.Tenants != nil {
		c.Tenants.Version = 0
	}
	b, _ := proto.MarshalOptions{Deterministic: true}.Marshal(c)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

// JSON renders the document in proto3 JSON. protojson deliberately varies its
// whitespace between runs, so the output is re-encoded canonically (sorted
// keys, fixed indentation) to keep the ConfigMap byte-stable.
func JSON(doc *controlpb.ControlDocument) ([]byte, error) {
	raw, err := protojson.Marshal(doc)
	if err != nil {
		return nil, err
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	return json.MarshalIndent(v, "", "  ")
}

// ParseJSON is the inverse of JSON (used to check what is live).
func ParseJSON(b []byte) (*controlpb.ControlDocument, error) {
	doc := &controlpb.ControlDocument{}
	return doc, protojson.UnmarshalOptions{DiscardUnknown: true}.Unmarshal(b, doc)
}
