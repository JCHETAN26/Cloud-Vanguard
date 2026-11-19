package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/render"
)

var ctx = context.Background()

func versionOf(doc []byte) int64 {
	d, err := render.ParseJSON(doc)
	if err != nil {
		return 0
	}
	return d.GetVersion()
}

func newCluster(ns, name string, replicas int32) *cachev1alpha1.CacheCluster {
	return &cachev1alpha1.CacheCluster{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec:       cachev1alpha1.CacheClusterSpec{Replicas: replicas, UpstreamPrimary: "http://vllm.inference:8000"},
	}
}

func newPolicy(ns, name, tenant string) *cachev1alpha1.TenantCachePolicy {
	return &cachev1alpha1.TenantCachePolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec:       cachev1alpha1.TenantCachePolicySpec{TenantID: tenant, RPMLimit: 600},
	}
}

func getCluster(t *testing.T, ns, name string) *cachev1alpha1.CacheCluster {
	t.Helper()
	cc := &cachev1alpha1.CacheCluster{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, cc); err != nil {
		t.Fatal(err)
	}
	return cc
}

// controlDoc reads the rendered control document from the cluster's ConfigMap.
func controlDoc(ns, cluster string) (*controlpb.ControlDocument, error) {
	cm := &corev1.ConfigMap{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: cluster + "-control"}, cm); err != nil {
		return nil, err
	}
	return render.ParseJSON([]byte(cm.Data[render.ControlKey]))
}

func tenantIn(doc *controlpb.ControlDocument, id string) *controlpb.TenantPolicy {
	for _, t := range doc.GetTenants().GetTenants() {
		if t.GetTenantId() == id {
			return t
		}
	}
	return nil
}

func envValue(c corev1.Container, name string) (string, bool) {
	for _, e := range c.Env {
		if e.Name == name {
			if e.ValueFrom != nil && e.ValueFrom.FieldRef != nil {
				return "fieldRef:" + e.ValueFrom.FieldRef.FieldPath, true
			}
			if e.ValueFrom != nil && e.ValueFrom.SecretKeyRef != nil {
				return "secret:" + e.ValueFrom.SecretKeyRef.Name, true
			}
			return e.Value, true
		}
	}
	return "", false
}

func ownedBy(obj metav1.Object, cc *cachev1alpha1.CacheCluster) error {
	for _, ref := range obj.GetOwnerReferences() {
		if ref.UID == cc.UID && ref.Controller != nil && *ref.Controller &&
			ref.BlockOwnerDeletion != nil && *ref.BlockOwnerDeletion {
			return nil
		}
	}
	return fmt.Errorf("%s has no controller ownerReference to %s", obj.GetName(), cc.Name)
}

// ---------------------------------------------------------------------------
// CacheCluster
// ---------------------------------------------------------------------------

func TestCacheClusterDefaultsAndValidation(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	cc := newCluster(ns, "defaults", 2)
	if err := k8sClient.Create(ctx, cc); err != nil {
		t.Fatal(err)
	}
	got := getCluster(t, ns, "defaults")
	if got.Spec.Vnodes != 1000 || got.Spec.DefaultTTLSeconds != 3600 || got.Spec.MaxCacheBytes == nil || got.Spec.MaxCacheBytes.String() != "256Mi" ||
		got.Spec.Image == "" {
		t.Fatalf("API server defaults not applied: %+v", got.Spec)
	}

	bad := []*cachev1alpha1.CacheCluster{newCluster(ns, "tls", 1), newCluster(ns, "zero", 0), newCluster(ns, "vn", 1)}
	bad[0].Spec.UpstreamPrimary = "https://api.example.com"
	bad[2].Spec.Vnodes = 200000
	for _, b := range bad {
		if err := k8sClient.Create(ctx, b); err == nil || !apierrors.IsInvalid(err) {
			t.Errorf("%s: expected validation error, got %v", b.Name, err)
		}
	}
}

func TestCacheClusterReconcilesWorkloads(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	cc := newCluster(ns, "vc", 3)
	cc.Spec.UpstreamFallback = "http://openai-shim.inference:8000"
	if err := k8sClient.Create(ctx, cc); err != nil {
		t.Fatal(err)
	}
	cc = getCluster(t, ns, "vc")

	sts := &appsv1.StatefulSet{}
	eventually(t, "StatefulSet", func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "vc"}, sts)
	})
	if *sts.Spec.Replicas != 3 || sts.Spec.ServiceName != "vc-peers" || sts.Spec.PodManagementPolicy != appsv1.ParallelPodManagement {
		t.Fatalf("statefulset spec: replicas=%d serviceName=%s policy=%s", *sts.Spec.Replicas, sts.Spec.ServiceName,
			sts.Spec.PodManagementPolicy)
	}
	c := sts.Spec.Template.Spec.Containers[0]
	for name, want := range map[string]string{
		"NODE_ID": "fieldRef:metadata.name", "UPSTREAM_URL": "http://vllm.inference:8000",
		"FALLBACK_UPSTREAM_URL": "http://openai-shim.inference:8000", "RING_VNODES": "1000", "CACHE_MAX_MB": "256",
		"CACHE_TTL_S": "3600", "CONTROL_FILE": "/etc/vanguard/control/control.json", "ADMIN_TOKEN": "secret:vc-admin",
		"PEER_TOKEN": "secret:vc-admin", "ADMIN_ADDR": "0.0.0.0",
	} {
		if got, ok := envValue(c, name); !ok || got != want {
			t.Errorf("env %s = %q, want %q", name, got, want)
		}
	}
	if err := ownedBy(sts, cc); err != nil {
		t.Error(err)
	}

	peers, client_ := &corev1.Service{}, &corev1.Service{}
	eventually(t, "services", func() error {
		if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "vc-peers"}, peers); err != nil {
			return err
		}
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "vc"}, client_)
	})
	if peers.Spec.ClusterIP != corev1.ClusterIPNone || !peers.Spec.PublishNotReadyAddresses {
		t.Errorf("peer service must be headless and publish not-ready addresses: %+v", peers.Spec)
	}
	if client_.Spec.ClusterIP == corev1.ClusterIPNone || client_.Spec.Ports[0].Port != 8080 {
		t.Errorf("client service: %+v", client_.Spec)
	}

	secret := &corev1.Secret{}
	eventually(t, "admin secret", func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "vc-admin"}, secret)
	})
	if len(secret.Data["token"]) != 64 {
		t.Errorf("admin token should be 32 random bytes hex-encoded, got %d chars", len(secret.Data["token"]))
	}

	var doc *controlpb.ControlDocument
	eventually(t, "control document", func() error {
		var err error
		doc, err = controlDoc(ns, "vc")
		return err
	})
	ring := doc.GetRing()
	if ring.GetVnodes() != 1000 || len(ring.GetNodes()) != 3 ||
		ring.GetNodes()[2].GetId() != "vc-2" ||
		ring.GetNodes()[2].GetAddress() != "vc-2.vc-peers."+ns+".svc.cluster.local:8080" {
		t.Fatalf("ring: %v", ring)
	}
	eventually(t, "status", func() error {
		got := getCluster(t, ns, "vc")
		if got.Status.ConfigVersion != 1 || got.Status.ObservedGeneration != got.Generation ||
			meta.FindStatusCondition(got.Status.Conditions, ConditionAvailable) == nil {
			return fmt.Errorf("status %+v", got.Status)
		}
		return nil
	})
}

func TestCacheClusterScalingBumpsConfigVersionOnce(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	if err := k8sClient.Create(ctx, newCluster(ns, "scale", 3)); err != nil {
		t.Fatal(err)
	}
	eventually(t, "v1", func() error {
		if v := getCluster(t, ns, "scale").Status.ConfigVersion; v != 1 {
			return fmt.Errorf("configVersion %d", v)
		}
		return nil
	})

	cc := getCluster(t, ns, "scale")
	cc.Spec.Replicas = 5
	cc.Spec.Vnodes = 500
	if err := k8sClient.Update(ctx, cc); err != nil {
		t.Fatal(err)
	}
	eventually(t, "scaled statefulset and ring", func() error {
		sts := &appsv1.StatefulSet{}
		if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "scale"}, sts); err != nil {
			return err
		}
		doc, err := controlDoc(ns, "scale")
		if err != nil {
			return err
		}
		if *sts.Spec.Replicas != 5 || len(doc.GetRing().GetNodes()) != 5 || doc.GetRing().GetVnodes() != 500 {
			return fmt.Errorf("replicas=%d ring=%d vnodes=%d", *sts.Spec.Replicas, len(doc.GetRing().GetNodes()),
				doc.GetRing().GetVnodes())
		}
		if env, _ := envValue(sts.Spec.Template.Spec.Containers[0], "RING_VNODES"); env != "500" {
			return fmt.Errorf("RING_VNODES=%s", env)
		}
		return nil
	})
	// Exactly one version bump for one change, and no churn afterwards.
	consistently(t, "config version stays 2", 2_000_000_000, func() error {
		if v := getCluster(t, ns, "scale").Status.ConfigVersion; v != 2 {
			return fmt.Errorf("configVersion %d", v)
		}
		return nil
	})
}

// makeReadyPod simulates what the kubelet would do: a running, ready pod with an IP.
func makeReadyPod(t *testing.T, ns, cluster string, ordinal int, ip string) {
	t.Helper()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%s-%d", cluster, ordinal), Namespace: ns,
			Labels: render.Labels(&cachev1alpha1.CacheCluster{ObjectMeta: metav1.ObjectMeta{Name: cluster}})},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "vanguard-proxy", Image: "vanguard-proxy:test"}}},
	}
	if err := k8sClient.Create(ctx, pod); err != nil {
		t.Fatal(err)
	}
	pod.Status.PodIP = ip
	pod.Status.Phase = corev1.PodRunning
	pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	if err := k8sClient.Status().Update(ctx, pod); err != nil {
		t.Fatal(err)
	}
}

func TestCacheClusterStatusAndConfigPush(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	if err := k8sClient.Create(ctx, newCluster(ns, "push", 3)); err != nil {
		t.Fatal(err)
	}
	sts := &appsv1.StatefulSet{}
	eventually(t, "statefulset", func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "push"}, sts)
	})
	ips := []string{"10.1.0.10", "10.1.0.11", "10.1.0.12"}
	fakePush.setFailing(ips[2]+":9901", true)
	for i, ip := range ips {
		makeReadyPod(t, ns, "push", i, ip)
	}
	sts.Status.Replicas, sts.Status.ReadyReplicas = 3, 3
	if err := k8sClient.Status().Update(ctx, sts); err != nil {
		t.Fatal(err)
	}

	// One pod's admin API is down: 2/3 synced, ConfigSynced=False.
	eventually(t, "partial sync", func() error {
		st := getCluster(t, ns, "push").Status
		c := meta.FindStatusCondition(st.Conditions, ConditionConfigSynced)
		if st.ReadyReplicas != 3 || len(st.Endpoints) != 3 || st.SyncedReplicas != 2 || c == nil || c.Status != metav1.ConditionFalse {
			return fmt.Errorf("status %+v", st)
		}
		if a := meta.FindStatusCondition(st.Conditions, ConditionAvailable); a == nil || a.Status != metav1.ConditionTrue {
			return fmt.Errorf("Available should be True: %+v", a)
		}
		return nil
	})
	secret := &corev1.Secret{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "push-admin"}, secret); err != nil {
		t.Fatal(err)
	}
	doc, token := fakePush.last(ips[0] + ":9901")
	if token != string(secret.Data["token"]) || versionOf(doc) != getCluster(t, ns, "push").Status.ConfigVersion {
		t.Fatalf("push used token %q / version %d", token, versionOf(doc))
	}
	// The pod recovers: the 5 s retry brings it in sync.
	fakePush.setFailing(ips[2]+":9901", false)
	eventually(t, "full sync", func() error {
		st := getCluster(t, ns, "push").Status
		c := meta.FindStatusCondition(st.Conditions, ConditionConfigSynced)
		if st.SyncedReplicas != 3 || c == nil || c.Status != metav1.ConditionTrue {
			return fmt.Errorf("status %+v", st)
		}
		return nil
	})
}

// ---------------------------------------------------------------------------
// TenantCachePolicy
// ---------------------------------------------------------------------------

func sha(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

func getPolicy(t *testing.T, ns, name string) *cachev1alpha1.TenantCachePolicy {
	t.Helper()
	p := &cachev1alpha1.TenantCachePolicy{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, p); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTenantPolicyLifecycle(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	if err := k8sClient.Create(ctx, newCluster(ns, "vc", 2)); err != nil {
		t.Fatal(err)
	}
	keys := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "acme-keys", Namespace: ns},
		StringData: map[string]string{"api-keys": "sk-acme-1\nsk-acme-2\n"}}
	if err := k8sClient.Create(ctx, keys); err != nil {
		t.Fatal(err)
	}
	p := newPolicy(ns, "acme", "acme")
	p.Spec.APIKeysSecretRef = &cachev1alpha1.SecretKeyRef{Name: "acme-keys"}
	p.Spec.RPMLimit, p.Spec.TPMLimit = 1200, 50000
	p.Spec.SemanticMode, p.Spec.SemanticThreshold, p.Spec.TTLSeconds = "neural", "0.97", ptr.To(int32(600))
	if err := k8sClient.Create(ctx, p); err != nil {
		t.Fatal(err)
	}

	// Rendered into the control document, keys hashed, status Applied.
	eventually(t, "policy rendered and applied", func() error {
		doc, err := controlDoc(ns, "vc")
		if err != nil {
			return err
		}
		tp := tenantIn(doc, "acme")
		if tp == nil || tp.GetRpmLimit() != 1200 || tp.GetTpmLimit() != 50000 || tp.GetSemanticMode() != "neural" ||
			tp.GetSemanticThreshold() != 0.97 || tp.GetTtlSeconds() != 600 || len(tp.GetApiKeySha256()) != 2 {
			return fmt.Errorf("tenant in doc: %v", tp)
		}
		for _, h := range tp.GetApiKeySha256() {
			if h != sha("sk-acme-1") && h != sha("sk-acme-2") {
				return fmt.Errorf("unexpected key hash %s", h)
			}
		}
		st := getPolicy(t, ns, "acme").Status
		if !st.Applied || st.ObservedGeneration != 1 || len(st.AppliedClusters) != 1 {
			return fmt.Errorf("policy status %+v", st)
		}
		return nil
	})
	v1 := getCluster(t, ns, "vc").Status.ConfigVersion

	// Update: new generation is rendered, version bumps, status follows.
	p = getPolicy(t, ns, "acme")
	p.Spec.RPMLimit = 300
	if err := k8sClient.Update(ctx, p); err != nil {
		t.Fatal(err)
	}
	eventually(t, "update rendered", func() error {
		doc, err := controlDoc(ns, "vc")
		if err != nil {
			return err
		}
		tp := tenantIn(doc, "acme")
		st := getPolicy(t, ns, "acme").Status
		if tp.GetRpmLimit() != 300 || tp.GetSourceGeneration() != 2 || !st.Applied || st.ObservedGeneration != 2 ||
			getCluster(t, ns, "vc").Status.ConfigVersion <= v1 {
			return fmt.Errorf("rpm=%d gen=%d status=%+v", tp.GetRpmLimit(), tp.GetSourceGeneration(), st)
		}
		return nil
	})

	// Key rotation in the Secret propagates without touching the policy.
	keys.StringData = map[string]string{"api-keys": "sk-acme-3\n"}
	if err := k8sClient.Update(ctx, keys); err != nil {
		t.Fatal(err)
	}
	eventually(t, "rotated keys", func() error {
		doc, err := controlDoc(ns, "vc")
		if err != nil {
			return err
		}
		if h := tenantIn(doc, "acme").GetApiKeySha256(); len(h) != 1 || h[0] != sha("sk-acme-3") {
			return fmt.Errorf("hashes %v", h)
		}
		return nil
	})

	// Deletion removes the tenant from every pod's configuration.
	if err := k8sClient.Delete(ctx, getPolicy(t, ns, "acme")); err != nil {
		t.Fatal(err)
	}
	eventually(t, "tenant removed", func() error {
		doc, err := controlDoc(ns, "vc")
		if err != nil {
			return err
		}
		if tenantIn(doc, "acme") != nil {
			return fmt.Errorf("acme still rendered")
		}
		return nil
	})
}

func TestTenantPolicyConflictsValidationAndMissingCluster(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	if err := k8sClient.Create(ctx, newCluster(ns, "vc", 1)); err != nil {
		t.Fatal(err)
	}
	first := newPolicy(ns, "first", "shared")
	first.Spec.RPMLimit = 100
	if err := k8sClient.Create(ctx, first); err != nil {
		t.Fatal(err)
	}
	eventually(t, "first applied", func() error {
		if !getPolicy(t, ns, "first").Status.Applied {
			return fmt.Errorf("not applied")
		}
		return nil
	})
	second := newPolicy(ns, "second", "shared")
	second.Spec.RPMLimit = 999
	if err := k8sClient.Create(ctx, second); err != nil {
		t.Fatal(err)
	}
	eventually(t, "conflict reported", func() error {
		st := getPolicy(t, ns, "second").Status
		c := meta.FindStatusCondition(st.Conditions, ConditionConflict)
		if st.Applied || c == nil || c.Status != metav1.ConditionTrue {
			return fmt.Errorf("status %+v", st)
		}
		return nil
	})
	consistently(t, "oldest policy keeps the tenant", 1_000_000_000, func() error {
		doc, err := controlDoc(ns, "vc")
		if err != nil {
			return err
		}
		if tenantIn(doc, "shared").GetRpmLimit() != 100 {
			return fmt.Errorf("conflicting policy leaked into the document")
		}
		return nil
	})

	// Schema validation rejects bad thresholds / modes before they reach any pod.
	bad := newPolicy(ns, "bad", "bad")
	bad.Spec.SemanticThreshold = "1.5"
	if err := k8sClient.Create(ctx, bad); err == nil || !apierrors.IsInvalid(err) {
		t.Errorf("threshold 1.5 accepted: %v", err)
	}
	bad = newPolicy(ns, "bad2", "bad")
	bad.Spec.SemanticMode = "gpu"
	if err := k8sClient.Create(ctx, bad); err == nil || !apierrors.IsInvalid(err) {
		t.Errorf("semanticMode gpu accepted: %v", err)
	}

	orphan := newPolicy(ns, "orphan", "orphan")
	orphan.Spec.ClusterRef = "does-not-exist"
	if err := k8sClient.Create(ctx, orphan); err != nil {
		t.Fatal(err)
	}
	eventually(t, "ClusterNotFound", func() error {
		c := meta.FindStatusCondition(getPolicy(t, ns, "orphan").Status.Conditions, ConditionApplied)
		if c == nil || c.Reason != "ClusterNotFound" {
			return fmt.Errorf("condition %+v", c)
		}
		return nil
	})
}

func TestCacheClusterDeletionCleanup(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	if err := k8sClient.Create(ctx, newCluster(ns, "doomed", 2)); err != nil {
		t.Fatal(err)
	}
	p := newPolicy(ns, "tenant", "t1")
	if err := k8sClient.Create(ctx, p); err != nil {
		t.Fatal(err)
	}
	eventually(t, "policy applied", func() error {
		if !getPolicy(t, ns, "tenant").Status.Applied {
			return fmt.Errorf("not applied")
		}
		return nil
	})
	cc := getCluster(t, ns, "doomed")
	// Every child is owned by the cluster (controller + blockOwnerDeletion), so
	// the garbage collector removes them all when the cluster goes. envtest has
	// no GC, so the contract is asserted on the ownerReferences.
	children := []client.Object{&appsv1.StatefulSet{}, &corev1.Service{}, &corev1.Service{}, &corev1.ConfigMap{}, &corev1.Secret{}}
	names := []string{"doomed", "doomed", "doomed-peers", "doomed-control", "doomed-admin"}
	for i, obj := range children {
		if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: names[i]}, obj); err != nil {
			t.Fatal(err)
		}
		if err := ownedBy(obj, cc); err != nil {
			t.Error(err)
		}
	}
	if err := k8sClient.Delete(ctx, cc); err != nil {
		t.Fatal(err)
	}
	// Policies that targeted it stop reporting Applied.
	eventually(t, "policy no longer applied", func() error {
		st := getPolicy(t, ns, "tenant").Status
		c := meta.FindStatusCondition(st.Conditions, ConditionApplied)
		if st.Applied || c == nil || c.Reason != "NoClusters" {
			return fmt.Errorf("status %+v", st)
		}
		return nil
	})
}

// The published samples must be accepted by the CRD schemas and reconcile.
func TestSamplesApplyAndReconcile(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	raw, err := os.ReadFile(filepath.Join("..", "..", "config", "samples", "cachecluster.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range strings.Split(string(raw), "\n---\n") {
		obj := &unstructured.Unstructured{}
		if err := yaml.Unmarshal([]byte(chunk), &obj.Object); err != nil {
			t.Fatal(err)
		}
		obj.SetNamespace(ns)
		if err := k8sClient.Create(ctx, obj); err != nil {
			t.Fatalf("%s/%s rejected: %v", obj.GetKind(), obj.GetName(), err)
		}
	}
	eventually(t, "sample policy applied", func() error {
		st := getPolicy(t, ns, "acme").Status
		if !st.Applied {
			return fmt.Errorf("status %+v", st)
		}
		doc, err := controlDoc(ns, "vc")
		if err != nil {
			return err
		}
		if tp := tenantIn(doc, "acme"); tp == nil || len(tp.GetApiKeySha256()) != 2 || tp.GetSemanticThreshold() != 0.97 {
			return fmt.Errorf("tenant %v", tp)
		}
		return nil
	})
}
