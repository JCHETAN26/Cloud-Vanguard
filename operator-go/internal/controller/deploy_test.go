package controller

import (
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/render"
)

// The deploy/ bundle (`kubectl kustomize deploy/`: CRDs, operator RBAC and
// Deployment, CSI driver, StorageClass, OpenTelemetry Collector) is applied to
// the envtest kube-apiserver, then both reconcilers run *as the operator's
// ServiceAccount* on the published sample. A permission missing from the
// shipped RBAC fails here as Forbidden instead of in a cluster.

const snapshotGroup = "snapshot.storage.k8s.io" // external-snapshotter CRDs: not part of envtest

func decodeYAML(t *testing.T, raw []byte) []*unstructured.Unstructured {
	t.Helper()
	dec := utilyaml.NewYAMLOrJSONDecoder(bytes.NewReader(raw), 4096)
	var out []*unstructured.Unstructured
	for {
		m := map[string]interface{}{}
		if err := dec.Decode(&m); err != nil {
			if errors.Is(err, io.EOF) {
				return out
			}
			t.Fatal(err)
		}
		if len(m) > 0 {
			out = append(out, &unstructured.Unstructured{Object: m})
		}
	}
}

func renderDeployBundle(t *testing.T) []*unstructured.Unstructured {
	t.Helper()
	kubectl := filepath.Join(assetsDir(), "kubectl") // shipped with the envtest binaries
	if _, err := os.Stat(kubectl); err != nil {
		t.Skipf("kubectl not found next to the envtest binaries (%v)", err)
	}
	cmd := exec.Command(kubectl, "kustomize", filepath.Join("..", "..", "..", "deploy"))
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("kubectl kustomize deploy/: %v\n%s", err, stderr.String())
	}
	return decodeYAML(t, out)
}

// applyAll server-side-applies objects, retrying for ordering (namespaces and
// CRDs before their contents). Objects of the snapshot group are returned as
// skipped when their CRDs are absent; anything else that fails is fatal.
func applyAll(t *testing.T, objs []*unstructured.Unstructured) (applied int, skipped []string) {
	t.Helper()
	pending := objs
	var lastErr map[string]error
	for pass := 0; pass < 4 && len(pending) > 0; pass++ {
		var next []*unstructured.Unstructured
		lastErr = map[string]error{}
		for _, o := range pending {
			id := o.GetKind() + "/" + o.GetName()
			obj := o.DeepCopy()
			obj.SetManagedFields(nil)
			err := k8sClient.Patch(ctx, obj, client.Apply, client.FieldOwner("deploy-test"), client.ForceOwnership)
			switch {
			case err == nil:
				applied++
			case meta.IsNoMatchError(err) && o.GroupVersionKind().Group == snapshotGroup:
				skipped = append(skipped, id)
			default:
				lastErr[id] = err
				next = append(next, o)
			}
		}
		pending = next
		if len(pending) > 0 {
			time.Sleep(500 * time.Millisecond) // CRDs become servable asynchronously
		}
	}
	for id, err := range lastErr {
		t.Errorf("apply %s: %v", id, err)
	}
	if t.Failed() {
		t.FailNow()
	}
	return applied, skipped
}

func operatorIdentity(t *testing.T) (namespace, serviceAccount string) {
	t.Helper()
	d := &appsv1.Deployment{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: "cloudvanguard-system", Name: "cloudvanguard-operator"}, d); err != nil {
		t.Fatal(err)
	}
	return d.Namespace, d.Spec.Template.Spec.ServiceAccountName
}

func impersonating(t *testing.T, ns, sa string) client.WithWatch {
	t.Helper()
	cfg := rest.CopyConfig(restCfg)
	cfg.Impersonate = rest.ImpersonationConfig{
		UserName: "system:serviceaccount:" + ns + ":" + sa,
		Groups:   []string{"system:serviceaccounts", "system:serviceaccounts:" + ns, "system:authenticated"},
	}
	c, err := client.NewWithWatch(cfg, client.Options{Scheme: k8sClient.Scheme()})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestDeployBundleAppliesAndOperatorRBACSuffices(t *testing.T) {
	requireEnv(t)
	objs := renderDeployBundle(t)
	kinds := map[string]int{}
	for _, o := range objs {
		kinds[o.GetKind()]++
	}
	applied, skipped := applyAll(t, objs)
	if applied+len(skipped) != len(objs) || len(skipped) != kinds["VolumeSnapshotClass"] {
		t.Fatalf("applied %d, skipped %v of %d objects", applied, skipped, len(objs))
	}
	names := make([]string, 0, len(kinds))
	for k, n := range kinds {
		names = append(names, k+"="+strconv.Itoa(n))
	}
	sort.Strings(names)
	t.Logf("deploy/ bundle: %d objects applied to kube-apiserver, skipped %v (no external-snapshotter CRDs); kinds %s",
		applied, skipped, strings.Join(names, " "))

	// The rendered operator Deployment runs the image the Dockerfiles build.
	d := &appsv1.Deployment{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: "cloudvanguard-system", Name: "cloudvanguard-operator"}, d); err != nil {
		t.Fatal(err)
	}
	if img := d.Spec.Template.Spec.Containers[0].Image; img != "ghcr.io/jchetan26/cloudvanguard-operator:0.4.0" {
		t.Errorf("operator image %q", img)
	}

	ns, sa := operatorIdentity(t)
	op := impersonating(t, ns, sa)

	// The API server enforces RBAC (otherwise everything below proves nothing).
	if err := op.List(ctx, &corev1.NodeList{}); !apierrors.IsForbidden(err) {
		t.Fatalf("listing nodes as %s/%s: want Forbidden, got %v", ns, sa, err)
	}

	// The manager's informers list+watch these cluster-wide.
	for _, l := range []client.ObjectList{&cachev1alpha1.CacheClusterList{}, &cachev1alpha1.TenantCachePolicyList{},
		&appsv1.StatefulSetList{}, &corev1.ServiceList{}, &corev1.ConfigMapList{}, &corev1.SecretList{}, &corev1.PodList{}} {
		if err := op.List(ctx, l); err != nil {
			t.Errorf("list %T: %v", l, err)
			continue
		}
		w, err := op.Watch(ctx, l)
		if err != nil {
			t.Errorf("watch %T: %v", l, err)
			continue
		}
		w.Stop()
	}

	// Leader election (--leader-elect, 2 replicas): a Lease and events in the operator's namespace only.
	lease := &coordinationv1.Lease{ObjectMeta: metav1.ObjectMeta{Namespace: ns, Name: "cloudvanguard-operator.cache.cloudvanguard.io"},
		Spec: coordinationv1.LeaseSpec{HolderIdentity: ptr.To("pod-a"), LeaseDurationSeconds: ptr.To(int32(15))}}
	if err := op.Create(ctx, lease); err != nil {
		t.Fatalf("leader-election lease: %v", err)
	}
	lease.Spec.HolderIdentity = ptr.To("pod-b")
	if err := op.Update(ctx, lease); err != nil {
		t.Fatalf("leader-election lease update: %v", err)
	}
	ev := &corev1.Event{ObjectMeta: metav1.ObjectMeta{Namespace: ns, Name: "le.1"},
		InvolvedObject: corev1.ObjectReference{Kind: "Lease", Namespace: ns, Name: lease.Name}, Reason: "LeaderElection"}
	if err := op.Create(ctx, ev); err != nil {
		t.Fatalf("leader-election event: %v", err)
	}
	other := &coordinationv1.Lease{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "not-ours"}}
	if err := op.Create(ctx, other); !apierrors.IsForbidden(err) {
		t.Errorf("lease outside %s: want Forbidden, got %v", ns, err)
	}

	// The published sample, reconciled by both controllers running as the operator.
	if err := k8sClient.Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: rbacNamespace}}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "deploy", "samples", "cachecluster-persistent.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var policies []string
	for _, o := range decodeYAML(t, raw) {
		if o.GroupVersionKind().Group == snapshotGroup {
			continue
		}
		o.SetNamespace(rbacNamespace)
		if err := k8sClient.Create(ctx, o); err != nil {
			t.Fatalf("sample %s/%s: %v", o.GetKind(), o.GetName(), err)
		}
		if o.GetKind() == "TenantCachePolicy" {
			policies = append(policies, o.GetName())
		}
	}
	ccr := &CacheClusterReconciler{Client: op, APIReader: op, Scheme: k8sClient.Scheme(), Pusher: fakePush, ResyncPeriod: time.Hour}
	tpr := &TenantCachePolicyReconciler{Client: op, Scheme: k8sClient.Scheme()}
	req := func(name string) ctrl.Request {
		return ctrl.Request{NamespacedName: types.NamespacedName{Namespace: rbacNamespace, Name: name}}
	}
	for i := 0; i < 3; i++ {
		if _, err := ccr.Reconcile(ctx, req("vc")); err != nil {
			t.Fatalf("CacheCluster reconcile %d as %s: %v", i, sa, err)
		}
		for _, p := range policies {
			if _, err := tpr.Reconcile(ctx, req(p)); err != nil {
				t.Fatalf("TenantCachePolicy reconcile %d as %s: %v", i, sa, err)
			}
		}
	}

	sts := &appsv1.StatefulSet{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: rbacNamespace, Name: "vc"}, sts); err != nil {
		t.Fatalf("StatefulSet not created by the operator identity: %v", err)
	}
	if v := otelVars(sts); len(v) != 8 {
		t.Errorf("OTEL env %v", v)
	}
	if len(sts.Spec.VolumeClaimTemplates) != 1 || *sts.Spec.VolumeClaimTemplates[0].Spec.StorageClassName != "cloudvanguard-block" {
		t.Errorf("volumeClaimTemplates %+v", sts.Spec.VolumeClaimTemplates)
	}
	for _, o := range []client.Object{
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: render.ControlConfigMapName(getCluster(t, rbacNamespace, "vc"))}},
		&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: render.AdminSecretName(getCluster(t, rbacNamespace, "vc"))}},
	} {
		if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: rbacNamespace, Name: o.GetName()}, o); err != nil {
			t.Errorf("%T %s: %v", o, o.GetName(), err)
		}
	}
	svcs := &corev1.ServiceList{}
	if err := k8sClient.List(ctx, svcs, client.InNamespace(rbacNamespace)); err != nil || len(svcs.Items) != 2 {
		t.Errorf("services: %d %v", len(svcs.Items), err)
	}
	cc := getCluster(t, rbacNamespace, "vc")
	if cc.Status.ConfigVersion == 0 || meta.FindStatusCondition(cc.Status.Conditions, ConditionTracingReady) == nil {
		t.Errorf("status not written as the operator: %+v", cc.Status)
	}
	for _, p := range policies {
		if !getPolicy(t, rbacNamespace, p).Status.Applied {
			t.Errorf("policy %s not applied", p)
		}
	}

	// Deletion cleanup path as the operator: deleting the StatefulSet is allowed; the operator recreates it.
	if err := op.Delete(ctx, sts); err != nil {
		t.Fatalf("delete StatefulSet as operator: %v", err)
	}
	if _, err := ccr.Reconcile(ctx, req("vc")); err != nil {
		t.Fatalf("reconcile after delete: %v", err)
	}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: rbacNamespace, Name: "vc"}, &appsv1.StatefulSet{}); err != nil {
		t.Fatalf("StatefulSet not recreated: %v", err)
	}
}
