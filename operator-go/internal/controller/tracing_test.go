package controller

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/render"
)

func tracingSpec(endpoint, ratio string) *cachev1alpha1.TracingSpec {
	return &cachev1alpha1.TracingSpec{Enabled: true, Endpoint: endpoint, SamplerRatio: ratio}
}

func getSTS(t *testing.T, ns, name string) *appsv1.StatefulSet {
	t.Helper()
	sts := &appsv1.StatefulSet{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, sts); err != nil {
		t.Fatal(err)
	}
	return sts
}

func otelVars(sts *appsv1.StatefulSet) map[string]string {
	out := map[string]string{}
	for _, e := range sts.Spec.Template.Spec.Containers[0].Env {
		if strings.HasPrefix(e.Name, "OTEL_") {
			out[e.Name] = e.Value
		}
	}
	return out
}

func updateCluster(t *testing.T, ns, name string, mutate func(*cachev1alpha1.CacheCluster)) {
	t.Helper()
	eventually(t, "update "+name, func() error {
		cc := getCluster(t, ns, name)
		mutate(cc)
		return k8sClient.Update(ctx, cc)
	})
}

// Tracing settings reach the pod environment; the StatefulSet's pod template
// changes exactly when the rendered settings change, and never otherwise.
func TestTracingPropagatesWithoutRolloutChurn(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	cc := newCluster(ns, "tr", 3)
	cc.Spec.Tracing = tracingSpec("http://otel-collector.observability:4318", "0.10")
	cc.Spec.Tracing.Batch = &cachev1alpha1.TracingBatchSpec{MaxQueueSize: ptr.To(int32(8192))}
	if err := k8sClient.Create(ctx, cc); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector.observability:4318",
		"OTEL_TRACES_SAMPLER":         "parentbased_traceidratio",
		"OTEL_TRACES_SAMPLER_ARG":     "0.1",
		"OTEL_SERVICE_NAME":           "vanguard-proxy",
		"OTEL_BSP_MAX_QUEUE_SIZE":     "8192",
	}
	eventually(t, "OTEL_* env on the StatefulSet", func() error {
		sts := &appsv1.StatefulSet{}
		if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "tr"}, sts); err != nil {
			return err
		}
		if got := otelVars(sts); fmt.Sprint(got) != fmt.Sprint(want) {
			return fmt.Errorf("env %v, want %v", got, want)
		}
		return nil
	})
	waitConverged(t, ns, "tr")
	sts := getSTS(t, ns, "tr")
	gen, hash := sts.Generation, sts.Annotations[render.TemplateHashAnnotation]
	if hash == "" {
		t.Fatal("no pod-template-hash annotation")
	}

	// Unrelated changes do not touch the pod template: a tenant policy (new
	// control document version) and a re-spelled but equal sampler ratio.
	if err := k8sClient.Create(ctx, newPolicy(ns, "acme", "acme")); err != nil {
		t.Fatal(err)
	}
	updateCluster(t, ns, "tr", func(c *cachev1alpha1.CacheCluster) { c.Spec.Tracing.SamplerRatio = "0.1000" })
	eventually(t, "reconcile of the re-spelled ratio", func() error {
		c := getCluster(t, ns, "tr")
		if c.Status.ObservedGeneration != c.Generation {
			return fmt.Errorf("observedGeneration %d < %d", c.Status.ObservedGeneration, c.Generation)
		}
		return nil
	})
	consistently(t, "pod template unchanged", 1500*time.Millisecond, func() error {
		s := getSTS(t, ns, "tr")
		if s.Generation != gen || s.Annotations[render.TemplateHashAnnotation] != hash {
			return fmt.Errorf("StatefulSet generation %d -> %d (rollout without a real change)", gen, s.Generation)
		}
		return nil
	})

	// A real change rolls exactly once.
	updateCluster(t, ns, "tr", func(c *cachev1alpha1.CacheCluster) { c.Spec.Tracing.SamplerRatio = "0.5" })
	eventually(t, "new sampler ratio rendered", func() error {
		if v := otelVars(getSTS(t, ns, "tr"))["OTEL_TRACES_SAMPLER_ARG"]; v != "0.5" {
			return fmt.Errorf("OTEL_TRACES_SAMPLER_ARG=%q", v)
		}
		return nil
	})
	waitConverged(t, ns, "tr")
	if g := getSTS(t, ns, "tr").Generation; g != gen+1 {
		t.Fatalf("StatefulSet generation %d -> %d, want exactly one template update", gen, g)
	}

	// Disabling removes every OTEL_* variable.
	updateCluster(t, ns, "tr", func(c *cachev1alpha1.CacheCluster) { c.Spec.Tracing.Enabled = false })
	eventually(t, "OTEL_* removed", func() error {
		if v := otelVars(getSTS(t, ns, "tr")); len(v) != 0 {
			return fmt.Errorf("still set: %v", v)
		}
		return nil
	})
}

// status.tracing and TracingReady follow what the pods report.
func TestTracingStatusReflectsPodReports(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	cc := newCluster(ns, "ts", 2)
	cc.Spec.Tracing = tracingSpec("http://collector:4318", "0.25")
	if err := k8sClient.Create(ctx, cc); err != nil {
		t.Fatal(err)
	}
	waitConverged(t, ns, "ts")
	ip0, ip1 := "10.9.0.1", "10.9.0.2"
	makeReadyPod(t, ns, "ts", 0, ip0)
	makeReadyPod(t, ns, "ts", 1, ip1)
	time.Sleep(500 * time.Millisecond) // let the manager's pod-triggered reconciles settle
	ep0, ep1 := ip0+":9901", ip1+":9901"
	good := func() *controlpb.TracingStatus {
		return &controlpb.TracingStatus{Enabled: true, TracesEndpoint: "http://collector:4318/v1/traces",
			Sampler: "parentbased_traceidratio", SamplerRatio: 0.25, ServiceName: "vanguard-proxy"}
	}
	r, _ := directReconciler()
	check := func(status metav1.ConditionStatus, reason string, configured, failing int32) {
		t.Helper()
		reconcileN(t, r, ns, "ts", 1)
		c := getCluster(t, ns, "ts")
		cond := meta.FindStatusCondition(c.Status.Conditions, ConditionTracingReady)
		if cond == nil || cond.Status != status || cond.Reason != reason {
			t.Fatalf("TracingReady = %+v, want %s/%s", cond, status, reason)
		}
		tr := c.Status.Tracing
		if tr == nil || tr.ConfiguredReplicas != configured || tr.ExportFailingReplicas != failing {
			t.Fatalf("status.tracing = %+v, want configured=%d failing=%d", tr, configured, failing)
		}
	}

	// Pods that report nothing (old proxies, or not restarted yet).
	fakePush.setTracing(ep0, nil)
	fakePush.setTracing(ep1, nil)
	check(metav1.ConditionFalse, "RolloutPending", 0, 0)
	// One pod still runs the old ratio.
	fakePush.setTracing(ep0, good())
	old := good()
	old.SamplerRatio = 1
	fakePush.setTracing(ep1, old)
	check(metav1.ConditionFalse, "RolloutPending", 1, 0)
	// Both configured.
	fakePush.setTracing(ep1, good())
	check(metav1.ConditionTrue, "Exporting", 2, 0)
	c := getCluster(t, ns, "ts")
	if c.Status.Tracing.TracesEndpoint != "http://collector:4318/v1/traces" || c.Status.Tracing.SamplerRatio != "0.25" ||
		!c.Status.Tracing.Enabled {
		t.Fatalf("status.tracing = %+v", c.Status.Tracing)
	}
	// The collector goes away.
	failing := good()
	failing.LastExportFailed, failing.LastExportError, failing.ExportErrors = true, "connect: connection refused", 3
	fakePush.setTracing(ep1, failing)
	check(metav1.ConditionFalse, "ExportFailing", 2, 1)
	if e := getCluster(t, ns, "ts").Status.Tracing.LastExportError; !strings.Contains(e, "connection refused") ||
		!strings.Contains(e, "ts-1") {
		t.Fatalf("lastExportError = %q", e)
	}
	// It recovers.
	fakePush.setTracing(ep1, good())
	check(metav1.ConditionTrue, "Exporting", 2, 0)
	// Steady state: no further status writes.
	r2, counter := directReconciler()
	reconcileN(t, r2, ns, "ts", 3)
	if w := counter.Writes(); len(w) != 0 {
		t.Fatalf("steady-state reconciles wrote %v", w)
	}
	// Disabled.
	updateCluster(t, ns, "ts", func(c *cachev1alpha1.CacheCluster) { c.Spec.Tracing.Enabled = false })
	fakePush.setTracing(ep0, &controlpb.TracingStatus{})
	fakePush.setTracing(ep1, &controlpb.TracingStatus{})
	check(metav1.ConditionFalse, "Disabled", 2, 0)
}

// The API server rejects tracing settings the proxy would refuse.
func TestTracingInvalidSpecsAreRejectedByTheAPI(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	bad := map[string]*cachev1alpha1.TracingSpec{
		"no-endpoint":   {Enabled: true},
		"https":         tracingSpec("https://collector:4318", ""),
		"signal-path":   tracingSpec("http://collector:4318/v1/traces", ""),
		"ratio-high":    tracingSpec("http://collector:4318", "1.5"),
		"ratio-text":    tracingSpec("http://collector:4318", "half"),
		"ratio-precise": tracingSpec("http://collector:4318", "0.1234567"),
		"batch-over-q": {Enabled: true, Endpoint: "http://collector:4318",
			Batch: &cachev1alpha1.TracingBatchSpec{MaxExportBatchSize: ptr.To(int32(4096))}},
		"zero-delay": {Enabled: true, Endpoint: "http://collector:4318",
			Batch: &cachev1alpha1.TracingBatchSpec{ScheduleDelayMillis: ptr.To(int32(0))}},
		"long-name": {Enabled: true, Endpoint: "http://collector:4318", ServiceName: strings.Repeat("x", 129)},
	}
	for name, spec := range bad {
		cc := newCluster(ns, name, 1)
		cc.Spec.Tracing = spec
		if err := k8sClient.Create(ctx, cc); err == nil || !apierrors.IsInvalid(err) {
			t.Errorf("%s: expected validation error, got %v", name, err)
		}
	}
	// Disabled tracing needs no endpoint; an explicit larger queue allows a larger batch.
	ok := map[string]*cachev1alpha1.TracingSpec{
		"disabled": {Enabled: false},
		"big-batch": {Enabled: true, Endpoint: "http://collector:4318", Batch: &cachev1alpha1.TracingBatchSpec{
			MaxQueueSize: ptr.To(int32(8192)), MaxExportBatchSize: ptr.To(int32(4096))}},
	}
	for name, spec := range ok {
		cc := newCluster(ns, name, 1)
		cc.Spec.Tracing = spec
		if err := k8sClient.Create(ctx, cc); err != nil {
			t.Errorf("%s: rejected: %v", name, err)
		}
	}
	if c := getCluster(t, ns, "big-batch"); c.Spec.Tracing.SamplerRatio != "1" {
		t.Errorf("samplerRatio default not applied: %q", c.Spec.Tracing.SamplerRatio)
	}
}

// Defense in depth: if an invalid spec gets past the API (schema not
// installed, or relaxed later), the operator leaves the running pods alone
// and reports the problem. Uses the fake client, which applies no schema.
func TestInvalidTracingLeavesStatefulSetUnchanged(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := cachev1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	cc := newCluster("prod", "vc", 2)
	cc.UID, cc.Generation = "uid-1", 3
	cc.Spec.Tracing = &cachev1alpha1.TracingSpec{Enabled: true, Endpoint: "http://c:4318",
		Batch: &cachev1alpha1.TracingBatchSpec{MaxExportBatchSize: ptr.To(int32(4096))}} // > default queue
	running := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "vc", Namespace: "prod", ResourceVersion: "",
			CreationTimestamp: metav1.Now(), Annotations: map[string]string{render.TemplateHashAnnotation: "previous"}},
		Spec: appsv1.StatefulSetSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "vanguard-proxy", Image: "previous"}}}}},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(cc, running).
		WithStatusSubresource(&cachev1alpha1.CacheCluster{}).Build()
	r := &CacheClusterReconciler{Client: c, Scheme: scheme}
	if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "prod", Name: "vc"}}); err != nil {
		t.Fatal(err)
	}
	sts := &appsv1.StatefulSet{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: "prod", Name: "vc"}, sts); err != nil {
		t.Fatal(err)
	}
	if sts.Spec.Template.Spec.Containers[0].Image != "previous" || sts.Annotations[render.TemplateHashAnnotation] != "previous" {
		t.Fatalf("StatefulSet was modified despite an invalid spec: %+v", sts.Spec.Template.Spec.Containers[0])
	}
	got := &cachev1alpha1.CacheCluster{}
	if err := c.Get(ctx, types.NamespacedName{Namespace: "prod", Name: "vc"}, got); err != nil {
		t.Fatal(err)
	}
	cond := meta.FindStatusCondition(got.Status.Conditions, ConditionTracingReady)
	if cond == nil || cond.Reason != "InvalidConfig" || !strings.Contains(cond.Message, "maxExportBatchSize") {
		t.Fatalf("TracingReady = %+v", cond)
	}
	if got.Status.ObservedGeneration != 3 {
		t.Fatalf("observedGeneration = %d", got.Status.ObservedGeneration)
	}
}

// The published deploy/ sample passes the real API server's schema and CEL
// validation and renders its tracing settings into the pods.
func TestDeploySampleWithTracing(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "deploy", "samples", "cachecluster-persistent.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	applied := 0
	for _, chunk := range strings.Split(string(raw), "\n---\n") {
		obj := &unstructured.Unstructured{}
		if err := yaml.Unmarshal([]byte(chunk), &obj.Object); err != nil {
			t.Fatal(err)
		}
		if obj.GetKind() == "VolumeSnapshot" { // external-snapshotter CRDs are not part of envtest
			continue
		}
		obj.SetNamespace(ns)
		if err := k8sClient.Create(ctx, obj); err != nil {
			t.Fatalf("%s/%s rejected: %v", obj.GetKind(), obj.GetName(), err)
		}
		applied++
	}
	if applied != 3 {
		t.Fatalf("applied %d objects, want CacheCluster + Secret + TenantCachePolicy", applied)
	}
	eventually(t, "sample tracing env", func() error {
		sts := &appsv1.StatefulSet{}
		if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "vc"}, sts); err != nil {
			return err
		}
		v := otelVars(sts)
		if v["OTEL_EXPORTER_OTLP_ENDPOINT"] != "http://otel-collector.observability.svc.cluster.local:4318" ||
			v["OTEL_TRACES_SAMPLER_ARG"] != "0.1" || v["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] != "1024" || len(v) != 8 {
			return fmt.Errorf("OTEL env %v", v)
		}
		return nil
	})
}
