package controller

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// countingClient records every write a reconciler sends to the API server.
type countingClient struct {
	client.Client
	mu     sync.Mutex
	writes []string
}

func (c *countingClient) record(verb string, obj client.Object) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.writes = append(c.writes, fmt.Sprintf("%s %T %s", verb, obj, obj.GetName()))
}

func (c *countingClient) Writes() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.writes...)
}

func (c *countingClient) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.writes = nil
}

func (c *countingClient) Create(ctx2 context.Context, obj client.Object, opts ...client.CreateOption) error {
	c.record("create", obj)
	return c.Client.Create(ctx2, obj, opts...)
}

func (c *countingClient) Update(ctx2 context.Context, obj client.Object, opts ...client.UpdateOption) error {
	c.record("update", obj)
	return c.Client.Update(ctx2, obj, opts...)
}

func (c *countingClient) Patch(ctx2 context.Context, obj client.Object, p client.Patch, opts ...client.PatchOption) error {
	c.record("patch", obj)
	return c.Client.Patch(ctx2, obj, p, opts...)
}

func (c *countingClient) Delete(ctx2 context.Context, obj client.Object, opts ...client.DeleteOption) error {
	c.record("delete", obj)
	return c.Client.Delete(ctx2, obj, opts...)
}

func (c *countingClient) Status() client.SubResourceWriter {
	return &countingStatus{SubResourceWriter: c.Client.Status(), c: c}
}

type countingStatus struct {
	client.SubResourceWriter
	c *countingClient
}

func (s *countingStatus) Update(ctx2 context.Context, obj client.Object, opts ...client.SubResourceUpdateOption) error {
	s.c.record("status-update", obj)
	return s.SubResourceWriter.Update(ctx2, obj, opts...)
}

func (s *countingStatus) Patch(ctx2 context.Context, obj client.Object, p client.Patch, opts ...client.SubResourcePatchOption) error {
	s.c.record("status-patch", obj)
	return s.SubResourceWriter.Patch(ctx2, obj, p, opts...)
}

// directReconciler reconciles through an uncached, write-counting client, so
// a test can drive Reconcile synchronously and see exactly what it writes.
func directReconciler() (*CacheClusterReconciler, *countingClient) {
	cc := &countingClient{Client: k8sClient}
	return &CacheClusterReconciler{Client: cc, Scheme: k8sClient.Scheme(), Pusher: fakePush, ResyncPeriod: time.Hour}, cc
}

func reconcileN(t *testing.T, r *CacheClusterReconciler, ns, name string, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Namespace: ns, Name: name}}); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
	}
}

// waitConverged waits until the manager's reconciler has created everything
// and published status, so direct reconciles do not race its first pass.
func waitConverged(t *testing.T, ns, name string) {
	t.Helper()
	eventually(t, "initial reconcile of "+name, func() error {
		cc := getCluster(t, ns, name)
		if cc.Status.ConfigVersion == 0 || cc.Status.ObservedGeneration != cc.Generation {
			return fmt.Errorf("status not published yet: %+v", cc.Status)
		}
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, &appsv1.StatefulSet{})
	})
	time.Sleep(300 * time.Millisecond)
}

// A steady-state reconcile must not write anything: no StatefulSet update (no
// rollout risk, no resourceVersion churn), no Service/ConfigMap update and no
// status patch when nothing changed.
func TestReconcileIsANoOpWhenNothingChanged(t *testing.T) {
	requireEnv(t)
	ns := newNamespace(t)
	if err := k8sClient.Create(ctx, newCluster(ns, "steady", 3)); err != nil {
		t.Fatal(err)
	}
	waitConverged(t, ns, "steady")
	r, counter := directReconciler()
	reconcileN(t, r, ns, "steady", 2) // converge
	sts := &appsv1.StatefulSet{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "steady"}, sts); err != nil {
		t.Fatal(err)
	}
	rv, gen := sts.ResourceVersion, sts.Generation
	counter.Reset()
	reconcileN(t, r, ns, "steady", 5)
	if w := counter.Writes(); len(w) != 0 {
		t.Errorf("5 steady-state reconciles issued %d writes: %v", len(w), w)
	}
	if err := k8sClient.Get(ctx, types.NamespacedName{Namespace: ns, Name: "steady"}, sts); err != nil {
		t.Fatal(err)
	}
	if sts.ResourceVersion != rv || sts.Generation != gen {
		t.Errorf("StatefulSet changed: resourceVersion %s -> %s, generation %d -> %d", rv, sts.ResourceVersion, gen, sts.Generation)
	}
}
