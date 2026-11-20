package controller

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
)

// The controller tests run both reconcilers in a real manager against a real
// kube-apiserver + etcd (envtest). There is no kubelet, scheduler or garbage
// collector: tests create Pods and set StatefulSet status themselves, and
// check ownerReferences instead of waiting for cascading deletion.

// rbacNamespace is invisible to the shared test manager (field selector on its
// cache): deploy_test.go reconciles there as the operator's ServiceAccount,
// with no admin-privileged reconciler racing it.
const rbacNamespace = "rbac-as-operator"

var (
	restCfg    *rest.Config
	k8sClient  client.Client
	fakePush   = &recordingPusher{}
	envMissing string
)

// recordingPusher stands in for the pods' admin API.
type recordingPusher struct {
	mu      sync.Mutex
	pushes  map[string][]byte // endpoint -> last document
	tokens  map[string]string
	failing map[string]bool
	tracing map[string]*controlpb.TracingStatus // endpoint -> reported tracing state
}

func (p *recordingPusher) Push(_ context.Context, endpoint, token string, doc []byte) (PushResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.failing[endpoint] {
		return PushResult{}, fmt.Errorf("connection refused")
	}
	if p.pushes == nil {
		p.pushes, p.tokens = map[string][]byte{}, map[string]string{}
	}
	p.pushes[endpoint] = append([]byte(nil), doc...)
	p.tokens[endpoint] = token
	return PushResult{ActiveVersion: versionOf(doc), Tracing: p.tracing[endpoint]}, nil
}

// setTracing makes the pod at `endpoint` report this tracing state.
func (p *recordingPusher) setTracing(endpoint string, ts *controlpb.TracingStatus) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.tracing == nil {
		p.tracing = map[string]*controlpb.TracingStatus{}
	}
	p.tracing[endpoint] = ts
}

func (p *recordingPusher) setFailing(endpoint string, fail bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.failing == nil {
		p.failing = map[string]bool{}
	}
	p.failing[endpoint] = fail
}

func (p *recordingPusher) last(endpoint string) ([]byte, string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.pushes[endpoint], p.tokens[endpoint]
}

func assetsDir() string {
	if dir := os.Getenv("KUBEBUILDER_ASSETS"); dir != "" {
		return dir
	}
	// `make envtest` installs into bin/k8s/k8s/<version>-<os>-<arch>.
	matches, _ := filepath.Glob(filepath.Join("..", "..", "bin", "k8s", "k8s", "*"))
	sort.Strings(matches)
	if len(matches) == 0 {
		return ""
	}
	return matches[len(matches)-1]
}

func TestMain(m *testing.M) {
	logf.SetLogger(zap.New(zap.WriteTo(io.Discard)))
	assets := assetsDir()
	if assets == "" {
		envMissing = "envtest binaries not found: run `make envtest` (or set KUBEBUILDER_ASSETS)"
		os.Exit(m.Run())
	}
	testEnv := &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
		BinaryAssetsDirectory: assets,
	}
	cfg, err := testEnv.Start()
	if err != nil {
		fmt.Fprintln(os.Stderr, "envtest start:", err)
		os.Exit(1)
	}
	scheme := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(cachev1alpha1.AddToScheme(scheme))

	restCfg = cfg
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{Scheme: scheme, Metrics: metricsserver.Options{BindAddress: "0"},
		Cache: cache.Options{DefaultFieldSelector: fields.OneTermNotEqualSelector("metadata.namespace", rbacNamespace)}})
	if err != nil {
		fmt.Fprintln(os.Stderr, "manager:", err)
		os.Exit(1)
	}
	if err := (&CacheClusterReconciler{Client: mgr.GetClient(), APIReader: mgr.GetAPIReader(), Scheme: scheme, Pusher: fakePush,
		ResyncPeriod: time.Hour}).SetupWithManager(mgr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := (&TenantCachePolicyReconciler{Client: mgr.GetClient(), Scheme: scheme}).SetupWithManager(mgr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		if err := mgr.Start(ctx); err != nil {
			fmt.Fprintln(os.Stderr, "manager stopped:", err)
		}
	}()
	// Assertions read straight from the API server, not the manager's cache.
	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	code := m.Run()
	cancel()
	_ = testEnv.Stop()
	os.Exit(code)
}

// --- helpers ----------------------------------------------------------------

func requireEnv(t *testing.T) {
	t.Helper()
	if envMissing != "" {
		t.Skip(envMissing)
	}
}

var nsCounter int

func newNamespace(t *testing.T) string {
	t.Helper()
	nsCounter++
	name := fmt.Sprintf("t%d-%d", nsCounter, time.Now().UnixNano()%100000)
	if err := k8sClient.Create(context.Background(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}); err != nil {
		t.Fatal(err)
	}
	return name
}

// eventually retries f until it returns nil or the timeout expires.
func eventually(t *testing.T, what string, f func() error) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		err := f()
		if err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s: %v", what, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// consistently fails if f ever returns an error during the window.
func consistently(t *testing.T, what string, window time.Duration, f func() error) {
	t.Helper()
	deadline := time.Now().Add(window)
	for time.Now().Before(deadline) {
		if err := f(); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
}
