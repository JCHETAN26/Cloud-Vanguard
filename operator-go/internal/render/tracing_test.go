package render

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

func tracedCluster() *cachev1alpha1.CacheCluster {
	cc := cluster()
	cc.Spec.Tracing = &cachev1alpha1.TracingSpec{
		Enabled:      true,
		Endpoint:     "http://otel-collector.observability:4318/",
		SamplerRatio: "0.250",
		ServiceName:  "vanguard-prod",
		Batch: &cachev1alpha1.TracingBatchSpec{
			ScheduleDelayMillis: ptr.To(int32(2000)),
			MaxQueueSize:        ptr.To(int32(8192)),
			MaxExportBatchSize:  ptr.To(int32(1024)),
			ExportTimeoutMillis: ptr.To(int32(5000)),
		},
	}
	return cc
}

func otelEnv(env []corev1.EnvVar) []corev1.EnvVar {
	var out []corev1.EnvVar
	for _, e := range env {
		if strings.HasPrefix(e.Name, "OTEL_") {
			out = append(out, e)
		}
	}
	return out
}

func TestResolveTracingCanonicalForm(t *testing.T) {
	tr, err := ResolveTracing(tracedCluster())
	if err != nil {
		t.Fatal(err)
	}
	if tr.Endpoint != "http://otel-collector.observability:4318" || tr.TracesURL != tr.Endpoint+"/v1/traces" ||
		tr.Ratio != 0.25 || tr.RatioText != "0.25" || tr.ServiceName != "vanguard-prod" {
		t.Fatalf("unexpected canonical form: %+v", tr)
	}
	want := []string{
		"OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.observability:4318",
		"OTEL_TRACES_SAMPLER=parentbased_traceidratio",
		"OTEL_TRACES_SAMPLER_ARG=0.25",
		"OTEL_SERVICE_NAME=vanguard-prod",
		"OTEL_BSP_SCHEDULE_DELAY=2000",
		"OTEL_BSP_MAX_QUEUE_SIZE=8192",
		"OTEL_BSP_MAX_EXPORT_BATCH_SIZE=1024",
		"OTEL_EXPORTER_OTLP_TIMEOUT=5000",
	}
	env := tr.Env()
	if len(env) != len(want) {
		t.Fatalf("env = %v", env)
	}
	for i, e := range env {
		if e.Name+"="+e.Value != want[i] {
			t.Errorf("env[%d] = %s=%s, want %s", i, e.Name, e.Value, want[i])
		}
	}

	// Defaults: ratio 1, service name vanguard-proxy, no batch variables.
	cc := cluster()
	cc.Spec.Tracing = &cachev1alpha1.TracingSpec{Enabled: true, Endpoint: "http://c:4318"}
	tr, err = ResolveTracing(cc)
	if err != nil || tr.RatioText != "1" || tr.ServiceName != DefaultTracingServiceName || len(tr.Env()) != 4 {
		t.Fatalf("defaults: %+v %v", tr, err)
	}
	// Disabled (or absent): no OTEL_* variables at all.
	for _, spec := range []*cachev1alpha1.TracingSpec{nil, {Enabled: false, Endpoint: "http://c:4318"}} {
		cc.Spec.Tracing = spec
		tr, err = ResolveTracing(cc)
		if err != nil || tr.Enabled || tr.Env() != nil {
			t.Fatalf("disabled %+v: %+v %v", spec, tr, err)
		}
	}
}

func TestResolveTracingRejectsWhatTheProxyWouldRefuse(t *testing.T) {
	cases := map[string]func(*cachev1alpha1.TracingSpec){
		"no endpoint":        func(s *cachev1alpha1.TracingSpec) { s.Endpoint = "" },
		"https endpoint":     func(s *cachev1alpha1.TracingSpec) { s.Endpoint = "https://c:4318" },
		"signal path":        func(s *cachev1alpha1.TracingSpec) { s.Endpoint = "http://c:4318/v1/traces" },
		"ratio > 1":          func(s *cachev1alpha1.TracingSpec) { s.SamplerRatio = "1.5" },
		"ratio not a number": func(s *cachev1alpha1.TracingSpec) { s.SamplerRatio = "half" },
		"negative ratio":     func(s *cachev1alpha1.TracingSpec) { s.SamplerRatio = "-0.1" },
		"service name 129":   func(s *cachev1alpha1.TracingSpec) { s.ServiceName = strings.Repeat("x", 129) },
		"batch > queue":      func(s *cachev1alpha1.TracingSpec) { s.Batch.MaxExportBatchSize = ptr.To(int32(9000)) },
		"batch > default q": func(s *cachev1alpha1.TracingSpec) {
			s.Batch.MaxQueueSize = nil
			s.Batch.MaxExportBatchSize = ptr.To(int32(4096))
		},
		"zero schedule delay": func(s *cachev1alpha1.TracingSpec) { s.Batch.ScheduleDelayMillis = ptr.To(int32(0)) },
		"timeout too large":   func(s *cachev1alpha1.TracingSpec) { s.Batch.ExportTimeoutMillis = ptr.To(int32(600001)) },
	}
	for name, mutate := range cases {
		cc := tracedCluster()
		mutate(cc.Spec.Tracing)
		if _, err := ResolveTracing(cc); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestTemplateHashTracksOnlyRenderedChanges(t *testing.T) {
	cc := tracedCluster()
	tr, _ := ResolveTracing(cc)
	base := TemplateHash(PodTemplate(cc, tr))
	if base != TemplateHash(PodTemplate(cc, tr)) {
		t.Fatal("hash is not deterministic")
	}
	// Same value, different spelling: same pods.
	cc.Spec.Tracing.SamplerRatio = "0.25"
	tr2, _ := ResolveTracing(cc)
	if TemplateHash(PodTemplate(cc, tr2)) != base {
		t.Fatal("reformatting samplerRatio changed the pod template")
	}
	// A real change: new template.
	cc.Spec.Tracing.SamplerRatio = "0.5"
	tr3, _ := ResolveTracing(cc)
	if TemplateHash(PodTemplate(cc, tr3)) == base {
		t.Fatal("changing samplerRatio did not change the pod template")
	}
	// Replicas are not part of the template.
	cc.Spec.Tracing.SamplerRatio = "0.25"
	cc.Spec.Replicas = 7
	if TemplateHash(PodTemplate(cc, tr2)) != base {
		t.Fatal("replicas changed the pod template hash")
	}
}

func TestTracingMatchesPodReports(t *testing.T) {
	tr, _ := ResolveTracing(tracedCluster())
	good := &controlpb.TracingStatus{Enabled: true, TracesEndpoint: "http://otel-collector.observability:4318/v1/traces",
		Sampler: "parentbased_traceidratio", SamplerRatio: 0.25, ServiceName: "vanguard-prod"}
	if !tr.Matches(good) {
		t.Fatal("matching report rejected")
	}
	for name, mutate := range map[string]func(*controlpb.TracingStatus){
		"disabled":     func(s *controlpb.TracingStatus) { s.Enabled = false },
		"old endpoint": func(s *controlpb.TracingStatus) { s.TracesEndpoint = "http://old:4318/v1/traces" },
		"old ratio":    func(s *controlpb.TracingStatus) { s.SamplerRatio = 1 },
		"sampler":      func(s *controlpb.TracingStatus) { s.Sampler = "always_on" },
		"service":      func(s *controlpb.TracingStatus) { s.ServiceName = "vanguard-proxy" },
	} {
		s := protoClone(good)
		mutate(s)
		if tr.Matches(s) {
			t.Errorf("%s: mismatching report accepted", name)
		}
	}
	if tr.Matches(nil) {
		t.Error("a pod that reports nothing matched enabled tracing")
	}
	var off Tracing
	if !off.Matches(nil) || !off.Matches(&controlpb.TracingStatus{}) || off.Matches(good) {
		t.Error("disabled tracing: wrong match result")
	}
}

func protoClone(s *controlpb.TracingStatus) *controlpb.TracingStatus {
	return proto.Clone(s).(*controlpb.TracingStatus)
}

// Golden pod environment with every tracing field set. It is the contract
// with the C++ data plane: proxy-cpp's OperatorEnvGolden test loads exactly
// these variables through vanguard-proxy's config loader.
func TestGoldenTracingEnv(t *testing.T) {
	cc := tracedCluster()
	tr, err := ResolveTracing(cc)
	if err != nil {
		t.Fatal(err)
	}
	type kv struct {
		Name  string `json:"name"`
		Value string `json:"value"`
	}
	var vars []kv
	for _, e := range PodTemplate(cc, tr).Spec.Containers[0].Env {
		if e.ValueFrom == nil { // NODE_ID and tokens come from the downward API / Secret
			vars = append(vars, kv{e.Name, e.Value})
		}
	}
	got, err := json.MarshalIndent(vars, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join("testdata", "tracing_env_golden.json")
	if *update {
		if err := os.WriteFile(path, append(got, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test ./internal/render -update)", err)
	}
	if string(want) != string(got)+"\n" {
		t.Fatalf("rendered env differs from %s:\n%s", path, got)
	}
	if len(otelEnv(PodTemplate(cc, tr).Spec.Containers[0].Env)) != 8 {
		t.Fatal("expected 8 OTEL_* variables")
	}
}

// The proxy's ApplyResult (golden file written by proxy-cpp's tests from its
// real serializer) must parse strictly as vanguard.control.v1.ApplyResult and
// match the configuration that produced it.
func TestGoldenApplyResultFromProxy(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "proto", "vanguard", "control", "v1", "testdata",
		"apply_result_tracing.golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	res := &controlpb.ApplyResult{}
	if err := (protojson.UnmarshalOptions{DiscardUnknown: false}).Unmarshal(b, res); err != nil {
		t.Fatalf("proxy ApplyResult is not valid vanguard.control.v1.ApplyResult JSON: %v", err)
	}
	ts := res.GetTracing()
	if res.GetVersion() != 42 || res.GetNodeId() != "vc-0" || !ts.GetEnabled() || ts.GetSampler() != TracingSampler {
		t.Fatalf("unexpected ApplyResult: %v", res)
	}
	cc := cluster()
	cc.Spec.Tracing = &cachev1alpha1.TracingSpec{Enabled: true, Endpoint: "http://otel-collector.observability:4318",
		SamplerRatio: "0.25", ServiceName: "vanguard-prod"}
	tr, err := ResolveTracing(cc)
	if err != nil {
		t.Fatal(err)
	}
	if !tr.Matches(ts) {
		t.Fatalf("operator does not recognise the proxy's report of its own settings: %v vs %+v", ts, tr)
	}
}
