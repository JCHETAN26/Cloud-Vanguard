package render

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
)

const (
	// DefaultTracingServiceName matches vanguard-proxy's own default.
	DefaultTracingServiceName = "vanguard-proxy"
	// TracingSampler is always parent-based: a caller's sampling decision wins,
	// so traces that enter the gateway are never cut in half.
	TracingSampler = "parentbased_traceidratio"

	// Proxy defaults for the batch span processor (proxy-cpp/include/vanguard/config.hpp).
	defaultMaxQueueSize       = 2048
	defaultMaxExportBatchSize = 512
)

// Same shape the CRD enforces (spec.tracing.endpoint / samplerRatio). The
// operator re-validates because a CRD installed without the current schema
// (or a future relaxation) must not turn into a crash-looping proxy.
var (
	endpointPattern = regexp.MustCompile(`^http://[^\s/]+(/\S*)?$`)
	ratioPattern    = regexp.MustCompile(`^(0(\.[0-9]{1,6})?|1(\.0{1,6})?)$`)
)

// Tracing is the validated, canonical tracing configuration of a cluster: the
// single source for the pods' OTEL_* environment, the status, and the
// comparison with what each pod reports.
type Tracing struct {
	Enabled     bool
	Endpoint    string  // collector base URL, no trailing slash
	TracesURL   string  // Endpoint + "/v1/traces" (what the proxy reports)
	Ratio       float64 // sampling ratio
	RatioText   string  // canonical decimal form, e.g. "0.1"
	ServiceName string
	Batch       cachev1alpha1.TracingBatchSpec
}

// ResolveTracing validates spec.tracing and returns its canonical form.
func ResolveTracing(cc *cachev1alpha1.CacheCluster) (Tracing, error) {
	spec := cc.Spec.Tracing
	if spec == nil || !spec.Enabled {
		return Tracing{}, nil
	}
	t := Tracing{Enabled: true}
	endpoint := strings.TrimRight(spec.Endpoint, "/")
	switch {
	case endpoint == "":
		return Tracing{}, fmt.Errorf("spec.tracing.endpoint is required when tracing is enabled")
	case !endpointPattern.MatchString(endpoint):
		return Tracing{}, fmt.Errorf("spec.tracing.endpoint %q must be an http:// URL", spec.Endpoint)
	case strings.HasSuffix(endpoint, "/v1/traces"):
		return Tracing{}, fmt.Errorf("spec.tracing.endpoint %q must be the collector base URL (/v1/traces is appended)", spec.Endpoint)
	}
	t.Endpoint = endpoint
	t.TracesURL = endpoint + "/v1/traces"

	ratio := spec.SamplerRatio
	if ratio == "" {
		ratio = "1"
	}
	if !ratioPattern.MatchString(ratio) {
		return Tracing{}, fmt.Errorf("spec.tracing.samplerRatio %q must be a decimal in [0, 1]", spec.SamplerRatio)
	}
	t.Ratio, _ = strconv.ParseFloat(ratio, 64)
	// Canonical text: "0.10" and "0.1" render the same env, so reformatting the
	// spec does not roll the pods.
	t.RatioText = strconv.FormatFloat(t.Ratio, 'f', -1, 64)

	t.ServiceName = spec.ServiceName
	if t.ServiceName == "" {
		t.ServiceName = DefaultTracingServiceName
	}
	if len(t.ServiceName) > 128 {
		return Tracing{}, fmt.Errorf("spec.tracing.serviceName is longer than 128 characters")
	}
	if b := spec.Batch; b != nil {
		check := func(name string, v *int32, max int32) error {
			if v != nil && (*v < 1 || *v > max) {
				return fmt.Errorf("spec.tracing.batch.%s must be in [1, %d]", name, max)
			}
			return nil
		}
		for _, err := range []error{
			check("scheduleDelayMillis", b.ScheduleDelayMillis, 3600000),
			check("maxQueueSize", b.MaxQueueSize, 16777216),
			check("maxExportBatchSize", b.MaxExportBatchSize, 1048576),
			check("exportTimeoutMillis", b.ExportTimeoutMillis, 600000),
		} {
			if err != nil {
				return Tracing{}, err
			}
		}
		// The proxy refuses to start when batch > queue: catch it here instead
		// of rolling out crash-looping pods.
		queue, batch := int32(defaultMaxQueueSize), int32(defaultMaxExportBatchSize)
		if b.MaxQueueSize != nil {
			queue = *b.MaxQueueSize
		}
		if b.MaxExportBatchSize != nil {
			batch = *b.MaxExportBatchSize
		}
		if batch > queue {
			return Tracing{}, fmt.Errorf("spec.tracing.batch.maxExportBatchSize (%d) exceeds maxQueueSize (%d)", batch, queue)
		}
		t.Batch = *b
	}
	return t, nil
}

// Env is the vanguard-proxy environment for this configuration, in a fixed
// order. Disabled tracing sets nothing (the proxy's default is off).
func (t Tracing) Env() []corev1.EnvVar {
	if !t.Enabled {
		return nil
	}
	env := []corev1.EnvVar{
		{Name: "OTEL_EXPORTER_OTLP_ENDPOINT", Value: t.Endpoint},
		{Name: "OTEL_TRACES_SAMPLER", Value: TracingSampler},
		{Name: "OTEL_TRACES_SAMPLER_ARG", Value: t.RatioText},
		{Name: "OTEL_SERVICE_NAME", Value: t.ServiceName},
	}
	add := func(name string, v *int32) {
		if v != nil {
			env = append(env, corev1.EnvVar{Name: name, Value: strconv.Itoa(int(*v))})
		}
	}
	add("OTEL_BSP_SCHEDULE_DELAY", t.Batch.ScheduleDelayMillis)
	add("OTEL_BSP_MAX_QUEUE_SIZE", t.Batch.MaxQueueSize)
	add("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", t.Batch.MaxExportBatchSize)
	add("OTEL_EXPORTER_OTLP_TIMEOUT", t.Batch.ExportTimeoutMillis)
	return env
}

// Matches reports whether a pod's reported tracing state is this
// configuration. A pod that reports nothing (a proxy that predates tracing)
// matches only "disabled".
func (t Tracing) Matches(s *controlpb.TracingStatus) bool {
	if s == nil || !s.GetEnabled() {
		return !t.Enabled
	}
	return t.Enabled &&
		s.GetTracesEndpoint() == t.TracesURL &&
		s.GetSampler() == TracingSampler &&
		math.Abs(s.GetSamplerRatio()-t.Ratio) < 1e-9 &&
		s.GetServiceName() == t.ServiceName
}
