package v1alpha1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// CacheClusterSpec describes a fleet of vanguard-proxy gateways that share one
// consistent-hash ring.
type CacheClusterSpec struct {
	// Number of gateway pods (ring members).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=256
	// +kubebuilder:default=3
	Replicas int32 `json:"replicas"`

	// Primary OpenAI-compatible upstream, e.g. a vLLM service.
	// +kubebuilder:validation:Pattern=`^http://[^\s/]+(/\S*)?$`
	UpstreamPrimary string `json:"upstreamPrimary"`

	// Optional fallback upstream used when the primary's circuit is open.
	// +kubebuilder:validation:Pattern=`^http://[^\s/]+(/\S*)?$`
	// +optional
	UpstreamFallback string `json:"upstreamFallback,omitempty"`

	// Virtual nodes per ring member.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100000
	// +kubebuilder:default=1000
	// +optional
	Vnodes int32 `json:"vnodes,omitempty"`

	// Response cache capacity per pod. A pointer so that "unset" is omitted on
	// the wire and the API server applies the default (a zero Quantity value
	// would serialize as "0").
	// +kubebuilder:default="256Mi"
	// +optional
	MaxCacheBytes *resource.Quantity `json:"maxCacheBytes,omitempty"`

	// Default cache entry TTL (tenants may override via TenantCachePolicy).
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=3600
	// +optional
	DefaultTTLSeconds int32 `json:"defaultTtlSeconds,omitempty"`

	// Warm-cache persistence: each pod gets a raw block volume (CSI,
	// volumeMode Block) that vanguard-proxy snapshots its cache to, so a
	// restarted or rescheduled pod resumes with a warm cache.
	// +optional
	Persistence *PersistenceSpec `json:"persistence,omitempty"`

	// OpenTelemetry tracing: vanguard-proxy exports spans over OTLP/HTTP to a
	// collector. Rendered into the pods' OTEL_* environment, so changing it
	// rolls the pods; leaving it unchanged never does.
	// +optional
	Tracing *TracingSpec `json:"tracing,omitempty"`

	// Container image for vanguard-proxy.
	// +kubebuilder:default="ghcr.io/jchetan26/vanguard-proxy:0.3.0"
	// +optional
	Image string `json:"image,omitempty"`
}

// PersistenceSpec configures per-pod warm-cache volumes.
type PersistenceSpec struct {
	// StorageClass provisioning the volumes, e.g. one backed by csi.cloudvanguard.io.
	// +optional
	StorageClassName *string `json:"storageClassName,omitempty"`

	// Volume size per pod. Each snapshot slot gets just under half of it.
	// +kubebuilder:default="1Gi"
	// +optional
	Size *resource.Quantity `json:"size,omitempty"`

	// How often pods snapshot their cache (they also snapshot on SIGTERM).
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=60
	// +optional
	SnapshotIntervalSeconds int32 `json:"snapshotIntervalSeconds,omitempty"`
}

// TracingSpec configures span export from every gateway pod.
// +kubebuilder:validation:XValidation:rule="!self.enabled || has(self.endpoint)",message="endpoint is required when tracing is enabled"
type TracingSpec struct {
	// Export spans. When false the OTEL_* variables are not set at all.
	// +kubebuilder:default=false
	Enabled bool `json:"enabled"`

	// OTLP/HTTP base URL of the collector, e.g. http://otel-collector.observability:4318.
	// Spans are POSTed to <endpoint>/v1/traces. Plain http only: run a
	// collector (or agent) in-cluster and let it handle TLS to the backend.
	// +kubebuilder:validation:Pattern=`^http://[^\s/]+(/\S*)?$`
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:XValidation:rule="!self.endsWith('/v1/traces') && !self.endsWith('/v1/traces/')",message="give the collector base URL; /v1/traces is appended"
	// +optional
	Endpoint string `json:"endpoint,omitempty"`

	// Fraction of new traces to sample, as a decimal string in [0, 1]
	// (OTEL_TRACES_SAMPLER_ARG). Sampling is parent-based: an incoming
	// traceparent's decision always wins, so a trace is never cut in half.
	// +kubebuilder:validation:Pattern=`^(0(\.[0-9]{1,6})?|1(\.0{1,6})?)$`
	// +kubebuilder:default="1"
	// +optional
	SamplerRatio string `json:"samplerRatio,omitempty"`

	// service.name resource attribute (default vanguard-proxy). Each pod also
	// reports service.instance.id = its pod name.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	// +optional
	ServiceName string `json:"serviceName,omitempty"`

	// Batch span processor settings (OTEL_BSP_*); unset fields keep the
	// proxy defaults (5000 ms, queue 2048, batch 512, timeout 10000 ms).
	// +optional
	Batch *TracingBatchSpec `json:"batch,omitempty"`
}

// TracingBatchSpec tunes span batching and export.
// +kubebuilder:validation:XValidation:rule="(has(self.maxExportBatchSize) ? self.maxExportBatchSize : 512) <= (has(self.maxQueueSize) ? self.maxQueueSize : 2048)",message="maxExportBatchSize must not exceed maxQueueSize (defaults 512 and 2048)"
type TracingBatchSpec struct {
	// Max delay between exports (OTEL_BSP_SCHEDULE_DELAY).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=3600000
	// +optional
	ScheduleDelayMillis *int32 `json:"scheduleDelayMillis,omitempty"`

	// Ended spans buffered per pod; more are dropped and counted (OTEL_BSP_MAX_QUEUE_SIZE).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=16777216
	// +optional
	MaxQueueSize *int32 `json:"maxQueueSize,omitempty"`

	// Spans per OTLP request (OTEL_BSP_MAX_EXPORT_BATCH_SIZE).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1048576
	// +optional
	MaxExportBatchSize *int32 `json:"maxExportBatchSize,omitempty"`

	// OTLP request timeout (OTEL_EXPORTER_OTLP_TIMEOUT).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=600000
	// +optional
	ExportTimeoutMillis *int32 `json:"exportTimeoutMillis,omitempty"`
}

// TracingStatus is the tracing state reported by the ready pods.
type TracingStatus struct {
	// Tracing is enabled in the applied spec.
	Enabled bool `json:"enabled"`

	// OTLP traces URL the pods are configured to export to.
	// +optional
	TracesEndpoint string `json:"tracesEndpoint,omitempty"`

	// Sampling ratio in effect (canonical form of spec.tracing.samplerRatio).
	// +optional
	SamplerRatio string `json:"samplerRatio,omitempty"`

	// Ready pods whose proxy reports exactly the configured tracing settings.
	// +optional
	ConfiguredReplicas int32 `json:"configuredReplicas,omitempty"`

	// Ready pods whose most recent OTLP export failed.
	// +optional
	ExportFailingReplicas int32 `json:"exportFailingReplicas,omitempty"`

	// Error of the most recent failed export (one failing pod's, if several).
	// +optional
	LastExportError string `json:"lastExportError,omitempty"`
}

// CacheClusterStatus is the observed state of a CacheCluster.
type CacheClusterStatus struct {
	// Generation of the spec most recently reconciled.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Pods passing their readiness probe.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Data-plane endpoints (host:port) of ready ring members.
	// +optional
	Endpoints []string `json:"endpoints,omitempty"`

	// Version of the control document (ring + tenant policies) last rendered.
	// +optional
	ConfigVersion int64 `json:"configVersion,omitempty"`

	// Hash of the rendered control document.
	// +optional
	ConfigHash string `json:"configHash,omitempty"`

	// Ready pods confirmed to run ConfigVersion (pushed via the admin API).
	// +optional
	SyncedReplicas int32 `json:"syncedReplicas,omitempty"`

	// Tracing as reported by the pods (nil until the first reconcile).
	// +optional
	Tracing *TracingStatus `json:"tracing,omitempty"`

	// Conditions: Available, ConfigSynced, TracingReady.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.readyReplicas
// +kubebuilder:resource:shortName=cc
// +kubebuilder:printcolumn:name="Replicas",type=integer,JSONPath=`.spec.replicas`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Synced",type=integer,JSONPath=`.status.syncedReplicas`
// +kubebuilder:printcolumn:name="Config",type=integer,JSONPath=`.status.configVersion`
// +kubebuilder:printcolumn:name="Tracing",type=string,JSONPath=`.status.conditions[?(@.type=="TracingReady")].reason`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// CacheCluster is a ring of vanguard-proxy gateways.
type CacheCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CacheClusterSpec   `json:"spec,omitempty"`
	Status CacheClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// CacheClusterList contains a list of CacheCluster.
type CacheClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CacheCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CacheCluster{}, &CacheClusterList{})
}
