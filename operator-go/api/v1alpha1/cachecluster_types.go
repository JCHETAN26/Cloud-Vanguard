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

	// Conditions: Available, ConfigSynced.
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
