package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SecretKeyRef points at a key in a Secret in the policy's namespace.
type SecretKeyRef struct {
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`
	// Key holding one API key per line.
	// +kubebuilder:default="api-keys"
	// +optional
	Key string `json:"key,omitempty"`
}

// TenantCachePolicySpec is the desired configuration of one tenant on the gateway.
type TenantCachePolicySpec struct {
	// Tenant identifier used for rate limiting, cache isolation and metrics.
	// +kubebuilder:validation:Pattern=`^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$`
	TenantID string `json:"tenantId"`

	// CacheCluster this policy applies to. Empty = every CacheCluster in the namespace.
	// +optional
	ClusterRef string `json:"clusterRef,omitempty"`

	// Secret with the tenant's API keys. Only SHA-256 hashes of the keys are
	// distributed to the gateways.
	// +optional
	APIKeysSecretRef *SecretKeyRef `json:"apiKeysSecretRef,omitempty"`

	// Cluster-wide requests per minute (0 = unlimited). Each of N gateways enforces 1/N.
	// +kubebuilder:validation:Minimum=0
	// +optional
	RPMLimit int64 `json:"rpmLimit,omitempty"`

	// Cluster-wide estimated tokens per minute sent upstream (0 = unlimited).
	// +kubebuilder:validation:Minimum=0
	// +optional
	TPMLimit int64 `json:"tpmLimit,omitempty"`

	// Semantic cache mode for this tenant (default: the gateway's).
	// +kubebuilder:validation:Enum=lexical;neural;off
	// +optional
	SemanticMode string `json:"semanticMode,omitempty"`

	// Minimum cosine similarity for a semantic hit, as a decimal string in
	// (0, 1], e.g. "0.96". Default: 0.94 lexical, 0.96 neural.
	// +kubebuilder:validation:Pattern=`^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$`
	// +optional
	SemanticThreshold string `json:"semanticThreshold,omitempty"`

	// Cache TTL for this tenant's entries (default: the cluster's defaultTtlSeconds).
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=2592000
	// +optional
	TTLSeconds *int32 `json:"ttlSeconds,omitempty"`
}

// TenantCachePolicyStatus reports whether the policy is live on the gateways.
type TenantCachePolicyStatus struct {
	// True once ObservedGeneration is rendered into every target cluster's control document.
	// +optional
	Applied bool `json:"applied"`

	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Clusters currently serving this policy.
	// +optional
	AppliedClusters []string `json:"appliedClusters,omitempty"`

	// Conditions: Applied, Conflict.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=tcp
// +kubebuilder:printcolumn:name="Tenant",type=string,JSONPath=`.spec.tenantId`
// +kubebuilder:printcolumn:name="RPM",type=integer,JSONPath=`.spec.rpmLimit`
// +kubebuilder:printcolumn:name="TPM",type=integer,JSONPath=`.spec.tpmLimit`
// +kubebuilder:printcolumn:name="Semantic",type=string,JSONPath=`.spec.semanticMode`
// +kubebuilder:printcolumn:name="Applied",type=boolean,JSONPath=`.status.applied`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// TenantCachePolicy configures limits and caching for one tenant.
type TenantCachePolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TenantCachePolicySpec   `json:"spec,omitempty"`
	Status TenantCachePolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TenantCachePolicyList contains a list of TenantCachePolicy.
type TenantCachePolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TenantCachePolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TenantCachePolicy{}, &TenantCachePolicyList{})
}
