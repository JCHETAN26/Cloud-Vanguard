package render

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
)

const (
	mib = 1024 * 1024
	// WarmCacheVolume is the volumeClaimTemplate name; WarmCacheDevicePath is
	// where the raw block device appears in the container.
	WarmCacheVolume     = "warm-cache"
	WarmCacheDevicePath = "/dev/vanguard-cache"
	// DiskGroupID is the "disk" group (gid 6 on Debian, Ubuntu, RHEL and Fedora). The container runtime
	// passes a raw block volume through with the host node's ownership, root:disk 0660, and does not
	// apply fsGroup to devices, so the non-root proxy (uid 10001) needs this supplemental group to open
	// the warm-cache device. Found on the first kind run: "open /dev/vanguard-cache: Permission denied".
	DiskGroupID int64 = 6
	// TemplateHashAnnotation (on the StatefulSet) records the hash of the pod
	// template the operator last rendered. The template is only rewritten when
	// that hash changes, so steady-state reconciles send no StatefulSet update.
	TemplateHashAnnotation = "cache.cloudvanguard.io/pod-template-hash"
)

// CacheMiB converts spec.maxCacheBytes to the proxy's CACHE_MAX_MB (rounded up, min 1).
func CacheMiB(cc *cachev1alpha1.CacheCluster) int64 {
	if cc.Spec.MaxCacheBytes == nil {
		return 256
	}
	b := cc.Spec.MaxCacheBytes.Value()
	if b <= 0 {
		return 256
	}
	return (b + mib - 1) / mib
}

func servicePorts() []corev1.ServicePort {
	return []corev1.ServicePort{
		{Name: "http", Port: DataPort, TargetPort: intstr.FromString("http"), Protocol: corev1.ProtocolTCP},
		{Name: "admin", Port: AdminPort, TargetPort: intstr.FromString("admin"), Protocol: corev1.ProtocolTCP},
	}
}

// MutatePeerService shapes the headless Service that gives every pod a stable
// DNS name (<pod>.<name>-peers...). Not-ready pods are published too: ring
// membership is the StatefulSet's ordinals, and a peer that is not ready yet
// is handled by the proxy's local fallback rather than by reshuffling keys.
func MutatePeerService(svc *corev1.Service, cc *cachev1alpha1.CacheCluster) {
	svc.Labels = Labels(cc)
	svc.Spec.ClusterIP = corev1.ClusterIPNone
	svc.Spec.PublishNotReadyAddresses = true
	svc.Spec.Selector = Labels(cc)
	svc.Spec.Ports = servicePorts()
}

// MutateClientService shapes the ClusterIP Service tenants call.
func MutateClientService(svc *corev1.Service, cc *cachev1alpha1.CacheCluster) {
	svc.Labels = Labels(cc)
	svc.Spec.Selector = Labels(cc)
	svc.Spec.Ports = servicePorts()[:1]
}

// MutateStatefulSet shapes the gateway StatefulSet. Selector, serviceName and
// volumeClaimTemplates are immutable and only set on creation. The pod
// template is replaced only when its rendered hash differs from the one
// recorded on the StatefulSet: the API server fills in defaults (probe
// timeouts, DNS policy, ...), so comparing our sparse template with the stored
// one would always differ and send an Update on every reconcile.
//
// Trade-off: a manual edit of the pod template is not reverted until the
// rendered template itself changes.
func MutateStatefulSet(sts *appsv1.StatefulSet, cc *cachev1alpha1.CacheCluster, tracing Tracing) {
	labels := Labels(cc)
	sts.Labels = labels
	if sts.CreationTimestamp.IsZero() {
		sts.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
		sts.Spec.ServiceName = PeerServiceName(cc)
		// Pods are independent ring members: start and stop them in parallel.
		sts.Spec.PodManagementPolicy = appsv1.ParallelPodManagement
	}
	sts.Spec.Replicas = ptr.To(cc.Spec.Replicas)

	if p := cc.Spec.Persistence; p != nil && sts.CreationTimestamp.IsZero() {
		size := resource.MustParse("1Gi")
		if p.Size != nil {
			size = *p.Size
		}
		sts.Spec.VolumeClaimTemplates = []corev1.PersistentVolumeClaim{{
			ObjectMeta: metav1.ObjectMeta{Name: WarmCacheVolume, Labels: labels},
			Spec: corev1.PersistentVolumeClaimSpec{
				AccessModes:      []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
				VolumeMode:       ptr.To(corev1.PersistentVolumeBlock),
				StorageClassName: p.StorageClassName,
				Resources:        corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: size}},
			},
		}}
		// Keep warm caches when pods are scaled away or the cluster is deleted.
		sts.Spec.PersistentVolumeClaimRetentionPolicy = &appsv1.StatefulSetPersistentVolumeClaimRetentionPolicy{
			WhenDeleted: appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
			WhenScaled:  appsv1.RetainPersistentVolumeClaimRetentionPolicyType,
		}
	}

	template := PodTemplate(cc, tracing)
	hash := TemplateHash(template)
	if sts.CreationTimestamp.IsZero() || sts.Annotations[TemplateHashAnnotation] != hash {
		sts.Spec.Template = template
		if sts.Annotations == nil {
			sts.Annotations = map[string]string{}
		}
		sts.Annotations[TemplateHashAnnotation] = hash
	}
}

// TemplateHash is a stable digest of a rendered pod template.
func TemplateHash(t corev1.PodTemplateSpec) string {
	b, err := json.Marshal(t) // struct fields in declaration order, map keys sorted
	if err != nil {
		panic(err) // a PodTemplateSpec always marshals
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:8])
}

// PodTemplate renders the gateway pod template.
func PodTemplate(cc *cachev1alpha1.CacheCluster, tracing Tracing) corev1.PodTemplateSpec {
	labels := Labels(cc)
	vnodes := cc.Spec.Vnodes
	if vnodes <= 0 {
		vnodes = DefaultVnodes
	}
	env := []corev1.EnvVar{
		{Name: "NODE_ID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
		{Name: "LISTEN_PORT", Value: strconv.Itoa(DataPort)},
		{Name: "ADMIN_ADDR", Value: "0.0.0.0"},
		{Name: "ADMIN_PORT", Value: strconv.Itoa(AdminPort)},
		{Name: "ADMIN_TOKEN", ValueFrom: secretRef(cc)},
		{Name: "PEER_TOKEN", ValueFrom: secretRef(cc)},
		{Name: "UPSTREAM_URL", Value: cc.Spec.UpstreamPrimary},
		{Name: "RING_VNODES", Value: strconv.Itoa(int(vnodes))},
		{Name: "CACHE_MAX_MB", Value: strconv.FormatInt(CacheMiB(cc), 10)},
		{Name: "CACHE_TTL_S", Value: strconv.Itoa(int(cc.Spec.DefaultTTLSeconds))},
		{Name: "CONTROL_FILE", Value: ControlMountPath + "/" + ControlKey},
	}
	if cc.Spec.UpstreamFallback != "" {
		env = append(env, corev1.EnvVar{Name: "FALLBACK_UPSTREAM_URL", Value: cc.Spec.UpstreamFallback})
	}
	var devices []corev1.VolumeDevice
	var podSecurity *corev1.PodSecurityContext
	if p := cc.Spec.Persistence; p != nil {
		podSecurity = &corev1.PodSecurityContext{SupplementalGroups: []int64{DiskGroupID}}
		env = append(env,
			corev1.EnvVar{Name: "SNAPSHOT_PATH", Value: WarmCacheDevicePath},
			corev1.EnvVar{Name: "SNAPSHOT_MODE", Value: "block"},
			corev1.EnvVar{Name: "SNAPSHOT_INTERVAL_S", Value: strconv.Itoa(int(p.SnapshotIntervalSeconds))})
		devices = []corev1.VolumeDevice{{Name: WarmCacheVolume, DevicePath: WarmCacheDevicePath}}
	}
	env = append(env, tracing.Env()...)
	probe := func(delay int32) *corev1.Probe {
		return &corev1.Probe{
			ProbeHandler:        corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstr.FromString("http")}},
			InitialDelaySeconds: delay,
			PeriodSeconds:       5,
		}
	}
	memory := resource.NewQuantity((CacheMiB(cc)+128)*mib, resource.BinarySI)

	return corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: labels}, Spec: corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:  "vanguard-proxy",
			Image: cc.Spec.Image,
			Ports: []corev1.ContainerPort{
				{Name: "http", ContainerPort: DataPort, Protocol: corev1.ProtocolTCP},
				{Name: "admin", ContainerPort: AdminPort, Protocol: corev1.ProtocolTCP},
			},
			Env:            env,
			ReadinessProbe: probe(1),
			LivenessProbe:  probe(10),
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("250m"), corev1.ResourceMemory: *memory},
			},
			VolumeMounts:  []corev1.VolumeMount{{Name: "control", MountPath: ControlMountPath, ReadOnly: true}},
			VolumeDevices: devices,
		}},
		// Room for the SIGTERM snapshot to complete.
		SecurityContext:               podSecurity,
		TerminationGracePeriodSeconds: ptr.To(int64(60)),
		Volumes: []corev1.Volume{{
			Name: "control",
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: ControlConfigMapName(cc)},
				Optional:             ptr.To(true),
			}},
		}},
		// Spread ring members across nodes so one machine failure loses one shard.
		Affinity: &corev1.Affinity{PodAntiAffinity: &corev1.PodAntiAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []corev1.WeightedPodAffinityTerm{{
				Weight: 100,
				PodAffinityTerm: corev1.PodAffinityTerm{
					LabelSelector: &metav1.LabelSelector{MatchLabels: labels},
					TopologyKey:   "kubernetes.io/hostname",
				},
			}},
		}},
	}}
}

func secretRef(cc *cachev1alpha1.CacheCluster) *corev1.EnvVarSource {
	return &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
		LocalObjectReference: corev1.LocalObjectReference{Name: AdminSecretName(cc)},
		Key:                  AdminTokenKey,
	}}
}
