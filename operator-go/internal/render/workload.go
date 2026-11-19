package render

import (
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
)

const mib = 1024 * 1024

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

// MutateStatefulSet shapes the gateway StatefulSet. Selector and serviceName
// are immutable and only set on creation.
func MutateStatefulSet(sts *appsv1.StatefulSet, cc *cachev1alpha1.CacheCluster) {
	labels := Labels(cc)
	sts.Labels = labels
	if sts.CreationTimestamp.IsZero() {
		sts.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
		sts.Spec.ServiceName = PeerServiceName(cc)
	}
	sts.Spec.Replicas = ptr.To(cc.Spec.Replicas)
	// Pods are independent ring members: start and stop them in parallel.
	sts.Spec.PodManagementPolicy = appsv1.ParallelPodManagement

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
	probe := func(delay int32) *corev1.Probe {
		return &corev1.Probe{
			ProbeHandler:        corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstr.FromString("http")}},
			InitialDelaySeconds: delay,
			PeriodSeconds:       5,
		}
	}
	memory := resource.NewQuantity((CacheMiB(cc)+128)*mib, resource.BinarySI)

	sts.Spec.Template.Labels = labels
	sts.Spec.Template.Spec = corev1.PodSpec{
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
			VolumeMounts: []corev1.VolumeMount{{Name: "control", MountPath: ControlMountPath, ReadOnly: true}},
		}},
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
	}
}

func secretRef(cc *cachev1alpha1.CacheCluster) *corev1.EnvVarSource {
	return &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
		LocalObjectReference: corev1.LocalObjectReference{Name: AdminSecretName(cc)},
		Key:                  AdminTokenKey,
	}}
}
