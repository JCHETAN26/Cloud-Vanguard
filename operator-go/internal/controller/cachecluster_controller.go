package controller

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/render"
)

const (
	ConditionAvailable    = "Available"
	ConditionConfigSynced = "ConfigSynced"
)

// CacheClusterReconciler runs a ring of vanguard-proxy pods and keeps every
// pod's ring topology and tenant policies in sync with the API.
//
// It is the single writer of each cluster's control document: it renders the
// ring (from spec.replicas) and the winning TenantCachePolicies into the
// <name>-control ConfigMap (durable; mounted into pods and polled), then
// pushes the same document to every ready pod's admin API (fast path).
type CacheClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
	Pusher Pusher
	// ResyncPeriod re-pushes periodically to repair drift (e.g. a pod that
	// restarted and has not re-read its ConfigMap yet).
	ResyncPeriod time.Duration
}

// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=cacheclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=cacheclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=cacheclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=tenantcachepolicies,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services;configmaps;secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch

func (r *CacheClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	cc := &cachev1alpha1.CacheCluster{}
	if err := r.Get(ctx, req.NamespacedName, cc); err != nil {
		// Deleted: owned objects are garbage-collected through ownerReferences.
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !cc.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	token, err := r.ensureAdminSecret(ctx, cc)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Render the control document; bump the version only if content changed.
	policies, err := r.resolvePolicies(ctx, cc)
	if err != nil {
		return ctrl.Result{}, err
	}
	// The ConfigMap carries the hash/version it was rendered with; reading it
	// (rather than only our status, which may lag in the cache) keeps the
	// version from being bumped twice for one change.
	doc := render.ControlDocument(cc, policies, 0)
	hash := render.Hash(doc)
	version := r.nextVersion(ctx, cc, hash)
	doc = render.ControlDocument(cc, policies, version)
	docJSON, err := render.JSON(doc)
	if err != nil {
		return ctrl.Result{}, err
	}

	if err := r.apply(ctx, cc, &corev1.ConfigMap{ObjectMeta: r.meta(cc, render.ControlConfigMapName(cc))}, func(o client.Object) {
		cm := o.(*corev1.ConfigMap)
		cm.Labels = render.Labels(cc)
		cm.Annotations = map[string]string{render.HashAnnotation: hash, render.VersionAnnotation: strconv.FormatInt(version, 10)}
		cm.Data = map[string]string{render.ControlKey: string(docJSON)}
	}); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.apply(ctx, cc, &corev1.Service{ObjectMeta: r.meta(cc, render.PeerServiceName(cc))}, func(o client.Object) {
		render.MutatePeerService(o.(*corev1.Service), cc)
	}); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.apply(ctx, cc, &corev1.Service{ObjectMeta: r.meta(cc, render.ClientServiceName(cc))}, func(o client.Object) {
		render.MutateClientService(o.(*corev1.Service), cc)
	}); err != nil {
		return ctrl.Result{}, err
	}
	sts := &appsv1.StatefulSet{ObjectMeta: r.meta(cc, render.StatefulSetName(cc))}
	if err := r.apply(ctx, cc, sts, func(o client.Object) {
		render.MutateStatefulSet(o.(*appsv1.StatefulSet), cc)
	}); err != nil {
		return ctrl.Result{}, err
	}

	// Push to ready pods.
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(cc.Namespace), client.MatchingLabels(render.Labels(cc))); err != nil {
		return ctrl.Result{}, err
	}
	var endpoints []string
	var synced int32
	var pushErr error
	for i := range pods.Items {
		pod := &pods.Items[i]
		if !podReady(pod) || pod.Status.PodIP == "" || !pod.DeletionTimestamp.IsZero() {
			continue
		}
		endpoints = append(endpoints, fmt.Sprintf("%s.%s.%s.svc.cluster.local:%d", pod.Name,
			render.PeerServiceName(cc), cc.Namespace, render.DataPort))
		if r.Pusher == nil {
			continue
		}
		active, err := r.Pusher.Push(ctx, fmt.Sprintf("%s:%d", pod.Status.PodIP, render.AdminPort), token, docJSON)
		if err != nil {
			log.Info("config push failed", "pod", pod.Name, "error", err.Error())
			pushErr = err
			continue
		}
		if active >= version {
			synced++
		}
	}
	sort.Strings(endpoints)

	// Status.
	patch := client.MergeFrom(cc.DeepCopy())
	cc.Status.ObservedGeneration = cc.Generation
	cc.Status.ReadyReplicas = sts.Status.ReadyReplicas
	cc.Status.Endpoints = endpoints
	cc.Status.ConfigVersion = version
	cc.Status.ConfigHash = hash
	cc.Status.SyncedReplicas = synced
	setCondition(&cc.Status.Conditions, cc.Generation, ConditionAvailable,
		cc.Status.ReadyReplicas >= cc.Spec.Replicas, "AllReplicasReady", "ReplicasNotReady",
		fmt.Sprintf("%d/%d ring members ready", cc.Status.ReadyReplicas, cc.Spec.Replicas))
	syncedOK := pushErr == nil && synced == int32(len(endpoints))
	msg := fmt.Sprintf("config version %d live on %d/%d ready pods", version, synced, len(endpoints))
	if pushErr != nil {
		msg += ": " + pushErr.Error()
	}
	setCondition(&cc.Status.Conditions, cc.Generation, ConditionConfigSynced, syncedOK, "Pushed", "PushPending", msg)
	if err := r.Status().Patch(ctx, cc, patch); err != nil {
		return ctrl.Result{}, err
	}

	if !syncedOK {
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}
	return ctrl.Result{RequeueAfter: r.ResyncPeriod}, nil
}

// nextVersion returns the version for a document with content `hash`: the
// current one if nothing changed, otherwise one more than anything published.
func (r *CacheClusterReconciler) nextVersion(ctx context.Context, cc *cachev1alpha1.CacheCluster, hash string) int64 {
	current := cc.Status.ConfigVersion
	cm := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{Namespace: cc.Namespace, Name: render.ControlConfigMapName(cc)}, cm)
	if err == nil {
		published, _ := strconv.ParseInt(cm.Annotations[render.VersionAnnotation], 10, 64)
		if cm.Annotations[render.HashAnnotation] == hash && published > 0 {
			return published
		}
		if published > current {
			current = published
		}
	} else if cc.Status.ConfigHash == hash && current > 0 {
		return current
	}
	return current + 1
}

func (r *CacheClusterReconciler) meta(cc *cachev1alpha1.CacheCluster, name string) metav1.ObjectMeta {
	return metav1.ObjectMeta{Name: name, Namespace: cc.Namespace}
}

// apply creates or updates an owned object.
func (r *CacheClusterReconciler) apply(ctx context.Context, cc *cachev1alpha1.CacheCluster, obj client.Object,
	mutate func(client.Object)) error {
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, obj, func() error {
		mutate(obj)
		return controllerutil.SetControllerReference(cc, obj, r.Scheme)
	})
	return err
}

// ensureAdminSecret creates the per-cluster token that authenticates the
// operator to the proxies' admin API and the proxies to each other.
func (r *CacheClusterReconciler) ensureAdminSecret(ctx context.Context, cc *cachev1alpha1.CacheCluster) (string, error) {
	secret := &corev1.Secret{}
	key := types.NamespacedName{Namespace: cc.Namespace, Name: render.AdminSecretName(cc)}
	err := r.Get(ctx, key, secret)
	if err == nil {
		return string(secret.Data[render.AdminTokenKey]), nil
	}
	if !apierrors.IsNotFound(err) {
		return "", err
	}
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	token := hex.EncodeToString(buf)
	secret = &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: key.Name, Namespace: key.Namespace, Labels: render.Labels(cc)},
		Data:       map[string][]byte{render.AdminTokenKey: []byte(token)},
	}
	if err := controllerutil.SetControllerReference(cc, secret, r.Scheme); err != nil {
		return "", err
	}
	if err := r.Create(ctx, secret); err != nil {
		return "", err
	}
	return token, nil
}

// resolvePolicies returns the winning policies for the cluster with the
// hashes of their API keys. A missing key Secret leaves that tenant without
// dynamic keys (limits still apply to keys configured statically).
func (r *CacheClusterReconciler) resolvePolicies(ctx context.Context, cc *cachev1alpha1.CacheCluster) ([]render.ResolvedPolicy, error) {
	list := &cachev1alpha1.TenantCachePolicyList{}
	if err := r.List(ctx, list, client.InNamespace(cc.Namespace)); err != nil {
		return nil, err
	}
	winners, _ := render.SelectPolicies(cc.Name, list.Items)
	out := make([]render.ResolvedPolicy, 0, len(winners))
	for _, p := range winners {
		if _, err := render.ParseThreshold(p.Spec.SemanticThreshold); err != nil {
			continue // invalid policies are reported by the policy controller, never rendered
		}
		rp := render.ResolvedPolicy{Policy: p}
		if ref := p.Spec.APIKeysSecretRef; ref != nil {
			secret := &corev1.Secret{}
			err := r.Get(ctx, types.NamespacedName{Namespace: cc.Namespace, Name: ref.Name}, secret)
			if err != nil && !apierrors.IsNotFound(err) {
				return nil, err
			}
			if err == nil {
				k := ref.Key
				if k == "" {
					k = "api-keys"
				}
				rp.KeyHashes = render.HashKeys(secret.Data[k])
			}
		}
		out = append(out, rp)
	}
	return out, nil
}

func podReady(pod *corev1.Pod) bool {
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodReady {
			return c.Status == corev1.ConditionTrue
		}
	}
	return false
}

func setCondition(conds *[]metav1.Condition, generation int64, typ string, ok bool, yes, no, msg string) {
	c := metav1.Condition{Type: typ, Status: metav1.ConditionFalse, Reason: no, Message: msg, ObservedGeneration: generation}
	if ok {
		c.Status, c.Reason = metav1.ConditionTrue, yes
	}
	meta.SetStatusCondition(conds, c)
}

// SetupWithManager wires watches: owned objects, pods (readiness changes ->
// push), policies and their key Secrets (re-render).
func (r *CacheClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cachev1alpha1.CacheCluster{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.StatefulSet{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ConfigMap{}).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(podToCluster)).
		Watches(&cachev1alpha1.TenantCachePolicy{}, handler.EnqueueRequestsFromMapFunc(r.policyToClusters)).
		Watches(&corev1.Secret{}, handler.EnqueueRequestsFromMapFunc(r.secretToClusters)).
		Named("cachecluster").
		Complete(r)
}

func podToCluster(_ context.Context, obj client.Object) []reconcile.Request {
	labels := obj.GetLabels()
	if labels[render.ManagedByLabel] != render.ManagedByValue || labels[render.InstanceLabel] == "" {
		return nil
	}
	return []reconcile.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: labels[render.InstanceLabel]}}}
}

func (r *CacheClusterReconciler) policyToClusters(ctx context.Context, obj client.Object) []reconcile.Request {
	p, ok := obj.(*cachev1alpha1.TenantCachePolicy)
	if !ok {
		return nil
	}
	if p.Spec.ClusterRef != "" {
		return []reconcile.Request{{NamespacedName: types.NamespacedName{Namespace: p.Namespace, Name: p.Spec.ClusterRef}}}
	}
	return r.allClusters(ctx, p.Namespace)
}

func (r *CacheClusterReconciler) secretToClusters(ctx context.Context, obj client.Object) []reconcile.Request {
	list := &cachev1alpha1.TenantCachePolicyList{}
	if err := r.List(ctx, list, client.InNamespace(obj.GetNamespace())); err != nil {
		return nil
	}
	seen := map[string]bool{}
	var out []reconcile.Request
	for i := range list.Items {
		p := &list.Items[i]
		if p.Spec.APIKeysSecretRef == nil || p.Spec.APIKeysSecretRef.Name != obj.GetName() {
			continue
		}
		for _, req := range r.policyToClusters(ctx, p) {
			if !seen[req.Name] {
				seen[req.Name] = true
				out = append(out, req)
			}
		}
	}
	return out
}

func (r *CacheClusterReconciler) allClusters(ctx context.Context, namespace string) []reconcile.Request {
	list := &cachev1alpha1.CacheClusterList{}
	if err := r.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return nil
	}
	out := make([]reconcile.Request, 0, len(list.Items))
	for _, c := range list.Items {
		out = append(out, reconcile.Request{NamespacedName: types.NamespacedName{Namespace: c.Namespace, Name: c.Name}})
	}
	return out
}
