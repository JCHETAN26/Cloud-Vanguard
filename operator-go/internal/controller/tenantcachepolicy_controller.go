package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	cachev1alpha1 "github.com/JCHETAN26/Cloud-Vanguard/operator-go/api/v1alpha1"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/render"
)

const (
	ConditionApplied  = "Applied"
	ConditionConflict = "Conflict"
)

// TenantCachePolicyReconciler validates policies and reports whether each one
// is live. Rendering and distribution belong to the CacheCluster controller
// (single writer per cluster); this controller reads the rendered control
// documents back and marks a policy Applied once its current generation is in
// every target cluster's document.
type TenantCachePolicyReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=tenantcachepolicies,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=tenantcachepolicies/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=cache.cloudvanguard.io,resources=cacheclusters,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch

func (r *TenantCachePolicyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	p := &cachev1alpha1.TenantCachePolicy{}
	if err := r.Get(ctx, req.NamespacedName, p); err != nil {
		// Deleted: the CacheCluster controller re-renders without it.
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !p.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	patch := client.MergeFrom(p.DeepCopy())
	applied, requeue, appliedTo := false, false, []string(nil)
	reason, message := "", ""
	conflictWith := ""

	if _, err := render.ParseThreshold(p.Spec.SemanticThreshold); err != nil {
		reason, message = "InvalidSpec", err.Error()
	} else {
		targets, err := r.targetClusters(ctx, p)
		if err != nil {
			return ctrl.Result{}, err
		}
		all := &cachev1alpha1.TenantCachePolicyList{}
		if err := r.List(ctx, all, client.InNamespace(p.Namespace)); err != nil {
			return ctrl.Result{}, err
		}
		switch {
		case len(targets) == 0 && p.Spec.ClusterRef != "":
			reason, message, requeue = "ClusterNotFound", fmt.Sprintf("CacheCluster %q not found", p.Spec.ClusterRef), true
		case len(targets) == 0:
			reason, message = "NoClusters", "no CacheCluster in this namespace"
		default:
			var pending []string
			for _, cc := range targets {
				if _, conflicts := render.SelectPolicies(cc.Name, all.Items); conflicts[p.Name] != "" {
					conflictWith = conflicts[p.Name]
					continue
				}
				live, err := r.liveIn(ctx, &cc, p)
				if err != nil {
					return ctrl.Result{}, err
				}
				if live {
					appliedTo = append(appliedTo, cc.Name)
				} else {
					pending = append(pending, cc.Name)
				}
			}
			sort.Strings(appliedTo)
			switch {
			case conflictWith != "":
				reason = "Conflict"
				message = fmt.Sprintf("tenant %q is already configured by TenantCachePolicy %q (oldest wins)",
					p.Spec.TenantID, conflictWith)
			case len(pending) > 0:
				reason, message, requeue = "Pending", "waiting for "+strings.Join(pending, ", "), true
			default:
				applied = true
				reason = "Applied"
				message = fmt.Sprintf("generation %d live in %s", p.Generation, strings.Join(appliedTo, ", "))
			}
		}
	}

	p.Status.Applied = applied
	p.Status.ObservedGeneration = p.Generation
	p.Status.AppliedClusters = appliedTo
	setCondition(&p.Status.Conditions, p.Generation, ConditionApplied, applied, reason, reason, message)
	setCondition(&p.Status.Conditions, p.Generation, ConditionConflict, conflictWith != "", "DuplicateTenant",
		"Unique", conflictMessage(conflictWith))
	if err := r.Status().Patch(ctx, p, patch); err != nil {
		return ctrl.Result{}, err
	}
	if requeue {
		return ctrl.Result{RequeueAfter: 2 * time.Second}, nil
	}
	return ctrl.Result{}, nil
}

func conflictMessage(winner string) string {
	if winner == "" {
		return "no other policy configures this tenant"
	}
	return "superseded by " + winner
}

func (r *TenantCachePolicyReconciler) targetClusters(ctx context.Context, p *cachev1alpha1.TenantCachePolicy) ([]cachev1alpha1.CacheCluster, error) {
	if p.Spec.ClusterRef != "" {
		cc := &cachev1alpha1.CacheCluster{}
		err := r.Get(ctx, types.NamespacedName{Namespace: p.Namespace, Name: p.Spec.ClusterRef}, cc)
		if apierrors.IsNotFound(err) || (err == nil && !cc.DeletionTimestamp.IsZero()) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		return []cachev1alpha1.CacheCluster{*cc}, nil
	}
	list := &cachev1alpha1.CacheClusterList{}
	if err := r.List(ctx, list, client.InNamespace(p.Namespace)); err != nil {
		return nil, err
	}
	var out []cachev1alpha1.CacheCluster
	for _, cc := range list.Items {
		if cc.DeletionTimestamp.IsZero() {
			out = append(out, cc)
		}
	}
	return out, nil
}

// liveIn reports whether the cluster's rendered control document contains
// this policy at its current generation.
func (r *TenantCachePolicyReconciler) liveIn(ctx context.Context, cc *cachev1alpha1.CacheCluster, p *cachev1alpha1.TenantCachePolicy) (bool, error) {
	cm := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{Namespace: cc.Namespace, Name: render.ControlConfigMapName(cc)}, cm)
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	doc, err := render.ParseJSON([]byte(cm.Data[render.ControlKey]))
	if err != nil {
		return false, nil
	}
	for _, t := range doc.GetTenants().GetTenants() {
		if t.GetTenantId() == p.Spec.TenantID && t.GetSourceGeneration() == p.Generation {
			return true, nil
		}
	}
	return false, nil
}

// SetupWithManager watches policies and the rendered ConfigMaps (so status
// flips to Applied as soon as the cluster controller publishes).
func (r *TenantCachePolicyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cachev1alpha1.TenantCachePolicy{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Watches(&corev1.ConfigMap{}, handler.EnqueueRequestsFromMapFunc(r.configMapToPolicies)).
		Watches(&cachev1alpha1.CacheCluster{}, handler.EnqueueRequestsFromMapFunc(r.clusterToPolicies)).
		Named("tenantcachepolicy").
		Complete(r)
}

func (r *TenantCachePolicyReconciler) configMapToPolicies(ctx context.Context, obj client.Object) []reconcile.Request {
	if obj.GetLabels()[render.ManagedByLabel] != render.ManagedByValue {
		return nil
	}
	return r.policiesIn(ctx, obj.GetNamespace())
}

func (r *TenantCachePolicyReconciler) clusterToPolicies(ctx context.Context, obj client.Object) []reconcile.Request {
	return r.policiesIn(ctx, obj.GetNamespace())
}

func (r *TenantCachePolicyReconciler) policiesIn(ctx context.Context, namespace string) []reconcile.Request {
	list := &cachev1alpha1.TenantCachePolicyList{}
	if err := r.List(ctx, list, client.InNamespace(namespace)); err != nil {
		return nil
	}
	out := make([]reconcile.Request, 0, len(list.Items))
	for _, p := range list.Items {
		out = append(out, reconcile.Request{NamespacedName: types.NamespacedName{Namespace: p.Namespace, Name: p.Name}})
	}
	return out
}
