package controller

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/controlpb"
)

// Pusher delivers a control document to one vanguard-proxy pod. The
// ConfigMap mounted at CONTROL_FILE is the durable path (pods converge from it
// after restarts); pushing makes changes live in milliseconds instead of
// waiting for the kubelet's ConfigMap sync (up to ~1 minute).
type Pusher interface {
	// Push PUTs `doc` to http://<endpoint>/admin/v1/config and returns what
	// the pod reports: the active version and its tracing state.
	Push(ctx context.Context, endpoint, token string, doc []byte) (PushResult, error)
}

// PushResult is a pod's answer to a push (vanguard.control.v1.ApplyResult).
type PushResult struct {
	ActiveVersion int64
	Tracing       *controlpb.TracingStatus // nil from proxies that predate tracing
}

// HTTPPusher is the production Pusher.
type HTTPPusher struct {
	Client *http.Client
}

// NewHTTPPusher returns a pusher with a short timeout: a slow pod must not
// stall reconciliation of the whole cluster.
func NewHTTPPusher() *HTTPPusher {
	return &HTTPPusher{Client: &http.Client{Timeout: 3 * time.Second}}
}

// The proxy answers with ApplyResult in proto3 JSON. Unknown fields are
// ignored so a newer proxy can add fields without breaking older operators.
var applyResultJSON = protojson.UnmarshalOptions{DiscardUnknown: true}

func (p *HTTPPusher) Push(ctx context.Context, endpoint, token string, doc []byte) (PushResult, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, "http://"+endpoint+"/admin/v1/config", bytes.NewReader(doc))
	if err != nil {
		return PushResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := p.Client.Do(req)
	if err != nil {
		return PushResult{}, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	res := &controlpb.ApplyResult{}
	parseErr := applyResultJSON.Unmarshal(body, res)
	switch resp.StatusCode {
	case http.StatusOK, http.StatusConflict:
		// 409: the pod already runs a newer document (e.g. pushed by a newer
		// operator replica); it is ahead, not broken.
		if parseErr != nil {
			return PushResult{}, fmt.Errorf("push to %s: unreadable ApplyResult: %w", endpoint, parseErr)
		}
		return PushResult{ActiveVersion: res.GetVersion(), Tracing: res.GetTracing()}, nil
	default:
		return PushResult{}, fmt.Errorf("push to %s: HTTP %d: %s", endpoint, resp.StatusCode, res.GetMessage())
	}
}
