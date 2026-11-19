package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

// Pusher delivers a control document to one vanguard-proxy pod. The
// ConfigMap mounted at CONTROL_FILE is the durable path (pods converge from it
// after restarts); pushing makes changes live in milliseconds instead of
// waiting for the kubelet's ConfigMap sync (up to ~1 minute).
type Pusher interface {
	// Push PUTs `doc` to http://<endpoint>/admin/v1/config and returns the
	// version the pod reports as active.
	Push(ctx context.Context, endpoint, token string, doc []byte) (int64, error)
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

type applyResult struct {
	Applied bool   `json:"applied"`
	Version string `json:"version"`
	Message string `json:"message"`
}

func (p *HTTPPusher) Push(ctx context.Context, endpoint, token string, doc []byte) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, "http://"+endpoint+"/admin/v1/config", bytes.NewReader(doc))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := p.Client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var res applyResult
	_ = json.Unmarshal(body, &res)
	version, _ := strconv.ParseInt(res.Version, 10, 64)
	switch resp.StatusCode {
	case http.StatusOK:
		return version, nil
	case http.StatusConflict:
		// The pod already runs a newer document (e.g. pushed by a newer
		// operator replica); it is ahead, not broken.
		return version, nil
	default:
		return 0, fmt.Errorf("push to %s: HTTP %d: %s", endpoint, resp.StatusCode, res.Message)
	}
}
