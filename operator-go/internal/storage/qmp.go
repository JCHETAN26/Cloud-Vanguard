package storage

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"
)

// QMPClient speaks the QEMU Machine Protocol over a unix socket
// (qemu-system-* -qmp unix:/path,server=on,wait=off).
type QMPClient struct {
	conn   net.Conn
	reader *bufio.Reader
	mu     sync.Mutex
	// Events seen while waiting for command replies (e.g. DEVICE_DELETED).
	Events []QMPEvent
}

// QMPEvent is an asynchronous QMP event.
type QMPEvent struct {
	Event string          `json:"event"`
	Data  json.RawMessage `json:"data"`
}

type qmpMessage struct {
	QMP    json.RawMessage `json:"QMP"`
	Return json.RawMessage `json:"return"`
	Error  *struct {
		Class string `json:"class"`
		Desc  string `json:"desc"`
	} `json:"error"`
	Event string          `json:"event"`
	Data  json.RawMessage `json:"data"`
}

// DialQMP connects, reads the greeting and negotiates capabilities.
func DialQMP(ctx context.Context, socket string) (*QMPClient, error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "unix", socket)
	if err != nil {
		return nil, fmt.Errorf("qmp dial %s: %w", socket, err)
	}
	c := &QMPClient{conn: conn, reader: bufio.NewReader(conn)}
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	} else {
		_ = conn.SetDeadline(time.Now().Add(10 * time.Second))
	}
	var greeting qmpMessage
	if err := c.read(&greeting); err != nil || greeting.QMP == nil {
		conn.Close()
		return nil, fmt.Errorf("qmp greeting: %v", err)
	}
	if _, err := c.Execute("qmp_capabilities", nil); err != nil {
		conn.Close()
		return nil, err
	}
	return c, nil
}

func (c *QMPClient) read(m *qmpMessage) error {
	line, err := c.reader.ReadBytes('\n')
	if err != nil {
		return err
	}
	return json.Unmarshal(line, m)
}

// Execute runs one command and returns its "return" payload.
func (c *QMPClient) Execute(command string, args any) (json.RawMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	req := map[string]any{"execute": command}
	if args != nil {
		req["arguments"] = args
	}
	b, _ := json.Marshal(req)
	if _, err := c.conn.Write(append(b, '\n')); err != nil {
		return nil, err
	}
	for {
		var m qmpMessage
		if err := c.read(&m); err != nil {
			return nil, fmt.Errorf("qmp %s: %w", command, err)
		}
		switch {
		case m.Event != "":
			c.Events = append(c.Events, QMPEvent{Event: m.Event, Data: m.Data})
		case m.Error != nil:
			return nil, fmt.Errorf("qmp %s: %s: %s", command, m.Error.Class, m.Error.Desc)
		case m.Return != nil:
			return m.Return, nil
		}
	}
}

// Close closes the connection.
func (c *QMPClient) Close() error { return c.conn.Close() }
