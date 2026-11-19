// Command csi-driver runs the CloudVanguard CSI plugin (csi.cloudvanguard.io).
//
// Controller pods run --mode=controller next to the external-provisioner /
// attacher / snapshotter / resizer sidecars; node pods run --mode=node in a
// privileged DaemonSet with node-driver-registrar. --mode=all serves both on
// one socket (single-host dev clusters, tests).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/csi"
	"github.com/JCHETAN26/Cloud-Vanguard/operator-go/internal/storage"
)

func main() {
	endpoint := flag.String("endpoint", "unix:///csi/csi.sock", "CSI endpoint (unix:// or tcp://)")
	nodeID := flag.String("node-id", os.Getenv("NODE_ID"), "this node's ID (node mode)")
	mode := flag.String("mode", "all", "controller | node | all")
	stateDir := flag.String("state-dir", "/var/lib/cloudvanguard-csi", "image + metadata directory (controller)")
	imageTool := flag.String("image-tool", "auto", "auto | qemu-img | raw")
	attacher := flag.String("attacher", "local", "local (host-visible raw images) | qmp (hot-plug into node VMs)")
	qmpTemplate := flag.String("qmp-socket-template", "/run/vanguard/qmp/{node}.sock", "QMP socket per node VM")
	mounter := flag.String("mounter", "host", "host (mount/bind, privileged) | link (unprivileged; block volumes only are real)")
	knownNodes := flag.String("known-nodes", "", "comma-separated node IDs accepted by ControllerPublish (local attacher; empty = any)")
	flag.Parse()

	d := &csi.Driver{NodeID: *nodeID}
	if *mode == "controller" || *mode == "all" {
		var tool storage.ImageTool
		switch *imageTool {
		case "qemu-img":
			tool = storage.QemuImg{}
		case "raw":
			tool = storage.RawFileTool{}
		default:
			tool = storage.DefaultImageTool()
		}
		var att storage.Attacher = storage.LocalAttacher{}
		if *attacher == "qmp" {
			att = storage.QMPAttacher{SocketTemplate: *qmpTemplate}
			d.NodeExists = func(node string) bool {
				_, err := os.Stat(strings.ReplaceAll(*qmpTemplate, "{node}", node))
				return err == nil
			}
		} else if *knownNodes != "" {
			allowed := map[string]bool{}
			for _, n := range strings.Split(*knownNodes, ",") {
				allowed[strings.TrimSpace(n)] = true
			}
			d.NodeExists = func(node string) bool { return allowed[node] }
		}
		store, err := storage.NewStore(*stateDir, tool, att)
		if err != nil {
			fmt.Fprintln(os.Stderr, "state dir:", err)
			os.Exit(1)
		}
		d.Store = store
		fmt.Fprintf(os.Stderr, "controller: images in %s via %s, attacher %s\n", *stateDir, tool.Name(), att.Name())
	}
	if *mode == "node" || *mode == "all" {
		if *nodeID == "" {
			fmt.Fprintln(os.Stderr, "--node-id (or NODE_ID) is required in node mode")
			os.Exit(2)
		}
		if *mounter == "link" {
			d.Mounter = csi.LinkMounter{}
		} else {
			d.Mounter = csi.HostMounter{}
		}
		fmt.Fprintf(os.Stderr, "node %s: mounter %s\n", *nodeID, d.Mounter.Name())
	}

	lis, err := csi.Listen(*endpoint)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	fmt.Fprintf(os.Stderr, "%s %s serving on %s (mode %s)\n", csi.DriverName, csi.DriverVersion, *endpoint, *mode)
	if err := d.Serve(ctx, lis); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
