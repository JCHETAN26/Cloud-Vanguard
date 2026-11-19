#!/usr/bin/env bash
# Reports which container / Kubernetes / hypervisor tools are available, so
# test scripts can choose real vs simulated layers and reports can say which
# ran. Prints a table; with --json prints one JSON object. Always exits 0.
set -u

probe() {  # name -> version string or ""
  case "$1" in
    kvm)       [[ -c /dev/kvm && -r /dev/kvm && -w /dev/kvm ]] && echo "/dev/kvm" ;;
    loop)      { [[ -e /dev/loop-control ]] && losetup -f >/dev/null 2>&1 && echo "loop devices"; } || true ;;
    mount)     { [[ "$(id -u)" == 0 ]] || sudo -n true 2>/dev/null; } && grep -q CapEff /proc/self/status 2>/dev/null \
                 && [[ "$(awk '/CapEff/{print $2}' /proc/self/status)" != "0000000000000000" ]] && echo "CAP_SYS_ADMIN" ;;
    docker)    command -v docker >/dev/null && docker info >/dev/null 2>&1 && docker --version ;;
    kind)      command -v kind >/dev/null && kind version ;;
    k3s)       command -v k3s >/dev/null && k3s --version | head -1 ;;
    kubectl)   command -v kubectl >/dev/null && kubectl version --client 2>/dev/null | head -1 ;;
    qemu-img)  command -v qemu-img >/dev/null && qemu-img --version | head -1 ;;
    qemu-system) command -v qemu-system-x86_64 >/dev/null && qemu-system-x86_64 --version | head -1 ;;
    go)        command -v go >/dev/null && go version ;;
    cmake)     command -v cmake >/dev/null && cmake --version | head -1 ;;
    python3)   command -v python3 >/dev/null && python3 --version ;;
  esac
}

TOOLS=(kvm loop mount docker kind k3s kubectl qemu-img qemu-system go cmake python3)
if [[ "${1:-}" == "--json" ]]; then
  printf '{'
  sep=""
  for t in "${TOOLS[@]}"; do
    v="$(probe "$t" 2>/dev/null | tr -d '"' | head -1)"
    printf '%s"%s":"%s"' "$sep" "$t" "$v"
    sep=","
  done
  printf '}\n'
  exit 0
fi
printf '%-12s %-9s %s\n' TOOL STATUS DETAIL
for t in "${TOOLS[@]}"; do
  v="$(probe "$t" 2>/dev/null | head -1)"
  if [[ -n "$v" ]]; then printf '%-12s %-9s %s\n' "$t" present "$v"; else printf '%-12s %-9s\n' "$t" MISSING; fi
done
