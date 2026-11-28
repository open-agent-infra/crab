#!/usr/bin/env bash
# Provision a dedicated "service VM" that runs the crab gateway accessible from
# the LAN. Clones the dev VM disk and starts an independent QEMU instance with
# the gateway port bound on 0.0.0.0 so other machines/servers can reach it.
#
# Usage:
#   bash tools/vm/provision-service-vm.sh          # provision & start
#   bash tools/vm/provision-service-vm.sh --stop   # shut down the service VM
#   bash tools/vm/provision-service-vm.sh --status # show running state
#   bash tools/vm/provision-service-vm.sh --reset  # wipe ALL crab state & rebuild
#
# Idempotent: re-running on a live VM syncs code and confirms service health.
#
# --reset wipes everything on a *running* service VM and rebuilds a clean stack:
#   stops gateway + daemon, deletes all sandboxes (runc containers + per-sandbox
#   ZFS datasets + bundle/checkpoint/metadata dirs), deletes the gateway registry
#   SQLite (all tenants + API keys), restarts the daemon (--config) + gateway, and
#   recreates a fresh default tenant + API key (printed once). Image rootfs caches
#   are preserved so the first post-reset launch stays fast.

set -euo pipefail

# =============================================================================
# Configuration (adjust as needed)
# =============================================================================

SERVICE_VM_DIR=${SERVICE_VM_DIR:-$HOME/crab-service-vm}
SERVICE_VM_SSH_PORT=${SERVICE_VM_SSH_PORT:-2223}       # host-side SSH port
SERVICE_VM_GATEWAY_PORT=${SERVICE_VM_GATEWAY_PORT:-8900}  # host-side, bound 0.0.0.0
SERVICE_VM_CPUS=${SERVICE_VM_CPUS:-4}
SERVICE_VM_MEM=${SERVICE_VM_MEM:-8G}
# `user` keeps the zero-setup QEMU SLIRP path used by development. `tap`
# attaches the guest to an operator-created tap/bridge and gives production
# deployments a host-equivalent egress path.
SERVICE_VM_NET_MODE=${SERVICE_VM_NET_MODE:-user}
SERVICE_VM_TAP_IFACE=${SERVICE_VM_TAP_IFACE:-}
SERVICE_VM_TAP_SSH_HOST=${SERVICE_VM_TAP_SSH_HOST:-}
SERVICE_VM_TAP_SSH_PORT=${SERVICE_VM_TAP_SSH_PORT:-22}

TENANT_NAME=${TENANT_NAME:-default}
MAX_SANDBOXES=${MAX_SANDBOXES:-20}
MAX_MEMORY=${MAX_MEMORY:-6G}  # leave ~2G for system overhead

# Gateway listen port inside the VM (matches the host-side forward target)
VM_GATEWAY_PORT=8900

# Reset-related paths inside the VM (source of truth for --reset teardown).
# The service VM runs the daemon as `python3 -m crab.daemon --config <path>`.
CRAB_CONFIG_PATH=${CRAB_CONFIG_PATH:-/etc/crab/config.yaml}
GATEWAY_DATA_DIR=${GATEWAY_DATA_DIR:-/var/lib/crab/gateway}
# The deployed crab lives at /root/crab and is loaded via PYTHONPATH (the stock
# dist-packages copy is older); restart daemon/gateway the same way on --reset.
VM_CODE_PATH=${VM_CODE_PATH:-/root/crab}

# =============================================================================
# Source vm-lib.sh to reuse shared variables and helpers
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=vm-lib.sh
source "$SCRIPT_DIR/vm-lib.sh"

# After sourcing vm-lib.sh we have: QEMU_BIN, VM_DATA_DIR, SSH_KEY (dev VM key),
# and the SSH_OPTS pattern. The dev VM disk is at $DISK (=$IMAGES_DIR/$VM_NAME.qcow2).

DEV_VM_DISK="$DISK"  # from vm-lib.sh: ~/crab-vm/images/crab-dev.qcow2

# Service VM paths
SVC_DISK="$SERVICE_VM_DIR/crab-service.qcow2"
SVC_PIDFILE="$SERVICE_VM_DIR/crab-service.pid"
SVC_MONITOR="$SERVICE_VM_DIR/crab-service.monitor"
SVC_CONSOLE="$SERVICE_VM_DIR/crab-service-console.log"

# =============================================================================
# Helpers
# =============================================================================

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# SSH and host reachability differ between the forwarded SLIRP path and a
# bridge-attached guest. The tap interface, bridge membership, DHCP/static
# address, firewall, and host routing are intentionally operator-owned.
case "$SERVICE_VM_NET_MODE" in
    user)
        SVC_SSH_DEST=root@127.0.0.1
        SVC_SSH_PORT=$SERVICE_VM_SSH_PORT
        SVC_GATEWAY_CHECK_HOST=127.0.0.1
        SVC_GATEWAY_PUBLIC_HOST=""
        QEMU_NETDEV="user,id=net0,hostfwd=tcp:127.0.0.1:${SERVICE_VM_SSH_PORT}-:22,hostfwd=tcp:0.0.0.0:${SERVICE_VM_GATEWAY_PORT}-:${VM_GATEWAY_PORT}"
        ;;
    tap)
        [ -n "$SERVICE_VM_TAP_IFACE" ] \
            || die "SERVICE_VM_TAP_IFACE is required when SERVICE_VM_NET_MODE=tap"
        [ -n "$SERVICE_VM_TAP_SSH_HOST" ] \
            || die "SERVICE_VM_TAP_SSH_HOST is required when SERVICE_VM_NET_MODE=tap"
        SVC_SSH_DEST="root@${SERVICE_VM_TAP_SSH_HOST}"
        SVC_SSH_PORT=$SERVICE_VM_TAP_SSH_PORT
        SVC_GATEWAY_CHECK_HOST=$SERVICE_VM_TAP_SSH_HOST
        SVC_GATEWAY_PUBLIC_HOST=$SERVICE_VM_TAP_SSH_HOST
        QEMU_NETDEV="tap,id=net0,ifname=${SERVICE_VM_TAP_IFACE},script=no,downscript=no"
        ;;
    *)
        die "unsupported SERVICE_VM_NET_MODE=$SERVICE_VM_NET_MODE (expected user or tap)"
        ;;
esac

SVC_SSH_OPTS=(-i "$SSH_KEY" -p "$SVC_SSH_PORT" -o StrictHostKeyChecking=no
              -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5)

svc_vm_running() {
    [ -f "$SVC_PIDFILE" ] && kill -0 "$(cat "$SVC_PIDFILE")" 2>/dev/null
}

svc_vm_ssh() {
    ssh "${SVC_SSH_OPTS[@]}" "$SVC_SSH_DEST" "$@"
}

# ssh -f backgrounds the client immediately; needed for launching daemons that
# would otherwise hold the SSH channel open indefinitely.
svc_vm_ssh_daemon() {
    ssh -f "${SVC_SSH_OPTS[@]}" "$SVC_SSH_DEST" "$@"
}

svc_vm_wait_ssh() {
    local max_wait=${1:-60}
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if svc_vm_ssh true 2>/dev/null; then
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    return 1
}

svc_vm_rsync_repo() {
    # Mirror vm_rsync_repo from vm-lib.sh but targeting the service VM port
    rsync -a --delete -e "ssh ${SVC_SSH_OPTS[*]}" \
        --exclude .venv --exclude .cache --exclude build --exclude dist \
        --exclude '__pycache__' --exclude '.mypy_cache' --exclude '.pytest_cache' \
        --exclude '.ruff_cache' \
        "$REPO_ROOT/" "$SVC_SSH_DEST:/root/crab/"
}

port_in_use() {
    local port=$1
    # Check if something is already listening on the port
    if command -v ss >/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    elif command -v netstat >/dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    fi
    return 1
}

get_host_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "<host-ip>"
}

# =============================================================================
# --stop: shut down the service VM
# =============================================================================

if [ "${1:-}" = "--stop" ]; then
    if ! svc_vm_running; then
        echo "Service VM is not running."
        exit 0
    fi
    log "Shutting down service VM..."
    svc_vm_ssh poweroff 2>/dev/null || true
    for _ in $(seq 1 30); do
        svc_vm_running || { echo "Service VM stopped."; exit 0; }
        sleep 2
    done
    # Fallback: hard kill
    log "Graceful shutdown timed out, killing process"
    kill "$(cat "$SVC_PIDFILE")" 2>/dev/null || true
    sleep 2
    svc_vm_running && die "Failed to stop service VM"
    echo "Service VM stopped."
    exit 0
fi

# =============================================================================
# --status: show current state
# =============================================================================

if [ "${1:-}" = "--status" ]; then
    if svc_vm_running; then
        local_pid=$(cat "$SVC_PIDFILE")
        echo "Service VM: RUNNING (PID $local_pid)"
        echo "  Network: $SERVICE_VM_NET_MODE"
        echo "  SSH:     ssh -p $SVC_SSH_PORT $SVC_SSH_DEST"
        if [ "$SERVICE_VM_NET_MODE" = tap ]; then
            echo "  Gateway: http://${SVC_GATEWAY_PUBLIC_HOST}:${VM_GATEWAY_PORT}"
        else
            echo "  Gateway: http://$(get_host_ip):$SERVICE_VM_GATEWAY_PORT"
        fi
        # Try to check gateway health
        if svc_vm_ssh "curl -sf http://localhost:$VM_GATEWAY_PORT/healthz" >/dev/null 2>&1; then
            echo "  Gateway health: OK"
        else
            echo "  Gateway health: NOT RESPONDING"
        fi
    else
        echo "Service VM: NOT RUNNING"
        [ -f "$SVC_DISK" ] && echo "  Disk exists: $SVC_DISK"
    fi
    exit 0
fi

# =============================================================================
# --reset: wipe ALL crab state on a running service VM and rebuild a clean stack
# =============================================================================

if [ "${1:-}" = "--reset" ]; then
    svc_vm_running || die "Service VM is not running; start it first: bash tools/vm/provision-service-vm.sh"
    svc_vm_wait_ssh 30 || die "SSH to service VM not available"

    log "RESET: syncing latest repo into service VM..."
    svc_vm_rsync_repo

    log "RESET: stopping gateway + daemon..."
    svc_vm_ssh "fuser -k ${VM_GATEWAY_PORT}/tcp 2>/dev/null || true"
    # The bracket trick (crab[.]daemon) keeps pkill's own cmdline from matching.
    svc_vm_ssh 'pkill -f "crab[.]gateway" 2>/dev/null || true; pkill -f "crab[.]daemon" 2>/dev/null || true; sleep 2; rm -f /run/crab/crab.sock'

    log "RESET: deleting all sandboxes (runc containers + ZFS datasets + state dirs)..."
    svc_vm_ssh "CRAB_CONFIG_PATH='$CRAB_CONFIG_PATH' bash -s" <<'REMOTE'
set -u
CONFIG="${CRAB_CONFIG_PATH:-/etc/crab/config.yaml}"

# Derive storage roots from the daemon config (fall back to standard defaults).
eval "$(python3 - "$CONFIG" <<'PY'
import sys, yaml
cfg = {}
try:
    with open(sys.argv[1]) as fh:
        cfg = yaml.safe_load(fh) or {}
except Exception:
    pass
p = cfg.get("storage_planes", {}) or {}
print('RUNTIME_ROOT="%s"' % p.get("runtime_root", "/var/lib/crab/runtime"))
print('STORAGE_ROOT="%s"' % p.get("storage_root", "/var/lib/crab/checkpoints"))
PY
)"
echo "  runtime_root=$RUNTIME_ROOT  storage_root=$STORAGE_ROOT"

# 1. runc containers.
if command -v runc >/dev/null 2>&1; then
    for c in $(runc list -q 2>/dev/null); do
        echo "  runc delete $c"
        runc delete --force "$c" 2>/dev/null || true
    done
fi

# 2. Per-sandbox ZFS datasets, identified by the /sbx- component. This catches
#    every dataset tree (e.g. crab/sandboxes/sbx-*, stale crab/<run>/sbx-*) while
#    leaving image rootfs caches (…-cache-…, no /sbx-) intact.
if command -v zfs >/dev/null 2>&1; then
    for ds in $(zfs list -H -o name 2>/dev/null | grep '/sbx-' || true); do
        echo "  zfs destroy -rf $ds"
        zfs destroy -rf "$ds" 2>/dev/null || true
    done
fi

# 3. Bundle / checkpoint / metadata directories.
if [ -n "${RUNTIME_ROOT:-}" ]; then
    rm -rf "$RUNTIME_ROOT"/bundles/* "$RUNTIME_ROOT"/checkpoints/* \
           "$RUNTIME_ROOT"/runtime-state/* "$RUNTIME_ROOT"/sandbox-meta/* 2>/dev/null || true
fi
if [ -n "${STORAGE_ROOT:-}" ]; then
    rm -rf "$STORAGE_ROOT"/artifacts/* "$STORAGE_ROOT"/manifests/* \
           "$STORAGE_ROOT"/journal/* 2>/dev/null || true
fi
echo "  sandbox state dirs cleaned"
REMOTE

    log "RESET: deleting gateway registry (all tenants + API keys)..."
    svc_vm_ssh "rm -f '$GATEWAY_DATA_DIR'/gateway.sqlite3 '$GATEWAY_DATA_DIR'/gateway.sqlite3-wal '$GATEWAY_DATA_DIR'/gateway.sqlite3-shm"

    log "RESET: restarting daemon (--config $CRAB_CONFIG_PATH)..."
    svc_vm_ssh 'mkdir -p /run/crab'
    svc_vm_ssh_daemon "cd / && setsid env PYTHONPATH='$VM_CODE_PATH' python3 -m crab.daemon --config '$CRAB_CONFIG_PATH' </dev/null >/var/log/crabd.log 2>&1"
    for _ in $(seq 1 30); do
        svc_vm_ssh '[ -S /run/crab/crab.sock ]' 2>/dev/null && break
        sleep 1
    done
    svc_vm_ssh '[ -S /run/crab/crab.sock ]' 2>/dev/null \
        || { svc_vm_ssh 'tail -20 /var/log/crabd.log' >&2; die "daemon socket not ready after reset"; }
    log "daemon socket ready"

    log "RESET: restarting gateway..."
    svc_vm_ssh_daemon "cd / && setsid env PYTHONPATH='$VM_CODE_PATH' python3 -m crab.gateway serve \
        --bind 0.0.0.0:${VM_GATEWAY_PORT} \
        --daemon-socket /run/crab/crab.sock \
        --log-level INFO </dev/null >/var/log/crab-gateway.log 2>&1"
    for _ in $(seq 1 30); do
        svc_vm_ssh "curl -sf http://localhost:${VM_GATEWAY_PORT}/healthz" >/dev/null 2>&1 && break
        sleep 1
    done
    svc_vm_ssh "curl -sf http://localhost:${VM_GATEWAY_PORT}/healthz" >/dev/null 2>&1 \
        || { svc_vm_ssh 'tail -30 /var/log/crab-gateway.log' >&2; die "gateway not healthy after reset"; }
    log "gateway healthy"

    # Recreate a fresh default tenant + API key.
    log "RESET: creating fresh tenant '$TENANT_NAME'..."
    TENANT_CREATE_OUT=$(svc_vm_ssh "cd / && PYTHONPATH='$VM_CODE_PATH' python3 -m crab.gateway tenants create '$TENANT_NAME' \
        --max-sandboxes $MAX_SANDBOXES --max-memory '$MAX_MEMORY'" 2>&1) || true
    TENANT_ID=$(svc_vm_ssh "cd / && PYTHONPATH='$VM_CODE_PATH' python3 -m crab.gateway tenants list" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data.get('tenants', []):
    if t.get('name') == '$TENANT_NAME':
        print(t['id']); break
" 2>/dev/null || echo "")
    [ -n "$TENANT_ID" ] || die "could not determine tenant id after reset (output: $TENANT_CREATE_OUT)"

    API_KEY_OUTPUT=$(svc_vm_ssh "cd / && PYTHONPATH='$VM_CODE_PATH' python3 -m crab.gateway keys create --tenant '$TENANT_ID'") \
        || die "failed to create API key after reset"
    API_KEY=$(echo "$API_KEY_OUTPUT" | grep -oP 'crab_sk_\w+' | head -1 || echo "")
    [ -n "$API_KEY" ] || API_KEY="(check gateway output: $API_KEY_OUTPUT)"

    if [ "$SERVICE_VM_NET_MODE" = tap ]; then
        HOST_IP=$SVC_GATEWAY_PUBLIC_HOST
        RESET_GATEWAY_PORT=$VM_GATEWAY_PORT
    else
        HOST_IP=$(get_host_ip)
        RESET_GATEWAY_PORT=$SERVICE_VM_GATEWAY_PORT
    fi
    cat <<EOF

====== Crab Service VM RESET complete ======
Gateway: http://${HOST_IP}:${RESET_GATEWAY_PORT}
Tenant:  ${TENANT_NAME} (id: ${TENANT_ID})
API Key: ${API_KEY}

All previous tenants, API keys, and sandboxes were wiped; image caches kept.
Export the new credentials to use the tutorial:
  export CRAB_GATEWAY_URL=http://${HOST_IP}:${RESET_GATEWAY_PORT}
  export CRAB_API_KEY=${API_KEY}
=============================================
EOF
    exit 0
fi

# =============================================================================
# Pre-flight checks
# =============================================================================

[ -f "$DEV_VM_DISK" ] || die "Dev VM disk not found: $DEV_VM_DISK (run provision-vm.sh first)"
[ -f "$SSH_KEY" ] || die "SSH key not found: $SSH_KEY (run provision-vm.sh first)"
[ -x "$QEMU_BIN" ] || die "QEMU binary not found: $QEMU_BIN"

# Check port conflicts (only when VM is NOT already running)
if ! svc_vm_running && [ "$SERVICE_VM_NET_MODE" = user ]; then
    if port_in_use "$SERVICE_VM_SSH_PORT"; then
        die "Port $SERVICE_VM_SSH_PORT is already in use (needed for service VM SSH). Choose another SERVICE_VM_SSH_PORT."
    fi
    if port_in_use "$SERVICE_VM_GATEWAY_PORT"; then
        die "Port $SERVICE_VM_GATEWAY_PORT is already in use (needed for gateway). Choose another SERVICE_VM_GATEWAY_PORT."
    fi
fi

# =============================================================================
# Step 1: Create service VM directory and disk
# =============================================================================

mkdir -p "$SERVICE_VM_DIR"

if [ ! -f "$SVC_DISK" ]; then
    log "Copying dev VM disk to service VM (this may take a minute)..."
    DISK_SIZE=$(stat --printf="%s" "$DEV_VM_DISK" 2>/dev/null || stat -f%z "$DEV_VM_DISK")
    DISK_SIZE_GB=$((DISK_SIZE / 1024 / 1024 / 1024))

    if [ "$DISK_SIZE_GB" -lt 20 ]; then
        # Small enough for a full copy (independent, no backing file dependency)
        cp "$DEV_VM_DISK" "$SVC_DISK"
        log "Full copy complete (${DISK_SIZE_GB}G)"
    else
        # Large disk: use COW backing file for space efficiency
        log "Disk is ${DISK_SIZE_GB}G, using qemu-img backing file (COW)"
        qemu-img create -f qcow2 -F qcow2 -b "$DEV_VM_DISK" "$SVC_DISK"
    fi
else
    log "Service VM disk already exists: $SVC_DISK"
fi

# =============================================================================
# Step 2: Boot the service VM (if not already running)
# =============================================================================

if svc_vm_running; then
    log "Service VM already running (PID $(cat "$SVC_PIDFILE")), skipping boot"
else
    log "Starting service VM (${SERVICE_VM_CPUS} CPUs, ${SERVICE_VM_MEM} RAM)"

    # Convert memory to MB for QEMU -m flag
    MEM_VALUE="${SERVICE_VM_MEM%[GgMm]}"
    MEM_SUFFIX="${SERVICE_VM_MEM: -1}"
    case "$MEM_SUFFIX" in
        G|g) MEMORY_ARG="${MEM_VALUE}G" ;;
        M|m) MEMORY_ARG="${MEM_VALUE}M" ;;
        *)   MEMORY_ARG="${SERVICE_VM_MEM}" ;;
    esac

    "$QEMU_BIN" \
        -name crab-service \
        -machine q35,accel=kvm -cpu host \
        -smp "$SERVICE_VM_CPUS" -m "$MEMORY_ARG" \
        -drive "file=$SVC_DISK,if=virtio,format=qcow2" \
        -netdev "$QEMU_NETDEV" \
        -device "virtio-net-pci,netdev=net0" \
        -display none \
        -serial "file:$SVC_CONSOLE" \
        -pidfile "$SVC_PIDFILE" \
        -monitor "unix:$SVC_MONITOR,server,nowait" \
        -daemonize

    log "QEMU started, PID: $(cat "$SVC_PIDFILE")"
fi

# =============================================================================
# Step 3: Wait for SSH
# =============================================================================

log "Waiting for SSH at $SVC_SSH_DEST port $SVC_SSH_PORT..."
svc_vm_wait_ssh 60 || die "SSH to service VM timed out after 60s"
log "SSH is up"

# =============================================================================
# Step 4: Sync latest code into the service VM
# =============================================================================

log "Syncing repository into service VM..."
svc_vm_rsync_repo
log "Code sync complete"

# =============================================================================
# Step 5: Ensure the crab daemon is running
# =============================================================================

log "Ensuring crab daemon (crabd) is running..."
# Split into separate SSH calls to avoid pkill/pgrep self-match (the SSH command
# would contain both the kill pattern and the nohup start command in its argv).
DAEMON_OK=$(svc_vm_ssh '
    if [ -S /run/crab/crab.sock ] && python3 -c "
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    s.connect(\"/run/crab/crab.sock\")
    s.close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "yes"
    else
        echo "no"
    fi
')
if [ "$DAEMON_OK" = "yes" ]; then
    log "crabd already running (socket responsive)"
else
    # Kill stale daemon (separate SSH call avoids self-match)
    svc_vm_ssh 'pkill -f "crab[.]daemon[.]server" 2>/dev/null || true; rm -f /run/crab/crab.sock; mkdir -p /run/crab'
    log "Starting crabd inside VM..."
    svc_vm_ssh_daemon 'cd / && setsid python3 -m crab.daemon.server --socket /run/crab/crab.sock \
        --log-level INFO </dev/null >/var/log/crabd.log 2>&1'
    # Wait for socket to appear
    log "Waiting for daemon socket..."
    for _ in $(seq 1 30); do
        if svc_vm_ssh '[ -S /run/crab/crab.sock ]' 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if ! svc_vm_ssh '[ -S /run/crab/crab.sock ]' 2>/dev/null; then
        log "ERROR: crabd socket not found after 30s"
        svc_vm_ssh 'tail -20 /var/log/crabd.log' >&2
        die "Daemon startup failed inside service VM"
    fi
    log "crabd started, socket ready"
fi

# =============================================================================
# Step 6: Start the gateway (if not already running)
# =============================================================================

log "Ensuring crab-gateway is running on port $VM_GATEWAY_PORT..."
# Gateway check + start is split into two SSH calls to avoid the pkill/pgrep
# self-match problem (a single SSH command containing both the kill pattern and
# the nohup start command would match itself via /proc/PID/cmdline).
GW_HEALTHY=$(svc_vm_ssh "curl -sf http://localhost:$VM_GATEWAY_PORT/healthz >/dev/null 2>&1 && echo yes || echo no")
if [ "$GW_HEALTHY" = "yes" ]; then
    log "crab-gateway already running and healthy"
else
    # Kill any stale gateway using fuser (port-based, no self-match risk)
    svc_vm_ssh 'fuser -k 8900/tcp 2>/dev/null || true; sleep 1'
    log "Starting crab-gateway inside VM..."
    svc_vm_ssh_daemon "cd / && setsid python3 -m crab.gateway serve \
        --bind 0.0.0.0:$VM_GATEWAY_PORT \
        --daemon-socket /run/crab/crab.sock \
        --log-level INFO </dev/null >/var/log/crab-gateway.log 2>&1"
    # Wait for gateway to become healthy
    log "Waiting for gateway to be ready..."
    for _ in $(seq 1 30); do
        if svc_vm_ssh "curl -sf http://localhost:$VM_GATEWAY_PORT/healthz" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ! svc_vm_ssh "curl -sf http://localhost:$VM_GATEWAY_PORT/healthz" >/dev/null 2>&1; then
        log "ERROR: gateway did not become healthy within 30s"
        svc_vm_ssh 'tail -30 /var/log/crab-gateway.log' >&2
        die "Gateway startup failed inside service VM"
    fi
    log "crab-gateway ready"
fi

# =============================================================================
# Step 7: Wait for gateway health from host side (confirm hostfwd works)
# =============================================================================

log "Verifying gateway reachable from host..."
if [ "$SERVICE_VM_NET_MODE" = tap ]; then
    HOST_GATEWAY_PORT=$VM_GATEWAY_PORT
else
    HOST_GATEWAY_PORT=$SERVICE_VM_GATEWAY_PORT
fi
for _ in $(seq 1 10); do
    if curl -sf "http://${SVC_GATEWAY_CHECK_HOST}:${HOST_GATEWAY_PORT}/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -sf "http://${SVC_GATEWAY_CHECK_HOST}:${HOST_GATEWAY_PORT}/healthz" >/dev/null 2>&1 \
    || die "Gateway not reachable at ${SVC_GATEWAY_CHECK_HOST}:${HOST_GATEWAY_PORT}"
log "Gateway reachable on host"

# =============================================================================
# Step 8: Create tenant and API key (idempotent)
# =============================================================================

log "Ensuring tenant '$TENANT_NAME' exists..."
# All gateway admin commands run from / to avoid namespace-package issues
TENANT_ID=$(svc_vm_ssh "cd / && python3 -m crab.gateway tenants list 2>/dev/null" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data.get('tenants', data if isinstance(data, list) else []):
        if t.get('name') == '$TENANT_NAME':
            print(t['id'])
            sys.exit(0)
except Exception:
    pass
" 2>/dev/null || echo "")

if [ -n "$TENANT_ID" ]; then
    log "Tenant '$TENANT_NAME' already exists (id: $TENANT_ID)"
else
    log "Creating tenant '$TENANT_NAME'..."
    TENANT_CREATE_OUT=$(svc_vm_ssh "cd / && python3 -m crab.gateway tenants create '$TENANT_NAME' \
        --max-sandboxes $MAX_SANDBOXES --max-memory '$MAX_MEMORY'" 2>&1) || true
    TENANT_ID=$(echo "$TENANT_CREATE_OUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('id', data.get('tenant', {}).get('id', '')))
except Exception:
    pass
" 2>/dev/null || echo "")
    if [ -z "$TENANT_ID" ]; then
        # Fallback: re-list
        TENANT_ID=$(svc_vm_ssh "cd / && python3 -m crab.gateway tenants list" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data.get('tenants', []):
    if t.get('name') == '$TENANT_NAME':
        print(t['id']); break
" 2>/dev/null || echo "unknown")
    fi
    log "Created tenant '$TENANT_NAME' (id: $TENANT_ID)"
fi

[ -n "$TENANT_ID" ] || die "Could not determine tenant ID"

# =============================================================================
# Step 9: Create API key
# =============================================================================

log "Creating API key for tenant '$TENANT_NAME'..."
API_KEY_OUTPUT=$(svc_vm_ssh "cd / && python3 -m crab.gateway keys create --tenant '$TENANT_ID'") || die "Failed to create API key"

API_KEY=$(echo "$API_KEY_OUTPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # The key output may have 'key' or 'plaintext' field
    print(data.get('key', data.get('plaintext', '')))
except Exception:
    import re
    for line in sys.stdin:
        m = re.search(r'crab_sk_\\w+', line)
        if m:
            print(m.group(0))
            break
" 2>/dev/null || echo "")

if [ -z "$API_KEY" ]; then
    # Try to extract from raw output
    API_KEY=$(echo "$API_KEY_OUTPUT" | grep -oP 'crab_sk_\w+' | head -1 || echo "")
fi
if [ -z "$API_KEY" ]; then
    API_KEY="(check output: $API_KEY_OUTPUT)"
fi

# =============================================================================
# Step 10: Pre-create an example sandbox
# =============================================================================

log "Creating example sandbox..."
SANDBOX_ID="<skipped>"
# Only attempt sandbox creation if we have a valid-looking API key
if echo "$API_KEY" | grep -q "crab_sk_"; then
    SANDBOX_OUTPUT=$(svc_vm_ssh "
        cd / && python3 -c \"
import sys; sys.path.insert(0, '/root/crab')
from crab import Engine
engine = Engine.connect(url='http://localhost:$VM_GATEWAY_PORT', api_key='$API_KEY')
sb = engine.sandbox.run('ubuntu:22.04', detach=True)
print(sb.id)
\" 2>&1
    ") || SANDBOX_OUTPUT=""
    if [ -n "$SANDBOX_OUTPUT" ]; then
        SANDBOX_ID=$(echo "$SANDBOX_OUTPUT" | grep -oP 'sbx-\w+' | head -1)
        [ -z "$SANDBOX_ID" ] && SANDBOX_ID="<creation failed: $SANDBOX_OUTPUT>"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

if [ "$SERVICE_VM_NET_MODE" = tap ]; then
    HOST_IP=$SVC_GATEWAY_PUBLIC_HOST
    PUBLIC_GATEWAY_PORT=$VM_GATEWAY_PORT
else
    HOST_IP=$(get_host_ip)
    PUBLIC_GATEWAY_PORT=$SERVICE_VM_GATEWAY_PORT
fi

cat <<EOF

====== Crab Service VM Ready ======
Network: ${SERVICE_VM_NET_MODE}
SSH:     ssh -i $SSH_KEY -p $SVC_SSH_PORT $SVC_SSH_DEST
Gateway: http://${HOST_IP}:${PUBLIC_GATEWAY_PORT}
API Key: ${API_KEY}
Tenant:  ${TENANT_NAME} (id: ${TENANT_ID})
Example sandbox: ${SANDBOX_ID}

From other servers:
  engine = Engine.connect(url="http://${HOST_IP}:${PUBLIC_GATEWAY_PORT}", api_key="${API_KEY}")
  sandbox = Sandbox.connect("${SANDBOX_ID}", engine=engine)

Manage:
  Stop:   bash tools/vm/provision-service-vm.sh --stop
  Status: bash tools/vm/provision-service-vm.sh --status
  Logs:   ssh -i $SSH_KEY -p $SVC_SSH_PORT $SVC_SSH_DEST 'tail -f /var/log/crab-gateway.log'
===================================
EOF
