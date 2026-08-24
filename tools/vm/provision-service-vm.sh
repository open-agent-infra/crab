#!/usr/bin/env bash
# Provision a dedicated "service VM" that runs the crab gateway accessible from
# the LAN. Clones the dev VM disk and starts an independent QEMU instance with
# the gateway port bound on 0.0.0.0 so other machines/servers can reach it.
#
# Usage:
#   bash tools/vm/provision-service-vm.sh          # provision & start
#   bash tools/vm/provision-service-vm.sh --stop   # shut down the service VM
#   bash tools/vm/provision-service-vm.sh --status # show running state
#
# Idempotent: re-running on a live VM syncs code and confirms service health.

set -euo pipefail

# =============================================================================
# Configuration (adjust as needed)
# =============================================================================

SERVICE_VM_DIR=${SERVICE_VM_DIR:-$HOME/crab-service-vm}
SERVICE_VM_SSH_PORT=${SERVICE_VM_SSH_PORT:-2223}       # host-side SSH port
SERVICE_VM_GATEWAY_PORT=${SERVICE_VM_GATEWAY_PORT:-8900}  # host-side, bound 0.0.0.0
SERVICE_VM_CPUS=${SERVICE_VM_CPUS:-4}
SERVICE_VM_MEM=${SERVICE_VM_MEM:-8G}

TENANT_NAME=${TENANT_NAME:-default}
MAX_SANDBOXES=${MAX_SANDBOXES:-20}
MAX_MEMORY=${MAX_MEMORY:-6G}  # leave ~2G for system overhead

# Gateway listen port inside the VM (matches the host-side forward target)
VM_GATEWAY_PORT=8900

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

# SSH config for service VM (reuses the same key baked into the dev VM image)
SVC_SSH_DEST=root@127.0.0.1
SVC_SSH_OPTS=(-i "$SSH_KEY" -p "$SERVICE_VM_SSH_PORT" -o StrictHostKeyChecking=no
              -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5)

# =============================================================================
# Helpers
# =============================================================================

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

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
        echo "  SSH:     ssh -p $SERVICE_VM_SSH_PORT root@127.0.0.1"
        echo "  Gateway: http://$(get_host_ip):$SERVICE_VM_GATEWAY_PORT"
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
# Pre-flight checks
# =============================================================================

[ -f "$DEV_VM_DISK" ] || die "Dev VM disk not found: $DEV_VM_DISK (run provision-vm.sh first)"
[ -f "$SSH_KEY" ] || die "SSH key not found: $SSH_KEY (run provision-vm.sh first)"
[ -x "$QEMU_BIN" ] || die "QEMU binary not found: $QEMU_BIN"

# Check port conflicts (only when VM is NOT already running)
if ! svc_vm_running; then
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
        -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:${SERVICE_VM_SSH_PORT}-:22,hostfwd=tcp:0.0.0.0:${SERVICE_VM_GATEWAY_PORT}-:${VM_GATEWAY_PORT}" \
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

log "Waiting for SSH on port $SERVICE_VM_SSH_PORT..."
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
for _ in $(seq 1 10); do
    if curl -sf "http://127.0.0.1:$SERVICE_VM_GATEWAY_PORT/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -sf "http://127.0.0.1:$SERVICE_VM_GATEWAY_PORT/healthz" >/dev/null 2>&1 \
    || die "Gateway not reachable on host port $SERVICE_VM_GATEWAY_PORT"
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

HOST_IP=$(get_host_ip)

cat <<EOF

====== Crab Service VM Ready ======
SSH:     ssh -i $SSH_KEY -p $SERVICE_VM_SSH_PORT root@127.0.0.1
Gateway: http://${HOST_IP}:${SERVICE_VM_GATEWAY_PORT}
API Key: ${API_KEY}
Tenant:  ${TENANT_NAME} (id: ${TENANT_ID})
Example sandbox: ${SANDBOX_ID}

From other servers:
  engine = Engine.connect(url="http://${HOST_IP}:${SERVICE_VM_GATEWAY_PORT}", api_key="${API_KEY}")
  sandbox = Sandbox.connect("${SANDBOX_ID}", engine=engine)

Manage:
  Stop:   bash tools/vm/provision-service-vm.sh --stop
  Status: bash tools/vm/provision-service-vm.sh --status
  Logs:   ssh -i $SSH_KEY -p $SERVICE_VM_SSH_PORT root@127.0.0.1 'tail -f /var/log/crab-gateway.log'
===================================
EOF
