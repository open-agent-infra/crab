#!/usr/bin/env bash
# deploy-cloud.sh — One-click Crab cloud service deployment.
#
# Installs system dependencies, builds and configures Crab, enables systemd
# services, creates a gateway tenant + API key, and prints connection info.
#
# Usage (on the target machine, as root):
#   curl -sL https://raw.githubusercontent.com/open-agent-infra/crab/experimental/scripts/deploy-cloud.sh | bash
#   # or with options:
#   bash deploy-cloud.sh --repo URL --branch BRANCH --tenant NAME --max-sandboxes N --max-memory SIZE --gateway-port PORT
set -Eeuo pipefail

# ─── defaults ───────────────────────────────────────────────────────────────

REPO_URL="https://github.com/open-agent-infra/crab"
BRANCH="experimental"
TENANT_NAME="default"
MAX_SANDBOXES=20
MAX_MEMORY="8G"
GATEWAY_PORT=8900
CRAB_ROOT=/root/crab
VENV=/opt/crab/venv
CONFIG=/etc/crab/config.yaml
ZPOOL_NAME=crab
ZPOOL_FILE=/var/lib/crab/crab.zpool
ZPOOL_SIZE=32G

# ─── argument parsing ───────────────────────────────────────────────────────

while (($#)); do
  case "$1" in
    --repo)       REPO_URL=$2; shift 2 ;;
    --branch)     BRANCH=$2; shift 2 ;;
    --tenant)     TENANT_NAME=$2; shift 2 ;;
    --max-sandboxes) MAX_SANDBOXES=$2; shift 2 ;;
    --max-memory) MAX_MEMORY=$2; shift 2 ;;
    --gateway-port) GATEWAY_PORT=$2; shift 2 ;;
    -h|--help)
      sed -n '3,11p' "$0"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ─── helpers ────────────────────────────────────────────────────────────────

die() { echo "deploy-cloud.sh: ERROR: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# ─── preflight checks ──────────────────────────────────────────────────────

[[ $EUID -eq 0 ]] || die "must run as root"
[[ $(uname -m) = x86_64 ]] || die "only x86-64 is supported"
source /etc/os-release 2>/dev/null || true
[[ ${ID:-} = ubuntu ]] || die "only Ubuntu is supported (found: ${ID:-unknown})"

log "Deploying Crab from $REPO_URL branch=$BRANCH"
log "Target: tenant=$TENANT_NAME max_sandboxes=$MAX_SANDBOXES max_memory=$MAX_MEMORY port=$GATEWAY_PORT"

# ─── step 1: install system packages ───────────────────────────────────────

log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
packages=(
  ca-certificates curl git gcc make pkg-config clang llvm
  libbpf-dev libelf-dev zlib1g-dev
  python3 python3-dev python3-pip python3-venv
  runc criu docker.io zfsutils-linux
)
apt-get install -y --no-install-recommends "${packages[@]}"

# ─── step 2: start docker ──────────────────────────────────────────────────

log "Enabling Docker"
systemctl enable --now docker >/dev/null 2>&1 || true
docker info >/dev/null 2>&1 || die "Docker is not reachable"

# ─── step 3: load ZFS + create pool ────────────────────────────────────────

log "Preparing ZFS"
modprobe zfs || die "cannot load zfs kernel module"
if ! zpool list -H -o name "$ZPOOL_NAME" >/dev/null 2>&1; then
  [[ ! -e $ZPOOL_FILE ]] || die "refusing to reuse existing pool file $ZPOOL_FILE"
  install -d -m 0755 /var/lib/crab /var/lib/crab/logs /opt/crab /etc/crab
  truncate -s "$ZPOOL_SIZE" "$ZPOOL_FILE"
  zpool create -m none "$ZPOOL_NAME" "$ZPOOL_FILE" || {
    rm -f "$ZPOOL_FILE"
    die "failed to create zpool $ZPOOL_NAME"
  }
  log "Created ZFS pool $ZPOOL_NAME ($ZPOOL_SIZE sparse-file backed)"
else
  log "Using existing ZFS pool $ZPOOL_NAME"
fi
install -d -m 0755 /var/lib/crab /var/lib/crab/logs /opt/crab /etc/crab

# ─── step 4: clone or update source ────────────────────────────────────────

log "Fetching Crab source → $CRAB_ROOT"
if [[ -d $CRAB_ROOT/.git ]]; then
  cd "$CRAB_ROOT"
  git fetch origin
  git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
  git pull --ff-only origin "$BRANCH" || git reset --hard "origin/$BRANCH"
else
  rm -rf "$CRAB_ROOT"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$CRAB_ROOT"
fi
cd "$CRAB_ROOT"

# ─── step 5: build eBPF host inspector ─────────────────────────────────────

log "Building eBPF host inspector"
make -C "$CRAB_ROOT/crab/host_inspector/bpf" clean all

# ─── step 6: create venv + install crab ─────────────────────────────────────

log "Creating Python venv and installing Crab"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade -q pip setuptools wheel
"$VENV/bin/python" -m pip install -q "$CRAB_ROOT[tls]"
ln -sfn "$VENV/bin/crab" /usr/local/bin/crab
ln -sfn "$VENV/bin/crabd" /usr/local/bin/crabd
ln -sfn "$VENV/bin/crab-gateway" /usr/local/bin/crab-gateway

# ─── step 7: write config ──────────────────────────────────────────────────

log "Writing $CONFIG"
sed "s|^zfs_dataset_prefix: crab/sandboxes$|zfs_dataset_prefix: ${ZPOOL_NAME}/sandboxes|" \
  "$CRAB_ROOT/config/crab.yaml" > "$CONFIG"

# ─── step 8: systemd units ─────────────────────────────────────────────────

log "Installing systemd units"
cat > /etc/systemd/system/crabd.service <<EOF
[Unit]
Description=Crab Daemon
After=network.target docker.service zfs-mount.service
Requires=docker.service

[Service]
Type=simple
ExecStart=$VENV/bin/python -m crab.daemon --config $CONFIG
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/lib/crab/logs/daemon.log
StandardError=append:/var/lib/crab/logs/daemon.log

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/crab-gateway.service <<EOF
[Unit]
Description=Crab Gateway
After=crabd.service
Requires=crabd.service

[Service]
Type=simple
ExecStart=$VENV/bin/crab-gateway serve --bind 0.0.0.0:$GATEWAY_PORT --daemon-socket /run/crab/crab.sock
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/lib/crab/logs/gateway.log
StandardError=append:/var/lib/crab/logs/gateway.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable crabd crab-gateway

# ─── step 9: start services ────────────────────────────────────────────────

log "Starting Crab services"
# Stop any existing instances
pkill -f "python.*crab.daemon" 2>/dev/null || true
pkill -f "crab.gateway\|crab-gateway" 2>/dev/null || true
sleep 2

install -d -m 0755 /run/crab
systemctl start crabd
sleep 5
systemctl start crab-gateway
sleep 3

# Verify
if ! curl -sf "localhost:$GATEWAY_PORT/healthz" >/dev/null; then
  log "WARNING: healthz check failed; check logs in /var/lib/crab/logs/"
  systemctl status crabd --no-pager || true
  systemctl status crab-gateway --no-pager || true
  die "service did not start properly"
fi
log "Services running — healthz OK"

# ─── step 10: pull default image ───────────────────────────────────────────

log "Pulling default sandbox image (ubuntu:22.04)"
docker pull ubuntu:22.04 >/dev/null 2>&1 || log "WARNING: docker pull ubuntu:22.04 failed (non-fatal)"

# ─── step 11: create tenant + API key ──────────────────────────────────────

log "Creating tenant '$TENANT_NAME' and minting API key"
sleep 2  # let gateway fully settle

TENANT_JSON=$("$VENV/bin/crab-gateway" tenants create "$TENANT_NAME" \
  --max-sandboxes "$MAX_SANDBOXES" --max-memory "$MAX_MEMORY" 2>&1) || true
TENANT_ID=$(echo "$TENANT_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['tenant']['id'])" 2>/dev/null) || \
  TENANT_ID=$("$VENV/bin/crab-gateway" tenants list | python3 -c "
import sys, json
for t in json.load(sys.stdin)['tenants']:
    if t['name'] == '$TENANT_NAME':
        print(t['id']); break
")

KEY_JSON=$("$VENV/bin/crab-gateway" keys create --tenant "$TENANT_ID")
API_KEY=$(echo "$KEY_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])")

# ─── done ───────────────────────────────────────────────────────────────────

PUBLIC_IP=$(curl -s --connect-timeout 5 ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              Crab Cloud Service — Deployment Complete           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "  Endpoint:   http://$PUBLIC_IP:$GATEWAY_PORT"
echo "  API Key:    $API_KEY"
echo "  Tenant:     $TENANT_NAME ($TENANT_ID)"
echo "║                                                                ║"
echo "  SDK usage (on your machine):                                    "
echo "    pip install crab                                              "
echo "    from crab import Sandbox                                      "
echo "    sbx = Sandbox(base_url=\"http://$PUBLIC_IP:$GATEWAY_PORT\",   "
echo "                  api_key=\"$API_KEY\")                           "
echo "║                                                                ║"
echo "  Manage:                                                         "
echo "    systemctl status crabd crab-gateway                           "
echo "    journalctl -u crabd -f                                        "
echo "    crab-gateway tenants list                                     "
echo "║                                                                ║"
echo "  IMPORTANT: Ensure your cloud firewall / security group allows   "
echo "  inbound TCP on port $GATEWAY_PORT.                              "
echo "║                                                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "API_KEY=$API_KEY"
