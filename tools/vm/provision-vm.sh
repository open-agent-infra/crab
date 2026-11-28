#!/usr/bin/env bash
# Provision the crab development VM with plain QEMU (no libvirt; see
# tools/vm/vm-lib.sh for why).
#
# Usage:  bash tools/vm/provision-vm.sh
#
# Host prerequisites: qemu-kvm, qemu-img, genisoimage (or cloud-localds),
# curl, rsync, ssh — all runnable as a normal user (/dev/kvm must be rw).
#
# All images and downloads live under $VM_DATA_DIR (default: ~/crab-vm).
# The script refuses to overwrite an existing VM disk.
#
# Crab needs runc + CRIU + ZFS + eBPF + root — exactly why development happens
# on the host and testing inside this disposable VM (see .cache/tasks/
# filesystem-provider-refactor.md §"Development & Test Environment").

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=vm-lib.sh
source "$SCRIPT_DIR/vm-lib.sh"

DISK_GB=${DISK_GB:-100}
IMAGE_URL=${IMAGE_URL:-https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img}

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

# --- host checks -------------------------------------------------------------
for cmd in qemu-img ssh-keygen ssh rsync curl; do
    command -v "$cmd" >/dev/null || die "missing host command: $cmd"
done
[ -x "$QEMU_BIN" ] || die "missing QEMU binary: $QEMU_BIN (set QEMU_BIN)"
# cloud-localds (Debian) or genisoimage (RHEL-family) builds the seed ISO.
command -v cloud-localds >/dev/null || command -v genisoimage >/dev/null \
    || die "missing host command: cloud-localds or genisoimage"
[ -r /dev/kvm ] && [ -w /dev/kvm ] || die "/dev/kvm is missing or not accessible"
[ -f "$DISK" ] && die "VM disk already exists: $DISK (delete it to rebuild)"
vm_running && die "a '$VM_NAME' QEMU process is already running (pid $(cat "$PIDFILE"))"

mkdir -p "$WORK_DIR" "$IMAGES_DIR"

# --- artifacts ---------------------------------------------------------------
BASE_IMAGE="$WORK_DIR/$(basename "$IMAGE_URL")"
if [ ! -f "$BASE_IMAGE" ]; then
    log "downloading base image"
    curl -fL --retry 3 -o "$BASE_IMAGE.tmp" "$IMAGE_URL"
    mv "$BASE_IMAGE.tmp" "$BASE_IMAGE"
fi

if [ ! -f "$SSH_KEY" ]; then
    log "generating dedicated SSH key"
    ssh-keygen -t ed25519 -N '' -C crab-dev-vm -f "$SSH_KEY"
fi

log "creating VM disk (${DISK_GB}G, backed by base image)"
qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMAGE" "$DISK" "${DISK_GB}G"

log "rendering cloud-init seed"
sed "s|__SSH_PUBLIC_KEY__|$(cat "$SSH_KEY.pub")|" \
    "$SCRIPT_DIR/cloud-init-user-data.yaml" > "$WORK_DIR/user-data"
printf 'instance-id: %s\nlocal-hostname: %s\n' "$VM_NAME" "$VM_NAME" > "$WORK_DIR/meta-data"
# slirp's DNS relay (10.0.2.3) cannot forward to a host-loopback resolver
# (this host's first nameserver is 127.0.0.1), so hand the guest the host's
# real upstream nameservers instead of the DHCP-provided relay.
HOST_DNS=$(awk '/^nameserver/ && $2 !~ /^127\./ {printf "%s%s", sep, $2; sep=", "}' /etc/resolv.conf)
[ -n "$HOST_DNS" ] || HOST_DNS=10.0.2.3
log "guest DNS servers: $HOST_DNS"
cat > "$WORK_DIR/network-config" <<EOF
version: 2
ethernets:
  all:
    match: {name: "e*"}
    dhcp4: true
    dhcp4-overrides: {use-dns: false}
    nameservers: {addresses: [$HOST_DNS]}
EOF
if command -v cloud-localds >/dev/null; then
    cloud-localds --network-config "$WORK_DIR/network-config" \
        "$SEED" "$WORK_DIR/user-data" "$WORK_DIR/meta-data"
else
    # genisoimage equivalent: a NoCloud seed is just an ISO labelled `cidata`.
    genisoimage -quiet -output "$SEED" -volid cidata -joliet -rock \
        "$WORK_DIR/user-data" "$WORK_DIR/meta-data" "$WORK_DIR/network-config"
fi

# --- first boot ---------------------------------------------------------------
log "booting VM (first boot, with cloud-init seed)"
vm_boot seed

log "waiting for SSH (cloud-init also installs packages; console: $CONSOLE_LOG)"
vm_wait_ssh 120 || die "SSH to $SSH_DEST:$SSH_PORT failed (check $CONSOLE_LOG)"

log "waiting for cloud-init to finish (package installation)"
ssh "${SSH_OPTS[@]}" "$SSH_DEST" cloud-init status --wait || die "cloud-init failed"

# --- sync repository + in-VM setup --------------------------------------------
log "syncing repository into VM"
vm_rsync_repo "$REPO_ROOT"

log "running in-VM setup (crab installer + editable dev install)"
scp "${SSH_OPTS[@]/#-p/-P}" "$SCRIPT_DIR/vm-setup.sh" "$SCRIPT_DIR/vm-verify.sh" "$SSH_DEST:/root/"
# Reuse the host's registry mirrors: Docker Hub is not directly reachable
# from this network and the guest inherits the same egress path.
if [ -f /etc/docker/daemon.json ]; then
    log "copying host docker daemon.json (registry mirrors) into the VM"
    scp "${SSH_OPTS[@]/#-p/-P}" /etc/docker/daemon.json "$SSH_DEST:/tmp/host-daemon.json"
    ssh "${SSH_OPTS[@]}" "$SSH_DEST" \
        'python3 -c "import json,sys; d=json.load(open(\"/tmp/host-daemon.json\")); json.dump({\"registry-mirrors\": d.get(\"registry-mirrors\", [])}, open(\"/etc/docker/daemon.json\",\"w\"))" && systemctl restart docker'
fi
ssh "${SSH_OPTS[@]}" "$SSH_DEST" bash /root/vm-setup.sh

log "running verification checklist"
ssh "${SSH_OPTS[@]}" "$SSH_DEST" bash /root/vm-verify.sh

# --- snapshot ------------------------------------------------------------------
log "taking pristine snapshot"
vm_shutdown || die "VM did not shut down cleanly"
# Internal qcow2 snapshot; the seed ISO is a separate drive and not included,
# and is no longer needed after the first boot.
qemu-img snapshot -c pristine "$DISK"

log "restarting VM (without seed)"
vm_boot
vm_wait_ssh 60 || die "VM did not come back after snapshot"

cat <<EOF

Provisioning complete.

  Connect:   ssh -i $SSH_KEY -p $SSH_PORT root@127.0.0.1
  Run tests: bash tools/vm/run-in-vm.sh 'python3 -m unittest -v tests.test_storage'
  Rollback:  bash tools/vm/run-in-vm.sh --revert-pristine
  Repo:      /root/crab (disposable mirror; edit on the host, never in the VM)
EOF
