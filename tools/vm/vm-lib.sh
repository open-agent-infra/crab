#!/usr/bin/env bash
# Shared helpers for the crab-dev VM. Sourced by provision-vm.sh and
# run-in-vm.sh — not executable on its own.
#
# The VM is plain QEMU (no libvirt): on this host an automated compliance job
# periodically runs `dnf remove -y unbound-libs`, which cascades through
# gnutls-dane -> swtpm-tools -> libvirt-daemon-driver-qemu and uninstalls the
# libvirt stack out from under us. qemu-kvm itself survives, /dev/kvm is 0666,
# and user-mode (slirp) networking needs no root, so raw QEMU is both simpler
# and durable here. SSH reaches the guest via a hostfwd port on 127.0.0.1.

QEMU_BIN=${QEMU_BIN:-/usr/libexec/qemu-kvm}
VM_NAME=${VM_NAME:-crab-dev}
VM_DATA_DIR=${VM_DATA_DIR:-$HOME/crab-vm}
SSH_PORT=${SSH_PORT:-2222}
VCPUS=${VCPUS:-8}
MEMORY_MB=${MEMORY_MB:-16384}

IMAGES_DIR="$VM_DATA_DIR/images"
WORK_DIR="$VM_DATA_DIR/work"
DISK="$IMAGES_DIR/$VM_NAME.qcow2"
SEED="$IMAGES_DIR/$VM_NAME-seed.img"
PIDFILE="$WORK_DIR/$VM_NAME.pid"
MONITOR_SOCK="$WORK_DIR/$VM_NAME.monitor"
CONSOLE_LOG="$WORK_DIR/$VM_NAME-console.log"
SSH_KEY="$WORK_DIR/id_ed25519"
SSH_DEST=root@127.0.0.1
SSH_OPTS=(-i "$SSH_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5)

vm_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

# vm_boot [seed] — start the VM daemonized; pass "seed" to attach the
# cloud-init seed ISO (first boot only).
vm_boot() {
    local args=(
        -name "$VM_NAME"
        -machine q35,accel=kvm -cpu host
        -smp "$VCPUS" -m "$MEMORY_MB"
        -drive "file=$DISK,if=virtio,format=qcow2"
        -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$SSH_PORT-:22"
        -device "virtio-net-pci,netdev=net0"
        -display none
        -serial "file:$CONSOLE_LOG"
        -pidfile "$PIDFILE"
        -monitor "unix:$MONITOR_SOCK,server,nowait"
        -daemonize
    )
    [ "${1:-}" = seed ] && args+=(-drive "file=$SEED,if=virtio,format=raw,readonly=on")
    "$QEMU_BIN" "${args[@]}"
}

# vm_wait_ssh [attempts] — poll until SSH answers (5s between attempts).
vm_wait_ssh() {
    local attempts=${1:-60}
    for _ in $(seq 1 "$attempts"); do
        ssh "${SSH_OPTS[@]}" "$SSH_DEST" true 2>/dev/null && return 0
        sleep 5
    done
    return 1
}

# vm_shutdown — graceful poweroff over SSH, hard quit via monitor as fallback.
vm_shutdown() {
    vm_running || return 0
    ssh "${SSH_OPTS[@]}" "$SSH_DEST" poweroff 2>/dev/null || true
    for _ in $(seq 1 60); do
        vm_running || return 0
        sleep 2
    done
    echo "vm_shutdown: graceful poweroff timed out, sending monitor quit" >&2
    if command -v nc >/dev/null; then
        echo quit | nc -U "$MONITOR_SOCK" >/dev/null 2>&1 || true
    else
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
    fi
    for _ in $(seq 1 15); do
        vm_running || return 0
        sleep 2
    done
    return 1
}

# vm_rsync_repo <repo_root> — mirror the host repo into /root/crab.
vm_rsync_repo() {
    rsync -a --delete -e "ssh ${SSH_OPTS[*]}" \
        --exclude .venv --exclude .cache --exclude build --exclude dist \
        --exclude '__pycache__' --exclude '.mypy_cache' --exclude '.pytest_cache' \
        --exclude '.ruff_cache' \
        "$1/" "$SSH_DEST:/root/crab/"
}

# vm_ensure_btrfs_noatime — enforce noatime on the guest's btrfs root.
#
# Read-induced atime bumps surface in `btrfs send` diffs as utimes-only
# entries, so a relatime mount fills every changeset with image binaries the
# sandbox merely executed (`/usr/bin/mv`, `libacl.so` …). vm-setup.sh sets
# this at provision time, but a pristine snapshot taken before that rule
# existed carries a stale `btrfs loop 0 0` fstab line, and every
# --revert-pristine restores it. So the repair runs on each boot instead of
# once at provision time. Quiet when the mount is already correct.
vm_ensure_btrfs_noatime() {
    ssh "${SSH_OPTS[@]}" "$SSH_DEST" '
        B=/var/lib/crab/crab.btrfs
        M=/var/lib/crab/btrfs
        mountpoint -q "$M" || exit 0
        findmnt -no OPTIONS "$M" | grep -qw noatime && exit 0
        echo "vm: remounting $M with noatime (required for clean changesets)" >&2
        mount -o remount,noatime "$M" || exit 1
        sed -i "s|^$B $M btrfs loop 0 0\$|$B $M btrfs loop,noatime 0 0|" /etc/fstab
    ' || echo "vm-lib: WARNING could not enforce noatime on the btrfs root" >&2
}
