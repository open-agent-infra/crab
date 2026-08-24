#!/usr/bin/env bash
# In-VM one-time setup. Run as root inside the crab-dev VM.
#
# Two installs on purpose:
#   1. scripts/install-ubuntu.sh — the real operator path: verifies host deps,
#      creates the file-backed `crab` zpool, builds the eBPF inspector, and
#      installs a release copy into /opt/crab/venv with /etc/crab/config.yaml.
#      Exercising the shipped installer keeps it honest.
#   2. An *editable* install of /root/crab into the system Python so that
#      run-in-vm.sh test invocations always execute the freshly rsynced tree.

set -euo pipefail

REPO=/root/crab
cd "$REPO"

# Docker Hub is unreachable from this network; without a mirror every
# `docker build/pull` (installer check, real-host integration tests) times
# out. provision-vm.sh copies the host's /etc/docker/daemon.json into the VM
# before running this script; fall back to a known-good mirror otherwise.
if [ ! -f /etc/docker/daemon.json ]; then
    echo "==> configuring docker registry mirror (fallback)"
    printf '{\n  "registry-mirrors": ["https://docker.m.daocloud.io"]\n}\n' > /etc/docker/daemon.json
    systemctl restart docker
fi

echo "==> running the shipped installer (zpool + eBPF + /opt/crab venv)"
bash "$REPO/scripts/install-ubuntu.sh" --skip-packages

echo "==> installing the repo as editable into the system Python (dev/test copy)"
# PyPI directly is very slow over the slirp NIC on this network; the Aliyun
# mirror is fast. --ignore-installed typing_extensions: the debian-installed
# copy has no RECORD file and pip refuses to upgrade it otherwise.
printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\n' > /etc/pip.conf
python3 -m pip install --break-system-packages -q --ignore-installed typing_extensions
# pyarrow + swebench: not crab dependencies, but several benchmark test
# modules import them; without them ~11 test modules fail at collection.
python3 -m pip install --break-system-packages -e "$REPO" pyarrow swebench

echo "==> preparing a btrfs playground for the FilesystemProvider work"
# Mirrors the installer's sparse zpool-file pattern; used by the future
# BtrfsProvider integration tests (.cache/tasks/filesystem-provider-refactor.md §4).
BTRFS_FILE=/var/lib/crab/crab.btrfs
BTRFS_MNT=/var/lib/crab/btrfs
if ! mountpoint -q "$BTRFS_MNT"; then
    if [ ! -f "$BTRFS_FILE" ]; then
        truncate -s 32G "$BTRFS_FILE"
        mkfs.btrfs -q "$BTRFS_FILE"
    fi
    mkdir -p "$BTRFS_MNT"
    mount -o loop,noatime "$BTRFS_FILE" "$BTRFS_MNT"
    grep -q "$BTRFS_FILE" /etc/fstab || \
        echo "$BTRFS_FILE $BTRFS_MNT btrfs loop,noatime 0 0" >> /etc/fstab
elif ! findmnt -no OPTIONS "$BTRFS_MNT" | grep -qw noatime; then
    # atime updates leak into `btrfs send` diffs as utimes-only noise;
    # the changeset provider needs noatime for zfs-parity semantics.
    mount -o remount,noatime "$BTRFS_MNT"
    sed -i "s|^$BTRFS_FILE $BTRFS_MNT btrfs loop 0 0\$|$BTRFS_FILE $BTRFS_MNT btrfs loop,noatime 0 0|" /etc/fstab
fi

echo "==> setup complete"
