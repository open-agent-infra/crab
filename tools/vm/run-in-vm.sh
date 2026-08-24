#!/usr/bin/env bash
# Sync the host repository into the crab-dev VM, then run one shell command
# inside the VM's repo copy. Plain QEMU, no libvirt (see vm-lib.sh).
#
# Usage:
#   bash tools/vm/run-in-vm.sh '<shell command>'
#   bash tools/vm/run-in-vm.sh                    # interactive shell in the VM
#   bash tools/vm/run-in-vm.sh --revert-pristine  # roll back to post-provision baseline
#
# Examples:
#   bash tools/vm/run-in-vm.sh 'python3 -m unittest discover -s tests -v'
#   bash tools/vm/run-in-vm.sh 'python3 -m unittest -v tests.test_storage tests.test_runc_runtime_checkpoint'
#   bash tools/vm/run-in-vm.sh 'bash /root/vm-verify.sh'
#
# The host repository is the source of truth; /root/crab inside the VM is a
# disposable mirror that this script overwrites (rsync --delete) on every
# invocation. Never edit code inside the VM.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# shellcheck source=vm-lib.sh
source "$SCRIPT_DIR/vm-lib.sh"

die() { echo "run-in-vm: $*" >&2; exit 1; }

[ -f "$SSH_KEY" ] || die "missing SSH key: $SSH_KEY (run tools/vm/provision-vm.sh first)"
[ -f "$DISK" ] || die "missing VM disk: $DISK (run tools/vm/provision-vm.sh first)"

# --- optional: revert to the pristine qcow2 snapshot --------------------------
if [ "${1:-}" = --revert-pristine ]; then
    echo "run-in-vm: reverting '$VM_NAME' to the pristine snapshot" >&2
    vm_shutdown || die "VM did not shut down for revert"
    qemu-img snapshot -a pristine "$DISK"
    vm_boot
    vm_wait_ssh 60 || die "VM did not come back after revert"
    vm_ensure_btrfs_noatime
    echo "run-in-vm: revert complete" >&2
    exit 0
fi

# --- ensure the VM is running -------------------------------------------------
if ! vm_running; then
    echo "run-in-vm: starting VM '$VM_NAME'" >&2
    vm_boot
fi
vm_wait_ssh 60 || die "VM did not become reachable over SSH"
# The pristine snapshot predates the noatime requirement, so this self-heals
# on every boot rather than trusting provision-time setup (see vm-lib.sh).
vm_ensure_btrfs_noatime

# --- sync the repository ------------------------------------------------------
vm_rsync_repo "$REPO_ROOT"

# Keep the in-VM copies of the setup/verify scripts current so host-side edits
# take effect without a manual scp.
rsync -a -e "ssh ${SSH_OPTS[*]}" \
    "$SCRIPT_DIR/vm-setup.sh" "$SCRIPT_DIR/vm-verify.sh" "$SSH_DEST:/root/"

# --- run ----------------------------------------------------------------------
if [ $# -eq 0 ]; then
    exec ssh -t "${SSH_OPTS[@]}" "$SSH_DEST" 'cd /root/crab && exec bash'
fi
[ $# -eq 1 ] || die "pass the VM command as one quoted shell string"
exec ssh "${SSH_OPTS[@]}" "$SSH_DEST" "cd /root/crab && $1"
