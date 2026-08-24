#!/usr/bin/env bash
# In-VM environment verification checklist. Run as root inside the crab-dev VM.
# Exits nonzero if any hard check fails. Safe to re-run at any time.

set -uo pipefail

REPO=/root/crab
FAILURES=0

check() {
    local label=$1; shift
    if "$@" >/dev/null 2>&1; then
        echo "ok    $label"
    else
        echo "FAIL  $label"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "== kernel & modules =="
echo "kernel: $(uname -r)"
check "kernel BTF present (/sys/kernel/btf/vmlinux)" test -e /sys/kernel/btf/vmlinux
check "zfs module loaded" bash -c 'lsmod | grep -q "^zfs"'
check "btrfs available" bash -c 'grep -qw btrfs /proc/filesystems'
check "overlay available" bash -c 'grep -qw overlay /proc/filesystems'

echo "== crab runtime stack =="
for tool in docker runc criu zfs zpool btrfs clang git rsync; do
    check "tool: $tool" command -v "$tool"
done
check "docker daemon reachable" docker info
check "criu check" criu check
check "crab zpool present" zpool list -H -o name crab
check "btrfs playground mounted" mountpoint -q /var/lib/crab/btrfs

echo "== python project =="
# cd to a neutral directory first: /root contains the repo checkout named
# `crab`, which Python would otherwise import as a namespace package
# (crab.__file__ == None) instead of the installed package.
check "package importable" bash -c 'cd / && python3 -c "import crab"'
check "editable install points at /root/crab" \
    bash -c 'cd / && python3 -c "import crab, sys; sys.exit(0 if str(crab.__file__).startswith(\"/root/crab/\") else 1)"'
check "operator CLI installed" command -v crab
check "config installed" test -f /etc/crab/config.yaml

echo "== eBPF host inspector =="
check "bpf objects built" bash -c "ls $REPO/crab/host_inspector/bpf/*.o"

if [ "$FAILURES" -ne 0 ]; then
    echo
    echo "verification FAILED: $FAILURES check(s); fix before handing the VM to the agent"
    exit 1
fi

echo
echo "== test suites (this takes a while) =="
cd "$REPO"
set -e
# Dependency-light set first (mirrors .github/workflows/ci.yml), then the full
# suite: real-host tests self-skip only when docker/runc/criu/zfs are missing,
# so inside this VM they run for real.
python3 -m unittest -v \
    tests.test_remote_engine_checkpoint \
    tests.test_image_runtime \
    tests.test_sdk_sandbox \
    tests.test_iflow_trace_replay
python3 -m unittest discover -s tests -v

echo
echo "verification PASSED: environment is ready for the agent"
