#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  command cat <<'EOF'
Install Crab's full Linux v0 stack on Ubuntu x86-64.

Usage:
  sudo ./scripts/install-ubuntu.sh [options]

Options:
  --fs-backend NAME  Filesystem backend: zfs (default), btrfs, or overlay
  --zpool NAME       Dedicated ZFS pool name (default: crab)
  --zpool-file PATH  Sparse backing file for a new pool
                     (default: /var/lib/crab/crab.zpool)
  --zpool-size SIZE  Sparse backing file size (default: 32G)
  --btrfs-file PATH  Sparse backing file for a new btrfs filesystem
                     (default: /var/lib/crab/crab.btrfs)
  --btrfs-root PATH  Mountpoint for the btrfs filesystem
                     (default: /var/lib/crab/btrfs)
  --btrfs-size SIZE  Sparse backing file size (default: 32G)
  --config PATH      Installed config path (default: /etc/crab/config.yaml)
  --skip-packages    Do not run apt; only verify/build/install/configure
  --no-create-pool   Require --zpool to name an existing pool
  -h, --help         Show this help

Safety:
  The script never selects an arbitrary existing pool and never repartitions
  a disk. When the requested pool does not exist, it creates a sparse-file
  pool at --zpool-file. An existing unimported backing file is never reused.
EOF
}

# Install Crab's full Linux v0 stack on Ubuntu x86-64.
#
# Usage:
#   sudo ./scripts/install-ubuntu.sh [options]
#
# Options:
#   --fs-backend NAME  Filesystem backend: zfs (default) or btrfs
#   --zpool NAME       Dedicated ZFS pool name (default: crab)
#   --zpool-file PATH  Sparse backing file for a new pool
#                      (default: /var/lib/crab/crab.zpool)
#   --zpool-size SIZE  Sparse backing file size (default: 32G)
#   --config PATH      Installed config path (default: /etc/crab/config.yaml)
#   --skip-packages    Do not run apt; only verify/build/install/configure
#   --no-create-pool   Require --zpool to name an existing pool
#   -h, --help         Show this help
#
# Safety:
#   The script never selects an arbitrary existing pool and never repartitions
#   a disk. When the requested pool does not exist, it creates a sparse-file
#   pool at --zpool-file. An existing unimported backing file is never reused.

die() {
  echo "install-ubuntu.sh: $*" >&2
  exit 1
}

log() {
  echo "==> $*"
}

require_root() {
  [[ ${EUID} -eq 0 ]] || die "run this installer with sudo"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

FS_BACKEND=zfs
ZPOOL_NAME=crab
ZPOOL_FILE=/var/lib/crab/crab.zpool
ZPOOL_SIZE=32G
BTRFS_FILE=/var/lib/crab/crab.btrfs
BTRFS_ROOT=/var/lib/crab/btrfs
BTRFS_SIZE=32G
CONFIG_PATH=/etc/crab/config.yaml
SKIP_PACKAGES=0
CREATE_POOL=1

while (($#)); do
  case "$1" in
    --fs-backend)
      (($# >= 2)) || die "--fs-backend requires a value"
      FS_BACKEND=$2
      shift 2
      ;;
    --zpool)
      (($# >= 2)) || die "--zpool requires a value"
      ZPOOL_NAME=$2
      shift 2
      ;;
    --btrfs-file)
      (($# >= 2)) || die "--btrfs-file requires a value"
      BTRFS_FILE=$2
      shift 2
      ;;
    --btrfs-root)
      (($# >= 2)) || die "--btrfs-root requires a value"
      BTRFS_ROOT=$2
      shift 2
      ;;
    --btrfs-size)
      (($# >= 2)) || die "--btrfs-size requires a value"
      BTRFS_SIZE=$2
      shift 2
      ;;
    --zpool-file)
      (($# >= 2)) || die "--zpool-file requires a value"
      ZPOOL_FILE=$2
      shift 2
      ;;
    --zpool-size)
      (($# >= 2)) || die "--zpool-size requires a value"
      ZPOOL_SIZE=$2
      shift 2
      ;;
    --config)
      (($# >= 2)) || die "--config requires a value"
      CONFIG_PATH=$2
      shift 2
      ;;
    --skip-packages)
      SKIP_PACKAGES=1
      shift
      ;;
    --no-create-pool)
      CREATE_POOL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

require_root

[[ ${ZPOOL_NAME} =~ ^[A-Za-z][A-Za-z0-9_.:-]*$ ]] || die "invalid zpool name: ${ZPOOL_NAME}"
[[ ${ZPOOL_FILE} = /* ]] || die "--zpool-file must be an absolute path"
[[ ${FS_BACKEND} = zfs || ${FS_BACKEND} = btrfs || ${FS_BACKEND} = overlay ]] || die "--fs-backend must be zfs, btrfs or overlay"
[[ ${BTRFS_FILE} = /* ]] || die "--btrfs-file must be an absolute path"
[[ ${BTRFS_ROOT} = /* ]] || die "--btrfs-root must be an absolute path"
[[ ${CONFIG_PATH} = /* ]] || die "--config must be an absolute path"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

[[ -f ${REPO_ROOT}/pyproject.toml ]] || die "cannot locate the Crab repository"
[[ -f ${REPO_ROOT}/config/crab.yaml ]] || die "missing config/crab.yaml"

source /etc/os-release
[[ ${ID:-} = ubuntu ]] || die "v0 installer supports Ubuntu only (found ${ID:-unknown})"
[[ $(uname -m) = x86_64 ]] || die "v0 installer supports x86-64 only"

if ((SKIP_PACKAGES == 0)); then
  log "Installing Ubuntu packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  packages=(
    ca-certificates curl gcc make pkg-config clang llvm
    libbpf-dev libelf-dev zlib1g-dev
    python3 python3-dev python3-pip python3-venv
  )
  command_exists docker || packages+=(docker.io)
  command_exists runc || packages+=(runc)
  command_exists criu || packages+=(criu)
  if [[ ${FS_BACKEND} = zfs ]]; then
    command_exists zfs || packages+=(zfsutils-linux)
  else
    command_exists btrfs || packages+=(btrfs-progs)
  fi
  if apt-cache show "linux-headers-$(uname -r)" >/dev/null 2>&1; then
    packages+=("linux-headers-$(uname -r)")
  fi
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

fs_binaries=(zfs zpool)
[[ ${FS_BACKEND} != zfs ]] && fs_binaries=(btrfs)
for binary in docker runc criu "${fs_binaries[@]}" clang gcc make python3; do
  command_exists "${binary}" || die "missing dependency after installation: ${binary}"
done

if command_exists systemctl; then
  systemctl enable --now docker >/dev/null 2>&1 || true
fi
docker info >/dev/null 2>&1 || die "Docker daemon is not reachable"

install -d -m 0755 /var/lib/crab /var/lib/crab/logs /opt/crab /etc/crab

if [[ ${FS_BACKEND} = zfs ]]; then
  if zpool list -H -o name "${ZPOOL_NAME}" >/dev/null 2>&1; then
    log "Using existing explicitly named zpool ${ZPOOL_NAME}"
  else
    ((CREATE_POOL == 1)) || die "zpool ${ZPOOL_NAME} does not exist"
    [[ ! -e ${ZPOOL_FILE} ]] || die "refusing to reuse existing pool file ${ZPOOL_FILE}"
    log "Creating dedicated sparse-file zpool ${ZPOOL_NAME} (${ZPOOL_SIZE})"
    install -d -m 0755 "$(dirname -- "${ZPOOL_FILE}")"
    truncate -s "${ZPOOL_SIZE}" "${ZPOOL_FILE}"
    if ! zpool create -m none "${ZPOOL_NAME}" "${ZPOOL_FILE}"; then
      rm -f -- "${ZPOOL_FILE}"
      die "failed to create zpool ${ZPOOL_NAME}"
    fi
  fi
else
  if [[ $(stat -f -c %T "${BTRFS_ROOT}" 2>/dev/null) = btrfs ]]; then
    log "Using existing btrfs filesystem at ${BTRFS_ROOT}"
    # atime updates leak into `btrfs send` diffs as utimes-only noise;
    # the changeset provider needs noatime for zfs-parity semantics.
    if ! findmnt -no OPTIONS --target "${BTRFS_ROOT}" | grep -qw noatime; then
      log "Remounting ${BTRFS_ROOT} with noatime (required for clean changesets)"
      mount -o remount,noatime "${BTRFS_ROOT}" || die "failed to remount ${BTRFS_ROOT} with noatime"
      # A remount alone does not survive a reboot, and the noise it lets
      # back in is silent: changesets simply start listing binaries the
      # sandbox executed. Persist it, or say plainly that we could not.
      BTRFS_MOUNT_TARGET=$(findmnt -no TARGET --target "${BTRFS_ROOT}")
      # --target resolves to the enclosing mount, which may be a parent
      # (e.g. /) rather than BTRFS_ROOT itself. Persisting noatime there
      # widens the change beyond Crab's directory, so flag it.
      if [[ "${BTRFS_MOUNT_TARGET}" != "${BTRFS_ROOT}" ]]; then
        log "WARNING: ${BTRFS_ROOT} is not its own mount; noatime applies to the"
        log "WARNING: enclosing filesystem ${BTRFS_MOUNT_TARGET}. Mount Crab's btrfs"
        log "WARNING: on a dedicated subvolume/loop file to scope this to Crab."
      fi
      if awk -v t="${BTRFS_MOUNT_TARGET}" '$1 !~ /^#/ && $2 == t { found = 1 } END { exit !found }' /etc/fstab; then
        FSTAB_TMP=$(mktemp)
        awk -v t="${BTRFS_MOUNT_TARGET}" 'BEGIN { OFS = " " }
          $1 !~ /^#/ && $2 == t && $4 !~ /(^|,)noatime(,|$)/ { $4 = $4 ",noatime" }
          { print }' /etc/fstab >"${FSTAB_TMP}" &&
          cat "${FSTAB_TMP}" >/etc/fstab
        rm -f -- "${FSTAB_TMP}"
        log "Persisted noatime on the ${BTRFS_MOUNT_TARGET} entry in /etc/fstab"
      else
        log "WARNING: ${BTRFS_MOUNT_TARGET} has no /etc/fstab entry, so noatime is not persistent."
        log "WARNING: add noatime to how that filesystem is mounted, or changesets will silently"
        log "WARNING: regain read-induced noise after the next reboot."
      fi
    fi
  else
    ((CREATE_POOL == 1)) || die "no btrfs filesystem mounted at ${BTRFS_ROOT}"
    # Same safety rule as the zpool flow: never adopt an existing backing
    # file whose provenance we don't know.
    [[ ! -e ${BTRFS_FILE} ]] || die "refusing to reuse existing btrfs file ${BTRFS_FILE}"
    log "Creating dedicated loop-backed btrfs filesystem at ${BTRFS_ROOT} (${BTRFS_SIZE})"
    install -d -m 0755 "$(dirname -- "${BTRFS_FILE}")" "${BTRFS_ROOT}"
    truncate -s "${BTRFS_SIZE}" "${BTRFS_FILE}"
    if ! mkfs.btrfs -q "${BTRFS_FILE}" || ! mount -o loop,noatime "${BTRFS_FILE}" "${BTRFS_ROOT}"; then
      rm -f -- "${BTRFS_FILE}"
      die "failed to create btrfs filesystem at ${BTRFS_ROOT}"
    fi
    grep -q "${BTRFS_FILE}" /etc/fstab ||       echo "${BTRFS_FILE} ${BTRFS_ROOT} btrfs loop,noatime 0 0" >>/etc/fstab
  fi
fi

log "Building the eBPF host inspector"
make -C "${REPO_ROOT}/crab/host_inspector/bpf" clean all

log "Installing Crab into /opt/crab/venv"
python3 -m venv /opt/crab/venv
/opt/crab/venv/bin/python -m pip install --upgrade pip setuptools wheel
/opt/crab/venv/bin/python -m pip install "${REPO_ROOT}"
ln -sfn /opt/crab/venv/bin/crab /usr/local/bin/crab
ln -sfn /opt/crab/venv/bin/crabd /usr/local/bin/crabd

log "Installing ${CONFIG_PATH}"
config_tmp=$(mktemp)
trap 'rm -f -- "${config_tmp}"' EXIT
if [[ ${FS_BACKEND} = zfs ]]; then
  sed "s|^zfs_dataset_prefix: crab/sandboxes$|zfs_dataset_prefix: ${ZPOOL_NAME}/sandboxes|" \
    "${REPO_ROOT}/config/crab.yaml" >"${config_tmp}"
else
  # btrfs and overlay share the btrfs mount; overlay derives its area
  # from btrfs_root (<btrfs_root>/overlay) so no extra key is needed.
  sed "s|^zfs_dataset_prefix: crab/sandboxes$|filesystem_backend: ${FS_BACKEND}\nbtrfs_root: ${BTRFS_ROOT}|" \
    "${REPO_ROOT}/config/crab.yaml" >"${config_tmp}"
fi
install -D -m 0644 "${config_tmp}" "${CONFIG_PATH}"

log "Checking CRIU and the filesystem backend"
criu check
if [[ ${FS_BACKEND} = zfs ]]; then
  zpool status "${ZPOOL_NAME}"
else
  btrfs filesystem show "${BTRFS_ROOT}"
fi
if [[ ${FS_BACKEND} = overlay ]]; then
  # The engine's ensure_root re-checks this at startup; fail early here.
  modprobe overlay 2>/dev/null || true
  grep -qw overlay /proc/filesystems || die "kernel lacks overlayfs support"
fi

echo
echo "Crab is installed. Run the real checkpoint/restore smoke test:"
echo "  sudo ${REPO_ROOT}/scripts/smoke-rollback.sh --config ${CONFIG_PATH}"
