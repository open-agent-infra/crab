#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  command cat <<'EOF'
Install Crab's full Linux v0 stack on Ubuntu x86-64.

Usage:
  sudo ./scripts/install-ubuntu.sh [options]

Options:
  --zpool NAME       Dedicated ZFS pool name (default: crab)
  --zpool-file PATH  Sparse backing file for a new pool
                     (default: /var/lib/crab/crab.zpool)
  --zpool-size SIZE  Sparse backing file size (default: 32G)
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

ZPOOL_NAME=crab
ZPOOL_FILE=/var/lib/crab/crab.zpool
ZPOOL_SIZE=32G
CONFIG_PATH=/etc/crab/config.yaml
SKIP_PACKAGES=0
CREATE_POOL=1

while (($#)); do
  case "$1" in
    --zpool)
      (($# >= 2)) || die "--zpool requires a value"
      ZPOOL_NAME=$2
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
  command_exists zfs || packages+=(zfsutils-linux)
  if apt-cache show "linux-headers-$(uname -r)" >/dev/null 2>&1; then
    packages+=("linux-headers-$(uname -r)")
  fi
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

for binary in docker runc criu zfs zpool clang gcc make python3; do
  command_exists "${binary}" || die "missing dependency after installation: ${binary}"
done

if command_exists systemctl; then
  systemctl enable --now docker >/dev/null 2>&1 || true
fi
docker info >/dev/null 2>&1 || die "Docker daemon is not reachable"

install -d -m 0755 /var/lib/crab /var/lib/crab/logs /opt/crab /etc/crab

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
sed "s|^zfs_dataset_prefix: crab/sandboxes$|zfs_dataset_prefix: ${ZPOOL_NAME}/sandboxes|" \
  "${REPO_ROOT}/config/crab.yaml" >"${config_tmp}"
install -D -m 0644 "${config_tmp}" "${CONFIG_PATH}"

log "Checking CRIU and ZFS"
criu check
zpool status "${ZPOOL_NAME}"

echo
echo "Crab is installed. Run the real checkpoint/restore smoke test:"
echo "  sudo ${REPO_ROOT}/scripts/smoke-rollback.sh --config ${CONFIG_PATH}"
