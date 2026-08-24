# Install Crab on Ubuntu

Crab v0 supports Ubuntu 24.04/26.04 on x86-64 and runs as root. The full
backend requires Docker, runc, CRIU, ZFS, and the eBPF host inspector.

## One-command installation

From a Crab checkout:

```bash
sudo ./scripts/install-ubuntu.sh
```

The script:

1. installs missing Ubuntu runtime and build dependencies;
2. starts Docker when systemd is available;
3. creates `/var/lib/crab` and `/opt/crab`;
4. creates a dedicated 32 GiB sparse-file ZFS pool named `crab` when it does
   not already exist;
5. builds the eBPF host-inspector helper from source;
6. installs Crab in `/opt/crab/venv` and links `crab`/`crabd` into
   `/usr/local/bin`;
7. installs [the default config](../config/crab.yaml) at
   `/etc/crab/config.yaml`;
8. runs `criu check` and `zpool status`.

The 32 GiB pool file is sparse: it consumes blocks as data is written rather
than allocating 32 GiB immediately.

## Pool options

Use a different dedicated pool name and backing file:

```bash
sudo ./scripts/install-ubuntu.sh \
  --zpool my-crab \
  --zpool-file /var/lib/crab/my-crab.zpool \
  --zpool-size 64G
```

Use an existing explicitly named pool without allowing pool creation:

```bash
sudo ./scripts/install-ubuntu.sh --zpool mypool --no-create-pool
```

The installer never picks the first pool on the machine. If the requested
pool does not exist and `--no-create-pool` is set, installation stops.

For hosts where dependencies are managed separately:

```bash
sudo ./scripts/install-ubuntu.sh --skip-packages
```

## Verify the installation

Run the real process-and-filesystem recovery test:

```bash
sudo ./scripts/smoke-rollback.sh
```

A successful run ends with:

```text
PASS: filesystem content and the background process were restored.
```

The smoke script removes its sandbox and stops the daemon if it started the
daemon. Pass `--keep` to retain the sandbox for inspection.

## Installed state

The default profile uses:

| Purpose | Location |
| --- | --- |
| Configuration | `/etc/crab/config.yaml` |
| Python environment | `/opt/crab/venv` |
| Runtime bundles and runc state | `/var/lib/crab/runtime` |
| Checkpoint manifests | `/var/lib/crab/checkpoints` |
| Root filesystem image cache | `/var/lib/crab/images` |
| Logs and telemetry | `/var/lib/crab/logs` |
| Root daemon socket | `/run/crab/crab.sock` |

## Troubleshooting

Check each host component:

```bash
sudo docker info
sudo runc --version
sudo criu check
sudo zpool status crab
sudo crab daemon start --foreground --config /etc/crab/config.yaml
```

The foreground daemon command is the best first diagnostic because it keeps
the host-inspector, runc, CRIU, and ZFS errors in the terminal.

Crab v0 does not rehydrate live sandboxes after a daemon restart. Remove
sandboxes before stopping the daemon, and do not restart it while a workflow is
active.
