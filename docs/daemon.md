# Crab daemon and CLI

`crabd` is the single host process that owns runc state, ZFS datasets, the
host inspector, and—when enabled—the LLM interceptor, forwarder, and sandbox
network. Both the Python SDK and `crab` CLI call it over HTTP on a Unix socket.

Crab v0 supports one root-owned daemon per host. Do not start a second daemon
against the same runtime paths or ZFS dataset prefix.

## Start and inspect the daemon

The installer creates `/etc/crab/config.yaml` and `/run/crab/crab.sock` is the
default root socket:

```bash
sudo crab daemon start --config /etc/crab/config.yaml
sudo crab daemon status
sudo crab info
```

For troubleshooting, keep logs in the terminal:

```bash
sudo crab daemon start --foreground --config /etc/crab/config.yaml
```

Use `--socket PATH` for a non-default socket. SDK clients can set
`CRAB_DAEMON_SOCKET` to the same path. The socket is mode `0600`; v0 has no
additional authentication or multi-user authorization layer.

The daemon is independent of SDK client processes. Closing an SDK connection
does not stop the daemon or destroy its sandboxes.

## CLI overview

```text
crab [--socket PATH] [--timeout SECONDS] [--json] <command>

  daemon start [--config FILE] [--foreground] [--log-file PATH]
  daemon stop
  daemon status
  info

  sandbox run [OPTIONS] IMAGE [-- COMMAND ...]
  sandbox ls
  sandbox exec SANDBOX_ID [OPTIONS] -- COMMAND ...
  sandbox stop|pause|resume SANDBOX_ID
  sandbox rm SANDBOX_ID [SANDBOX_ID ...]

  checkpoint create SANDBOX_ID
  checkpoint ls SANDBOX_ID
  checkpoint rm SANDBOX_ID CHECKPOINT_ID [--cascade]
  restore SANDBOX_ID CHECKPOINT_ID
```

Run `crab <group> <command> --help` for the authoritative option list.
`--json` is available for commands that return structured lists or status.

### Launch a sandbox

Arguments recognized by `sandbox run` must appear before the image:

```bash
SBX=$(sudo crab sandbox run --detach --name demo ubuntu:22.04)
sudo crab sandbox exec "$SBX" -- uname -a
```

Other useful options:

- `-e KEY=VALUE`: add a sandbox environment value; repeat as needed.
- `--work-dir PATH`: bind a host directory at `/work`. This directory is not
  rolled back by Crab checkpoints.
- `--network`: allocate a sandbox network namespace.
- `--rm`: destroy the sandbox after its command exits.
- `--detach`: create the sandbox and print its ID without executing a command.

Exec output is currently buffered and printed when the command finishes. PTY
and streaming exec are not implemented.

### Checkpoint and restore

```bash
CKPT=$(sudo crab checkpoint create "$SBX")
sudo crab checkpoint ls "$SBX"

sudo crab restore "$SBX" "$CKPT"

sudo crab checkpoint rm "$SBX" "$CKPT"
```

`checkpoint ls` reports whether each manifest contains process state,
filesystem state, or both. Automatic checkpoints may be filesystem-only when
the inspector sees only filesystem changes. Manual `checkpoint create`
requests a full process-and-filesystem checkpoint.

Incremental process checkpoints form dependency chains. Deleting a parent
with live descendants is rejected unless `--cascade` is supplied.

### Stop, remove, and daemon shutdown

- `sandbox stop` stops the container but keeps its bundle and ZFS dataset so
  it can still be restored or removed.
- `sandbox rm` destroys the live container and its per-sandbox ZFS dataset.
  Stored checkpoint manifests are not automatically pruned.
- `daemon stop` removes every sandbox tracked by that daemon session, then
  stops the inspector, interceptor, forwarder, network bridge, and socket.

`daemon stop` is therefore a destructive sandbox-lifecycle operation, not just
a way to disconnect the API. Remove or preserve anything important before
calling it.

The following host state remains after a clean daemon stop:

- the ZFS pool and parent dataset;
- checkpoint manifests and artifacts under `storage_root`;
- exported image rootfs cache;
- runtime/log directories that are not per-sandbox live datasets.

## SDK connection

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox(image="ubuntu:22.04", engine=engine)
    try:
        print(sandbox.commands.run("uname -a").stdout)
    finally:
        sandbox.kill()
```

`Sandbox(...)` without an explicit engine lazily connects to the default
daemon socket. Supplying an `Engine` explicitly makes ownership and connection
lifetime clearer in applications.

## Restart and crash behavior

The daemon does not yet persist and rebuild its sandbox registry. After a
clean stop, tracked sandboxes have already been destroyed. After a crash or
`kill -9`, runc records and ZFS datasets may remain on disk, but a restarted
daemon does not adopt them.

Inspect leftovers with the same explicit paths and dataset prefix from the
daemon config. Do not run broad cleanup commands against an arbitrary runc
root or ZFS pool.

## Current limitations

- one root-owned, single-user daemon per host;
- no remote or multi-host daemon API;
- no restart rehydration;
- no exec streaming or PTY;
- no rootless runc/CRIU/ZFS backend;
- checkpoint data outlives `sandbox rm` until removed explicitly.
