# Sandbox runtime baseline and images

This page defines the launch contract for new runc sandboxes. It applies to
the local daemon, authenticated gateway, bare-image SDK path, and Compose
templates. Existing running sandboxes are not rewritten during an upgrade.

## DNS and process capabilities

At create time a host-network sandbox materializes the host's
`/etc/resolv.conf`, including a reachable systemd-resolved loopback stub, so
split-DNS and resolver-routing policy are preserved. An isolated namespace
instead prefers `/run/systemd/resolve/resolv.conf` and rejects loopback or
unspecified nameservers. The selected file is written only after cloning the
immutable shared image rootfs, so host-specific DNS data is never baked into
an image-ID cache entry. Creation fails if no resolver usable in the selected
network mode is available.

The file then belongs to sandbox state:

- stop/start leaves it unchanged;
- checkpoint/restore and fork preserve the checkpointed version;
- a user's later edit is not overwritten by a lifecycle operation.

OCI bundles use an explicit Docker-like, non-privileged capability baseline.
It includes `CAP_SETUID` and `CAP_SETGID` so APT can use its `_apt` user, and
does not include `CAP_SYS_ADMIN`. All OCI capability vectors are explicit and
`process.noNewPrivileges` is `false`. Fork and restore retain the bundle
profile.

## Network selection

The Python and remote create APIs treat `network` as a tri-state value:

| Request | Effective behavior |
| --- | --- |
| `true` | Require a dedicated Crab network namespace and fail creation if allocation fails. |
| `false` | Share the daemon host's network namespace. |
| omitted or `null` | Use the daemon policy. |

The default policy selects an isolated namespace whenever runc sandbox
networking is enabled. The packaged daemon configuration enables it, so host
networking is an explicit `network=false` compatibility opt-out. Disabling
daemon sandbox networking changes the omitted/`null` default back to host mode;
an explicit `true` request still fails rather than silently downgrading.
`Sandbox.describe().metadata` reports `network_mode`, the original
`network_requested` value, and the lease/netns identity when isolated.

Port exposure requires an isolated namespace. Gateway rejects host-network
sandboxes with HTTP 400 because they have no guest address that identifies only
that sandbox; forwarding to host loopback could expose an unrelated host or
sandbox service.

The namespace boundary separates loopback, interfaces, routes, and port spaces.
All isolated sandboxes currently attach to one Crab bridge, however, and Crab
does not yet install an east-west deny policy between bridge ports. It must not
be interpreted as tenant-aware network segmentation.

Forks inherit the source's effective network mode. A fork of an isolated
source receives a distinct lease and netns; a host-network source remains in
host mode. Lease setup failure fails and rolls back the fork.

Remote adapters should send an explicit policy instead of depending forever
on the daemon default. In particular, `disagg-agent-platform` should add the
nullable `network` field to its Crab create wire type.

## Exec deadlines

`timeout_s` is enforced by the daemon with a dedicated cgroup for each exec
that has a finite timeout. On expiry Crab terminates that exec's complete descendant tree,
waits until the cgroup is empty, and then raises `SandboxExecTimeout`. A
cleanup failure is instead reported as `SandboxExecCleanupError`; it is never
presented as a completed timeout. Sibling execs and container init are outside
the per-exec cgroup. Untimed execs stay in the sandbox cgroup so host-inspector
filesystem events retain their normal sandbox attribution. Because the kernel
reports the child cgroup ID for a timeout-isolated exec, Crab conservatively
marks that sandbox filesystem-dirty when the exec completes; the backend diff
remains authoritative and the next checkpoint reset clears the marker.

Buffered execs continue as attached server work if a client disappears and
finish normally; they are not converted into detached work. A disconnected
streaming exec closes its generator and terminates its payload. Batch actions
use the same cleanup boundary and do not checkpoint a failed or timed-out
command. Timeout errors may include partial stdout/stderr.

Successfully launched background work remains supported. It must detach from
the attached command's stdio (for example, `nohup service >/tmp/service.log
2>&1 &`); after the foreground exec succeeds, surviving descendants are moved
back to the sandbox cgroup before the temporary exec cgroup is removed.

## Public image resolution

The initial on-demand contract is public Docker Hub, Linux/amd64. Both tags
and `sha256` digest-pinned references are accepted. Unqualified references
such as `python:3.12-slim` normalize to
`docker.io/library/python:3.12-slim`. The default pull policy is
`if-not-present`; sealed deployments can select `never`.

Crab resolves an immutable image ID/digest before export. The exported rootfs
and shared ZFS/btrfs/overlay base are keyed by immutable content plus Crab's
rootfs-preparation schema, not by a mutable tag. Pull and export locks ensure
concurrent cold creates publish one atomic cache result. Sandbox metadata
reports the requested reference, normalized reference, image ID, and digest.

Compatible images must:

- be Linux/amd64;
- provide `/bin/sh`;
- provide `sleep` at `/bin/sleep` or `/usr/bin/sleep`.

Crab preserves image environment and user defaults. Bare-image sandboxes keep
the SDK's existing `/work` command directory unless a command supplies an
explicit `cwd`; the image's `WorkingDir` is not selected automatically. Crab
deliberately replaces the image entrypoint/command with its long-running
`sleep infinity` sandbox init, so the image's normal application command is
not started automatically. Distroless and foreign-platform images fail early
with a typed compatibility error.

The daemon distinguishes malformed/denied references, missing images,
authentication requirements, registry rate limits, pull timeout, platform or
runtime incompatibility, oversized images, insufficient disk, and generic
pull failures. Private registry credentials are not supported by this
version.

## Cache and pull policy

Configure bounded pulls and optional prewarming under `images`:

```yaml
images:
  pull_policy: if-not-present       # or never
  allowed_registries: [docker.io]
  allowed_references: []            # empty means all allowed-registry refs
  pull_timeout_seconds: 600
  max_image_bytes: 8589934592       # compressed Docker-reported and expanded limit
  cache_max_bytes: 21474836480
  min_free_bytes: 2147483648
  cache_retention_seconds: 2592000
  prewarm:
    - python:3.12-slim              # best effort after daemon start
  required_prewarm: []              # failure prevents daemon readiness
```

Reference allowlist entries use shell-style matching against requested and
normalized names. Cache garbage collection evicts unlocked entries by age and
least-recent access when retention, disk reserve, or budget requires it.
Telemetry records reference, digest, cache hit/miss, duration, size when
Docker reports it, and a failure category. Registry credentials are never
included in those attributes or cache keys.

## Service-VM egress

QEMU user-mode networking (SLIRP) is the zero-setup development default in
`tools/vm/provision-service-vm.sh`. It can use a different resolver, CDN
endpoint, or proxy path than the host and can itself be the throughput
bottleneck. A slowdown already visible in the service VM is not evidence of a
runc networking regression.

Measure the exact same URL from the service VM and a fresh sandbox:

```bash
cd /root/crab/tools/vm
PYTHONPATH=/root/crab python3 diagnose-service-egress.py \
  'https://example.com/large-file' --network auto
```

The credential-free JSON output records DNS answers, remote address, connect
and first-byte timing, total duration, bytes, throughput, redirects, effective
image digest, and sandbox network mode. Use `--network host` and
`--network isolated` to compare both paths.

For production, attach the service VM to an operator-managed bridge:

```bash
SERVICE_VM_NET_MODE=tap \
SERVICE_VM_TAP_IFACE=tap-crab0 \
SERVICE_VM_TAP_SSH_HOST=192.0.2.50 \
bash tools/vm/provision-service-vm.sh
```

The operator must create and authorize the tap, attach it to the intended
bridge, assign the guest address through DHCP or static configuration, and
apply firewall/routing policy. Switching a running VM between SLIRP and tap
requires a controlled VM restart; the provisioning script never rewires host
networking automatically.

If policy requires an HTTP(S) proxy, configure Docker's service-VM pull path
and pass workload proxy environment explicitly to the sandbox. Do not put
proxy credentials in daemon YAML, image allowlists, diagnostic URLs, or
telemetry. Crab does not automatically copy arbitrary host proxy variables or
trust files into image exports. Its configured TLS-interception CA is an
explicit, content-keyed runtime input; use that path or an operator-controlled
per-sandbox secret mechanism for any additional trust material.
