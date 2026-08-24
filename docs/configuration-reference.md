# Configuration

Crab reads one YAML file when the daemon starts. The installer copies
[`config/crab.yaml`](../config/crab.yaml) to `/etc/crab/config.yaml`; treat that
file as the canonical v0 example.

```bash
sudo crab daemon start --config /etc/crab/config.yaml
```

Relative paths in a config file are resolved relative to that file, not the
caller's current directory.

## Runtime and storage

```yaml
runtime: runc
default_image: ubuntu:22.04
zfs_dataset_prefix: crab/sandboxes

storage_planes:
  runtime_root: /var/lib/crab/runtime
  storage_root: /var/lib/crab/checkpoints
  agent_state_root: /var/lib/crab/agents
  work_dir_host_root: /var/lib/crab/work
  image_cache_root: /var/lib/crab/images
```

- `runtime`: `runc` is the real v0 backend. `docker` selects an in-memory test
  implementation; it is not a Docker checkpoint/restore fallback.
- `zfs_dataset_prefix`: parent dataset for sandbox root filesystems. Name it
  explicitly; do not rely on automatic pool discovery in production.
- `storage_root`: checkpoint manifests and artifact metadata.
- `runtime_root`: runc bundles, runtime state, and prepared root filesystems.
- `agent_state_root`: host-side state used by built-in agent adapters.
- `work_dir_host_root`: host work directories created when an integration asks
  for one.
- `image_cache_root`: exported container-image rootfs cache.

`Sandbox(work_dir=...)` and `crab sandbox run --work-dir` create host bind
mounts. Their contents are outside ZFS rollback.

## Network and LLM interception

The safe default used by the no-key smoke test disables both:

```yaml
network:
  enable_sandbox_network: false

interceptor:
  enabled: false
```

An in-sandbox agent that sends LLM requests through Crab needs both features:

```yaml
network:
  enable_sandbox_network: true
  expected_sandboxes: 4

interceptor:
  enabled: true
  host: 127.0.0.1
  port: 0

forwarder:
  host: 127.0.0.1
  port: 0
```

Binding an agent that declares `requires_network_namespace = True` fails
clearly when the sandbox has no network namespace. Use an agent-oriented
config, such as
[`examples/sdk/configs/iflow_replay_engine.runc.yaml`](../examples/sdk/configs/iflow_replay_engine.runc.yaml),
instead of silently weakening isolation or bypassing interception.

## Host inspector

```yaml
host_inspector:
  launch_mode: process
  host: 127.0.0.1
  port: 0
  log_level: INFO
  log_file: /var/lib/crab/logs/host-inspector.log
```

Use `launch_mode: process` for real change detection. `in_process` is a
lightweight test implementation and does not consume real eBPF events.

## Checkpoint scheduling

```yaml
scheduler:
  min_checkpoint_interval_seconds: 0.0
  force_checkpoint_after_seconds: 0.0
  require_change_signal: true
  checkpoint_full_baseline_on_first_checkpoint: true
  prefer_checkpoint_during_llm_request: true
  require_llm_request_for_checkpoint: false
  inspect_without_pause: false
  incremental_process_enabled: false
```

Important settings:

- `require_change_signal`: avoid checkpoints when neither process nor
  filesystem state changed.
- `checkpoint_full_baseline_on_first_checkpoint`: make the first automatic
  checkpoint contain a complete recovery baseline.
- `require_llm_request_for_checkpoint`: restrict automatic checkpoints to an
  active intercepted request window.
- `inspect_without_pause`: live inspection is opt-in; the safer default pauses
  before inspection.
- `incremental_process_enabled`: opt in to CRIU pre-dump chains. Keep it off
  until the workload and retention policy have been validated together.

Manual `crab checkpoint create` and `Sandbox.checkpoint()` force a checkpoint;
they do not wait for the automatic scheduler to become due.

## Executor

```yaml
executor:
  max_workers: 2
  checkpoint_workers: 1
  restore_workers: 1
  coordination_workers: 1
  composite_step_workers: 2
  checkpoint_queue_size: 128
  max_retries: 0
  retry_backoff_seconds: 0.05
```

`max_workers` is the fallback for omitted checkpoint and restore worker
counts. Start conservatively: CRIU and ZFS work can put substantial pressure on
memory and storage bandwidth.

## CRIU and runc

```yaml
runc:
  command_timeout_seconds: 120.0
  zfs_prepare_timeout_seconds: 300.0
  checkpoint:
    tcp_established: true
    shell_job: true
    tcp_skip_in_flight: true
    ext_unix_sk: true
  restore:
    detach: true
    tcp_established: true
    shell_job: true
    ext_unix_sk: true
    lazy_pages: false
```

These flags map to runc/CRIU behavior and are workload-sensitive. The shipped
values are the tested v0 profile. Lazy-page restore is experimental and
requires additional kernel and CRIU support.

## Telemetry and logs

```yaml
telemetry:
  output: /var/lib/crab/logs/telemetry.jsonl
  detail_level: basic
  capture_command_output: false
  keep_in_memory_copy: false

logging:
  file: /var/lib/crab/logs/engine.log
  level: info
  file_mode: append
```

Keep `capture_command_output: false` unless command output is known not to
contain credentials or user data. See [Telemetry](telemetry.md) for the event
format and metric names.

## Validate a config

The most useful validation is starting the daemon in the foreground:

```bash
sudo crab daemon start --foreground --config /path/to/config.yaml
```

This resolves paths, initializes the host inspector, checks runtime storage,
and leaves errors visible in the terminal. Stop it with `Ctrl-C`.
