# Crab documentation

## Start here

- [Installation](installation.md): supported Ubuntu hosts, dependency setup,
  ZFS pool safety, and troubleshooting.
- [Getting started](getting-started.md): a no-API-key process and filesystem
  rollback walkthrough.
- [Daemon and CLI](daemon.md): daemon ownership, commands, cleanup behavior,
  and current operational limits.
- [Configuration](configuration-reference.md): the supported daemon YAML
  surface and the difference between smoke-test and agent-oriented configs.
- [Sandbox runtime baseline](sandbox-runtime-baseline.md): DNS, capabilities,
  network tri-state, exec deadlines, image pulling/cache policy, and
  service-VM egress diagnostics.
- [Cloud deployment](deploy-cloud.md): one-click VM/cloud deployment and
  post-deploy operations (Chinese).
- [Multi-tenancy](multi-tenancy.md): tenant, API key, and quota management
  through `crab-gateway` (Chinese).

## Build with Crab

- [Python SDK](sdk.md): create sandboxes, run commands, inspect checkpoints,
  and restore state.
- [Bring your own agent](byo-agent.md): implement the `Agent` contract and
  route in-sandbox or host-driven LLM traffic correctly.
- [iFlow trace replay](sdk-iflow-replay.md): replay a real recorded workflow
  without a model API key.

## Internals

- [Architecture](architecture.md): scheduler, executor, runtime, storage,
  interception, and recovery flows.
- [Telemetry](telemetry.md): JSONL records, correlation keys, and metric
  interpretation.
- [Agent integration notes](agent-integration-notes.md): restore-safe process,
  mount, and file-descriptor patterns for integration authors.
- [Host inspector](../crab/host_inspector/README.md): direct inspector testing
  and its process/filesystem signal matrix.

Historical experiment reports, PR design notes, and the old exhaustive mixed
benchmark configuration catalog live in [`legacy/docs/`](../legacy/docs/).
They are retained for archaeology and research reproducibility, not as current
user instructions.
