# Crab SDK

The Python SDK talks to the same long-running daemon as the `crab` CLI. The
daemon owns runc, CRIU, ZFS datasets, checkpoint storage, and host inspection;
the SDK process owns your application and agent control flow.

Install Crab and start the daemon before creating a `Sandbox`. For the manual
checkpoint examples below, the default config is sufficient:

```bash
sudo ./scripts/install-ubuntu.sh
sudo crab daemon start --config /etc/crab/config.yaml
```

Because the installed v0 daemon and socket are root-owned, run SDK examples as
root as well, or explicitly arrange a supported socket/permission model.

## Launch, checkpoint, and restore

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox(image="ubuntu:22.04", engine=engine)
    try:
        sandbox.commands.run("echo before > /root/state.txt")
        checkpoint_id = sandbox.checkpoint()

        sandbox.commands.run("echo after > /root/state.txt")
        sandbox.restore(checkpoint_id)

        assert sandbox.files.read("/root/state.txt").strip() == "before"
    finally:
        sandbox.kill()
```

`Engine.connect()` uses `/run/crab/crab.sock` for a root daemon. Pass a socket
path explicitly, or set `CRAB_DAEMON_SOCKET`, when the daemon uses another
location.

## Sandbox API

Create a sandbox from an image:

```python
sandbox = Sandbox(
    image="ubuntu:22.04",
    name="my-sandbox",       # optional; generated when omitted
    env={"MODE": "test"},   # environment inherited by sandbox commands
    network=False,           # request a network namespace explicitly
    engine=engine,
)
```

The main operations are:

```python
result = sandbox.commands.run(
    ["sh", "-lc", "uname -a"],
    cwd="/root",
    timeout=30,
    check=True,
)

sandbox.files.write("/root/note.txt", "hello\n")
text = sandbox.files.read("/root/note.txt")
exists = sandbox.files.exists("/root/note.txt")

checkpoint_id = sandbox.checkpoint()
checkpoints = sandbox.checkpoints.list()
sandbox.restore(checkpoint_id)
sandbox.checkpoints.delete(checkpoint_id)

sandbox.pause()
sandbox.resume()
sandbox.kill()
```

`commands.run()` currently buffers output and does not allocate a PTY. File
helpers are intended for text-sized files; use sandbox commands for binary or
streaming transfers.

### Host work directories are not rolled back

`Sandbox(work_dir="./repo")` bind-mounts that host directory at `/work`.
Checkpoint restore does not revert it. This is useful when files must survive
a sandbox rollback, but it is the wrong choice when the repository itself must
be transactional.

For rollback of project files, place the checkout inside the sandbox's
ZFS-backed root filesystem, for example under `/root/project`.

## Inspect existing state

The SDK can reconnect to a sandbox that the same live daemon is already
tracking:

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox.connect("my-sandbox", engine=engine)
    print(sandbox.checkpoints.list())
```

The daemon does not currently rebuild its in-memory sandbox registry after a
restart, so this is not restart recovery.

## Agent integrations

Crab exposes a small `Agent` contract and two built-in profiles:

| Profile | Protocol | Execution model |
| --- | --- | --- |
| `ClaudeCodeAgent` | Anthropic | Installs and invokes Claude Code inside the sandbox |
| `IFlowAgent` | OpenAI-compatible | Uses the packaged iFlow runtime inside the sandbox |

Both built-ins require a sandbox network namespace and the LLM interceptor.
The safe default config disables those features because the no-key rollback
smoke test does not need them. Start an agent-oriented config instead, such as
the iFlow replay config:

```bash
sudo crab daemon start \
  --config examples/sdk/configs/iflow_replay_engine.runc.yaml
```

Then bind the agent to a networked sandbox and register its real or replay
upstream:

```python
from crab import Engine, Sandbox
from crab.agents_builtin.iflow import IFlowAgent

with Engine.connect() as engine:
    sandbox = Sandbox(
        image="crab-iflow-bench:workspace",
        network=True,
        engine=engine,
    )
    try:
        agent = IFlowAgent().bind(
            sandbox,
            llm_url="http://127.0.0.1:18080",
        )
        result = agent.run("Solve the task")
    finally:
        sandbox.kill()
```

For a complete no-API-key workflow, use the
[iFlow trace replay example](sdk-iflow-replay.md). To integrate another agent,
see [Bring your own agent](byo-agent.md).

## How LLM interception is routed

When an agent is bound with `llm_url=...`, Crab registers that upstream for
the sandbox and exposes the interceptor URL through provider-specific base URL
environment variables.

- In-sandbox agents are attributed through the sandbox network lease and
  source IP mapping.
- Host-driven agents must send `X-Agent-Sandbox-Id: <sandbox-id>` explicitly;
  setting `CRAB_SANDBOX_ID` alone does not modify arbitrary HTTP clients.

For example, a host-side Anthropic client can pass a default header:

```python
import anthropic

client = anthropic.Anthropic(
    base_url=agent.llm_base_url,
    default_headers={"X-Agent-Sandbox-Id": str(sandbox.sandbox_id)},
)
```

API keys remain the responsibility of the upstream client or agent. Avoid
putting them in Crab config files, telemetry, or command output.

## Templates

`DockerComposeTemplate` translates one Compose service into a runc sandbox and
can materialize Terminal-Bench task assets:

```python
from crab import Sandbox
from crab.templates import DockerComposeTemplate

template = DockerComposeTemplate(
    compose_file="/path/to/docker-compose.yaml",
    service_name="client",
    task_root="/path/to/task-root",
)
sandbox = Sandbox(template=template, engine=engine)
```

This is a translation layer, not a general Docker Compose runtime. Multi-
service behavior should be validated for each task.

## Current limitations

- `Sandbox.fork()` is reserved but not implemented in the public SDK path.
- Daemon restart rehydration is not implemented.
- Exec output is buffered; streaming and PTY support are not implemented.
- `resources`, `timeout`, and `labels` constructor arguments are currently
  advisory metadata, not enforced resource limits or lifecycle policies.
- Only OpenAI-compatible and Anthropic LLM base-URL conventions are built in.
