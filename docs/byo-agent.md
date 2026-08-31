# Bring Your Own Agent

This guide shows how to integrate a custom agent into Crab through the
SDK. The minimal contract is intentionally small — you declare which LLM
protocol your agent speaks and provide `install()` + `execute()` methods.

For the broader SDK surface, see [sdk.md](sdk.md).

## The contract

```python
from crab import Agent, TaskResult

class MyAgent(Agent):
    name = "my-agent"            # used for logs and registry lookup
    llm_protocol = "anthropic"   # "openai" or "anthropic"
    version = "1.0"              # optional, used for image caching keys

    def install(self, sbx):
        """Run once when the agent is bound to a sandbox.

        Use this to install your agent's CLI, copy assets, warm caches, etc.
        Strict failure semantics: raising here aborts agent attachment and
        propagates the exception to `agent.bind(sbx, ...)`.
        """

    def execute(self, sbx, task: str) -> TaskResult:
        """Execute a single task. Required.

        Runs in the caller's thread. For in-sandbox agents the body typically
        does one `sbx.commands.run(...)` driving the agent CLI inside the
        sandbox. For on-host agents the body runs its own loop.
        """

    def on_restore(self, sbx):
        """Optional. Called after a checkpoint restore."""

    def request_stop(self):
        """Optional. Cooperative cancellation hook."""
```

That's the whole contract.

## In-sandbox agents

This is the common case: an agent CLI in the sandbox drives its own LLM/tool
loop. The example assumes your image already contains a `my-agent` executable;
replace its command-line arguments with those of the agent you integrate.

```python
from crab import Agent, TaskResult, register_agent


class MyCliAgent(Agent):
    name = "my-cli-agent"
    llm_protocol = "openai"
    requires_network_namespace = True

    def install(self, sbx):
        sbx.commands.run("command -v my-agent >/dev/null", check=True)

    def execute(self, sbx, task):
        result = sbx.commands.run(
            argv=["my-agent", "--prompt", task],
            env=self.command_env(),
            check=False,
        )
        return TaskResult(
            exit_code=result.returncode,
            output=result.stdout,
            extra={"stderr": result.stderr},
        )


register_agent("my-cli-agent", MyCliAgent)
```

User-side:

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sbx = Sandbox(image="my-agent-image:latest", network=True, engine=engine)
    try:
        agent = MyCliAgent().bind(sbx, llm_url="https://api.example.com/v1")
        print(agent.run("Fix the failing tests"))
    finally:
        sbx.kill()
```

`self.command_env()` supplies `OPENAI_BASE_URL` and `CRAB_SANDBOX_ID` to
that sandbox command. With a Crab-managed network namespace, the interceptor
attributes traffic using the sandbox network lease and forwards it to the
registered upstream.

## On-host agents

For agents that orchestrate the sandbox from outside (think
`terminus`/`mini-swe`-style tmux drivers), the `execute()` body itself is the
LLM/tool loop and issues many small `sbx.commands.run(...)` calls.

```python
import os

import anthropic
from crab import Agent, TaskResult


class HostDrivenAgent(Agent):
    name = "host-driven-example"
    llm_protocol = "anthropic"

    def install(self, sbx):
        sbx.commands.run("apt-get update && apt-get install -y tmux", check=True)
        sbx.commands.run("tmux new-session -d -s work", check=True)

    def execute(self, sbx, task):
        # Agent.run has set ANTHROPIC_BASE_URL in this process while execute()
        # is running, so the anthropic SDK picks up the interceptor URL.
        # A host-driven client must attach the sandbox id itself.
        client = anthropic.Anthropic(
            default_headers={
                "X-Agent-Sandbox-Id": str(sbx.sandbox_id),
            }
        )

        history = [{"role": "user", "content": task}]
        for step in range(20):
            response = client.messages.create(
                model=os.environ["ANTHROPIC_MODEL"],
                max_tokens=4096,
                messages=history,
            )
            text = response.content[0].text
            if "TASK_DONE" in text:
                return TaskResult(exit_code=0, output=text)

            # The model emits a shell command; run it in the sandbox.
            cmd = self._extract_command(text)
            sandbox_result = sbx.commands.run(cmd, check=False)
            history.append({"role": "assistant", "content": text})
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"Exit code: {sandbox_result.returncode}\n"
                        f"Output:\n{sandbox_result.stdout[:2000]}"
                    ),
                }
            )

        return TaskResult(exit_code=1, output="step budget exhausted")

    def _extract_command(self, text):
        # Strip a fenced block, parse, return a single command. Skipped here
        # for brevity.
        ...
```

User-side is identical:

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sbx = Sandbox(image="ubuntu:22.04", engine=engine)
    try:
        agent = HostDrivenAgent().bind(sbx, llm_url="https://api.anthropic.com")
        agent.run("Set up the project and run the migration")
    finally:
        sbx.kill()
```

Host-driven clients must add `X-Agent-Sandbox-Id` themselves. Setting
`CRAB_SANDBOX_ID` does not automatically modify arbitrary HTTP clients.

## Multi-task lifecycle

Both styles support multiple sequential tasks on the same sandbox. The
agent's in-sandbox state persists between tasks because the sandbox is not
torn down between `run()` calls. If `/work` is a host bind mount, its files
also persist, but they are outside checkpoint rollback.

```python
sbx = Sandbox(image="ubuntu:22.04")
agent = MyAgent().bind(sbx, llm_url=...)
agent.run("Set up the project")
agent.run("Now add a CHANGELOG entry")
agent.run("Bump the version")
```

## Registering and resolving

Three ways to point the SDK at your agent:

```python
# Instance — what you'd write in a quick experiment.
agent = MyAgent().bind(sbx)

# Registered name — useful when application configuration selects an agent.
register_agent("my-agent", MyAgent)
agent = resolve_agent("my-agent").bind(sbx)

# Import path — useful for plugin-style application layouts.
agent = resolve_agent("my_pkg.agents:MyAgent").bind(sbx)
```

## What the engine does for you

When `agent.install()` or `agent.execute()` is called:

1. The agent exposes the provider-specific interceptor URL as
   `agent.llm_base_url`.
2. It exports the right env vars (`ANTHROPIC_BASE_URL` /
   `OPENAI_BASE_URL` / `*_API_BASE`) while `agent.run()` invokes
   `agent.execute(...)` so vendor SDKs pick them up.
3. `agent.command_env(...)` returns those values for in-sandbox CLIs that
   need them in `sbx.commands.run(..., env=...)`.
4. For networked in-sandbox agents, it attributes requests from the sandbox
   network lease. Host-driven agents must send `X-Agent-Sandbox-Id` explicitly
   so the per-sandbox forwarder can select the registered upstream.
5. It captures the request through the C/R scheduler — so checkpoints can
   be taken at semantically meaningful turn boundaries without you wiring
   anything.

The daemon config must enable both the interceptor and sandbox networking for
in-sandbox agents that declare `requires_network_namespace = True`. The
packaged config enables sandbox networking but leaves interception disabled;
see [Configuration](configuration-reference.md#network-and-llm-interception).
