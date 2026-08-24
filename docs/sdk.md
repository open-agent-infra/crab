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

## Cloud mode (crab-gateway)

The same `Sandbox` surface works over the network against a `crab-gateway`
(the multi-tenant HTTP facade in front of a daemon). Pass a gateway URL and
API key instead of a socket path:

```python
from crab import Engine, Sandbox

with Engine.connect(
    url="https://crab.example.com",
    api_key="crab_sk_...",           # or set CRAB_API_KEY
) as engine:
    sandbox = Sandbox.connect("sbx-abc123", engine=engine)
    print(sandbox.commands.run("uname -a").stdout)
```

`Engine.connect` dispatches on argument shape: a path (or nothing) means
the local Unix socket, an `http(s)://` URL means a gateway. Local and cloud
engines are interchangeable everywhere the SDK accepts an `engine=`
argument; `get_default_engine()` still means the local socket.

Gateway errors surface as typed exceptions from `crab.cloud_client`:
`CloudAuthError` (401), `SandboxNotFound` (404 — including another
tenant's ids), `QuotaExceeded` (409, with the quota arithmetic on
`.quota`), `SandboxLost` (410, daemon restarted), `DaemonUnreachableError`
(502), `GatewayTimeoutError` (504). A timeout of the client's own request
raises plain `TimeoutError`, as in local-daemon mode.

Cloud-mode limitations:

- ~~Creating a new sandbox with `Sandbox(image=...)` required client-side
  bundle prep.~~ **Resolved in PR-S5.3**: remote creation is now fully
  supported — the daemon performs bundle prep server-side (docker export,
  ZFS clone, runc spec) when it receives `{"image": "..."}` metadata
  without a `bundle_path`.
- ~~Host-coupled helpers (upstream, network lease, inspector filters,
  process merge) raised `CloudUnsupportedOperation`.~~ **Resolved in
  PR-S5.3**: all per-sandbox daemon routes are now proxied through the
  gateway; only `/shutdown` remains blocked (operator-only).
- Host paths (`engine.storage_root` and friends) are not resolvable and
  raise `RuntimeError` — the gateway's `/info` deliberately omits them
  (this is by design, not a limitation to resolve).

## Current limitations

- `Sandbox.fork(count, lazy=False, effects=None)` clones a running sandbox
  via checkpoint+restore: each fork is an independent, running sandbox
  sharing the parent's state at fork time (incremental chain sharing
  applies when available). `lazy=True` restores with CRIU lazy-pages for a
  faster return. Forks share the parent's `work_dir` host mount.
  `effects` declares what the fork is *for*, which decides whether its
  outbound writes are gated — see
  [Fork intent and outbound writes](#fork-intent-and-outbound-writes).
  Works both with a local in-process engine and against the daemon
  (`crab sandbox fork` from the CLI).
- `Sandbox.begin(label=None)` opens a snapshot-based transaction
  (`with sandbox.begin() as txn:` commits on clean exit, aborts on
  exception; `txn.exec/commit/abort`). Begin takes an adaptive base
  checkpoint and arms observation staging; abort restores the base and
  drops staged LLM responses (gated callers get a 409); commit delivers
  them and drops a freshly-taken base. Weak isolation: txn actions run in
  place. Auto-checkpoints are suppressed while a txn is open. Works both
  with a local in-process engine and against the daemon (`crab txn
  begin/commit/abort/status` from the CLI).
- `Sandbox.begin(isolation="fork")` opens a fork-backed transaction
  (strong isolation): begin forks the sandbox and `txn.exec` runs in the
  fork while the source keeps serving its pre-txn state; commit promotes
  the fork's whole filesystem *and process* state back onto the source's
  unchanged identity (`commit(force=True)` overrides the dirty-source
  gate and discards source-side writes made during the txn); abort just
  destroys the fork — the source is never restored.
- `Sandbox.changeset(since=None)` returns the changed rootfs paths
  (added/modified/removed/renamed) relative to a base checkpoint's
  filesystem snapshot; `since=None` diffs against the sandbox's fork
  point. Raw truth from the CoW backend (`zfs diff` / `btrfs send`).
  Works both locally and against the daemon (`crab sandbox changeset`).
- `Sandbox.merge(fork, policy="fail_fast", ignore_prefixes=None,
  merger=None)` three-way merges a fork's filesystem changes back into
  the source: fork-side changes apply where the source did not touch the
  same path since the fork point; conflicts resolve per policy
  (`fail_fast` aborts before any write, `prefer_fork`, `prefer_source`,
  `text_merge` runs a line-based diff3). Returns a `MergeReport`
  (applied/conflicted/skipped, `rolled_back`); apply failures raise
  `MergeError` carrying the report after a path-level rollback from the
  pre-merge snapshot. Works both locally and against the daemon
  (`crab sandbox merge`); custom `merger` hooks are local-only.
- `Sandbox.merge_processes(fork, strategy="auto", ...)` is the process
  half of consolidation (C4). `auto` probes this sandbox's live
  processes: with background processes running, the fork's journaled
  execs are **replayed** here verbatim and every command's outcome is
  diffed against the recorded returncode/stdout digest (deviations are
  counted; `stop_on_deviation=True` aborts at the first one; the fork
  stays alive); with none, the fork is **promoted** wholesale onto this
  sandbox's identity: the source's fs changes since the fork point are
  first applied onto the fork (`policy`: fail_fast /
  prefer_incoming / prefer_existing / text_merge), then the fork —
  files and processes — takes over via the B3 swap, restored with CRIU
  lazy-pages unless `lazy_pages=False`, its history adopted per
  `observations`, and the fork destroyed. `force=True` promotes over
  live source processes (they die). Returns a ProcessMergeReport.
  Works both locally and against the daemon
  (`crab sandbox merge-processes`).

### Promotion and the sandbox's network identity

Both promotion paths above — a fork-backed `commit()` and a
`merge_processes` that promotes — restore the fork's process image onto
the source's identity. When sandbox networking is on, the fork's dumped
sockets are bound to the *fork's* guest IP, so the source **adopts the
fork's network lease** during the swap: its `sandbox_id` is unchanged (SDK
handles keep working), but its **guest IP changes** to the fork's. This is
what lets an in-sandbox server or client survive the cutover — the address
moved with it. Egress attribution and `Sandbox.get_host` follow the new
address automatically. On this path the fork is dumped stopped, so a
failed promotion is recovered by re-restoring the promoted checkpoint, not
by resuming the fork; the fork's filesystem is retained until the swap is
proven. Networking-off deployments are unaffected (no lease to move).
- `Sandbox.consolidate_observations(fork, policy="append",
  summarizer=None)` adopts a fork's journal history into this sandbox's
  journal as `kind="observation"` records with provenance (fork id,
  origin seq/kind/txn): `append` copies every qualifying record,
  `dedupe` skips execs the source produced identically itself since the
  fork point, `none` copies nothing (combine with a local-only
  `summarizer` callable for a digest entry). Fork-txn commits adopt the
  fork's history automatically; `Sandbox.merge(fork,
  observations="append"|"dedupe")` opts a merge in and nests the
  resulting report under `MergeReport.observations`. Works both locally
  and against the daemon (`crab sandbox consolidate`).
- `Sandbox.actions(kind=None, limit=None)` reads the per-sandbox action
  journal: every exec attempt (argv, cwd, env, exit status, timing;
  stdout/stderr as size+sha256 only) plus lifecycle markers
  (launch/checkpoint/restore/fork/destroy), adopted fork history
  (`kind="observation"`, C3) and egress flows (`kind="egress"`, D1).
  Journals are JSONL files under
  `{storage_root}/journal/` and record env values verbatim — treat them
  with the same care as checkpoint images. Works both locally and
  against the daemon (`crab sandbox actions`); disable recording with
  `EngineConfig(enable_action_journal=False)`.
- **Egress interception (D1)** is opt-in via
  `EngineConfig(enable_egress_proxy=True)` (config: `egress.enabled`),
  and requires `runtime="runc"` with `enable_sandbox_network=True` —
  the bridge netns is the redirect hook point, so enabling it also puts
  sandboxes in that netns. Every outbound TCP connection is redirected
  into a host-side transparent proxy that records one journal row per
  flow: host (HTTP `Host` header or TLS SNI), destination ip/port,
  scheme (`http`/`tls`/`tcp`), method/path for plaintext HTTP, byte
  counts and duration. With **TLS interception** enabled
  (`egress.tls_interception.enabled: true`, requires `crab[tls]`), most
  HTTPS flows are decrypted in-proxy: the proxy mints a per-SNI leaf
  certificate, terminates the handshake, and re-reads the decrypted
  head — making HTTPS `GET`/`POST`/etc. classifiable and recordable
  exactly like plaintext HTTP (`scheme="https"`). The CA certificate is
  automatically injected into the sandbox rootfs and environment
  (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`,
  `NODE_EXTRA_CA_CERTS`) so that most runtimes trust the proxy.
  **Note:** For commands launched via `commands.run(env=...)` or sandboxes
  created with `Sandbox(env=...)`, these CA environment variables can be
  overridden — which will cause that command to **not** trust the
  interception CA (effectively opting out of decryption for that
  invocation). The init-process environment cannot be overridden this way
  and always trusts the injected CA.
  **Exceptions that cannot be intercepted**: certificate-pinning clients,
  Java (custom keystore not injected in v1), HTTP/2-only flows (ALPN
  without `http/1.1`), and hosts listed in `bypass_hosts`. These remain
  `scheme="tls"` (opaque). When interception is off (default), bytes are
  spliced untouched and all encrypted flows are opaque as before.
  Host-bound traffic is never redirected,
  so the LLM interceptor path is unaffected. Flows opened inside a
  transaction carry its `txn_id`.
- `Sandbox.egress(txn_id=None, since_seq=None)` returns the
  **effect ledger** (D1): the recorded flows plus counts
  (`total`/`idempotent_reads`/`mutating`/`opaque`) and `hosts`. Each
  flow is classified from what the proxy could see — GET/HEAD/OPTIONS/
  TRACE are `idempotent_read`, POST/PUT/PATCH/DELETE are `mutating`,
  and everything encrypted, tunnelled or raw is `opaque`. Deployments
  refine the opaque cases with host rules:
  `EngineConfig(egress_rules=({"host_glob": "*.internal.example",
  "classify": "idempotent_read"},))` (config: `egress.rules`); the
  first matching rule wins over the protocol default. Pass `txn_id` to
  scope the view. Classification is a pure function of the stored row
  and is **re-derived on every read**, so changing `egress_rules`
  reclassifies history too (and rows recorded before classification
  existed are classified retroactively); the stored journal row is
  never rewritten. Works locally and against the daemon
  (`crab sandbox egress`).
- **Egress recording (D2)** is opt-in via
  `EngineConfig(enable_egress_recording=True)` (config:
  `egress.recording.enabled`) and requires the egress proxy. Records
  **idempotent reads** (GET/HEAD/OPTIONS/TRACE) for both plaintext HTTP
  and — with TLS interception enabled — HTTPS. Each recorded exchange
  lands in a per-sandbox cassette under
  `{storage_root}/cassettes/<sandbox_id>/`, content-addressed, and the
  ledger flow gains `recorded`, `request_key`, `status` and `truncated`.
  Responses larger than `egress.recording.max_body_bytes` (default 1
  MiB) are marked `truncated` and are never replayable; `4xx`/`5xx` need
  `record_errors`; `206` needs both `record_partial` and `range` among
  `varying_headers` (otherwise two ranges would collide on one
  cassette). `Authorization`/`Cookie`/`X-Api-Key` request headers and
  `Set-Cookie` responses are dropped before writing — but credentials
  passed in a **query string** are stored (same exposure as the
  journal's `path`), so treat cassettes like checkpoint images.
  Cassettes are pruned when the sandbox is destroyed.
- `Sandbox.replay_egress(policy="cassette_first", cassette_source=None)`
  is a context manager that serves recorded reads from cassettes instead
  of the network (D2). A hit never opens an upstream connection;
  `cassette_first` falls through to the network on a miss, while
  `cassette_only` answers a miss with `504` + `X-Crab-Replay: miss` for
  hermetic replay. **Mutating and opaque/raw flows always pass through
  in both modes** (with TLS interception, intercepted HTTPS reads become
  replayable too) — replay is a read cache, not an effect gate (holding
  writes is D3). Eligibility is re-evaluated at replay time with the
  current `egress_rules`, so a host reclassified as `mutating`/`opaque`
  stops being served even though its cassettes remain on disk.
  `cassette_source` reads another sandbox's bucket — pass the fork whose
  reads you are re-running. On exit the context yields
  `.report` (`EgressReplayReport`: served/missed/passed_through/hosts).
  Ledger flows gain `replayed` and `replayed_from`.
- `merge_processes(strategy="replay")` **defaults to
  `egress_replay="cassette_first"` with the fork as the cassette source**,
  which is what makes replayed commands deterministic; pass
  `egress_replay="none"` for live traffic. The report nests the replay
  outcome under `egress_replay`. Replay the fork's cassettes **before**
  killing the fork — destroying it prunes them.
- **Effect gate (D3, PR-D3.1)**: the proxy can hold back a *mutating*
  flow instead of letting it fire. Policies are `allow` (today's
  behavior), `defer` (queue allow-listed writes and answer
  `202 Accepted` + `X-Crab-Effect: deferred`), `reject` (`503` +
  `X-Crab-Effect: rejected`, nothing sent) and `seal` (the write goes
  out but the transaction becomes non-abortable). **Reads are never
  gated.** Deferral cannot wait for the commit — the sandbox process is
  blocked on the response while commit arrives from outside it — so a
  deferred write is answered immediately with `202`, which is why it
  requires an explicit endpoint allow-list
  (`EngineConfig(effects_rules=({"host_glob": "*.internal", "method":
  "POST", "path_glob": "/events*"},))`, config `effects.rules`); an
  unlisted write is refused rather than silently queued
  (`effects.on_unlisted`). Encrypted and raw flows carry no method, so
  `effects.opaque_effects` (`allow` default / `reject` / `seal`) decides
  their fate — **without TLS interception a transaction using HTTPS
  cannot be guaranteed write-free**. Ledger flows gain `effect` and
  `effect_status`, and the ledger counts `deferred`/`rejected`/`flushed`.
  Wiring the policies to transaction lifecycle (queue flush on commit,
  drop on abort, `TxnNotAbortable` for `seal`) lands in PR-D3.2; until
  then no session is opened automatically and every flow behaves exactly
  as before.
- `Sandbox.begin(label, isolation=..., effects=...)` selects the egress
  effect policy for the transaction (D3): `allow` (default for snapshot
  txns — writes pass through), `defer` (allow-listed writes are queued
  and answered `202`, then sent **after** the commit succeeds), `reject`
  (**default for fork-backed txns** — writes get `503`, nothing is sent)
  or `seal` (writes pass but the txn can no longer be aborted).
  `isolation="fork"` with `effects="defer"` is refused: the commit
  destroys the fork that owns the queue. `Transaction.effects` reports
  the policy in force.
- `TxnCommitResult.effects` is an `EffectFlushReport`
  (attempted/succeeded/failed + per-entry status or error). The queue is
  flushed in enqueue order, one request at a time, **from the host**
  (not through the proxy, so it cannot be caught by its own gate). **A
  flush failure never unwinds the commit** — the filesystem is already
  committed, so the failure is reported instead of pretended away.
- `TxnAbortResult.deferred_dropped` counts writes the abort discarded;
  they never reached the world, which is what `defer` is for.
  `TxnAbortResult.mutating_egress` counts only writes that **actually
  left the host** (deferred/rejected/dropped ones are excluded), so a
  defer-only transaction reports 0.
- Aborting a `seal`ed transaction raises `TxnNotAbortable` (daemon: 409
  `txn_not_abortable`); `abort(force=True)` proceeds and accepts that the
  external write stands.
- `merge_processes(strategy="replay", replay_effects=...)` defaults to
  `"reject"`: a replayed command's write would be the fork's write
  fired **twice**, so it is refused and shows up as a deviation instead.
  Pass `"allow"` to re-issue it.

### Fork intent and outbound writes

Whether a fork's outbound writes are gated depends on what the fork is
*for*, so `Sandbox.fork(effects=...)` asks (F1):

- **omitted (default)** — an *independent branch*. An RL rollout or a
  tree-search arm is a first-class timeline whose external effects are
  intended: N forks legitimately produce N effects, and nothing is gated.
  Unchanged from before.
- `effects="reject"` — a *temporal branch*. A speculative fork serves its
  parent and must not write on its own, so mutating plaintext egress is
  refused with `503` and counted in the fork's ledger as `rejected`. If the
  guess pays off, the promoted identity issues the write — once.
- `effects="allow"` — explicit opt-out, identical to omitting it unless the
  deployment flipped the default below.

`"defer"` and `"seal"` raise `ValueError`: a bare fork has no commit to
flush a queue into and no abort for a seal to block, so accepting either
name would promise something this cannot deliver. Reads are never gated.
The session is per sandbox and released when the fork is destroyed.

**Requires the egress proxy.** Gating happens in the host-side proxy, so
`effects="reject"` only bites on an engine started with
`enable_egress_proxy=True` (which itself needs
`enable_sandbox_network=True`). Without it there is nothing to gate with:
the call still succeeds and the fork writes freely, and Crab logs a warning
naming the missing prerequisite rather than failing the fork.

Three config keys govern three different things — the middle two are one
word apart, so they are always shown together:

| config key | applies to | default |
|---|---|---|
| `effects.default_policy` | snapshot transactions | `allow` |
| `effects.fork_policy` | fork-backed transactions (`begin(isolation="fork")`) | `reject` |
| `effects.standalone_fork_policy` | `sandbox.fork()` outside any transaction | `allow` |

**Bounded by TLS**: `effects="reject"` on a fork prevents *plaintext*
mutating egress. HTTPS writes remain unclassifiable and are not blocked, so
a speculative fork that writes over HTTPS can still double-fire. Closing
that gap needs TLS interception, which is deliberately out of scope.

- `TxnAbortResult.mutating_egress` counts the mutating flows the
  transaction already fired: the filesystem rollback cannot undo them,
  so the abort reports rather than hides them (holding or rejecting
  such requests is D3).
- Daemon restart rehydration is not implemented.
- Exec output is buffered; streaming and PTY support are not implemented.
- `resources` is enforced (S3): `Sandbox(resources={"cpus": 2, "memory":
  "512M", "pids": 256})` lands in the runc spec's `linux.resources`
  (cpu quota/period, memory limit, pids limit) and forks inherit the
  source's limits. `memory` accepts bytes or binary-suffixed strings
  (`K`/`M`/`G`/`T`, 1024-based); invalid values fail loudly at
  construction. There is no live resize — limits are fixed at create.
  Operational semantics for gateway tenants: a tenant with no
  `max_memory_bytes`/`max_cpu` quota behaves exactly as in S1/S2 — the
  aggregate gate never engages, with or without per-sandbox `resources`.
  Once an aggregate cap is configured, every create/fork for that tenant
  must declare the corresponding `resources` limit; undeclared requests
  are refused with 409 `QuotaExceeded` (`requested_*` is `null` in the
  quota payload).
  `timeout` and `labels` remain advisory metadata, not lifecycle policies.
- Only OpenAI-compatible and Anthropic LLM base-URL conventions are built in.
