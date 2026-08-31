"""Argparse tree + output formatting for the `crab` CLI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from ..daemon import DaemonClient, DaemonRequestError, default_socket_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crab",
        description=(
            "Operator CLI for the Crab daemon. The daemon owns all runc / "
            "ZFS / host-inspector state; subcommands here introspect or "
            "manipulate that state."
        ),
    )
    parser.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Daemon Unix socket path. Defaults to $CRAB_DAEMON_SOCKET or "
        "the XDG runtime location.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for daemon requests, seconds (default 30).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of formatted text.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ----- daemon group ----------------------------------------------------
    daemon = sub.add_parser(
        "daemon",
        help="Start, stop, or inspect the Crab daemon process.",
    )
    daemon_sub = daemon.add_subparsers(dest="daemon_command", required=True)

    p_start = daemon_sub.add_parser(
        "start",
        help="Start the Crab daemon in the background. Like `dockerd`, "
        "this is the long-running host service.",
    )
    p_start.add_argument("--config", type=Path, default=None)
    p_start.add_argument("--pid-file", type=Path, default=None)
    p_start.add_argument("--log-file", type=Path, default=None)
    p_start.add_argument("--log-level", default="INFO")
    p_start.add_argument(
        "--foreground",
        action="store_true",
        help="Run the daemon in the foreground instead of detaching.",
    )
    p_start.set_defaults(func=_cmd_daemon_start)

    p_stop = daemon_sub.add_parser("stop", help="Stop the running daemon.")
    p_stop.add_argument(
        "--timeout",
        dest="stop_timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the daemon to exit after SIGTERM.",
    )
    p_stop.set_defaults(func=_cmd_daemon_stop)

    p_status = daemon_sub.add_parser("status", help="Report whether the daemon is reachable.")
    p_status.set_defaults(func=_cmd_daemon_status)

    # ----- info -----------------------------------------------------------
    p_info = sub.add_parser("info", help="Dump the daemon's /info payload.")
    p_info.set_defaults(func=_cmd_info)

    # ----- sandbox group --------------------------------------------------
    sandbox = sub.add_parser("sandbox", help="Inspect or operate on sandboxes.")
    sandbox_sub = sandbox.add_subparsers(dest="sandbox_command", required=True)

    p_run = sandbox_sub.add_parser(
        "run",
        help="Spawn a new sandbox from an image, optionally exec a command, "
        "and optionally destroy it on exit. Loosely mirrors `docker run`.",
    )
    p_run.add_argument("image", metavar="IMAGE", help="Container image tag (e.g. ubuntu:22.04).")
    p_run.add_argument("argv", nargs=argparse.REMAINDER, help="Optional command after `--`.")
    p_run.add_argument("--name", default=None, help="Sandbox name (auto-generated when omitted).")
    p_run.add_argument(
        "--work-dir",
        dest="work_dir",
        default=None,
        help="Host directory bound to /work inside the sandbox.",
    )
    p_run.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable to set in the sandbox (repeatable).",
    )
    p_run.add_argument(
        "--network",
        action="store_true",
        help="Allocate a sandbox network namespace (default off).",
    )
    p_run.add_argument(
        "--detach",
        "-d",
        action="store_true",
        help="Leave the sandbox running and print its id; do not exec a command.",
    )
    p_run.add_argument(
        "--rm",
        action="store_true",
        help="Destroy the sandbox after the command exits (ignored with --detach).",
    )
    p_run.set_defaults(func=_cmd_sandbox_run)

    p_ls = sandbox_sub.add_parser("ls", help="List sandboxes the daemon is tracking.")
    p_ls.set_defaults(func=_cmd_sandbox_ls)

    p_rm = sandbox_sub.add_parser("rm", help="Destroy one or more sandboxes (kill + delete state).")
    p_rm.add_argument("sandbox_ids", nargs="+", metavar="SANDBOX_ID")
    p_rm.set_defaults(func=_cmd_sandbox_rm)

    p_stop = sandbox_sub.add_parser(
        "stop",
        help="Stop a running sandbox gracefully. Bundle and ZFS state remain "
        "so the sandbox can later be restored or destroyed with `rm`.",
    )
    p_stop.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_stop.set_defaults(func=_cmd_sandbox_stop)

    p_pause = sandbox_sub.add_parser(
        "pause",
        help="Pause a running sandbox (freezes all processes via the cgroup).",
    )
    p_pause.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_pause.set_defaults(func=_cmd_sandbox_pause)

    p_resume = sandbox_sub.add_parser("resume", help="Resume a paused sandbox.")
    p_resume.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_resume.set_defaults(func=_cmd_sandbox_resume)

    p_exec = sandbox_sub.add_parser(
        "exec",
        help="Run a command inside an existing sandbox and print its output.",
    )
    p_exec.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_exec.add_argument("argv", nargs=argparse.REMAINDER, help="Use `-- cmd ...`")
    p_exec.add_argument("--cwd", default=None)
    p_exec.add_argument("--user", default=None)
    p_exec.add_argument("--timeout", dest="exec_timeout", type=float, default=None)
    p_exec.set_defaults(func=_cmd_sandbox_exec)

    p_fork = sandbox_sub.add_parser(
        "fork",
        help="Fork a running sandbox into N independent running copies "
        "(checkpoint + restore). Prints one fork id per line.",
    )
    p_fork.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_fork.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of forks to create (default 1).",
    )
    p_fork.add_argument(
        "--lazy",
        action="store_true",
        help="Restore with CRIU lazy-pages: return as soon as metadata is "
        "in place and stream memory on demand.",
    )
    p_fork.add_argument(
        "--checkpoint",
        dest="fork_checkpoint_id",
        metavar="CHECKPOINT_ID",
        default=None,
        help="Fork from an existing checkpoint of the sandbox instead of its "
        "live state: no new checkpoint is taken and the source is not "
        "restored, so every fork branches from that past point.",
    )
    p_fork.add_argument(
        "--effects",
        choices=["allow", "reject"],
        default=None,
        help="Effect policy for the fork's outbound writes. Omitted: an "
        "independent branch, ungated (default). 'reject': a speculative "
        "branch that must not write — mutating plaintext egress is refused "
        "with 503. HTTPS writes stay unclassifiable either way.",
    )
    p_fork.set_defaults(func=_cmd_sandbox_fork)

    p_merge = sandbox_sub.add_parser(
        "merge",
        help="Three-way merge a fork's filesystem changes back into its "
        "source sandbox (C2). Conflicts resolve per --policy.",
    )
    p_merge.add_argument("sandbox_id", metavar="SOURCE_ID", help="Source sandbox id.")
    p_merge.add_argument("fork_id", metavar="FORK_ID", help="Fork sandbox id to merge from.")
    p_merge.add_argument(
        "--policy",
        default="fail_fast",
        choices=["fail_fast", "prefer_fork", "prefer_source", "text_merge"],
        help="Conflict policy (default fail_fast: any conflict aborts "
        "before a single write).",
    )
    p_merge.add_argument(
        "--ignore-prefix",
        action="append",
        dest="ignore_prefixes",
        default=None,
        metavar="PREFIX",
        help="Override the default ignore prefixes (/tmp, /var/tmp, /run); "
        "repeatable.",
    )
    p_merge.set_defaults(func=_cmd_sandbox_merge)

    p_changeset = sandbox_sub.add_parser(
        "changeset",
        help="Changed rootfs paths relative to a base checkpoint "
        "(defaults to the sandbox's fork point). One `change<TAB>path` per line.",
    )
    p_changeset.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_changeset.add_argument(
        "--since",
        default=None,
        metavar="CHECKPOINT_ID",
        help="Base checkpoint id; omit to diff against the fork point.",
    )
    p_changeset.set_defaults(func=_cmd_sandbox_changeset)

    p_actions = sandbox_sub.add_parser(
        "actions",
        help="Read a sandbox's action journal (exec, lifecycle, and "
        "adopted observation records), oldest first.",
    )
    p_actions.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_actions.add_argument(
        "--kind",
        default=None,
        choices=["exec", "lifecycle", "observation"],
        help="Only records of this kind.",
    )
    p_actions.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the newest N records.",
    )
    p_actions.set_defaults(func=_cmd_sandbox_actions)

    p_consolidate = sandbox_sub.add_parser(
        "consolidate",
        help="Adopt a fork's journal history into its source sandbox "
        "(C3). Read it back with `crab sandbox actions --kind observation`.",
    )
    p_consolidate.add_argument("sandbox_id", metavar="SOURCE_ID")
    p_consolidate.add_argument("fork_id", metavar="FORK_ID")
    p_consolidate.add_argument(
        "--policy",
        default="append",
        choices=["append", "dedupe", "none"],
        help="append (default): copy every qualifying record; dedupe: "
        "skip execs the source produced identically itself; none: "
        "record the consolidation marker only.",
    )
    p_consolidate.set_defaults(func=_cmd_sandbox_consolidate)

    p_merge_procs = sandbox_sub.add_parser(
        "merge-processes",
        help="Process-half of consolidation (C4): replay a fork's "
        "journaled execs on its source, or promote the fork wholesale "
        "(auto picks based on the source's live processes).",
    )
    p_merge_procs.add_argument("sandbox_id", metavar="SOURCE_ID")
    p_merge_procs.add_argument("fork_id", metavar="FORK_ID")
    p_merge_procs.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "replay", "promote"],
        help="auto (default): replay when the source runs background "
        "processes, promote otherwise.",
    )
    p_merge_procs.add_argument(
        "--policy",
        default="fail_fast",
        help="Promotion only: conflict policy for the reverse fs apply.",
    )
    p_merge_procs.add_argument(
        "--stop-on-deviation",
        action="store_true",
        dest="stop_on_deviation",
        help="Replay only: abort at the first deviating command.",
    )
    p_merge_procs.add_argument(
        "--no-lazy-pages",
        action="store_false",
        dest="lazy_pages",
        default=True,
        help="Promotion only: restore eagerly instead of via lazy-pages.",
    )
    p_merge_procs.add_argument(
        "--force",
        action="store_true",
        help="Promotion only: proceed even when the source runs "
        "background processes (they die).",
    )
    p_merge_procs.set_defaults(func=_cmd_sandbox_merge_processes)

    p_egress = sandbox_sub.add_parser(
        "egress",
        help="Show the sandbox's effect ledger: recorded outbound flows "
        "with their effect class (idempotent_read/mutating/opaque).",
    )
    p_egress.add_argument("sandbox_id")
    p_egress.add_argument(
        "--txn", dest="txn_id", default=None, help="Scope to one transaction."
    )
    p_egress.add_argument(
        "--since-seq",
        dest="since_seq",
        type=int,
        default=None,
        help="Only flows recorded after this journal sequence number.",
    )
    p_egress.add_argument(
        "--recorded",
        action="store_true",
        help="Only flows whose bodies were recorded into a cassette.",
    )
    p_egress.set_defaults(func=_cmd_sandbox_egress)

    # ----- checkpoint group ----------------------------------------------
    # Checkpoints are first-class entities with their own ids; keep them
    # at the top level so the verbs read cleanly (`crab checkpoint ls`,
    # `crab checkpoint rm ...`) and so future cross-sandbox operations
    # (`crab checkpoint prune --orphaned`, etc.) fit naturally.
    checkpoint = sub.add_parser("checkpoint", help="Manage sandbox checkpoints.")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)

    c_ls = checkpoint_sub.add_parser("ls", help="List checkpoints for a sandbox.")
    c_ls.add_argument("sandbox_id", metavar="SANDBOX_ID")
    c_ls.set_defaults(func=_cmd_checkpoint_ls)

    c_create = checkpoint_sub.add_parser(
        "create",
        help="Take a checkpoint of a sandbox (process + filesystem).",
    )
    c_create.add_argument("sandbox_id", metavar="SANDBOX_ID")
    c_create.add_argument(
        "--leave-running",
        dest="leave_running",
        action="store_true",
        default=True,
        help="Leave the sandbox running after the checkpoint (default).",
    )
    c_create.add_argument(
        "--no-leave-running",
        dest="leave_running",
        action="store_false",
        help="Stop the sandbox after taking the checkpoint.",
    )
    c_create.set_defaults(func=_cmd_checkpoint_create)

    c_rm = checkpoint_sub.add_parser("rm", help="Delete a checkpoint.")
    c_rm.add_argument("sandbox_id", metavar="SANDBOX_ID")
    c_rm.add_argument("checkpoint_id", metavar="CHECKPOINT_ID")
    c_rm.add_argument(
        "--cascade",
        action="store_true",
        help="Also delete descendant checkpoints that depend on this one.",
    )
    c_rm.set_defaults(func=_cmd_checkpoint_rm)

    # ----- txn group ------------------------------------------------------
    txn = sub.add_parser("txn", help="Manage sandbox transactions (snapshot-based).")
    txn_sub = txn.add_subparsers(dest="txn_command", required=True)

    t_begin = txn_sub.add_parser(
        "begin",
        help="Open a transaction: adaptive base checkpoint + staged "
        "observations + auto-checkpoints suppressed. Prints the txn id.",
    )
    t_begin.add_argument("sandbox_id", metavar="SANDBOX_ID")
    t_begin.add_argument("--label", default=None)
    t_begin.add_argument(
        "--isolation",
        default="snapshot",
        choices=["snapshot", "fork"],
        help="snapshot (default): actions run in place, abort restores the "
        "base; fork: actions run in a fork, commit promotes it back onto "
        "this sandbox, abort just destroys it.",
    )
    t_begin.set_defaults(func=_cmd_txn_begin)

    t_commit = txn_sub.add_parser(
        "commit",
        help="Commit: deliver staged observations, drop a fresh base.",
    )
    t_commit.add_argument("sandbox_id", metavar="SANDBOX_ID")
    t_commit.add_argument("txn_id", metavar="TXN_ID")
    t_commit.add_argument(
        "--force",
        action="store_true",
        help="Fork-backed txns only: promote even when the source changed "
        "since the fork point (its writes are discarded).",
    )
    t_commit.set_defaults(func=_cmd_txn_commit)

    t_abort = txn_sub.add_parser(
        "abort",
        help="Abort: drop staged observations, restore the base checkpoint.",
    )
    t_abort.add_argument("sandbox_id", metavar="SANDBOX_ID")
    t_abort.add_argument("txn_id", metavar="TXN_ID")
    t_abort.set_defaults(func=_cmd_txn_abort)

    t_status = txn_sub.add_parser("status", help="Show the sandbox's active transaction.")
    t_status.add_argument("sandbox_id", metavar="SANDBOX_ID")
    t_status.set_defaults(func=_cmd_txn_status)

    # ----- restore (top-level, not under `checkpoint`) -------------------
    # `restore` acts on a sandbox (using one of its checkpoints), so it
    # reads more naturally as a top-level verb than as a nested
    # `checkpoint restore`. Mirrors how `kubectl rollout undo` and
    # `git restore` are top-level rather than nested under a noun.
    p_restore = sub.add_parser(
        "restore",
        help="Restore a sandbox from one of its checkpoints.",
    )
    p_restore.add_argument("sandbox_id", metavar="SANDBOX_ID")
    p_restore.add_argument("checkpoint_id", metavar="CHECKPOINT_ID")
    p_restore.set_defaults(func=_cmd_restore)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("CRAB_CLI_LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        # Daemon socket missing → friendly error pointing at `daemon start`.
        print(f"crab: {exc}", file=sys.stderr)
        return 2
    except DaemonRequestError as exc:
        print(f"crab: daemon returned {exc.status_code}: {exc.body.decode('utf-8', 'replace')}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


# ---------------------------------------------------------------------------
# `crab daemon ...`
# ---------------------------------------------------------------------------


def _resolve_socket(args: argparse.Namespace) -> Path:
    return (args.socket or default_socket_path()).expanduser()


def _cmd_daemon_start(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    if client.ping():
        print(f"crab daemon already running at {socket_path}", file=sys.stderr)
        return 0
    cmd: list[str] = [
        sys.executable,
        "-m",
        "crab.daemon",
        "--socket",
        str(socket_path),
        "--log-level",
        str(args.log_level),
    ]
    if args.config is not None:
        cmd.extend(["--config", str(args.config)])
    if args.pid_file is not None:
        cmd.extend(["--pid-file", str(args.pid_file)])

    if args.foreground:
        # Run in this terminal — caller sees logs directly.
        return subprocess.call(cmd)

    log_file = args.log_file
    if log_file is None:
        log_file = socket_path.parent / "crab-daemon.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Detach: stdout/stderr go to the log file; the child becomes its own
    # session leader so it survives the CLI exit. Same pattern dockerd
    # uses when launched via systemd's `Type=forking` (with the caveat
    # that production deployments should drive the daemon via a unit file
    # instead of this convenience wrapper).
    log_fh = open(log_file, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    finally:
        log_fh.close()

    # Wait for the socket to appear (or the daemon to exit prematurely).
    deadline = time.monotonic() + max(5.0, args.timeout)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(
                f"crab daemon exited during startup (rc={proc.returncode}); "
                f"see {log_file} for details",
                file=sys.stderr,
            )
            return 1
        if client.ping():
            print(f"crab daemon started (pid={proc.pid}, socket={socket_path})")
            print(f"logs: {log_file}")
            return 0
        time.sleep(0.1)
    print(
        f"crab daemon did not become ready within {args.timeout:.1f}s; "
        f"see {log_file}",
        file=sys.stderr,
    )
    return 1


def _cmd_daemon_stop(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    if not client.ping():
        print(f"crab daemon not running at {socket_path}", file=sys.stderr)
        return 0
    # Ask politely first.
    try:
        client.post_json("/shutdown", {})
    except Exception:
        # /shutdown may close the socket before the response is parsed —
        # fall through to the poll loop below either way.
        pass
    deadline = time.monotonic() + max(1.0, args.stop_timeout)
    while time.monotonic() < deadline:
        if not client.ping():
            print(f"crab daemon stopped (socket={socket_path})")
            return 0
        time.sleep(0.1)
    print(
        f"crab daemon did not stop within {args.stop_timeout:.1f}s; "
        "try `kill -TERM <pid>` against the daemon process.",
        file=sys.stderr,
    )
    return 1


def _cmd_daemon_status(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    try:
        info = client.get_json("/info")
    except FileNotFoundError:
        print(f"daemon: not running (socket {socket_path} not found)")
        return 1
    except DaemonRequestError as exc:
        print(f"daemon: error {exc.status_code} from /info", file=sys.stderr)
        return 1
    _print_payload(args, info, fallback=_format_info)
    return 0


# ---------------------------------------------------------------------------
# `crab info`
# ---------------------------------------------------------------------------


def _cmd_info(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    info = client.get_json("/info")
    _print_payload(args, info, fallback=_format_info)
    return 0


# ---------------------------------------------------------------------------
# `crab sandbox ...`
# ---------------------------------------------------------------------------


def _cmd_sandbox_ls(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    response = client.get_json("/sandboxes")
    sandboxes = list(response.get("sandboxes") or [])
    if args.json:
        print(json.dumps(sandboxes, indent=2))
        return 0
    if not sandboxes:
        print("(no sandboxes)")
        return 0
    rows = []
    for sbx in sandboxes:
        rows.append(
            {
                "ID": sbx.get("sandbox_id", ""),
                "RUNTIME": sbx.get("runtime_name", ""),
                "STATUS": sbx.get("status", ""),
            }
        )
    _print_table(rows, columns=["ID", "RUNTIME", "STATUS"])
    return 0


def _cmd_sandbox_rm(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    rc = 0
    for sid in args.sandbox_ids:
        try:
            client.delete(f"/sandboxes/{sid}")
            print(f"{sid}")
        except DaemonRequestError as exc:
            print(f"crab sandbox rm {sid}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def _cmd_sandbox_stop(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    client.post_json(f"/sandboxes/{args.sandbox_id}/stop", {})
    print(args.sandbox_id)
    return 0


def _cmd_sandbox_pause(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    client.post_json(f"/sandboxes/{args.sandbox_id}/pause", {})
    print(args.sandbox_id)
    return 0


def _cmd_sandbox_resume(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    client.post_json(f"/sandboxes/{args.sandbox_id}/resume", {})
    print(args.sandbox_id)
    return 0


def _cmd_sandbox_run(args: argparse.Namespace) -> int:
    """`docker run`-style sandbox creation.

    Drives the SDK against the daemon: connects via `Engine.connect`,
    constructs a real `Sandbox(image=..., env=..., work_dir=...)`, then
    optionally exec's `argv` and tears the sandbox down when `--rm` is
    set (or when no `argv` is given without `--detach`). Using the SDK
    keeps bundle preparation and image cache logic in one place
    instead of duplicating it in the CLI layer."""
    # Import lazily — pulling crab.Sandbox at module import time would
    # drag the engine surface into every CLI invocation, including
    # `crab daemon start` where the daemon's own Engine.start path
    # MUST be the one called.
    from ..engine import Engine
    from ..sandbox import Sandbox

    env: dict[str, str] = {}
    for assignment in args.env or []:
        if "=" not in assignment:
            print(f"crab sandbox run: --env requires KEY=VALUE, got {assignment!r}", file=sys.stderr)
            return 2
        key, value = assignment.split("=", 1)
        env[key.strip()] = value

    socket_path = _resolve_socket(args)
    os.environ["CRAB_DAEMON_SOCKET"] = str(socket_path)
    engine = Engine.connect(socket_path, timeout_seconds=args.timeout)
    sbx: Sandbox | None = None
    try:
        sbx = Sandbox(
            image=args.image,
            engine=engine,
            name=args.name,
            work_dir=args.work_dir,
            env=env or None,
            network=args.network if args.network else None,
        )
        if args.detach or not args.argv:
            print(str(sbx.sandbox_id))
            return 0
        # Strip a leading "--" so users can write `crab sandbox run IMAGE -- cmd ...`.
        cmd_argv = list(args.argv)
        if cmd_argv and cmd_argv[0] == "--":
            cmd_argv = cmd_argv[1:]
        if not cmd_argv:
            print(str(sbx.sandbox_id))
            return 0
        result = sbx.commands.run(argv=cmd_argv, capture_output=True, check=False)
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        return int(result.returncode)
    finally:
        try:
            if sbx is not None and (args.rm or (not args.detach and bool(args.argv))):
                sbx.kill()
        except Exception:
            logger.debug("sbx.kill() failed during `sandbox run` teardown", exc_info=True)
        try:
            engine.stop()
        except Exception:
            pass


def _cmd_sandbox_fork(args: argparse.Namespace) -> int:
    if args.count < 1:
        print("crab sandbox fork: --count must be >= 1", file=sys.stderr)
        return 2
    socket_path = _resolve_socket(args)
    # Fork = one checkpoint + N clone/restores; budget the HTTP timeout
    # the same way `checkpoint create` does, scaled by count.
    client = DaemonClient(
        socket_path,
        timeout_seconds=max(args.timeout, 300.0 * args.count),
    )
    payload: dict = {"count": int(args.count), "lazy": bool(args.lazy)}
    if getattr(args, "effects", None) is not None:
        payload["effects"] = str(args.effects)
    if getattr(args, "fork_checkpoint_id", None) is not None:
        payload["checkpoint_id"] = str(args.fork_checkpoint_id)
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/fork",
        payload,
    )
    forks = list(response.get("forks") or [])
    if args.json:
        print(json.dumps(forks, indent=2))
        return 0
    for fork in forks:
        print(fork.get("sandbox_id", ""))
    return 0


def _cmd_sandbox_merge(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    # Merge quiesces both sandboxes and runs two backend diffs plus the
    # apply window; budget generously like fork does.
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 600.0))
    payload: dict[str, Any] = {
        "fork_sandbox_id": args.fork_id,
        "policy": args.policy,
    }
    if args.ignore_prefixes is not None:
        payload["ignore_prefixes"] = list(args.ignore_prefixes)
    response = client.post_json(f"/sandboxes/{args.sandbox_id}/merge", payload)
    report = response.get("report") or {}
    conflicted = list(report.get("conflicted") or [])
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        applied = list(report.get("applied") or [])
        skipped = list(report.get("skipped") or [])
        print(
            f"applied={len(applied)} conflicted={len(conflicted)} "
            f"skipped={len(skipped)} policy={report.get('policy', '')}"
        )
        for entry in conflicted:
            print(f"conflict\t{entry.get('path', '')}\t{entry.get('reason', '')}")
    # Conflicts mean the merge (fully or partially) did not land; make
    # that visible to scripts via the exit code.
    return 1 if conflicted else 0


def _cmd_sandbox_changeset(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 300.0))
    payload: dict[str, Any] = {} if args.since is None else {"since": args.since}
    response = client.post_json(f"/sandboxes/{args.sandbox_id}/changeset", payload)
    changeset = response.get("changeset") or {}
    if args.json:
        print(json.dumps(changeset, indent=2))
        return 0
    for entry in changeset.get("entries") or []:
        suffix = ""
        if entry.get("renamed_from"):
            suffix = f"\t(from {entry['renamed_from']})"
        print(f"{entry.get('change', '')}\t{entry.get('path', '')}{suffix}")
    return 0


def _summarize_action_record(record: dict[str, Any]) -> str:
    kind = record.get("kind", "")
    payload = record.get("payload") or {}
    if kind == "exec":
        argv = " ".join(payload.get("argv") or [])
        return f"rc={payload.get('returncode')}\t{argv}"
    if kind == "lifecycle":
        return str(payload.get("event", ""))
    if kind == "observation":
        origin = payload.get("origin_kind", "")
        fork = payload.get("fork_sandbox_id", "")
        if origin == "exec":
            inner = payload.get("origin_payload") or {}
            detail = " ".join(inner.get("argv") or [])
        elif origin == "summary":
            detail = str(payload.get("summary", ""))[:80]
        else:
            detail = str((payload.get("origin_payload") or {}).get("event", ""))
        return f"{origin}\tfrom={fork}\t{detail}"
    return ""


def _cmd_sandbox_actions(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 60.0))
    payload: dict[str, Any] = {}
    if args.kind is not None:
        payload["kind"] = args.kind
    if args.limit is not None:
        payload["limit"] = int(args.limit)
    response = client.post_json(f"/sandboxes/{args.sandbox_id}/actions", payload)
    records = list(response.get("records") or [])
    if args.json:
        print(json.dumps(records, indent=2))
        return 0
    for record in records:
        print(f"{record.get('seq', '')}\t{record.get('kind', '')}\t{_summarize_action_record(record)}")
    return 0


def _cmd_sandbox_consolidate(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 60.0))
    payload: dict[str, Any] = {"fork_sandbox_id": args.fork_id, "policy": args.policy}
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/observations/consolidate", payload
    )
    report = response.get("report") or {}
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(
        f"consolidated={report.get('consolidated', 0)} "
        f"skipped_duplicates={report.get('skipped_duplicates', 0)} "
        f"policy={report.get('policy', '')}"
        + (" (already consolidated)" if report.get("already_consolidated") else "")
    )
    return 0


def _cmd_sandbox_merge_processes(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    # Replay runs the fork's whole exec history; budget generously.
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 600.0))
    payload: dict[str, Any] = {
        "fork_sandbox_id": args.fork_id,
        "strategy": args.strategy,
        "policy": args.policy,
        "stop_on_deviation": bool(args.stop_on_deviation),
        "lazy_pages": bool(args.lazy_pages),
        "force": bool(args.force),
    }
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/processes/merge", payload
    )
    report = response.get("report") or {}
    deviations = int(report.get("deviations", 0))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        line = (
            f"strategy={report.get('strategy', '')} "
            f"replayed={len(report.get('replayed') or [])} "
            f"deviations={deviations}"
        )
        if report.get("stopped_early"):
            line += " (stopped early)"
        if report.get("promoted_checkpoint_id"):
            line += f" promoted={report['promoted_checkpoint_id']}"
        print(line)
        for entry in report.get("replayed") or []:
            if entry.get("deviated"):
                print(
                    f"deviation\tseq={entry.get('origin_seq')}"
                    f"\trc={entry.get('returncode')} (expected {entry.get('expected_returncode')})"
                    f"\t{' '.join(entry.get('argv') or [])}"
                )
    # Deviations mean the source may not have converged; surface to scripts.
    return 1 if deviations else 0


def _cmd_sandbox_egress(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 60.0))
    payload: dict[str, Any] = {}
    if args.txn_id:
        payload["txn_id"] = args.txn_id
    if args.since_seq is not None:
        payload["since_seq"] = args.since_seq
    response = client.post_json(f"/sandboxes/{args.sandbox_id}/egress", payload)
    ledger = response.get("ledger") or {}
    flows = ledger.get("flows") or []
    if getattr(args, "recorded", False):
        flows = [flow for flow in flows if flow.get("recorded")]
    if args.json:
        print(json.dumps({**ledger, "flows": flows}, indent=2))
        return 0
    print(
        f"total={ledger.get('total', 0)} "
        f"reads={ledger.get('idempotent_reads', 0)} "
        f"mutating={ledger.get('mutating', 0)} "
        f"opaque={ledger.get('opaque', 0)} "
        f"recorded={ledger.get('recorded', 0)} "
        f"replayed={ledger.get('replayed', 0)}"
    )
    for flow in flows:
        target = f"{flow.get('host')}:{flow.get('dst_port')}"
        verb = flow.get("method") or flow.get("scheme")
        detail = flow.get("path") or ""
        marks = []
        if flow.get("recorded"):
            marks.append(f"rec:{flow.get('status')}")
        if flow.get("truncated"):
            marks.append("truncated")
        if flow.get("replayed_from_seq") is not None:
            marks.append(f"replayed<-{flow['replayed_from_seq']}")
        if flow.get("txn_id"):
            marks.append(f"txn={flow['txn_id']}")
        suffix = ("\t" + " ".join(marks)) if marks else ""
        print(
            f"{flow.get('seq')}\t{flow.get('classification')}\t{verb}\t{target}\t{detail}"
            + suffix
        )
    return 0


def _cmd_checkpoint_ls(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    response = client.get_json(f"/sandboxes/{args.sandbox_id}/checkpoints")
    checkpoints = list(response.get("checkpoints") or [])
    if args.json:
        print(json.dumps(checkpoints, indent=2))
        return 0
    if not checkpoints:
        print("(no checkpoints)")
        return 0
    rows = []
    for ckpt in checkpoints:
        materialization = ckpt.get("materialization")
        if not materialization:
            materialized_process = ckpt.get(
                "materialized_process", ckpt.get("has_process")
            )
            materialized_filesystem = ckpt.get(
                "materialized_filesystem", ckpt.get("has_filesystem")
            )
            if materialized_process and materialized_filesystem:
                materialization = "full"
            elif materialized_process:
                materialization = "process_only"
            elif materialized_filesystem:
                materialization = "filesystem_only"
            else:
                materialization = "reused"
        rows.append(
            {
                "CHECKPOINT_ID": ckpt.get("checkpoint_id", ""),
                "CREATED_AT": ckpt.get("created_at") or "",
                "LABEL": ckpt.get("label") or "",
                "PROCESS": "yes" if ckpt.get("has_process") else "no",
                "FILESYSTEM": "yes" if ckpt.get("has_filesystem") else "no",
                "MATERIALIZATION": materialization,
            }
        )
    _print_table(
        rows,
        columns=[
            "CHECKPOINT_ID",
            "CREATED_AT",
            "PROCESS",
            "FILESYSTEM",
            "MATERIALIZATION",
            "LABEL",
        ],
    )
    return 0


def _cmd_checkpoint_create(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    # Checkpointing involves CRIU + ZFS snapshot, which can run for tens
    # of seconds on a busy sandbox. Bump the HTTP timeout accordingly.
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 300.0))
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/checkpoints",
        {"leave_running": bool(args.leave_running)},
    )
    print(response.get("checkpoint_id") or "")
    return 0 if response.get("status") in {"succeeded", "completed", None, ""} else 1


def _cmd_checkpoint_rm(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    payload = {"cascade": True} if args.cascade else {}
    # `delete` doesn't have a body API on DaemonClient v1, so pipe the
    # cascade flag through a POST shaped to the same path.
    client._request_json(
        "DELETE",
        f"/sandboxes/{args.sandbox_id}/checkpoints/{args.checkpoint_id}",
        body=(json.dumps(payload).encode("utf-8") if payload else None),
    )
    print(args.checkpoint_id)
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    """`crab restore SANDBOX_ID CHECKPOINT_ID`.

    Top-level verb (not under `checkpoint`) because the operation acts
    on a sandbox using one of its checkpoints — same shape as `git
    restore` / `kubectl rollout undo`."""
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 300.0))
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/checkpoints/{args.checkpoint_id}/restore",
        {},
    )
    if args.json:
        print(json.dumps(response, indent=2))
    else:
        print(args.checkpoint_id)
    return 0 if response.get("status") in {"succeeded", "completed", None, ""} else 1


def _cmd_txn_begin(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    # Begin may take a base checkpoint (snapshot) or a full fork; budget
    # like `sandbox fork` when isolation=fork.
    floor = 600.0 if args.isolation == "fork" else 300.0
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, floor))
    payload: dict[str, Any] = {}
    if args.label is not None:
        payload["label"] = args.label
    if args.isolation != "snapshot":
        payload["isolation"] = args.isolation
    response = client.post_json(f"/sandboxes/{args.sandbox_id}/txn", payload)
    txn = response.get("txn") or {}
    if args.json:
        print(json.dumps(txn, indent=2))
    else:
        print(txn.get("txn_id", ""))
    return 0


def _cmd_txn_commit(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    # Fork-backed commits swap fs + processes; budget generously.
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 600.0))
    payload: dict[str, Any] = {"force": True} if args.force else {}
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/txn/{args.txn_id}/commit", payload
    )
    result = response.get("result") or {}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        line = (
            f"committed {result.get('txn_id', '')} "
            f"released={result.get('released_observations', 0)} "
            f"base_dropped={result.get('base_dropped', False)}"
        )
        if result.get("promoted_checkpoint_id"):
            line += f" promoted={result['promoted_checkpoint_id']}"
        print(line)
    return 0


def _cmd_txn_abort(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    # Abort restores the base checkpoint; budget like `restore`.
    client = DaemonClient(socket_path, timeout_seconds=max(args.timeout, 300.0))
    response = client.post_json(
        f"/sandboxes/{args.sandbox_id}/txn/{args.txn_id}/abort", {}
    )
    result = response.get("result") or {}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"aborted {result.get('txn_id', '')} "
            f"discarded={result.get('discarded_observations', 0)} "
            f"restored={result.get('restored_checkpoint_id', '')}"
        )
    return 0


def _cmd_txn_status(args: argparse.Namespace) -> int:
    socket_path = _resolve_socket(args)
    client = DaemonClient(socket_path, timeout_seconds=args.timeout)
    response = client.get_json(f"/sandboxes/{args.sandbox_id}/txn")
    txn = response.get("txn")
    if args.json:
        print(json.dumps(txn, indent=2))
        return 0
    if txn is None:
        print("(no active transaction)")
        return 0
    print(
        f"{txn.get('txn_id', '')} base={txn.get('base_checkpoint_id', '')} "
        f"fresh={txn.get('base_was_fresh', False)} started={txn.get('started_at', '')}"
        + (f" label={txn['label']}" if txn.get("label") else "")
    )
    return 0


def _cmd_sandbox_exec(args: argparse.Namespace) -> int:
    if not args.argv:
        print("crab sandbox exec: need a command after `--`", file=sys.stderr)
        return 2
    socket_path = _resolve_socket(args)
    # Streaming exec is out of scope for v1 — use a long-enough timeout so
    # bulk commands don't time out against the 30s default.
    timeout_seconds = max(args.timeout, args.exec_timeout or 0.0, 600.0)
    client = DaemonClient(socket_path, timeout_seconds=timeout_seconds)
    payload: dict[str, Any] = {"argv": list(args.argv), "capture_output": True}
    if args.cwd is not None:
        payload["cwd"] = args.cwd
    if args.user is not None:
        payload["user"] = args.user
    if args.exec_timeout is not None:
        payload["timeout_s"] = float(args.exec_timeout)
    response = client.post_json(f"/sandboxes/{args.sandbox_id}/exec", payload)
    result = response.get("result") or {}
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    return int(result.get("returncode") or 0)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_payload(args: argparse.Namespace, payload: dict[str, Any], *, fallback) -> None:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        fallback(payload)


def _format_info(info: dict[str, Any]) -> None:
    keys = [
        "pid",
        "runtime",
        "default_image",
        "storage_root",
        "runtime_root",
        "image_cache_root",
        "work_dir_host_root",
        "agent_state_root",
        "interceptor_base_url",
        "forwarder_base_url",
        "network_bridge_ip",
        "sandbox_count",
    ]
    width = max(len(k) for k in keys) + 2
    for key in keys:
        if key not in info:
            continue
        print(f"{(key + ':').ljust(width)}{info[key]}")


def _print_table(rows: list[dict[str, str]], *, columns: list[str]) -> None:
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    fmt = "  ".join(f"{{:<{widths[col]}}}" for col in columns)
    print(fmt.format(*columns))
    for row in rows:
        print(fmt.format(*[str(row.get(col, "")) for col in columns]))


# Re-exported for tests / scripting.
__all__ = ["main"]
