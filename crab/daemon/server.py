"""Crab daemon process — the long-running host service.

The daemon owns exactly one in-process Engine (`crab.engine.Engine`)
and exposes it over a Unix-socket HTTP API. Clients (the SDK proxy in
`crab.remote_engine.RemoteEngine` and the `crab` CLI) translate
each user-facing call into one of these endpoints.

Running model:
  - One daemon per host. The daemon is the sole owner of runc state,
    ZFS datasets, the host inspector subprocess, the LLM interceptor,
    the LLM forwarder, and the sandbox network bridge.
  - The daemon's lifecycle is independent of any client. `Sandbox(...)`
    and `sbx.kill()` only affect sandbox lifecycles — they never stop
    the daemon. The daemon stops only on SIGTERM/SIGINT or an explicit
    `POST /shutdown`.
  - State persistence across daemon restarts is deferred. A v1 restart
    starts with an empty sandbox registry; cleanup of leftover runc
    state is the operator's responsibility (`crab sandbox rm` is
    available once the daemon is back up).

Auth: file-permission gating on the Unix socket only (0600 to the
daemon user). Richer schemes are a follow-up.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

from ..engine import Engine, EngineConfig
from ..ids import SandboxId
from ..merging import MergeError
from ..process_merge import ProcessMergeConflict
from ..models import SandboxSnapshot, utc_now
from ..txn import TxnAbortError, TxnActiveError, TxnCommitConflict, TxnError, TxnMismatchError, TxnNotAbortable
from .transport import (
    DEFAULT_SOCKET_PERMS,
    default_socket_path,
    serve_unix_socket,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Route table — each handler runs in a worker thread and is given the
# parsed request body + path variables. Keeping routes small and explicit
# is preferable to a framework dependency.
# ---------------------------------------------------------------------------


class _Routes:
    """Routes the daemon serves. The handlers close over a `DaemonServer`
    instance and dispatch into the wrapped Engine."""

    def __init__(self, daemon: "DaemonServer") -> None:
        self._daemon = daemon

    def healthz(self, body: dict[str, Any], **_: str) -> dict[str, Any]:
        return {"ok": True, "started": self._daemon.engine is not None}

    def info(self, body: dict[str, Any], **_: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        cfg = eng.config
        return {
            "ok": True,
            "version": 1,
            "pid": os.getpid(),
            "runtime": cfg.runtime,
            "default_image": cfg.default_image,
            "storage_root": str(eng.storage_root),
            "runtime_root": str(eng.runtime_root),
            "image_cache_root": str(eng.image_cache_root),
            "work_dir_host_root": str(eng.work_dir_host_root),
            "agent_state_root": str(eng.agent_state_root),
            "interceptor_base_url": eng.interceptor_base_url,
            "forwarder_base_url": eng.forwarder_base_url,
            "network_bridge_ip": eng.network_bridge_ip,
            "sandbox_count": len(self._daemon.sandbox_ids()),
        }

    def shutdown(self, body: dict[str, Any], **_: str) -> dict[str, Any]:
        # Schedule the shutdown for after the response has been sent so
        # the caller observes a clean 200. The daemon's own signal-driven
        # shutdown path takes care of stopping the Engine.
        self._daemon.request_shutdown()
        return {"ok": True, "scheduled": True}

    def list_sandboxes(self, body: dict[str, Any], **_: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        rows: list[dict[str, Any]] = []
        for sandbox_id in self._daemon.sandbox_ids():
            try:
                description = eng.runtime.describe(sandbox_id)
            except KeyError:
                continue
            rows.append(_serialize_description(description))
        return {"ok": True, "sandboxes": rows}

    def describe_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        try:
            description = eng.runtime.describe(sid)
        except KeyError as exc:
            raise _NotFound(f"unknown sandbox: {sandbox_id}") from exc
        runtime_state = None
        try:
            runtime_state = eng.runtime.inspect_runtime(sid)
        except Exception:
            logger.debug("inspect_runtime failed for %s", sandbox_id, exc_info=True)
        return {
            "ok": True,
            "description": _serialize_description(description),
            "runtime_state": _serialize_runtime_state(runtime_state) if runtime_state else None,
        }

    def launch_sandbox(self, body: dict[str, Any], **_: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        runtime_name = str(body.get("runtime_name") or eng.runtime.name)
        metadata = dict(body.get("metadata") or {})
        # S5 full-access: if metadata carries 'image' without 'bundle_path',
        # the client is in remote mode and the daemon must do server-side
        # bundle preparation (docker export + runc spec + config).
        if "image" in metadata and "bundle_path" not in metadata:
            metadata = self._prepare_image_launch(eng, metadata)
        sandbox_id = eng.runtime.launch(runtime_name, metadata)
        # Track in the daemon-side registry so /sandboxes lists it and
        # /shutdown can tear it down. The SDK Sandbox is the lifecycle
        # owner; this registry is a cheap mirror.
        self._daemon.register_sandbox(sandbox_id)
        _seed_inspector_running(eng, sandbox_id)
        return {"ok": True, "sandbox_id": str(sandbox_id)}

    def _prepare_image_launch(
        self, eng: Any, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Server-side bundle prep for remote creates (S5 full-access).

        Performs the same work as Sandbox._prepare_runc_launch() but runs
        entirely on the daemon host which has docker + ZFS + root.

        The sequence is: (1) image export, (2) build metadata with
        rootfs_copy_paths/shared_rootfs_key, (3) call runtime.prepare_launch
        which does the ZFS dataset + rootfs materialization, (4) write
        config.json AFTER the ZFS work is complete.  This ordering ensures
        config.json is never disturbed by ZFS mount operations."""
        from integrations.sandboxes.runtime import bundle as sandbox_bundle
        from integrations.sandboxes.runtime import image as sandbox_image

        image_tag = str(metadata["image"])
        sandbox_id_str = str(metadata.get("sandbox_id") or str(SandboxId.new()))

        # Resolve paths from engine's runtime
        rt = eng.runtime
        bundle_dir = rt._paths.bundle_root / sandbox_id_str
        if bundle_dir.exists():
            import shutil
            shutil.rmtree(bundle_dir, ignore_errors=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Image export (creates cached rootfs tarball → directory)
        image_id = sandbox_image.inspect_image_id(tag=image_tag)
        image_defaults = sandbox_image.inspect_image_runtime_defaults(
            tag=image_tag, cache_root=eng.image_cache_root
        )
        exported_rootfs = sandbox_image.export_image_rootfs(
            tag=image_tag,
            output_dir=eng.image_cache_root / image_id,
            cache_root=eng.image_cache_root,
        )

        # 2. Resource limits
        resource_limits = None
        resources = metadata.get("resources")
        if resources and isinstance(resources, dict):
            resource_limits = sandbox_bundle.SandboxResourceLimits(
                cpus=resources.get("cpus"),
                memory_bytes=resources.get("memory_bytes"),
                pids_limit=resources.get("pids"),
            )

        # 3. Network lease (optional)
        network_namespace_path = None
        if metadata.get("network"):
            try:
                lease = eng.allocate_network_lease(SandboxId(sandbox_id_str))
                if lease:
                    network_namespace_path = lease.namespace_path
                    metadata["guest_ip"] = str(lease.guest_ip)
            except Exception:
                logger.debug("network lease allocation skipped", exc_info=True)

        # 4. Build launch metadata with rootfs directives
        rootfs_copy_paths = [{"source": str(exported_rootfs), "destination": "/"}]
        shared_rootfs_key = image_id[:32]
        metadata.update({
            "sandbox_id": sandbox_id_str,
            "bundle_path": str(bundle_dir),
            "work_dir_host_path": None,
            "rootfs_init_dirs": ["/work", "/tmp"],
            "rootfs_copy_paths": rootfs_copy_paths,
            "shared_rootfs_key": shared_rootfs_key,
            "shared_rootfs_persist": True,
            "sdk_image": image_tag,
            "sdk_process_cwd": "/work",
        })

        # 5. Let the runtime handle ZFS dataset creation + rootfs
        #    materialization (shared-clone or fresh copy).  This runs
        #    prepare_launch which may mount a ZFS dataset at bundle/rootfs.
        rt.prepare_launch(rt.name, metadata)

        # 6. Write config.json AFTER ZFS work is done (prepare_launch marks
        #    _crab_runtime_prepared=True, so the second call inside
        #    runtime.launch() will be a no-op).
        rt.write_bundle_spec(bundle_dir)
        sandbox_bundle.write_bundle_config(
            bundle_dir=bundle_dir,
            llm_base_url="",
            provider="openai",
            sandbox_name=sandbox_id_str,
            status_port=0,
            cgroup_path=f"crab-sdk/{sandbox_id_str}",
            work_dir_host_path=None,
            network_namespace_path=network_namespace_path,
            image_defaults=image_defaults,
            image_rootfs_dir=exported_rootfs,
            resource_limits=resource_limits,
        )

        # 7. Write process section (sleep infinity idle init)
        self._write_daemon_bundle_process(
            bundle_dir, image_defaults, sandbox_id_str,
            metadata.get("env"),
        )
        return metadata

    @staticmethod
    def _write_daemon_bundle_process(
        bundle_dir: "Path",
        image_defaults: Any,
        sandbox_id_str: str,
        user_env: dict[str, str] | None,
    ) -> None:
        """Write process section into config.json for daemon-side creates."""
        import json as _json
        config_path = bundle_dir / "config.json"
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        process = dict(cfg.get("process") or {})
        process["terminal"] = False
        process["cwd"] = "/work"
        process["args"] = ["/bin/sh", "-lc", "exec sleep infinity"]
        # Merge environment: image defaults + SDK basics + user env
        defaults_env = (
            getattr(image_defaults, "environment", ()) if image_defaults else ()
        )
        env_map: dict[str, str] = {}
        for item in defaults_env:
            key, sep, value = str(item).partition("=")
            if sep:
                env_map[key] = value
        base_env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/root",
            "PYTHONUNBUFFERED": "1",
            "CRAB_SANDBOX_ID": sandbox_id_str,
        }
        env_map.update(base_env)
        if user_env:
            env_map.update(user_env)
        process["env"] = [f"{k}={v}" for k, v in env_map.items()]
        cfg["process"] = process
        cfg["root"] = {"path": "rootfs", "readonly": False}
        config_path.write_text(_json.dumps(cfg, indent=2), encoding="utf-8")

    def exec_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        argv = list(body.get("argv") or [])
        if not argv:
            raise _BadRequest("exec requires non-empty argv")
        env_raw = body.get("env")
        env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else None
        cwd = body.get("cwd")
        user = body.get("user")
        timeout_s = body.get("timeout_s")
        capture_output = bool(body.get("capture_output", True))
        result = eng.runtime.exec(
            sid,
            list(argv),
            cwd=cwd,
            env=env,
            user=user,
            timeout_s=timeout_s,
            capture_output=capture_output,
        )
        return {
            "ok": True,
            "result": {
                "args": list(result.args),
                "returncode": int(result.returncode),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        }

    def exec_sandbox_stream(self, body: dict[str, Any], *, sandbox_id: str, wfile: Any) -> None:
        """Streaming exec: writes chunked NDJSON to wfile."""
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        argv = list(body.get("argv") or [])
        if not argv:
            raise _BadRequest("exec requires non-empty argv")
        env_raw = body.get("env")
        env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else None
        cwd = body.get("cwd")
        user = body.get("user")
        timeout_s = body.get("timeout_s")
        rc = -1
        try:
            for channel, text in eng.runtime.stream_exec(
                sid, list(argv), cwd=cwd, env=env, user=user, timeout_s=timeout_s
            ):
                if channel == "exit":
                    rc = int(text)
                else:
                    line = json.dumps({"ch": channel, "t": text}) + "\n"
                    _write_chunk(wfile, line.encode("utf-8"))
            done_line = json.dumps({"done": True, "rc": rc}) + "\n"
            _write_chunk(wfile, done_line.encode("utf-8"))
        except BrokenPipeError:
            return
        finally:
            try:
                wfile.write(b"0\r\n\r\n")
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def kill_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        # Fork/txn bookkeeping mirrors Sandbox.kill(): release an open txn
        # (discard staged observations, no restore), materialize chain-shared
        # bytes for live forks of this sandbox, and release its own chain
        # pin if it is a fork. In daemon mode the SDK's kill path runs
        # against the _SystemShim (no-op hooks), so the daemon must do it.
        # Each step is independently best-effort: a failure in one must
        # not skip the others.
        try:
            eng.system.release_txn(sid)
        except Exception:
            logger.exception("txn bookkeeping failed for %s during kill", sid)
        try:
            eng.system.prepare_source_destroy(sid)
        except Exception:
            logger.exception("fork source bookkeeping failed for %s during kill", sid)
        try:
            eng.system.release_fork(sid)
        except Exception:
            logger.exception("fork release bookkeeping failed for %s during kill", sid)
        try:
            eng.runtime.stop(sid)
        except Exception:
            logger.debug("runtime.stop failed for %s during kill", sid, exc_info=True)
        try:
            eng.runtime.delete(sid)
        except Exception:
            logger.exception("runtime.delete failed for %s during kill", sid)
            raise
        self._daemon.unregister_sandbox(sid)
        eng.unregister_upstream(sid)
        try:
            eng.release_network_lease(sid)
        except Exception:
            logger.debug("release_network_lease failed for %s", sid, exc_info=True)
        return {"ok": True, "sandbox_id": str(sid)}

    def write_bundle_spec(self, body: dict[str, Any], **_: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        bundle_dir = body.get("bundle_dir")
        if not isinstance(bundle_dir, str) or not bundle_dir:
            raise _BadRequest("write_bundle_spec requires bundle_dir")
        eng.runtime.write_bundle_spec(Path(bundle_dir))
        return {"ok": True}

    def register_upstream(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        url = body.get("url")
        if not isinstance(url, str) or not url:
            raise _BadRequest("register_upstream requires url")
        eng.register_upstream(SandboxId(sandbox_id), url)
        return {"ok": True}

    def unregister_upstream(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        eng.unregister_upstream(SandboxId(sandbox_id))
        return {"ok": True}

    def allocate_network_lease(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        lease = eng.allocate_network_lease(SandboxId(sandbox_id))
        return {
            "ok": True,
            "lease": {
                "namespace_path": str(lease.namespace_path),
                "guest_ip": str(lease.guest_ip),
                "bridge_ip": eng.network_bridge_ip,
            },
        }

    def release_network_lease(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        eng.release_network_lease(SandboxId(sandbox_id))
        return {"ok": True}

    # ----- sandbox lifecycle (stop/pause/resume) --------------------------

    def stop_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Graceful stop: signal the runc container to terminate but leave
        bundle/ZFS state in place so the sandbox can be restarted via
        restore or destroyed later with DELETE /sandboxes/{id}."""
        eng = self._daemon.require_engine()
        eng.runtime.stop(SandboxId(sandbox_id))
        return {"ok": True, "sandbox_id": sandbox_id}

    def pause_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        eng.runtime.pause(SandboxId(sandbox_id))
        return {"ok": True, "sandbox_id": sandbox_id}

    def resume_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        eng.runtime.resume(SandboxId(sandbox_id))
        return {"ok": True, "sandbox_id": sandbox_id}

    # ----- checkpoints ----------------------------------------------------

    def list_checkpoints(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        try:
            checkpoint_ids = eng.system.storage.list_checkpoints(sid)
        except Exception as exc:
            raise _BadRequest(str(exc)) from exc
        out: list[dict[str, Any]] = []
        for ckpt_id in checkpoint_ids:
            entry: dict[str, Any] = {"checkpoint_id": str(ckpt_id)}
            try:
                manifest = eng.system.storage.get_manifest(sid, ckpt_id)
            except Exception:
                manifest = None
            if manifest is not None:
                entry["created_at"] = (
                    manifest.created_at.isoformat() if manifest.created_at else None
                )
                entry["label"] = manifest.metadata.get("label") if manifest.metadata else None
                entry["has_process"] = bool(
                    getattr(manifest, "process_artifacts", None)
                )
                entry["has_filesystem"] = bool(
                    getattr(manifest, "filesystem_artifacts", None)
                )
            out.append(entry)
        return {"ok": True, "checkpoints": out}

    def create_checkpoint(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        leave_running = bool(body.get("leave_running", True))
        client_checkpoint_id = body.get("checkpoint_id")  # client pre-allocated id
        try:
            result = eng.system.checkpoint_once(
                sid, leave_running=leave_running,
                checkpoint_id=client_checkpoint_id,
            )
        except Exception as exc:
            raise _BadRequest(f"checkpoint failed: {exc}") from exc
        return {
            "ok": True,
            "sandbox_id": sandbox_id,
            "checkpoint_id": str(result.checkpoint_id) if result.checkpoint_id else None,
            "status": result.status,
        }

    def delete_checkpoint(
        self, body: dict[str, Any], *, sandbox_id: str, checkpoint_id: str
    ) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        cascade = bool(body.get("cascade", False))
        try:
            from ..ids import CheckpointId

            eng.system.storage.delete_checkpoint(
                sid, CheckpointId(checkpoint_id), cascade=cascade
            )
        except KeyError as exc:
            raise _NotFound(f"checkpoint not found: {checkpoint_id}") from exc
        return {"ok": True, "sandbox_id": sandbox_id, "checkpoint_id": checkpoint_id}

    def restore_checkpoint(
        self, body: dict[str, Any], *, sandbox_id: str, checkpoint_id: str
    ) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        from ..ids import CheckpointId

        try:
            result = eng.system.restore_once(sid, CheckpointId(checkpoint_id))
        except Exception as exc:
            raise _BadRequest(f"restore failed: {exc}") from exc
        try:
            eng.repair_network_lease(sid)
        except Exception:
            logger.debug("repair_network_lease failed after restore", exc_info=True)
        _seed_inspector_running(eng, sid)
        return {
            "ok": True,
            "sandbox_id": sandbox_id,
            "checkpoint_id": checkpoint_id,
            "status": getattr(result, "status", "succeeded"),
        }

    def fork_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Fork a running sandbox N times (checkpoint + per-fork restore).

        Runs the same local path the in-process Engine exposes; forks are
        registered in the daemon registry so /sandboxes lists them and
        shutdown tears them down."""
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        try:
            count = int(body.get("count", 1))
        except (TypeError, ValueError):
            raise _BadRequest("fork count must be an integer") from None
        if count < 1:
            raise _BadRequest("fork count must be >= 1")
        lazy = bool(body.get("lazy", False))
        effects = body.get("effects")
        if effects is not None and not isinstance(effects, str):
            raise _BadRequest("fork effects must be a string")
        try:
            fork_ids = eng.fork_sandbox(sid, count=count, lazy=lazy, effects=effects)
        except (ValueError, RuntimeError) as exc:
            raise _BadRequest(f"fork failed: {exc}") from exc
        forks: list[dict[str, Any]] = []
        for fork_id in fork_ids:
            self._daemon.register_sandbox(fork_id)
            _seed_inspector_running(eng, fork_id)
            forks.append({"sandbox_id": str(fork_id)})
        return {"ok": True, "sandbox_id": sandbox_id, "forks": forks}

    # ----- transactions ----------------------------------------------

    def begin_txn(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        label_raw = body.get("label")
        label = None if label_raw is None else str(label_raw)
        isolation = str(body.get("isolation") or "snapshot")
        try:
            begin_kwargs: dict[str, Any] = {"label": label, "isolation": isolation}
            if body.get("effects") is not None:
                begin_kwargs["effects"] = str(body["effects"])
            description = eng.system.begin_txn(sid, **begin_kwargs)
        except TxnActiveError as exc:
            raise _TxnConflict("txn_active", str(exc)) from exc
        except (TxnError, ValueError) as exc:
            raise _BadRequest(f"txn begin failed: {exc}") from exc
        return {"ok": True, "txn": _serialize_txn(description)}

    def commit_txn(
        self, body: dict[str, Any], *, sandbox_id: str, txn_id: str
    ) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        force = bool(body.get("force", False))
        commit_kwargs: dict[str, Any] = {"force": force}
        observations_raw = body.get("observations")
        if observations_raw is not None:
            commit_kwargs["observations"] = str(observations_raw)
        try:
            result = eng.system.commit_txn(sid, txn_id, **commit_kwargs)
        except TxnMismatchError as exc:
            raise _TxnConflict("txn_mismatch", str(exc)) from exc
        except TxnCommitConflict as exc:
            raise _TxnConflict("txn_commit_conflict", str(exc)) from exc
        except TxnError as exc:
            raise _BadRequest(f"txn commit failed: {exc}") from exc
        return {
            "ok": True,
            "result": {
                "txn_id": result.txn_id,
                "released_observations": result.released_observations,
                "base_dropped": result.base_dropped,
                "promoted_checkpoint_id": result.promoted_checkpoint_id,
                "observations_consolidated": result.observations_consolidated,
                # Deferred-write flush outcome (D3); absent on pre-D3 results.
                "effects": (
                    None
                    if getattr(result, "effects", None) is None
                    else result.effects.to_json()
                ),
            },
        }

    def abort_txn(
        self, body: dict[str, Any], *, sandbox_id: str, txn_id: str
    ) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        try:
            abort_kwargs: dict[str, Any] = {}
            if body.get("force"):
                abort_kwargs["force"] = True
            result = eng.system.abort_txn(sid, txn_id, **abort_kwargs)
        except TxnNotAbortable as exc:
            # `seal` traded abortability for letting the write out.
            raise _TxnConflict("txn_not_abortable", str(exc)) from exc
        except TxnMismatchError as exc:
            raise _TxnConflict("txn_mismatch", str(exc)) from exc
        except TxnAbortError as exc:
            # Restore failed; the txn stays open and abort is retryable.
            raise _TxnConflict("txn_abort_failed", str(exc)) from exc
        except TxnError as exc:
            raise _BadRequest(f"txn abort failed: {exc}") from exc
        return {
            "ok": True,
            "result": {
                "txn_id": result.txn_id,
                "discarded_observations": result.discarded_observations,
                "restored_checkpoint_id": result.restored_checkpoint_id,
                # Mutating egress the abort could not undo (D1); omitting it
                # would make remote aborts silently report zero.
                "mutating_egress": getattr(result, "mutating_egress", 0),
                # Deferred writes this abort discarded (D3).
                "deferred_dropped": getattr(result, "deferred_dropped", 0),
            },
        }

    def current_txn(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        description = eng.system.current_txn(SandboxId(sandbox_id))
        return {
            "ok": True,
            "txn": None if description is None else _serialize_txn(description),
        }

    # ----- filesystem merge / changesets (C2) -------------------------

    def merge_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Three-way merge a fork's filesystem changes back into its
        source. Refusals/failures surface as 409 with error_type
        ``merge_error`` and the serialized report when one exists, so
        the SDK shim can rehydrate an identical ``MergeError``."""
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        fork_raw = body.get("fork_sandbox_id")
        if not fork_raw:
            raise _BadRequest("merge requires fork_sandbox_id")
        kwargs: dict[str, Any] = {"policy": str(body.get("policy") or "fail_fast")}
        observations_raw = body.get("observations")
        if observations_raw is not None:
            kwargs["observations"] = str(observations_raw)
        prefixes_raw = body.get("ignore_prefixes")
        if prefixes_raw is not None:
            if not isinstance(prefixes_raw, list) or not all(
                isinstance(prefix, str) for prefix in prefixes_raw
            ):
                raise _BadRequest("ignore_prefixes must be a list of strings")
            kwargs["ignore_prefixes"] = tuple(prefixes_raw)
        try:
            report = eng.system.merge_from_fork(sid, SandboxId(str(fork_raw)), **kwargs)
        except MergeError as exc:
            raise _MergeConflict(
                str(exc),
                report=None if exc.report is None else exc.report.to_json(),
            ) from exc
        except ValueError as exc:
            raise _BadRequest(f"merge failed: {exc}") from exc
        return {"ok": True, "report": report.to_json()}

    def changeset_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Changed rootfs paths relative to a base checkpoint; without
        ``since`` the sandbox's fork point is resolved (C1 semantics).
        ``force=true`` skips the inspector gate optimization."""
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        from ..ids import CheckpointId

        since_raw = body.get("since")
        force = bool(body.get("force", False))
        try:
            if since_raw:
                result = eng.system.changeset_since(
                    sid,
                    CheckpointId(str(since_raw)),
                    use_inspector_gate=not force,
                )
            else:
                result = eng.system.fork_changeset(sid, force=force)
        except (FileNotFoundError, ValueError) as exc:
            raise _BadRequest(f"changeset failed: {exc}") from exc
        return {"ok": True, "changeset": result.to_json()}

    def consolidate_observations_sandbox(
        self, body: dict[str, Any], *, sandbox_id: str
    ) -> dict[str, Any]:
        """Adopt a fork's journal history into this sandbox's journal
        (C3). Summarizer hooks never cross the RPC boundary."""
        eng = self._daemon.require_engine()
        fork_raw = body.get("fork_sandbox_id")
        if not fork_raw:
            raise _BadRequest("consolidate requires fork_sandbox_id")
        policy = str(body.get("policy") or "append")
        try:
            report = eng.system.consolidate_observations(
                SandboxId(sandbox_id),
                SandboxId(str(fork_raw)),
                policy=policy,
                reason=str(body.get("reason") or "manual"),
            )
        except (ValueError, RuntimeError) as exc:
            raise _BadRequest(f"consolidate failed: {exc}") from exc
        return {"ok": True, "report": report.to_json()}

    def sandbox_actions(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Read the sandbox's action journal (exec/lifecycle/observation
        records). env values ride verbatim — same trust domain as the
        daemon socket."""
        eng = self._daemon.require_engine()
        journal = getattr(eng.system, "journal", None)
        if journal is None:
            raise _BadRequest("action journal is not enabled on this daemon")
        kind_raw = body.get("kind")
        since_raw = body.get("since_seq")
        records = journal.entries(
            SandboxId(sandbox_id),
            kind=None if not kind_raw else str(kind_raw),
            since_seq=None if since_raw is None else int(since_raw),
        )
        limit_raw = body.get("limit")
        if limit_raw is not None and int(limit_raw) >= 0:
            records = records[-int(limit_raw):]
        return {"ok": True, "records": [record.to_json() for record in records]}

    def sandbox_egress(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Effect ledger view (D1): recorded egress flows, optionally
        scoped to one transaction."""
        eng = self._daemon.require_engine()
        txn_raw = body.get("txn_id")
        since_raw = body.get("since_seq")
        try:
            ledger = eng.system.egress_ledger(
                SandboxId(sandbox_id),
                txn_id=None if not txn_raw else str(txn_raw),
                since_seq=None if since_raw is None else int(since_raw),
            )
        except RuntimeError as exc:
            raise _BadRequest(f"egress ledger unavailable: {exc}") from exc
        return {"ok": True, "ledger": ledger.to_json()}

    def sandbox_egress_replay(
        self, body: dict[str, Any], *, sandbox_id: str
    ) -> dict[str, Any]:
        """Arm or close a replay window (D2). ``mode="end"`` returns the
        report; ``cassette_source`` reads another sandbox's bucket (the
        fork whose reads are being re-run)."""
        eng = self._daemon.require_engine()
        mode = str(body.get("mode") or "begin")
        sid = SandboxId(sandbox_id)
        if mode == "end":
            report = eng.system.end_egress_replay(sid)
            return {
                "ok": True,
                "report": None if report is None else report.to_json(),
            }
        if mode != "begin":
            raise _BadRequest(f"unknown replay mode: {mode!r} (expected begin or end)")
        source = body.get("cassette_source")
        try:
            eng.system.begin_egress_replay(
                sid,
                policy=str(body.get("policy") or "cassette_first"),
                cassette_source=None if not source else str(source),
            )
        except (ValueError, RuntimeError) as exc:
            raise _BadRequest(f"egress replay failed: {exc}") from exc
        return {"ok": True}

    def merge_processes_sandbox(
        self, body: dict[str, Any], *, sandbox_id: str
    ) -> dict[str, Any]:
        """Process-half of consolidation (C4): replay the fork's execs
        on the source, or (PR-C4.2) promote the fork wholesale.
        Refusals surface as 409 error_type=process_merge_conflict."""
        eng = self._daemon.require_engine()
        fork_raw = body.get("fork_sandbox_id")
        if not fork_raw:
            raise _BadRequest("merge-processes requires fork_sandbox_id")
        kwargs: dict[str, Any] = {
            "strategy": str(body.get("strategy") or "auto"),
            "stop_on_deviation": bool(body.get("stop_on_deviation", False)),
            "force": bool(body.get("force", False)),
        }
        if body.get("policy") is not None:
            kwargs["policy"] = str(body["policy"])
        if body.get("observations") is not None:
            kwargs["observations"] = str(body["observations"])
        if body.get("lazy_pages") is not None:
            kwargs["lazy_pages"] = bool(body["lazy_pages"])
        if body.get("egress_replay") is not None:
            kwargs["egress_replay"] = str(body["egress_replay"])
        if body.get("replay_effects") is not None:
            kwargs["replay_effects"] = str(body["replay_effects"])
        try:
            report = eng.system.merge_processes(
                SandboxId(sandbox_id), SandboxId(str(fork_raw)), **kwargs
            )
        except ProcessMergeConflict as exc:
            raise _TxnConflict("process_merge_conflict", str(exc)) from exc
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            raise _BadRequest(f"merge-processes failed: {exc}") from exc
        return {"ok": True, "report": report.to_json()}

    def update_host_inspector_filters(
        self, body: dict[str, Any], *, sandbox_id: str
    ) -> dict[str, Any]:
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        runtime = eng.runtime
        if not hasattr(runtime, "update_host_inspector_filters"):
            return {"ok": True, "applied": False}
        ignore_process_rules = body.get("ignore_process_rules")
        ignored_path_prefixes = body.get("ignored_path_prefixes")
        runtime.update_host_inspector_filters(
            sid,
            ignore_process_rules=ignore_process_rules,
            ignored_path_prefixes=ignored_path_prefixes,
        )
        return {"ok": True, "applied": True}

    def inspect_sandbox(self, body: dict[str, Any], *, sandbox_id: str) -> dict[str, Any]:
        """Read-only peek at inspector state. Does NOT reset flags.

        Self-heals on KeyError: if the sandbox exists in the runtime but
        the inspector never got seeded (crash / race between launch and
        the seed call), lazily register a clean baseline snapshot and
        retry once. Only if the runtime itself doesn't know the sandbox
        do we return 404."""
        eng = self._daemon.require_engine()
        sid = SandboxId(sandbox_id)
        try:
            snapshot = eng.system.inspector.inspect(sid)
        except KeyError:
            try:
                eng.runtime.describe(sid)
            except KeyError as exc:
                raise _NotFound(f"unknown sandbox: {sandbox_id}") from exc
            _seed_inspector_running(eng, sid)
            try:
                snapshot = eng.system.inspector.inspect(sid)
            except KeyError as exc:
                raise _NotFound(
                    f"inspector snapshot could not be seeded for {sandbox_id}"
                ) from exc
        return {
            "ok": True,
            "sandbox_id": sandbox_id,
            "runtime_name": eng.runtime.name,
            "is_running": bool(getattr(snapshot, "is_running", True)),
            "filesystem_changed": bool(snapshot.filesystem_changed),
            "process_changed": bool(snapshot.process_changed),
        }


def _seed_inspector_running(engine: Engine, sandbox_id: SandboxId) -> None:
    """Mirror the SDK's local `Sandbox._mark_inspector_running`.

    In daemon mode the SDK side holds a no-op inspector shim, so the
    daemon must seed its own engine's inspector after launch/restore/fork
    — otherwise checkpoint flows fail with "sandbox snapshot not found"
    and the scheduler's running-guard blocks checkpoints."""
    try:
        engine.system.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name=engine.runtime.name,
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
    except Exception:
        logger.debug("failed to seed inspector snapshot for %s", sandbox_id, exc_info=True)


def _serialize_txn(description) -> dict[str, Any]:
    return {
        "txn_id": description.txn_id,
        "sandbox_id": description.sandbox_id,
        "base_checkpoint_id": description.base_checkpoint_id,
        "base_was_fresh": bool(description.base_was_fresh),
        "started_at": description.started_at,
        "label": description.label,
        "isolation": getattr(description, "isolation", "snapshot"),
        "effects": getattr(description, "effects", "allow"),
        "fork_sandbox_id": getattr(description, "fork_sandbox_id", None),
    }


def _serialize_description(description) -> dict[str, Any]:
    return {
        "sandbox_id": str(description.sandbox_id),
        "runtime_name": description.runtime_name,
        "status": description.status,
        "metadata": _jsonable(description.metadata),
    }


def _serialize_runtime_state(state) -> dict[str, Any]:
    return {
        "sandbox_id": str(state.sandbox_id),
        "runtime_name": state.runtime_name,
        "status": state.status,
        "pid": state.pid,
        "bundle_path": state.bundle_path,
        "metadata": _jsonable(state.metadata),
        "is_running": state.is_running,
    }


def _jsonable(value: Any) -> Any:
    """Best-effort coercion of metadata values to JSON-serializable form."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


# ---------------------------------------------------------------------------
# Request handler — dispatches by (method, path) to a callable.
# ---------------------------------------------------------------------------


class _BadRequest(Exception):
    pass


class _TxnConflict(Exception):
    """409 with a machine-readable error_type so the SDK shim can raise
    the matching Txn* exception on the client side."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class _MergeConflict(Exception):
    """409 with error_type ``merge_error`` plus the serialized merge
    report (when the failure produced one) so the SDK shim rehydrates a
    ``MergeError`` with an intact ``report``."""

    def __init__(self, message: str, *, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report


class _NotFound(Exception):
    pass


def _write_chunk(wfile: Any, data: bytes) -> None:
    """Write a single HTTP chunked-encoding frame."""
    wfile.write(f"{len(data):x}\r\n".encode("ascii"))
    wfile.write(data)
    wfile.write(b"\r\n")
    wfile.flush()


def _build_handler(daemon: "DaemonServer"):
    routes = _Routes(daemon)

    # Path patterns: literal segments or `{var}` placeholders. Kept tiny
    # so the dispatch table is easy to audit.
    table: list[tuple[str, str, str, Callable[..., dict[str, Any]]]] = [
        ("GET", "/healthz", "", routes.healthz),
        ("GET", "/info", "", routes.info),
        ("POST", "/shutdown", "", routes.shutdown),
        ("GET", "/sandboxes", "", routes.list_sandboxes),
        ("POST", "/sandboxes", "", routes.launch_sandbox),
        ("POST", "/runtime/write_bundle_spec", "", routes.write_bundle_spec),
        ("GET", "/sandboxes/{sandbox_id}", "", routes.describe_sandbox),
        ("GET", "/sandboxes/{sandbox_id}/inspector", "", routes.inspect_sandbox),
        ("DELETE", "/sandboxes/{sandbox_id}", "", routes.kill_sandbox),
        ("POST", "/sandboxes/{sandbox_id}/exec", "", routes.exec_sandbox),
        ("POST", "/sandboxes/{sandbox_id}/upstream", "", routes.register_upstream),
        ("DELETE", "/sandboxes/{sandbox_id}/upstream", "", routes.unregister_upstream),
        ("POST", "/sandboxes/{sandbox_id}/network/lease", "", routes.allocate_network_lease),
        ("DELETE", "/sandboxes/{sandbox_id}/network/lease", "", routes.release_network_lease),
        (
            "POST",
            "/sandboxes/{sandbox_id}/host_inspector/filters",
            "",
            routes.update_host_inspector_filters,
        ),
        # Lifecycle (non-destructive)
        ("POST", "/sandboxes/{sandbox_id}/stop", "", routes.stop_sandbox),
        ("POST", "/sandboxes/{sandbox_id}/pause", "", routes.pause_sandbox),
        ("POST", "/sandboxes/{sandbox_id}/resume", "", routes.resume_sandbox),
        # Checkpoints (keyed by sandbox; checkpoint ids are unique per-sandbox)
        ("GET", "/sandboxes/{sandbox_id}/checkpoints", "", routes.list_checkpoints),
        ("POST", "/sandboxes/{sandbox_id}/checkpoints", "", routes.create_checkpoint),
        (
            "DELETE",
            "/sandboxes/{sandbox_id}/checkpoints/{checkpoint_id}",
            "",
            routes.delete_checkpoint,
        ),
        (
            "POST",
            "/sandboxes/{sandbox_id}/checkpoints/{checkpoint_id}/restore",
            "",
            routes.restore_checkpoint,
        ),
        # Fork (checkpoint + restore into new sandboxes)
        ("POST", "/sandboxes/{sandbox_id}/fork", "", routes.fork_sandbox),
        # Transactions (snapshot-based; see crab/txn.py)
        ("GET", "/sandboxes/{sandbox_id}/txn", "", routes.current_txn),
        ("POST", "/sandboxes/{sandbox_id}/txn", "", routes.begin_txn),
        ("POST", "/sandboxes/{sandbox_id}/txn/{txn_id}/commit", "", routes.commit_txn),
        ("POST", "/sandboxes/{sandbox_id}/txn/{txn_id}/abort", "", routes.abort_txn),
        # Filesystem merge + changesets (C2; see crab/merging.py)
        ("POST", "/sandboxes/{sandbox_id}/merge", "", routes.merge_sandbox),
        ("POST", "/sandboxes/{sandbox_id}/changeset", "", routes.changeset_sandbox),
        # Observation consolidation + journal reads (C3)
        (
            "POST",
            "/sandboxes/{sandbox_id}/observations/consolidate",
            "",
            routes.consolidate_observations_sandbox,
        ),
        ("POST", "/sandboxes/{sandbox_id}/actions", "", routes.sandbox_actions),
        # Effect ledger (D1)
        ("POST", "/sandboxes/{sandbox_id}/egress", "", routes.sandbox_egress),
        ("POST", "/sandboxes/{sandbox_id}/egress/replay", "", routes.sandbox_egress_replay),
        # Process merge (C4)
        ("POST", "/sandboxes/{sandbox_id}/processes/merge", "", routes.merge_processes_sandbox),
    ]

    def _match(method: str, path: str):
        path = path.split("?", 1)[0]
        for route_method, pattern, _meta, fn in table:
            if route_method != method:
                continue
            variables = _try_match(pattern, path)
            if variables is not None:
                return fn, variables
        return None, None

    def _try_match(pattern: str, path: str) -> dict[str, str] | None:
        pat_parts = [p for p in pattern.strip("/").split("/") if p]
        path_parts = [p for p in path.strip("/").split("/") if p]
        if len(pat_parts) != len(path_parts):
            return None
        variables: dict[str, str] = {}
        for pat, real in zip(pat_parts, path_parts, strict=True):
            if pat.startswith("{") and pat.endswith("}"):
                variables[pat[1:-1]] = real
                continue
            if pat != real:
                return None
        return variables

    class Handler(BaseHTTPRequestHandler):
        server_version = "crab-daemon/1"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("crab-daemon: " + fmt, *args)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            try:
                # Detect ?stream=1 query param for streaming exec
                raw_path = self.path
                query_string = ""
                if "?" in raw_path:
                    _, query_string = raw_path.split("?", 1)
                fn, variables = _match(method, raw_path)
                if fn is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                    return
                body = self._read_body()
                # Stream fork: if this is exec + stream=1, use streaming handler
                if getattr(fn, "__func__", None) is _Routes.exec_sandbox and "stream=1" in query_string:
                    self._handle_stream_exec(body, variables or {})
                    return
                result = fn(body, **(variables or {}))
                self._send_json(HTTPStatus.OK, result)
            except _BadRequest as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except _TxnConflict as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": str(exc), "error_type": exc.error_type},
                )
            except _MergeConflict as exc:
                payload: dict[str, Any] = {
                    "ok": False,
                    "error": str(exc),
                    "error_type": "merge_error",
                }
                if exc.report is not None:
                    payload["report"] = exc.report
                self._send_json(HTTPStatus.CONFLICT, payload)
            except _NotFound as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except KeyError as exc:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"unknown sandbox: {exc.args[0] if exc.args else ''}"},
                )
            except Exception as exc:
                logger.exception("daemon request failed: %s %s", method, self.path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )

        def _read_body(self) -> dict[str, Any]:
            length_header = self.headers.get("Content-Length")
            if not length_header:
                return {}
            try:
                length = int(length_header)
            except ValueError:
                raise _BadRequest("invalid Content-Length") from None
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise _BadRequest(f"invalid JSON body: {exc}") from exc
            if not isinstance(decoded, dict):
                raise _BadRequest("request body must be a JSON object")
            return decoded

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _handle_stream_exec(self, body: dict[str, Any], variables: dict[str, str]) -> None:
            """Write chunked NDJSON streaming exec response."""
            try:
                # Send HTTP headers directly (bypass BaseHTTPRequestHandler's
                # buffered response to get chunked transfer-encoding).
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                routes.exec_sandbox_stream(body, wfile=self.wfile, **variables)
            except _BadRequest as exc:
                # If headers not yet sent this won't work cleanly, but
                # exec_sandbox_stream raises _BadRequest before any IO
                # if argv is empty, and we've already sent 200. Write an
                # error frame instead.
                err_line = json.dumps({"error": str(exc), "done": True, "rc": -1}) + "\n"
                try:
                    _write_chunk(self.wfile, err_line.encode("utf-8"))
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                logger.exception("stream exec failed: %s", self.path)
                err_line = json.dumps({"error": f"{type(exc).__name__}: {exc}", "done": True, "rc": -1}) + "\n"
                try:
                    _write_chunk(self.wfile, err_line.encode("utf-8"))
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

    return Handler


# ---------------------------------------------------------------------------
# DaemonServer — owns the Engine, the HTTP server, and the sandbox registry.
# ---------------------------------------------------------------------------


class DaemonServer:
    """The Crab daemon."""

    def __init__(
        self,
        *,
        engine_config: EngineConfig | str | os.PathLike[str] | None = None,
        socket_path: Path | None = None,
        pid_file: Path | None = None,
        socket_group: str | None = None,
    ) -> None:
        self._engine_config = engine_config
        self._socket_path = (socket_path or default_socket_path()).expanduser().resolve()
        self._pid_file = pid_file.expanduser().resolve() if pid_file else None
        # Opt-in group sharing (S1): when a group is named the socket is
        # chgrp'd and opened 0660 so a non-root gateway user in that group
        # can reach the daemon. Unset → 0600, zero behavior change.
        self._socket_group = socket_group
        self._socket_perms = DEFAULT_SOCKET_PERMS if socket_group is None else 0o660
        self._engine: Engine | None = None
        self._server = None
        self._serve_thread: threading.Thread | None = None
        self._sandboxes_lock = threading.Lock()
        self._sandboxes: set[SandboxId] = set()
        self._stop_event = threading.Event()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def engine(self) -> Engine | None:
        return self._engine

    def require_engine(self) -> Engine:
        engine = self._engine
        if engine is None or not engine.started:
            raise RuntimeError("daemon engine is not running")
        return engine

    def sandbox_ids(self) -> list[SandboxId]:
        with self._sandboxes_lock:
            return list(self._sandboxes)

    def register_sandbox(self, sandbox_id: SandboxId) -> None:
        with self._sandboxes_lock:
            self._sandboxes.add(sandbox_id)

    def unregister_sandbox(self, sandbox_id: SandboxId) -> None:
        with self._sandboxes_lock:
            self._sandboxes.discard(sandbox_id)

    def start(self) -> None:
        if self._engine is not None:
            raise RuntimeError("daemon is already started")
        logger.info("Starting Crab daemon; socket=%s", self._socket_path)
        engine = Engine.start(self._engine_config)
        self._engine = engine
        try:
            handler = _build_handler(self)
            self._server = serve_unix_socket(
                self._socket_path,
                handler,
                socket_perms=self._socket_perms,
                socket_group=self._socket_group,
            )
        except Exception:
            engine.stop()
            self._engine = None
            raise
        self._serve_thread = threading.Thread(
            target=self._server.serve_forever,
            name="crab-daemon-http",
            daemon=True,
        )
        self._serve_thread.start()
        if self._pid_file is not None:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            self._pid_file.write_text(str(os.getpid()))
        logger.info(
            "Crab daemon ready: socket=%s perms=%o pid=%d sandbox_count=%d",
            self._socket_path,
            self._socket_perms,
            os.getpid(),
            0,
        )

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                logger.exception("daemon HTTP server shutdown failed")
            self._server = None
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=5.0)
            self._serve_thread = None
        # Kill any sandboxes the daemon launched but the SDK never cleaned
        # up. This is the equivalent of `dockerd` stopping its containers
        # on shutdown when configured to do so.
        engine = self._engine
        if engine is not None:
            for sandbox_id in self.sandbox_ids():
                try:
                    engine.runtime.stop(sandbox_id)
                except Exception:
                    pass
                try:
                    engine.runtime.delete(sandbox_id)
                except Exception:
                    logger.exception("runtime.delete failed for %s on daemon stop", sandbox_id)
            try:
                engine.stop()
            except Exception:
                logger.exception("engine.stop failed during daemon shutdown")
            self._engine = None
        if self._pid_file is not None:
            try:
                self._pid_file.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            if self._socket_path.exists():
                self._socket_path.unlink()
        except OSError:
            pass

    def request_shutdown(self) -> None:
        """Schedule shutdown without blocking the calling thread.

        The HTTP handler calls this so the response is delivered before
        the server tears itself down; the main thread is woken via
        `_stop_event` to drive `stop()`."""
        self._stop_event.set()

    def serve_forever(self) -> None:
        """Block until SIGTERM/SIGINT or `request_shutdown()`."""
        self._install_signal_handlers()
        try:
            while not self._stop_event.wait(timeout=1.0):
                continue
        finally:
            self.stop()

    def _install_signal_handlers(self) -> None:
        def _on_signal(signum: int, _frame: Any) -> None:
            logger.info("daemon received signal %d; shutting down", signum)
            self._stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _on_signal)
            except ValueError:
                # Not on the main thread; skipped in test embedding.
                pass


# ---------------------------------------------------------------------------
# CLI entry point — used by `python -m crab.daemon` and the
# `crabd` console script.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crabd",
        description="Run the Crab daemon (long-running host service).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to EngineConfig YAML. Defaults to $CRAB_DAEMON_CONFIG or built-in defaults.",
    )
    parser.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Override the Unix socket path the daemon listens on.",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="Optional PID file path.",
    )
    parser.add_argument(
        "--socket-group",
        default=None,
        help=(
            "Group name to chgrp the daemon socket to; implies mode 0660 so "
            "group members (e.g. the crab-gateway user) can connect. "
            "Unset keeps the 0600 sole-user default."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("CRAB_DAEMON_LOG_LEVEL", "INFO"),
        help="Log level (DEBUG/INFO/WARNING/ERROR).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )

    config_arg = args.config
    if config_arg is None:
        env_config = os.environ.get("CRAB_DAEMON_CONFIG")
        if env_config:
            config_arg = Path(env_config)

    daemon = DaemonServer(
        engine_config=config_arg,
        socket_path=args.socket,
        pid_file=args.pid_file,
        socket_group=args.socket_group,
    )
    try:
        daemon.start()
    except Exception:
        logger.exception("daemon failed to start")
        return 1
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
