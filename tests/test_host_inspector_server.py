from __future__ import annotations

import io
import os
import subprocess
import unittest
from unittest.mock import Mock, patch

from crab import HostInspectorServiceClient, SandboxId
from crab.host_inspector.fs_helper import LibbpfFilesystemMonitor, _EventWorkerPool
from crab.host_inspector.process_filter import ProcessIdentity
from crab.host_inspector.protocol import HelperEvent
from crab.host_inspector.runtime_resolver import ResolvedSandbox
from crab.host_inspector.server import HostInspectorDaemon, HostInspectorServer


class FakeResolver:
    def __init__(self) -> None:
        self.item = ResolvedSandbox(
            runtime="docker",
            object_id="container-1",
            runtime_name="docker",
            is_running=True,
            init_pid=111,
            cgroup_path="/demo.scope",
            cgroup_id=6869,
        )

    def resolve(self, runtime: str, object_id: str) -> ResolvedSandbox:
        _ = (runtime, object_id)
        return self.item


class FakeFilesystemMonitor:
    def __init__(self) -> None:
        self.started = False
        self.on_event = None
        self.upserts: list[tuple[str, int]] = []
        self.removals: list[str] = []
        self.sync_calls = 0
        self.ignored_pid_adds: list[int] = []
        self.ignored_pid_removes: list[int] = []
        self.ignore_rule_pushes: list[tuple[str, tuple]] = []

    def start(self, on_event) -> None:
        self.started = True
        self.on_event = on_event

    def stop(self) -> None:
        self.started = False

    def upsert_sandbox(self, sandbox_id: str, cgroup_id: int) -> None:
        self.upserts.append((sandbox_id, cgroup_id))

    def remove_sandbox(self, sandbox_id: str) -> None:
        self.removals.append(sandbox_id)

    def sync(self, sandbox_id: str, timeout_s: float = 2.0) -> bool:
        self.sync_calls += 1
        return True

    def add_ignored_pid(self, pid: int) -> None:
        self.ignored_pid_adds.append(int(pid))

    def remove_ignored_pid(self, pid: int) -> None:
        self.ignored_pid_removes.append(int(pid))

    def set_ignore_process_rules(self, sandbox_id: str, rules) -> None:
        self.ignore_rule_pushes.append((sandbox_id, tuple(rules)))


def _event(
    *,
    sandbox_id: str = "sbx-1",
    syscall: str,
    pid: int | None = None,
    fd: int | None = None,
    fd_kind: str | None = None,
    flags: int | None = None,
    path: str | None = None,
    path_secondary: str | None = None,
    inode: int | None = None,
    device: int | None = None,
) -> HelperEvent:
    return HelperEvent(
        sandbox_id=sandbox_id,
        kind="filesystem_change",
        syscall=syscall,
        pid=pid,
        fd=fd,
        fd_kind=fd_kind,
        flags=flags,
        path=path,
        path_secondary=path_secondary,
        inode=inode,
        device=device,
        cgroup_id=6869,
        timestamp="2026-03-11T12:00:00+00:00",
    )

class HostInspectorServerTests(unittest.TestCase):
    def test_conservative_filesystem_marker_survives_status_until_reset(self) -> None:
        daemon = HostInspectorDaemon(
            resolver=FakeResolver(),
            fs_monitor=FakeFilesystemMonitor(),
            process_poll_interval_s=60.0,
        )
        server = HostInspectorServer(host="127.0.0.1", port=0, daemon=daemon)
        server.start()
        client = HostInspectorServiceClient(
            f"http://127.0.0.1:{server.port}", timeout_s=1.0
        )
        try:
            with patch(
                "crab.host_inspector.server.list_cgroup_pids",
                return_value={111},
            ), patch(
                "crab.host_inspector.server.reset_soft_dirty_for_pids",
                return_value={111},
            ), patch(
                "crab.host_inspector.server.dirty_pids", return_value=set()
            ), patch(
                "crab.host_inspector.server.all_deleted_mmap_paths",
                return_value=frozenset(),
            ):
                client.register_sandbox(
                    SandboxId("sbx-dirty"), "docker", "container-1"
                )
                client.reset_sandbox(SandboxId("sbx-dirty"), None)
                client.mark_filesystem_changed(
                    SandboxId("sbx-dirty"), reason="isolated exec"
                )

                dirty = client.get_proc_and_fs_status(
                    SandboxId("sbx-dirty")
                )["status"]
                self.assertTrue(dirty["filesystem_changed"])
                self.assertEqual(
                    dirty["metadata"]["unreconciled_fs_events"][-1]["reason"],
                    "isolated exec",
                )

                client.reset_sandbox(SandboxId("sbx-dirty"), None)
                clean = client.get_proc_and_fs_status(
                    SandboxId("sbx-dirty")
                )["status"]
                self.assertFalse(clean["filesystem_changed"])

                first_unregister = client.unregister_sandbox(
                    SandboxId("sbx-dirty")
                )
                second_unregister = client.unregister_sandbox(
                    SandboxId("sbx-dirty")
                )
                self.assertTrue(first_unregister["unregistered"])
                self.assertTrue(second_unregister["already_absent"])
        finally:
            client.close()
            server.stop()

    def test_server_closes_idle_connections_so_one_client_does_not_pin_only_worker(self) -> None:
        daemon = HostInspectorDaemon(resolver=FakeResolver(), fs_monitor=FakeFilesystemMonitor(), process_poll_interval_s=60.0)
        server = HostInspectorServer(host="127.0.0.1", port=0, daemon=daemon, max_workers=1)
        server.start()
        base_url = f"http://127.0.0.1:{server.port}"
        client_a = HostInspectorServiceClient(base_url, timeout_s=0.5)
        client_b = HostInspectorServiceClient(base_url, timeout_s=0.5)
        try:
            first = client_a.register_sandbox(SandboxId("sbx-a"), "docker", "container-1")
            second = client_b.register_sandbox(SandboxId("sbx-b"), "docker", "container-1")
        finally:
            client_a.close()
            client_b.close()
            server.stop()

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])

    def test_filesystem_monitor_stop_kills_helper_after_terminate_timeout(self) -> None:
        monitor = LibbpfFilesystemMonitor(helper_path="/bin/true")
        stdout = Mock()
        stderr = Mock()
        stdin = Mock()
        process = Mock()
        process.pid = 12345
        process.poll.side_effect = [None, None]
        process.stdin = stdin
        process.stdout = stdout
        process.stderr = stderr
        process.wait.side_effect = [subprocess.TimeoutExpired(["fs_monitor"], 5.0), 0]
        monitor._process = process
        stdout_thread = Mock()
        stderr_thread = Mock()
        monitor._stdout_thread = stdout_thread
        monitor._stderr_thread = stderr_thread

        monitor.stop()

        stdin.close.assert_called_once()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        stdout.close.assert_called_once()
        stderr.close.assert_called_once()
        stdout_thread.join.assert_called_once()
        stderr_thread.join.assert_called_once()
        self.assertIsNone(monitor._stdout_thread)
        self.assertIsNone(monitor._stderr_thread)

    def test_filesystem_monitor_reader_ignores_stream_close_during_shutdown(self) -> None:
        class RaisingStream:
            def __iter__(self):
                raise ValueError("I/O operation on closed file")

        monitor = LibbpfFilesystemMonitor(helper_path="/bin/true")
        process = Mock()
        process.stdout = RaisingStream()
        process.stderr = io.StringIO("")
        monitor._process = process

        monitor._read_stdout()

    def test_register_reset_status_and_unregister(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        registered = {"status": daemon.register("sbx-1", "docker", "container-1")}
        self.assertFalse(registered["status"]["process_changed"])
        self.assertFalse(registered["status"]["filesystem_changed"])
        self.assertEqual(fs_monitor.upserts, [("sbx-1", 6869)])

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111, 222},
        ):
            reset = {"status": daemon.reset("sbx-1")}
        self.assertFalse(reset["status"]["process_changed"])
        self.assertFalse(reset["status"]["filesystem_changed"])

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                fd=3,
                fd_kind="regular",
                flags=577,
                path="/tmp/demo.txt",
                inode=10,
                device=1,
            )
        )
        with patch("crab.host_inspector.server.os.lstat"), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111, 222},
        ), patch("crab.host_inspector.server.dirty_pids", return_value=set()):
            status = {"status": daemon.status("sbx-1")}
        self.assertFalse(status["status"]["process_changed"])
        self.assertTrue(status["status"]["filesystem_changed"])

        unregistered = daemon.unregister("sbx-1")
        self.assertTrue(unregistered["unregistered"])
        self.assertIn("sbx-1", fs_monitor.removals)

    def test_checkpoint_reset_stabilizes_idle_criu_residual(self) -> None:
        """A checkpoint reset (captures_process=True) must scrub the soft-dirty
        residue CRIU's --leave-running dump leaves behind so an idle sandbox
        does not latch a false process_changed=True.

        The first verification scan sees the residual page; a follow-up clear
        settles it (idle process, no ongoing writes) and the sandbox reports
        process_changed=False.
        """
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        scan_returns = [{111}, set()]
        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths", return_value=frozenset()
        ), patch(
            "crab.host_inspector.process_monitor.clear_soft_dirty"
        ) as clear, patch(
            "crab.host_inspector.process_monitor.dirty_pids", side_effect=scan_returns
        ), patch("crab.host_inspector.process_monitor.time.sleep"):
            reset = daemon.reset("sbx-1", captures_process=True)

        self.assertFalse(reset["process_changed"])
        self.assertEqual(reset["metadata"]["baseline_pids"], [111])
        # initial clear of the tracked pid + one re-clear of the CRIU residual.
        self.assertEqual(clear.call_count, 2)

    def test_checkpoint_reset_does_not_mask_busy_process(self) -> None:
        """Reverse guard: a process that keeps writing must survive the
        stabilization loop. Every verification scan still reports it dirty, so
        the re-clears are bounded and the next status() poll reports the real
        activity as process_changed=True.
        """
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths", return_value=frozenset()
        ), patch("crab.host_inspector.process_monitor.clear_soft_dirty"), patch(
            "crab.host_inspector.process_monitor.dirty_pids", return_value={111}
        ), patch("crab.host_inspector.process_monitor.time.sleep"):
            daemon.reset("sbx-1", captures_process=True)

        # After the (bounded) stabilization, a status() poll that still sees the
        # pid dirty must report the real write activity, not swallow it.
        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.dirty_pids", return_value={111}
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["dirty_pids"], [111])

    def test_non_checkpoint_reset_skips_stabilization(self) -> None:
        """A plain baseline reset (captures_process=False) has no CRIU dump, so
        it must keep the cheap single-clear path and never run the verification
        scan.
        """
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.process_monitor.clear_soft_dirty"
        ) as clear, patch(
            "crab.host_inspector.process_monitor.dirty_pids"
        ) as scan:
            reset = daemon.reset("sbx-1")

        self.assertFalse(reset["process_changed"])
        self.assertEqual(clear.call_count, 1)
        scan.assert_not_called()

    def test_status_reports_dirty_memory_for_current_live_pid(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value={111},
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["dirty_pids"], [111])

    def test_status_returns_false_after_transient_pid_exits(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertFalse(status["process_changed"])
        self.assertEqual(status["metadata"]["current_pids"], [111])

    def test_status_returns_true_for_live_pid_set_difference(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["current_pids"], [111, 222])

    def test_open_with_rdwr_only_does_not_latch_filesystem(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(_event(syscall="openat", fd=3, fd_kind="regular", flags=os.O_RDWR, path="/tmp/ro.txt"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_mutating_open_on_real_path_survives_racy_fd_kind(self) -> None:
        """`cat > file << 'EOF'` opens the real file then dup2/close/pipe-reuses
        the fd before the helper can stat /proc/<pid>/fd/<fd>. The helper then
        reports fd_kind=fifo (or socket) even though the syscall actually hit a
        regular file — the path in the BPF event is authoritative, so the
        mutating open must still latch filesystem_changed."""
        for racy_kind in ("fifo", "socket"):
            with self.subTest(racy_kind=racy_kind):
                resolver = FakeResolver()
                fs_monitor = FakeFilesystemMonitor()
                daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
                daemon.register("sbx-1", "docker", "container-1")

                with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
                    "crab.host_inspector.server.reset_soft_dirty_for_pids",
                    return_value={111},
                ):
                    daemon.reset("sbx-1")

                daemon._handle_fs_event(
                    _event(
                        syscall="openat",
                        pid=222,
                        fd=3,
                        fd_kind=racy_kind,
                        flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                        path="/testbed/django/db/backends/postgresql/client.py",
                        inode=4242,
                        device=64,
                    )
                )
                status = daemon.status("sbx-1")
                self.assertTrue(status["filesystem_changed"])

    def test_mutating_open_on_dev_null_still_dropped(self) -> None:
        """The racy-fd_kind bypass must not accept writes into pseudo
        filesystems. /dev/null is the common shell-redirection sink and must
        keep being ignored."""
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="char",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                path="/dev/null",
            )
        )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_write_to_non_regular_fd_does_not_latch_filesystem(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(_event(syscall="write", pid=999, fd=3, fd_kind="fifo", flags=0))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_unresolved_mutating_open_is_diagnostic_only(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                pid=999,
                fd=3,
                fd_kind="unknown",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
            )
        )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])
        self.assertTrue(status["metadata"]["unreconciled_fs_events"])

    def test_create_then_delete_same_file_clears_net_effect(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="regular",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                path="/work/tmp.txt",
                inode=101,
                device=1,
            )
        )
        daemon._handle_fs_event(_event(syscall="unlinkat", pid=222, path="/work/tmp.txt"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])
        self.assertEqual(status["metadata"]["live_dirty_entries"], [])

    def test_touch_file_latches_filesystem_changed(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="regular",
                flags=os.O_CREAT | os.O_WRONLY,
                path="/work/1.txt",
                inode=201,
                device=1,
            )
        )
        with patch("crab.host_inspector.server.os.lstat"):
            status = daemon.status("sbx-1")
        self.assertTrue(status["filesystem_changed"])
        self.assertEqual(status["metadata"]["live_dirty_entries"][0]["path"], "/work/1.txt")

    def test_mkdir_then_rmdir_same_dir_clears_net_effect(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(_event(syscall="mkdirat", pid=222, path="/work/demo", device=1, inode=301))
        daemon._handle_fs_event(_event(syscall="rmdir", pid=222, path="/work/demo"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_rename_new_file_then_delete_clears_net_effect(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="regular",
                flags=os.O_CREAT | os.O_WRONLY,
                path="/work/a.txt",
                inode=401,
                device=1,
            )
        )
        daemon._handle_fs_event(
            _event(
                syscall="renameat2",
                pid=222,
                path="/work/a.txt",
                path_secondary="/work/b.txt",
                inode=401,
                device=1,
            )
        )
        daemon._handle_fs_event(_event(syscall="unlink", pid=222, path="/work/b.txt"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_delete_preexisting_file_remains_sticky(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(_event(syscall="unlink", pid=222, path="/work/preexisting.txt"))
        status = daemon.status("sbx-1")
        self.assertTrue(status["filesystem_changed"])
        self.assertTrue(status["metadata"]["live_dirty_entries"][0]["deleted"])

    def test_created_file_missing_at_status_still_counts_as_change(self) -> None:
        # A file that was created after reset and then disappeared (e.g., a
        # temp file that was unlinked but whose delete event was lost) is still
        # evidence that the sandbox mutated its filesystem.  Kernel tracepoint
        # events are authoritative — we should not second-guess them with a
        # host-side lstat that may fail for multiple valid reasons.
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="unknown",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                path="/tmp/short-lived.tmp",
            )
        )
        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["filesystem_changed"])
        self.assertEqual(len(status["metadata"]["live_dirty_entries"]), 1)

    def test_ignored_pid_does_not_contribute_to_process_changed(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            side_effect=[
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "sleep 1")),
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "sleep 1")),
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "sleep 1")),
            ],
        ), patch("crab.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={222},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")
            status = daemon.status("sbx-1")

        self.assertFalse(status["process_changed"])
        self.assertEqual(status["metadata"]["tracked_pids"], [222])
        self.assertEqual(status["metadata"]["ignored_pids"], [111])
        self.assertEqual(status["metadata"]["current_pids"], [222])

    def test_ignored_pid_filesystem_event_is_dropped(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=ProcessIdentity(
                pid=111,
                executable_path="/opt/iflow-runtime/node/bin/node",
                executable_basename="node",
                cmdline=("node", "iflow"),
            ),
        ), patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value=set(),
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=ProcessIdentity(
                pid=111,
                executable_path="/opt/iflow-runtime/node/bin/node",
                executable_basename="node",
                cmdline=("node", "iflow"),
            ),
        ):
            daemon._handle_fs_event(
                _event(
                    syscall="openat",
                    pid=111,
                    fd=3,
                    fd_kind="regular",
                    flags=577,
                    path="/root/.iflow/history.json",
                    inode=501,
                    device=1,
                )
            )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_non_ignored_child_process_still_counts(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            side_effect=[
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "echo hi")),
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "echo hi")),
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "echo hi")),
            ],
        ), patch("crab.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={222},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value={222},
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["dirty_pids"], [222])

    def test_process_only_scope_keeps_fs_events_but_drops_process_changed(self) -> None:
        """scope=process_only excludes a pid from process tracking (no
        contribution to tracked_pids / dirty_pids / process_changed) but
        leaves its fs events flowing — both the eBPF kernel filter and
        `_handle_fs_event` keep delivering them.

        This mechanic is what the terminus tmux integration relies on
        for its pane-bash rule (basename="bash" AND ancestor="tmux"):
        the pane shell should not trip `process_changed` every turn from
        heap-dirty noise, but its file writes are real signal. This test
        exercises the scope=process_only mechanic generically against an
        ancestor-only rule; the production terminus rule narrows the
        match to bash so long-running non-bash tmux descendants (e.g.
        background daemons) still contribute to `process_changed`.
        """
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [
            {"ancestor_executable_basename": "tmux", "scope": "process_only"},
        ]

        # PID 333 has tmux as an ancestor (matches the rule); PID 444 has
        # only init (PID 1) as an ancestor (no match).
        identity_333 = ProcessIdentity(pid=333, executable_path="/bin/cat", executable_basename="cat", cmdline=("cat",))
        identity_444 = ProcessIdentity(pid=444, executable_path="/bin/sleep", executable_basename="sleep", cmdline=("sleep", "infinity"))
        ancestors_333 = frozenset({"tmux", "bash"})
        ancestors_444 = frozenset()

        def fake_identity(pid):
            return {333: identity_333, 444: identity_444}.get(pid)

        def fake_ancestors(pid, *, max_depth=32):
            return {333: ancestors_333, 444: ancestors_444}.get(pid, frozenset())

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            side_effect=fake_identity,
        ), patch(
            "crab.host_inspector.process_filter.read_ancestor_basenames",
            side_effect=fake_ancestors,
        ), patch("crab.host_inspector.server.list_cgroup_pids", return_value={333, 444}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={444},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            # Even if PID 333 were dirty, it must not surface — but mock
            # confirms only tracked pids are queried.
            return_value=set(),
        ):
            daemon.register("sbx-tmux", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-tmux")
            status = daemon.status("sbx-tmux")
            metadata = status["metadata"]

        # tmux-descendant pid is in `ignored_pids` but NOT in `fs_ignored_pids`,
        # so the eBPF kernel filter never receives it and fs events flow through.
        self.assertEqual(metadata["tracked_pids"], [444])
        self.assertEqual(metadata["ignored_pids"], [333])
        self.assertEqual(metadata["fs_ignored_pids"], [])
        # And process_changed stays False: tracked == baseline, dirty empty.
        self.assertFalse(status["process_changed"])
        # The fs-event handler must NOT drop events from the process-only pid.
        # Verified directly via pid_matches_ignore_rules with fs_only=True:
        from crab.host_inspector.process_filter import (
            pid_matches_ignore_rules,
            parse_process_ignore_rules,
        )
        parsed = parse_process_ignore_rules(rules)
        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            side_effect=fake_identity,
        ), patch(
            "crab.host_inspector.process_filter.read_ancestor_basenames",
            side_effect=fake_ancestors,
        ):
            self.assertTrue(pid_matches_ignore_rules(333, parsed))            # tracked-side: matched
            self.assertFalse(pid_matches_ignore_rules(333, parsed, fs_only=True))  # fs-side: not matched

    def test_scope_all_rule_matches_fs_only_path(self) -> None:
        """A scope=all rule (the default, used today for `sleep` and
        `tmux`) MUST still match in the fs-event path so the existing
        ignore-everywhere semantics are preserved.
        """
        from crab.host_inspector.process_filter import (
            pid_matches_ignore_rules,
            parse_process_ignore_rules,
        )
        rules = parse_process_ignore_rules([{"executable_basename": "sleep"}])
        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=ProcessIdentity(pid=555, executable_path="/bin/sleep", executable_basename="sleep", cmdline=("sleep", "infinity")),
        ):
            self.assertTrue(pid_matches_ignore_rules(555, rules))
            self.assertTrue(pid_matches_ignore_rules(555, rules, fs_only=True))

    def test_mmap_invalidation_scans_ignored_init_pid_under_tmux_rules(self) -> None:
        """fault-7 regression: under the terminus tmux rules at idle, the
        only cgroup pids are `sleep` (scope=all), the `tmux` server
        (scope=all), and the pane `bash` (scope=process_only via
        basename=bash + ancestor=tmux). All three land in `ignored_pids`
        and `tracked_pids` is empty.

        When apt-get install rewrites libc.so.6, the eBPF link event must
        still fire mmap_invalidation because the long-lived `sleep
        infinity` init pid mmap'd that libc — even though sleep is in the
        ignored set. Limiting the scan to tracked_pids meant the signal
        was lost and the next checkpoint stayed filesystem-only, leaving
        the only process image (taken at start, with old build-ID)
        unrecoverable on restore.
        """
        resolver = FakeResolver()  # init_pid=111
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "sleep"}]
        identity = ProcessIdentity(
            pid=111,
            executable_path="/usr/bin/sleep",
            executable_basename="sleep",
            cmdline=("sleep", "infinity"),
        )
        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value=set(),
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")

        # Sanity: with the rule active, the entire cgroup is in
        # ignored_pids and tracked_pids is empty — exactly fault-7.
        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            pre_status = daemon.status("sbx-1")
        self.assertEqual(pre_status["metadata"]["tracked_pids"], [])
        self.assertEqual(pre_status["metadata"]["ignored_pids"], [111])
        self.assertFalse(pre_status["process_changed"])

        # apt-get install link-renames the new libc into place. The eBPF
        # event lands on path_secondary; sleep (pid 111, ignored) has
        # libc.so.6 mmap'd.
        with patch(
            "crab.host_inspector.server.path_invalidates_mmap",
            return_value="/usr/lib/x86_64-linux-gnu/libc.so.6",
        ):
            daemon._handle_fs_event(
                _event(
                    syscall="link",
                    pid=999,
                    path="/usr/lib/x86_64-linux-gnu/libc.so.6.dpkg-new",
                    path_secondary="/usr/lib/x86_64-linux-gnu/libc.so.6",
                )
            )

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        # process_changed must be True so the scheduler promotes the next
        # checkpoint to a full one.
        self.assertTrue(status["process_changed"])

    def test_status_kernel_truth_fallback_includes_ignored_and_init_pids(self) -> None:
        """The status() fallback must scan tracked + ignored + init_pid.

        With tmux rules active, the eBPF link event for libc may be
        dropped under load (perf ringbuf overflow), so the kernel-truth
        scan of /proc/<pid>/maps is the last line of defense. fault-7
        proved that limiting it to tracked_pids leaves zero pids to scan
        when tmux suppresses everything.
        """
        resolver = FakeResolver()  # init_pid=111
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "sleep"}]
        identity = ProcessIdentity(
            pid=111, executable_path="/usr/bin/sleep", executable_basename="sleep", cmdline=("sleep", "infinity"),
        )
        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value=set(),
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths",
            return_value={"/usr/lib/x86_64-linux-gnu/libc.so.6"},
        ) as scan:
            status = daemon.status("sbx-1")

        scanned_pids = scan.call_args.args[0]
        self.assertIn(111, set(scanned_pids))  # init_pid is part of the scan
        self.assertTrue(status["process_changed"])

    def test_full_checkpoint_reset_baselines_acknowledged_deleted_mmaps(self) -> None:
        """After a full process checkpoint, reset(captures_process=True)
        snapshots the current (deleted) mmap set. CRIU stored that file
        content inline in the dump image, so subsequent restores from
        that image won't break — these paths must not re-fire
        mmap_invalidation."""
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths",
            return_value={"/usr/lib/x86_64-linux-gnu/libc.so.6"},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1", captures_process=True)

        with daemon._records_lock:
            record = daemon._records["sbx-1"]
        self.assertEqual(
            record.acknowledged_deleted_mmaps,
            frozenset({"/usr/lib/x86_64-linux-gnu/libc.so.6"}),
        )

        # Re-firing on the same path is suppressed: a subsequent fs
        # event for an already-acknowledged path must NOT set
        # mmap_invalidated again.
        with patch(
            "crab.host_inspector.server.path_invalidates_mmap",
            return_value="/usr/lib/x86_64-linux-gnu/libc.so.6",
        ):
            daemon._handle_fs_event(
                _event(
                    syscall="link",
                    pid=999,
                    path="/usr/lib/x86_64-linux-gnu/libc.so.6.dpkg-new",
                    path_secondary="/usr/lib/x86_64-linux-gnu/libc.so.6",
                )
            )
        with daemon._records_lock:
            self.assertFalse(daemon._records["sbx-1"].mmap_invalidated)

        # And status() with the same kernel-truth scan does not re-fire either.
        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths",
            return_value={"/usr/lib/x86_64-linux-gnu/libc.so.6"},
        ):
            status = daemon.status("sbx-1")
        self.assertFalse(status["process_changed"])

    def test_fs_only_reset_preserves_acknowledged_baseline(self) -> None:
        """An fs-only checkpoint does NOT refresh the process image, so
        its `acknowledged_deleted_mmaps` baseline must stay intact —
        otherwise a later full checkpoint's frozen content set would be
        forgotten and we'd repeat the latch-forever bug."""
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths",
            return_value={"/usr/lib/x86_64-linux-gnu/libc.so.6"},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1", captures_process=True)

        # Now an fs-only reset (e.g. after an fs-only checkpoint).
        # all_deleted_mmap_paths is intentionally NOT patched here — the
        # daemon must not call it for fs-only resets.
        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1", captures_process=False)

        with daemon._records_lock:
            self.assertEqual(
                daemon._records["sbx-1"].acknowledged_deleted_mmaps,
                frozenset({"/usr/lib/x86_64-linux-gnu/libc.so.6"}),
            )

    def test_new_deleted_path_after_full_checkpoint_still_fires(self) -> None:
        """A second, distinct rewrite (different file) after the
        baseline must still fire mmap_invalidation — the suppression is
        per-path, not blanket."""
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)

        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths",
            return_value={"/usr/lib/x86_64-linux-gnu/libc.so.6"},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1", captures_process=True)

        # status() finds a NEW deleted path that wasn't in the baseline.
        with patch("crab.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value=set(),
        ), patch(
            "crab.host_inspector.server.all_deleted_mmap_paths",
            return_value={
                "/usr/lib/x86_64-linux-gnu/libc.so.6",
                "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            },
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])

    def test_pid_identity_cache_amortizes_proc_reads_across_burst(self) -> None:
        """The fs-event hot path uses PidIdentityCache so a burst of
        events from the same pid does not re-read /proc each time. Without
        this, every event triggered 2 syscalls in `read_process_identity`
        plus an ancestor walk for ancestor-scoped rules — the dominant
        cost behind the `fs_monitor.sync(timeout_s=5.0)` barrier missing
        its budget on tmux pane workloads.
        """
        from crab.host_inspector.process_filter import (
            PidIdentityCache,
            classify_pid_against_rules,
            parse_process_ignore_rules,
        )

        identity = ProcessIdentity(
            pid=42,
            executable_path="/usr/bin/sleep",
            executable_basename="sleep",
            cmdline=("sleep", "infinity"),
        )
        rules = parse_process_ignore_rules([{"executable_basename": "sleep"}])
        cache = PidIdentityCache(max_entries=8, ttl_s=60.0)

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ) as identity_mock:
            for _ in range(100):
                matched_any, matched_fs = classify_pid_against_rules(42, rules, cache=cache)
                self.assertTrue(matched_any)
                self.assertTrue(matched_fs)
        # Without cache: 100 reads. With cache: 1 read amortized across the burst.
        self.assertEqual(identity_mock.call_count, 1)

    def test_pid_identity_cache_invalidates_on_ttl_expiry(self) -> None:
        """Cache entries expire on TTL so PID reuse cannot misclassify
        events from the new occupant after the window passes. PID reuse
        within the TTL is essentially impossible on a normal Linux system
        (4M PID space) — the small TTL trades that vanishing failure mode
        against the per-event syscall amortization.
        """
        from crab.host_inspector.process_filter import (
            PidIdentityCache,
            classify_pid_against_rules,
            parse_process_ignore_rules,
        )

        identity = ProcessIdentity(
            pid=42,
            executable_path="/usr/bin/sleep",
            executable_basename="sleep",
            cmdline=("sleep", "infinity"),
        )
        rules = parse_process_ignore_rules([{"executable_basename": "sleep"}])
        cache = PidIdentityCache(max_entries=8, ttl_s=0.0)

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ) as identity_mock:
            for _ in range(3):
                classify_pid_against_rules(42, rules, cache=cache)
        # ttl_s=0 → every lookup is a cache miss → re-reads.
        self.assertEqual(identity_mock.call_count, 3)

    def test_pid_identity_cache_satisfies_ancestor_rules_only_when_recorded(self) -> None:
        """Two-shot caching: an entry computed without ancestors is
        reusable for ancestor-less rule sets but a miss for ancestor-
        needing ones. This avoids re-reading proc when both kinds of
        rules coexist, while still guaranteeing ancestor data when a
        rule actually needs it.
        """
        from crab.host_inspector.process_filter import (
            PidIdentityCache,
            classify_pid_against_rules,
            parse_process_ignore_rules,
        )

        identity = ProcessIdentity(
            pid=42,
            executable_path="/bin/cat",
            executable_basename="cat",
            cmdline=("cat",),
        )
        cache = PidIdentityCache(max_entries=8, ttl_s=60.0)
        ancestor_rules = parse_process_ignore_rules(
            [{"ancestor_executable_basename": "tmux", "scope": "process_only"}]
        )
        plain_rules = parse_process_ignore_rules([{"executable_basename": "sleep"}])

        with patch(
            "crab.host_inspector.process_filter.read_process_identity",
            return_value=identity,
        ) as identity_mock, patch(
            "crab.host_inspector.process_filter.read_ancestor_basenames",
            return_value=frozenset({"tmux", "bash"}),
        ) as ancestors_mock:
            classify_pid_against_rules(42, plain_rules, cache=cache)
            classify_pid_against_rules(42, ancestor_rules, cache=cache)
            classify_pid_against_rules(42, ancestor_rules, cache=cache)

        # First call (no ancestor needed) reads identity once.
        # Second call needs ancestors → identity cached but ancestors not, so
        # both are re-read for this entry.
        # Third call has the ancestor-bearing entry in cache → no syscalls.
        self.assertEqual(identity_mock.call_count, 2)
        self.assertEqual(ancestors_mock.call_count, 1)

    def test_unresolved_proc_identity_defaults_to_tracked(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch("crab.host_inspector.process_filter.read_process_identity", return_value=None), patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ), patch(
            "crab.host_inspector.server.dirty_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["tracked_pids"], [111])
        self.assertEqual(status["metadata"]["ignored_pids"], [])


class LibbpfMonitorIgnoreRuleWireTests(unittest.TestCase):
    def test_set_ignore_process_rules_filters_to_scope_all_on_wire(self) -> None:
        """The C helper has no concept of scope=process_only — those
        rules suppress process_changed in Python but MUST keep
        delivering fs events. Confirm the wire encoding drops them
        before they hit the helper, even though the daemon hands the
        full rule set down."""
        from crab.host_inspector.process_filter import (
            ProcessIgnoreRule,
            SCOPE_ALL,
            SCOPE_PROCESS_ONLY,
        )

        monitor = LibbpfFilesystemMonitor(helper_path="/bin/true")
        sent: list[dict[str, object]] = []

        def fake_send(payload: dict[str, object]) -> None:
            sent.append(payload)

        with patch.object(monitor, "_send", side_effect=fake_send):
            monitor.set_ignore_process_rules(
                "sbx-1",
                (
                    ProcessIgnoreRule(executable_basename="tmux", cmdline_contains=("server-1", "session-x"), scope=SCOPE_ALL),
                    ProcessIgnoreRule(executable_basename="bash", scope=SCOPE_PROCESS_ONLY),
                    ProcessIgnoreRule(ancestor_executable_basename="tmux", scope=SCOPE_ALL),
                ),
            )

        # 1 clear + 2 add (process_only rule filtered out before send).
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[0]["op"], "clear_ignore_process_rules")
        self.assertEqual(sent[1]["op"], "add_ignore_process_rule")
        self.assertEqual(sent[1]["executable_basename"], "tmux")
        # cmdline_contains list joined on '|' for the helper's parser.
        self.assertEqual(sent[1]["cmdline_contains"], "server-1|session-x")
        self.assertEqual(sent[2]["op"], "add_ignore_process_rule")
        self.assertEqual(sent[2]["ancestor_executable_basename"], "tmux")
        self.assertEqual(sent[2]["executable_basename"], "")


class MmapPathCacheTests(unittest.TestCase):
    def test_cache_avoids_redundant_proc_maps_reads_on_miss(self) -> None:
        """Most fs events MISS the mmap check (build outputs / temp
        files aren't mmap'd by anything). Pre-cache, every miss
        re-read /proc/<pid>/maps for every live pid in the cgroup —
        the dominant per-worker cost on burst-y workloads (benchmark
        20260429_063012 showed worker stalls with target_depth < 5K).
        With the cache, one read per pid covers an entire TTL window
        of misses."""
        from crab.host_inspector.process_monitor import MmapPathCache, path_invalidates_mmap

        call_count = [0]

        def fake_parse(pid: int) -> set[str]:
            call_count[0] += 1
            return {"/usr/lib/libc.so.6"}

        with patch("crab.host_inspector.process_monitor.parse_mapped_paths", side_effect=fake_parse):
            cache = MmapPathCache(ttl_s=10.0)
            # 50 events against 3 pids, NONE matching — pre-cache,
            # this would have done 150 /proc/<pid>/maps reads. With
            # the cache, three reads (one per pid) feed all 50 events.
            for _ in range(50):
                self.assertIsNone(
                    path_invalidates_mmap([100, 101, 102], "/build/output.o", cache=cache)
                )

        self.assertEqual(call_count[0], 3)

    def test_cache_expires_after_ttl(self) -> None:
        """Stale mappings shouldn't pin forever — TTL ensures we
        eventually re-read so dlopen/exec changes get picked up."""
        from crab.host_inspector.process_monitor import MmapPathCache

        call_count = [0]

        def fake_parse(pid: int) -> set[str]:
            call_count[0] += 1
            return {"/usr/lib/libc.so.6"}

        with patch("crab.host_inspector.process_monitor.parse_mapped_paths", side_effect=fake_parse):
            cache = MmapPathCache(ttl_s=0.01)
            cache.get(100)
            cache.get(100)
            self.assertEqual(call_count[0], 1)
            __import__("time").sleep(0.02)
            cache.get(100)
            self.assertEqual(call_count[0], 2)


class IgnoreRulePushTests(unittest.TestCase):
    def test_register_pushes_fs_eligible_rules_to_helper(self) -> None:
        """The daemon must hand `parsed_ignore_rules` to the C helper at
        register time so the helper can drop pid-matched events at
        source. Pre-port, the helper had no rule knowledge and every
        such event paid for IPC + Python decode + sandbox_lock acq
        before the rule check fired. The helper-side `hasattr` check
        in server.py keeps backward-compat for fs_monitor stubs that
        don't implement the method."""
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules_in = [
            {"executable_basename": "sleep", "scope": "all"},
            {"executable_basename": "tmux", "cmdline_contains": ["server-1"], "scope": "all"},
            # process_only rules are NOT pushed: they suppress
            # process_changed but must keep delivering fs events. The
            # helper-side filter must not drop those.
            {"executable_basename": "bash", "scope": "process_only"},
        ]
        with patch(
            "crab.host_inspector.server.list_cgroup_pids",
            return_value=[111],
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules_in)

        self.assertEqual(len(fs_monitor.ignore_rule_pushes), 1)
        sandbox_id, rules_pushed = fs_monitor.ignore_rule_pushes[0]
        self.assertEqual(sandbox_id, "sbx-1")
        # All three rules cross to the helper layer; the helper-side
        # set_ignore_process_rules method (in fs_helper.py) is what
        # actually filters to scope=all on the wire.
        self.assertEqual(len(rules_pushed), 3)
        self.assertEqual(rules_pushed[0].executable_basename, "sleep")
        self.assertEqual(rules_pushed[1].cmdline_contains, ("server-1",))
        self.assertEqual(rules_pushed[2].scope, "process_only")


class EventWorkerPoolBarrierTests(unittest.TestCase):
    def test_barrier_does_not_wait_on_peer_sandbox(self) -> None:
        """One queue per sandbox: a peer sandbox parked inside on_event
        cannot block a barrier issued for an unrelated sandbox. Under
        the prior hash-bucket pool, two sandboxes sharing a bucket
        could serialize each other; with per-sandbox queues there is
        no cross-sandbox serialization at all.
        """
        import threading

        slow_event = threading.Event()
        applied: list[str] = []

        def on_event(evt) -> None:
            if evt.sandbox_id == "noisy":
                # Hold the noisy worker until the test releases it.
                slow_event.wait(timeout=2.0)
            applied.append(evt.sandbox_id)

        pool = _EventWorkerPool(on_event)
        try:
            pool.register_sandbox("noisy")
            pool.register_sandbox("clean")
            pool.dispatch(_event(sandbox_id="noisy", syscall="write", path="/x"))
            # Wait until the noisy worker has pulled its event off the
            # queue and is parked inside on_event before issuing the
            # barrier (qsize() drops to 0 once q.get() returns).
            for _ in range(200):
                if pool._queues["noisy"].qsize() == 0:
                    break
                threading.Event().wait(0.005)

            waiter = threading.Event()
            target_depth, peer_max, peer_sum, peer_count = pool.barrier_for_sandbox(
                "clean", sync_id=42, waiter=waiter
            )
            self.assertEqual(target_depth, 0)
            self.assertGreaterEqual(peer_count, 1)
            # The fence sits on clean's queue, which is empty — the
            # clean worker processes the marker immediately. Pre-fix
            # (hash buckets), if noisy and clean shared a bucket, this
            # waiter would only fire when the noisy worker finished.
            self.assertTrue(waiter.wait(timeout=1.0))
            # Sanity: noisy worker is still blocked, hasn't applied yet.
            self.assertEqual(applied, [])

            slow_event.set()
        finally:
            slow_event.set()
            pool.stop()

    def test_barrier_waits_for_target_sandbox_drain(self) -> None:
        """The barrier MUST wait for the caller's own queue to drain —
        that's the correctness contract status() relies on."""
        import threading

        gate = threading.Event()
        applied: list[str] = []

        def on_event(evt) -> None:
            gate.wait(timeout=2.0)
            applied.append(evt.sandbox_id)

        pool = _EventWorkerPool(on_event)
        try:
            pool.register_sandbox("target")
            pool.dispatch(_event(sandbox_id="target", syscall="write", path="/x"))

            waiter = threading.Event()
            target_depth, _, _, _ = pool.barrier_for_sandbox(
                "target", sync_id=99, waiter=waiter
            )
            self.assertGreaterEqual(target_depth, 1)
            self.assertFalse(waiter.wait(timeout=0.05))
            gate.set()
            self.assertTrue(waiter.wait(timeout=1.0))
            self.assertEqual(applied, ["target"])
        finally:
            gate.set()
            pool.stop()

    def test_unregister_drains_queued_events_before_teardown(self) -> None:
        """A poison-pill sentinel in unregister_sandbox lets already-
        queued events drain before the worker exits. This guards
        against losing the last fs events on shutdown when a sandbox
        is removed mid-burst."""
        import threading

        applied: list[str] = []
        gate = threading.Event()

        def on_event(evt) -> None:
            gate.wait(timeout=2.0)
            applied.append(evt.sandbox_id)

        pool = _EventWorkerPool(on_event)
        try:
            pool.register_sandbox("sbx")
            pool.dispatch(_event(sandbox_id="sbx", syscall="write", path="/a"))
            pool.dispatch(_event(sandbox_id="sbx", syscall="write", path="/b"))
            gate.set()
            pool.unregister_sandbox("sbx")
            self.assertEqual(applied, ["sbx", "sbx"])
        finally:
            pool.stop()


if __name__ == "__main__":
    unittest.main()
