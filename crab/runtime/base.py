from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(ABC):
    @abstractmethod
    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        raise NotImplementedError


class SubprocessCommandRunner(CommandRunner):
    def __init__(self, timeout_seconds: float = 60.0):
        self._timeout_seconds = timeout_seconds

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        timeout = self._timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        detached = self._should_use_file_stdio(command)
        if detached:
            # Container init inherits these fds from runc. A tempfile here uses O_TMPFILE, so
            # the inherited fd points into the host's mount ns and CRIU can't dump it
            # ("Can't lookup mount=N for fd=1 path=/tmp/#<inode> (deleted)"). /dev/null resolves
            # identically across mount namespaces.
            completed = subprocess.run(
                command,
                cwd=None if cwd is None else str(cwd),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
            )
            stdout = ""
            stderr = ""
        else:
            completed = subprocess.run(
                command,
                cwd=None if cwd is None else str(cwd),
                check=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            stdout = "" if completed.stdout is None else completed.stdout
            stderr = "" if completed.stderr is None else completed.stderr
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _should_use_file_stdio(self, command: list[str]) -> bool:
        if "-d" in command:
            return True
        if not command:
            return False
        executable = Path(command[0]).name
        return executable == "runc" and any(verb in command for verb in ("create", "start"))
