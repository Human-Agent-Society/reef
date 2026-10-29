"""Standard-library process ownership shared by executors and background jobs.

Scopes own direct children and signal their POSIX process groups. They cannot
reclaim detached descendants, containers, or remote sandboxes owned by a program.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, RLock
from types import MappingProxyType
from typing import Literal


class ProcessSetup(ABC):
    """A child-side setup operation with no input or result, used only by local process adapters.

    Implementations apply inherited resource limits before exec, or raise to
    fail the spawn. Instances are passed to Popen only at this boundary.
    """

    @abstractmethod
    def apply(self) -> None: ...


@dataclass(frozen=True)
class ProcessRequest:
    """One local launch, made by an executor or command adapter and read by ProcessScope.run.

    argv is a nonempty tuple of NUL-free strings with a nonempty executable.
    cwd is an absolute Path. env is a copied, read-only string mapping whose
    nonempty keys contain neither '=' nor NUL, and whose values contain no NUL.
    timeout_seconds is None (no deadline) or finite and positive; grace_seconds
    is finite and nonnegative. stdin_text is text or None for closed stdin.
    capture_stdout selects capture versus discard; stderr is always captured.
    new_session selects process-group versus direct-child signals. pass_fds is
    a tuple of distinct nonnegative integer file descriptors borrowed until spawn.
    setup is None or a ProcessSetup applied in the child before exec.
    Paths, executable availability and open descriptors are checked by the OS at spawn.
    """

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    timeout_seconds: float | None
    grace_seconds: float = 0.0
    stdin_text: str | None = None
    capture_stdout: bool = True
    new_session: bool = True
    pass_fds: tuple[int, ...] = ()
    setup: ProcessSetup | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or not all(isinstance(arg, str) and "\0" not in arg for arg in self.argv)
            or not self.argv[0]
        ):
            raise ValueError("argv must contain a nonempty executable and NUL-free arguments")
        if not isinstance(self.cwd, Path) or not self.cwd.is_absolute():
            raise ValueError("cwd must be an absolute Path")
        if not isinstance(self.env, Mapping) or any(
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or not isinstance(value, str)
            or "\0" in value
            for key, value in self.env.items()
        ):
            raise ValueError("env must contain valid environment names and string values")
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        for name, value in (("timeout_seconds", self.timeout_seconds), ("grace_seconds", self.grace_seconds)):
            if name == "timeout_seconds" and value is None:
                continue
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or (name == "timeout_seconds" and value == 0)
            ):
                raise ValueError(
                    f"{name} must be finite and {'positive' if name == 'timeout_seconds' else 'nonnegative'}"
                )
        if self.stdin_text is not None and not isinstance(self.stdin_text, str):
            raise ValueError("stdin_text must be text or None")
        if not isinstance(self.capture_stdout, bool) or not isinstance(self.new_session, bool):
            raise ValueError("output and session flags must be booleans")
        if (
            not isinstance(self.pass_fds, tuple)
            or any(type(fd) is not int or fd < 0 for fd in self.pass_fds)
            or len(set(self.pass_fds)) != len(self.pass_fds)
        ):
            raise ValueError("pass_fds must contain distinct nonnegative integers")
        if self.setup is not None and not isinstance(self.setup, ProcessSetup):
            raise ValueError("setup must implement ProcessSetup")


@dataclass(frozen=True)
class ProcessResult:
    """A reaped child's exit status, captured output and first accepted termination cause.

    reason is exited, timed_out, or stopped. exit_code is an integer (negative
    for a POSIX signal); stdout and stderr are text, empty when discarded.
    A stop or timeout remains the reason even if the child subsequently exits 0.
    ProcessScope.run makes these values; command adapters interpret them.
    """

    reason: Literal["exited", "timed_out", "stopped"]
    exit_code: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if self.reason not in ("exited", "timed_out", "stopped") or type(self.exit_code) is not int:
            raise ValueError("invalid process result")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("process output must be text")


class ProcessScopeClosed(RuntimeError):
    """A launch was refused because its scope has received a stop request."""


class RunningProcess:
    """One child owned by a scope, with its immutable request and mutable stop state.

    process is a Popen returned for request. reason starts as exited; the first
    termination request sets it to stopped or timed_out and sets kill_at to a
    monotonic deadline. Later requests do not change either. The scope's
    condition protects these fields. kill_sent records whether KILL was already
    sent, so a reaped group is never signalled twice. Only ProcessScope creates and reads this state.
    """

    def __init__(self, process: subprocess.Popen[str], request: ProcessRequest) -> None:
        if not isinstance(process, subprocess.Popen) or not isinstance(request, ProcessRequest):
            raise ValueError("a running process needs a child and its request")
        self.process = process
        self.request = request
        self.reason: Literal["exited", "timed_out", "stopped"] = "exited"
        self.kill_at: float | None = None
        self.kill_sent = False

    def signal(self, signum: int) -> None:
        """Signal the group or child; an already removed target needs no signal."""
        if signum == signal.SIGKILL and self.kill_sent:
            return
        with contextlib.suppress(ProcessLookupError):
            if self.request.new_session:
                os.killpg(self.process.pid, signum)
            else:
                self.process.send_signal(signum)
        if signum == signal.SIGKILL:
            self.kill_sent = True

    def request_stop(self, reason: Literal["timed_out", "stopped"], grace_seconds: float) -> None:
        """Accept only the first stop cause and send TERM, or KILL for zero grace."""
        if self.kill_at is None:
            self.reason = reason
            self.kill_at = time.monotonic() + grace_seconds
            self.signal(signal.SIGTERM if grace_seconds else signal.SIGKILL)


SCOPE_TOPOLOGY_LOCK = RLock()


class ProcessScope:
    """A thread-safe owner of local executions; initially open and empty.

    live maps each unique child PID to its RunningProcess until run has reaped
    it and finished handling its output. closed means new launches are refused,
    not that existing children have finished. run and request_stop serialize
    spawn/registration against closure. wait_closed returns True only after a
    stop request and an empty live set. Repeated stop requests are idempotent.
    children contains scopes attached for this job's local executions; attachment
    transfers shutdown responsibility to this scope. Only unattached scopes may
    be attached, and a scope cannot attach itself or an ancestor. It never owns
    a remote resource. Attached scopes remain owned until this scope is discarded.
    """

    def __init__(self) -> None:
        self.condition = Condition()
        self.live: dict[int, RunningProcess] = {}
        self.closed = False
        self.children: set[ProcessScope] = set()
        self.parent: ProcessScope | None = None

    def attach(self, child: ProcessScope) -> None:
        """Own an unattached child scope; repeated attachment to this parent is idempotent.

        Refuse cycles and other parents. A closing parent stops the child and
        rejects attachment, so no new execution escapes a concurrent shutdown.
        """
        with SCOPE_TOPOLOGY_LOCK, self.condition:
            ancestor: ProcessScope | None = self
            while ancestor is not None:
                if ancestor is child:
                    raise ValueError("process scopes cannot form a cycle")
                ancestor = ancestor.parent
            if child.parent is not None and child.parent is not self:
                raise ValueError("a process scope already has an owner")
            if self.closed:
                child.request_stop()
                raise ProcessScopeClosed("the job process scope is stopping")
            child.parent = self
            self.children.add(child)

    def pids(self) -> tuple[int, ...]:
        """Return the current direct-child PIDs in launch order."""
        with self.condition:
            return tuple(self.live)

    def request_stop(self, *, grace_seconds: float | None = None) -> tuple[int, ...]:
        """Refuse launches and stop current children using their grace, or a nonnegative override."""
        if grace_seconds is not None and (
            isinstance(grace_seconds, bool)
            or not isinstance(grace_seconds, int | float)
            or not math.isfinite(grace_seconds)
            or grace_seconds < 0
        ):
            raise ValueError("grace_seconds must be finite and nonnegative")
        with self.condition:
            self.closed = True
            for child in self.live.values():
                grace = child.request.grace_seconds if grace_seconds is None else grace_seconds
                child.request_stop("stopped", grace)
            for child_scope in self.children:
                child_scope.request_stop(grace_seconds=grace_seconds)
            self.condition.notify_all()
            return tuple(self.live)

    def wait_closed(self, timeout_seconds: float | None = None) -> bool:
        """Wait for closure and reaping; None waits without a deadline, otherwise return False on timeout."""
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be finite and nonnegative")
        with self.condition:
            deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
            if not self.condition.wait_for(lambda: self.closed and not self.live, timeout=timeout_seconds):
                return False
            children = tuple(self.children)
        for child in children:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not child.wait_closed(remaining):
                return False
        return True

    def run(self, request: ProcessRequest) -> ProcessResult:
        """Launch and reap one request. OS launch errors propagate; stopped scopes reject before spawning.

        Timeout and stop return distinct results after reaping. An exception in
        the calling thread forces child cleanup before it propagates.
        """
        with self.condition:
            if self.closed:
                raise ProcessScopeClosed("the process scope is stopping; no new process")
            process = subprocess.Popen(
                request.argv,
                cwd=request.cwd,
                env=dict(request.env),
                stdin=subprocess.DEVNULL if request.stdin_text is None else subprocess.PIPE,
                stdout=subprocess.PIPE if request.capture_stdout else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=request.new_session,
                pass_fds=request.pass_fds,
                preexec_fn=None if request.setup is None else request.setup.apply,
            )
            child = RunningProcess(process, request)
            self.live[process.pid] = child
        deadline = None if request.timeout_seconds is None else time.monotonic() + request.timeout_seconds
        stdin_text = request.stdin_text
        try:
            while True:
                with self.condition:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        child.request_stop("timed_out", request.grace_seconds)
                    if child.kill_at is not None and now >= child.kill_at:
                        child.signal(signal.SIGKILL)
                    next_deadline = child.kill_at if child.kill_at is not None else deadline
                    interval = 0.05 if next_deadline is None else max(0.001, min(0.05, next_deadline - now))
                try:
                    stdout, stderr = process.communicate(input=stdin_text, timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    stdin_text = None
            with self.condition:
                return ProcessResult(child.reason, process.returncode, stdout or "", stderr or "")
        except BaseException:
            child.signal(signal.SIGKILL)
            process.wait()
            raise
        finally:
            # Popen's context manager closes the pipes after communicate or an exception.
            with process:
                pass
            with self.condition:
                self.live.pop(process.pid)
                self.condition.notify_all()


CURRENT_PROCESS_SCOPE: ContextVar[ProcessScope | None] = ContextVar("job_process_scope", default=None)
