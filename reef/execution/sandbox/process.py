"""How a run's process runs, behind an interface.

``run_task`` owns the executor-agnostic work: it prepares the run root,
writes the rendered composition, builds the environment, reads the trajectory,
and collects residue. Only the launch of the harness process differs between a
development run and a hosted one, so that step is the seam:

* :class:`LocalExecutor` runs the binary as a plain subprocess in its own
  session, exactly as before. It is the default, used for development and the
  hermetic tests.
* :class:`SandboxExecutor` runs the same binary in a sandbox that the host's
  kernel enforces: a read-only base filesystem with only the run root exposed
  (its workspace and declared runtime state writable, rendered inputs
  read-only), only explicitly forwarded deployment environment, resource
  limits, and network disabled unless an egress allowlist is configured.
  :class:`BwrapExecutor` is the bubblewrap jail on Linux; ``SeatbeltExecutor``
  in :mod:`reef.execution.sandbox.seatbelt` is the Seatbelt profile on macOS.

A deployment selects the executor by config; a hosted deployment that requires
the sandbox refuses to start when the sandbox runtime is unavailable, through
:meth:`ProgramExecutor.preflight`.
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Self

from reef.core.errors import InternalError, ReefError
from reef.execution.processes import (
    CURRENT_PROCESS_SCOPE,
    ProcessRequest,
    ProcessScope,
    ProcessScopeClosedError,
    ProcessSetup,
)

ISOLATION_ENV = "REEF_SANDBOX_ISOLATION"

RUN_OWNER_LEASE: ContextVar[bool] = ContextVar("run_owner_lease", default=False)


class ProcessOwner:
    """An environment's local process scope, stopped by ExecutorEnvironment.

    scope starts open and empty. stop refuses future launches and requests
    immediate termination of every owned process; each launch reaps its child.
    """

    def __init__(self) -> None:
        self.scope = ProcessScope()

    def stop(self) -> None:
        self.scope.request_stop()


PROCESS_OWNER: ContextVar[ProcessOwner | None] = ContextVar("sandbox_process_owner", default=None)


class SandboxUnavailableError(ReefError):
    """A deployment requires the sandbox executor but it cannot isolate."""


@dataclass(frozen=True)
class ProcessOutcome:
    """What a launched run process produced."""

    exit_code: int
    stdout: str
    stderr: str


class ProgramTimeoutError(InternalError):
    """The process exceeded its wall-clock budget; raised for run_task to map."""


class ProgramLaunchError(InternalError):
    """The process could not be launched; raised for run_task to map."""


class SandboxProgram(ABC):
    """The program an executor runs, and what an executor needs to resolve or install it.

    A subclass sets every attribute in ``__init__``. An executor reads only
    what its kind needs: one on the Reef host resolves :meth:`host_binary`,
    and a remote one resolves ``binary`` in a template built for ``name`` at
    ``version``.
    """

    #: The program's name; it names a remote template built for the program.
    name: str
    #: The binary name a remote image resolves on its ``PATH``.
    binary: str
    #: The program starts its own containers on the Reef host.
    self_isolating: bool
    #: ``<package>@<version>`` when a remote template can install the program with npm, else ``""``.
    npm_package: str
    #: The pinned version a remote template is built for, else ``""``.
    version: str

    @abstractmethod
    def host_binary(self) -> str:
        """The binary on the Reef host, installed there when it is missing."""

    @abstractmethod
    def host_install_dir(self) -> Path | None:
        """The directory a Reef-installed binary needs bound into a jail, else ``None``."""


class ProgramExecutor(ABC):
    """Synchronous launch interface adapted by ``ExecutorEnvironment``.

    ``run_task`` prepares ``root`` (with ``workspace`` inside it) and the
    ``env`` overlay; an executor decides how the process sees them. It must
    kill the whole process tree on timeout and raise :class:`ProgramTimeoutError`
    or :class:`ProgramLaunchError` so ``run_task`` maps them to the public
    ``TaskRunError`` uniformly.

    Callers do not need to know which executor they hold. :meth:`for_program`
    binds the executor to the program it runs and resolves that program's
    binary where the process runs. :meth:`open_session` gives one run of
    several processes access to ports on the Reef host; :meth:`push`,
    :meth:`pull` and :meth:`close` serve that run.
    """

    #: The process runs directly on the Reef host: it sees the host's paths, environment and container runtime.
    runs_on_host: bool = False

    @abstractmethod
    def preflight(self) -> None:
        """Raise if this executor cannot establish its isolation on this host."""

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[ProgramExecutor, str]:
        """This executor bound to run ``program``, and the binary its processes launch.

        ``binary`` overrides the resolved binary. By default the program's binary is resolved on the Reef host and
        installed there when it is missing.
        """
        return self, binary if binary is not None else program.host_binary()

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> ProgramExecutor:
        """An executor for one run of several processes, which lasts at most ``timeout_s``.

        The processes reach the Reef host's loopback ``forward_ports`` in addition to this executor's own. Call
        :meth:`close` on the result when the run ends; a result that is this executor needs no close.
        """
        return self

    def push(self, root: Path) -> None:
        """Copy the Reef host's ``root`` to where the session's processes read it.

        An executor whose processes read the host's files directly has nothing to copy.
        """
        return

    def pull(self, root: Path, relative: str = "") -> None:
        """Refresh the Reef host's copy of ``root / relative`` while a process still runs.

        An executor whose processes write the host's files directly has nothing to copy.
        """
        return

    def close(self) -> None:
        """Release what :meth:`open_session` started."""
        return

    @abstractmethod
    def launch(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        timeout: float,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> ProcessOutcome: ...


def run_process(
    popen_argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float | None,
    limits: SandboxLimits | None = None,
    stdin_text: str | None = None,
    new_session: bool = True,
) -> ProcessOutcome:
    """Start a process in its own session and wait, killing the whole group on
    timeout. Coding agents spawn tool children, so a timeout kills the session,
    not just the direct child. Nested tool jails can retain their caller's
    process group with ``new_session=False``."""
    read_fd = write_fd = None
    command = list(popen_argv)
    if RUN_OWNER_LEASE.get() and new_session:
        # Detect missing executables before the guard changes their exit status.
        executable = command[0]
        if "/" in executable and not os.path.isabs(executable):
            executable = str(cwd / executable)
        if shutil.which(executable, path=env.get("PATH", "")) is None:
            raise ProgramLaunchError(f"harness binary {command[0]!r} not found or not executable")
        read_fd, write_fd = os.pipe()
        command = [sys.executable, str(Path(__file__).with_name("lease.py")), str(read_fd), *command]
    owner = PROCESS_OWNER.get() if new_session else None
    scope = owner.scope if owner is not None else ProcessScope()
    job_scope = CURRENT_PROCESS_SCOPE.get()
    try:
        if job_scope is not None:
            job_scope.attach(scope)
        result = scope.run(
            ProcessRequest(
                argv=tuple(command),
                cwd=cwd.absolute(),
                env=env,
                timeout_seconds=timeout,
                stdin_text=stdin_text,
                new_session=new_session,
                pass_fds=() if read_fd is None else (read_fd,),
                setup=limits,
            )
        )
    except FileNotFoundError as exc:
        raise ProgramLaunchError(f"harness binary {popen_argv[0]!r} not found") from exc
    except ProcessScopeClosedError as exc:
        raise ProgramLaunchError("the execution environment was stopped") from exc
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
    if result.reason == "timed_out":
        raise ProgramTimeoutError(f"run timed out after {timeout}s: {list(popen_argv)}")
    if result.reason == "stopped":
        raise ProgramLaunchError("the execution environment was stopped")
    return ProcessOutcome(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)


def inherited_environment() -> dict[str, str]:
    """The minimal parent environment a run keeps: how to find binaries.

    The running interpreter's scripts directory comes last on PATH, so a
    runner that ships with reef (reef-native) is found when
    ``reef serve`` runs from a venv that is not on PATH; every name PATH
    already finds resolves as before."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "SYSTEMROOT", "TMPDIR", "CUDA_VISIBLE_DEVICES")
    }
    scripts = str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    if scripts not in path.split(os.pathsep):
        env["PATH"] = os.pathsep.join([path, scripts]) if path else scripts
    return env


@dataclass(frozen=True)
class LocalExecutor(ProgramExecutor):
    """Run the binary as a plain subprocess with the caller's privileges.

    A minimal environment and separate process group support cleanup.
    Filesystem and network access are not restricted.
    """

    runs_on_host = True

    def preflight(self) -> None:
        return None

    def launch(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        timeout: float,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> ProcessOutcome:
        return run_process(argv, cwd=workspace, env={**inherited_environment(), **env}, timeout=timeout)


#: Resource limits every sandboxed run gets unless the config overrides one.
@dataclass(frozen=True)
class SandboxLimits(ProcessSetup):
    """Nonnegative integer resource limits applied to a child before exec.

    cpu_seconds, memory_bytes, processes and file_bytes bound CPU time, address
    space, process count and file size respectively. Zero leaves that OS limit
    unchanged. The sandbox adapter constructs this value; ProcessScope calls
    apply in the child, setting each nonzero soft and hard limit to its value.
    OS errors fail the spawn rather than silently dropping a limit.
    """

    cpu_seconds: int = 0  # 0 disables; else RLIMIT_CPU
    memory_bytes: int = 0  # 0 disables; else RLIMIT_AS
    processes: int = 0  # 0 disables; else RLIMIT_NPROC
    file_bytes: int = 0  # 0 disables; else RLIMIT_FSIZE

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.cpu_seconds, self.memory_bytes, self.processes, self.file_bytes)
        ):
            raise ValueError("resource limits must be nonnegative integers")

    def apply(self) -> None:
        """Apply configured limits before the child executes its command."""
        for kind, value in (
            (resource.RLIMIT_CPU, self.cpu_seconds),
            (resource.RLIMIT_AS, self.memory_bytes),
            (resource.RLIMIT_NPROC, self.processes),
            (resource.RLIMIT_FSIZE, self.file_bytes),
        ):
            if value:
                resource.setrlimit(kind, (value, value))


def parse_env_from(section: Mapping[str, object], environ: Mapping[str, str]) -> dict[str, str]:
    """``sandbox.env_from``: the named host variables and their values; each must be set and non-empty."""
    names = section.get("env_from", [])
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in names
    ):
        raise SandboxUnavailableError("sandbox.env_from must list environment variable names")
    env = {}
    for name in names:
        if not environ.get(name):
            raise SandboxUnavailableError(f"sandbox.env_from requires environment variable {name!r}")
        env[name] = environ[name]
    return env


@dataclass(frozen=True)
class SandboxExecutor(ProgramExecutor):
    """Run the binary on the Reef host inside a sandbox that the host's kernel enforces.

    Values: ``egress_hosts`` are the endpoints a run must reach; empty, the ``default`` network allows no network
    access, and nonempty, it opens the network. ``network`` is ``default`` or ``isolated``: ``isolated`` opens the
    internet and allows no host loopback port except ``forward_ports``, which are distinct TCP ports. ``limits`` bound
    each run's resources. ``base_paths`` are the host directories a run may read so its binary and runtime resolve.
    ``env`` is the deployment's explicit environment for every run; a run inherits no other host variable.

    Each subclass enforces these values with one host mechanism, which its ``kind`` names, and implements
    :meth:`preflight` and :meth:`launch`.
    Purpose: :mod:`reef.execution.sandbox.config` builds one with :meth:`from_config` from a deployment's sandbox
    section, as it builds a ``RemoteExecutor``; a caller binds it to a program with :meth:`for_program`.
    """

    #: The host mechanism's name, as a deployment selects it.
    kind: ClassVar[str] = ""

    egress_hosts: tuple[str, ...] = ()
    network: str = "default"
    forward_ports: tuple[int, ...] = ()
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    base_paths: tuple[str, ...] = ()
    #: Explicit deployment credentials/settings; never inherited from the candidate tree.
    env: Mapping[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, section: Mapping[str, object], environ: Mapping[str, str]) -> Self:
        """The sandbox that ``section``'s ``egress_hosts``, ``limits`` and ``env_from`` describe.

        ``env_from`` names variables of ``environ``. Raise ``ReefError`` for a malformed option.
        """
        egress_hosts = section.get("egress_hosts") or ()
        if not isinstance(egress_hosts, (list, tuple)):
            raise ReefError("evolution.sandbox.egress_hosts must be a list")
        limits = section.get("limits") or {}
        if not isinstance(limits, Mapping):
            raise ReefError("evolution.sandbox.limits must be a mapping")
        return cls(
            egress_hosts=tuple(str(host) for host in egress_hosts),
            limits=SandboxLimits(
                cpu_seconds=int(limits.get("cpu_seconds", 0)),
                memory_bytes=int(limits.get("memory_bytes", 0)),
                processes=int(limits.get("processes", 0)),
                file_bytes=int(limits.get("file_bytes", 0)),
            ),
            env=parse_env_from(section, environ),
        )

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[ProgramExecutor, str]:
        """Let a run read the whole install of a binary Reef installed: npm launchers are symlinks into packages,
        and git installs need both their editable source and their venv."""
        if binary is not None:
            return self, binary
        resolved = program.host_binary()
        install_dir = program.host_install_dir()
        if install_dir is None:
            return self, resolved
        return replace(self, base_paths=(*self.base_paths, str(install_dir))), resolved

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> ProgramExecutor:
        """An isolated network adds the run's ports; the default network already reaches the host."""
        ports = tuple(dict.fromkeys((*self.forward_ports, *forward_ports)))
        if self.network != "isolated" or ports == self.forward_ports:
            return self
        return replace(self, forward_ports=ports)
