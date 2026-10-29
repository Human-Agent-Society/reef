"""How a rollout's process runs, behind an interface.

``run_rollout`` owns the executor-agnostic work: it prepares the rollout root,
writes the rendered composition, builds the environment, reads the trajectory,
and collects residue. Only the launch of the harness process differs between a
development run and a hosted one, so that step is the seam:

* :class:`LocalExecutor` runs the binary as a plain subprocess in its own
  session, exactly as before. It is the default, used for development and the
  hermetic tests.
* :class:`SandboxExecutor` runs the same binary inside a bubblewrap jail: a
  fresh non-root namespace, a read-only base filesystem with only the rollout
  root exposed (its workspace and declared runtime state writable, rendered
  inputs read-only), only explicitly forwarded deployment environment, resource limits,
  network disabled unless an egress allowlist is configured, and death with
  the parent.

A deployment selects the executor by config; a hosted deployment that requires
the sandbox refuses to start when the sandbox runtime is unavailable, through
:meth:`ProgramExecutor.preflight`.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path

from reef.core.errors import ReefError
from reef.execution.processes import (
    CURRENT_PROCESS_SCOPE,
    ProcessRequest,
    ProcessScope,
    ProcessScopeClosed,
    ProcessSetup,
)

ISOLATION_ENV = "REEF_SANDBOX_ISOLATION"

ROLLOUT_OWNER_LEASE: ContextVar[bool] = ContextVar("rollout_owner_lease", default=False)


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


class SandboxUnavailable(ReefError):
    """A deployment requires the sandbox executor but it cannot isolate."""


@dataclass(frozen=True)
class ProcessOutcome:
    """What a launched rollout process produced."""

    exit_code: int
    stdout: str
    stderr: str


class RolloutTimeout(Exception):
    """The process exceeded its wall-clock budget; raised for run_rollout to map."""


class RolloutLaunchError(Exception):
    """The process could not be launched; raised for run_rollout to map."""


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

    ``run_rollout`` prepares ``root`` (with ``workspace`` inside it) and the
    ``env`` overlay; an executor decides how the process sees them. It must
    kill the whole process tree on timeout and raise :class:`RolloutTimeout`
    or :class:`RolloutLaunchError` so ``run_rollout`` maps them to the public
    ``RolloutError`` uniformly.

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

    def set_secret_env(self, env: Mapping[str, str]) -> None:
        """Give each later process of this executor ``env`` in its environment, and never on a command line.

        ``env`` maps variable names to values, such as a credential. An executor that cannot keep a value off
        every command line it builds raises :class:`SandboxUnavailable`, which every executor does by default.
        """
        raise SandboxUnavailable(f"{type(self).__name__} cannot keep an environment value off its command lines")

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
    if ROLLOUT_OWNER_LEASE.get() and new_session:
        # Detect missing executables before the guard changes their exit status.
        executable = command[0]
        if "/" in executable and not os.path.isabs(executable):
            executable = str(cwd / executable)
        if shutil.which(executable, path=env.get("PATH", "")) is None:
            raise RolloutLaunchError(f"harness binary {command[0]!r} not found or not executable")
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
        raise RolloutLaunchError(f"harness binary {popen_argv[0]!r} not found") from exc
    except ProcessScopeClosed as exc:
        raise RolloutLaunchError("the execution environment was stopped") from exc
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
    if result.reason == "timed_out":
        raise RolloutTimeout(f"rollout timed out after {timeout}s: {list(popen_argv)}")
    if result.reason == "stopped":
        raise RolloutLaunchError("the execution environment was stopped")
    return ProcessOutcome(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)


def inherited_environment() -> dict[str, str]:
    """The minimal parent environment a rollout keeps: how to find binaries.

    The running interpreter's scripts directory comes last on PATH, so a
    runner that ships with reef (reef-terminus, reef-native) is found when
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


#: What every jail unshares and mounts before its binds; the rollout jail and the nesting probe share it.
ISOLATION: tuple[str, ...] = (
    "--unshare-user",
    "--unshare-ipc",
    "--unshare-pid",
    "--unshare-uts",
    "--unshare-cgroup-try",
    "--die-with-parent",
    "--new-session",
    "--proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
    "--clearenv",
)


#: Resource limits every sandboxed rollout gets unless the config overrides one.
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


@dataclass(frozen=True)
class SandboxExecutor(ProgramExecutor):
    """Run the binary inside a bubblewrap jail.

    bubblewrap (``bwrap``) is an unprivileged, daemonless sandbox: it builds a
    new mount/pid/user namespace, binds a read-only base filesystem, exposes
    only the rollout root (workspace writable), clears the environment, and
    dies with its parent. Network is unshared (no egress) unless
    ``egress_hosts`` lists the model endpoint the rollout must reach, in which
    case the network namespace is shared so the endpoint is reachable; a
    single-host firewall is the follow-up (issue #18). The jail also carries
    ``REEF_NATIVE_ENFORCE=bwrap``, so the native loop runs each tool call in
    a nested jail derived from the tool's declared capabilities; preflight
    proves the host can nest one.
    """

    egress_hosts: tuple[str, ...] = ()
    #: ``isolated`` runs the jail in a network namespace of its own that pasta connects to the internet: no
    #: host address is reachable (``--no-map-gw``) except the loopback ports in ``forward_ports``, which pasta
    #: forwards to the host's loopback. The default leaves the network to ``egress_hosts``.
    network: str = "default"
    forward_ports: tuple[int, ...] = ()
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    #: Base directories bound read-only so the binary and its runtime resolve.
    base_paths: tuple[str, ...] = ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ssl", "/opt")
    #: Explicit deployment credentials/settings; never inherited from the candidate tree.
    env: Mapping[str, str] = field(default_factory=dict, repr=False)

    def preflight(self) -> None:
        if shutil.which("bwrap") is None:
            raise SandboxUnavailable(
                "the sandbox executor requires bubblewrap (bwrap) on PATH; install it or set "
                "evolution.executor: local for development"
            )
        if self.network not in ("default", "isolated"):
            raise SandboxUnavailable(f"unknown sandbox network {self.network!r}; use 'default' or 'isolated'")
        if self.network == "isolated":
            self.preflight_pasta()
        # The native loop nests one jail per tool call inside the rollout's, so a host that refuses a user namespace
        # inside another fails here instead of ending every call in SANDBOX_FAILED and tying every pairing.
        try:
            done = subprocess.run(
                self.nested_probe_argv(),
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(
                f"the sandbox executor cannot nest a bubblewrap jail on this host: {exc}"
            ) from exc
        if done.returncode != 0:
            raise SandboxUnavailable(
                "the sandbox executor cannot nest a bubblewrap jail on this host, which every sandboxed native "
                f"tool call needs: {done.stderr.strip()[-600:]}"
            )

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[ProgramExecutor, str]:
        """Bind the whole install of a binary Reef installed: npm launchers are symlinks into packages, and git
        installs need both their editable source and their venv."""
        if binary is not None:
            return self, binary
        resolved = program.host_binary()
        install_dir = program.host_install_dir()
        if install_dir is None:
            return self, resolved
        return replace(self, base_paths=(*self.base_paths, str(install_dir))), resolved

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> ProgramExecutor:
        """An isolated network forwards the run's ports; the default network already reaches the host."""
        ports = tuple(dict.fromkeys((*self.forward_ports, *forward_ports)))
        if self.network != "isolated" or ports == self.forward_ports:
            return self
        return replace(self, forward_ports=ports)

    def preflight_pasta(self) -> None:
        """An isolated network is one pasta sets up: prove it can on this host, as this user.

        pasta cannot map users under root (it drops to nobody), and a container may deny it the namespaces it
        needs; either way the jail would never start, so the host is refused here with pasta's own reason.
        """
        if shutil.which("pasta") is None:
            raise SandboxUnavailable(
                "an isolated sandbox network requires pasta (the passt package) on PATH; install it, or run "
                "without isolation where that is allowed"
            )
        if os.geteuid() == 0:
            raise SandboxUnavailable(
                "pasta cannot map users when started as root; run the Reef service as a non-root user with user "
                "namespaces allowed to isolate the network"
            )
        probe = ["pasta", "--config-net", "--no-map-gw", "--quiet", "-t", "none", "-u", "none", "-T", "none"]
        try:
            done = subprocess.run(
                [*probe, "-U", "none", "--", "/bin/true"],
                env={"PATH": os.environ.get("PATH", "")},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"pasta cannot set up an isolated network on this host: {exc}") from exc
        if done.returncode != 0:
            raise SandboxUnavailable(
                "pasta cannot set up an isolated network on this host (run the Reef service as a non-root user "
                f"with user namespaces allowed): {(done.stderr or done.stdout).strip()[-600:]}"
            )

    def nested_probe_argv(self) -> list[str]:
        """A rollout jail with a tool jail inside it around ``true``: the shape every sandboxed tool call takes."""
        bwrap = shutil.which("bwrap") or "bwrap"
        binds = [token for base in self.base_paths if Path(base).exists() for token in ("--ro-bind", base, base)]
        rollout = [bwrap, *ISOLATION, "--unshare-net", *binds]
        tool = [bwrap, *TOOL_ISOLATION, "--unshare-net", *binds]
        return [*rollout, "--", *tool, "--", "/bin/true"]

    def bwrap_argv(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> list[str]:
        cmd = ["bwrap", *ISOLATION]
        isolated = self.network == "isolated"
        if isolated:
            # pasta made the network namespace; bwrap stays in it rather than unsharing a disconnected one.
            ports = ",".join(str(port) for port in self.forward_ports) or "none"
            # Every forward is explicit: pasta's default ("auto") would expose each port the host listens on.
            forwards = ["-t", "none", "-u", "none", "-T", ports, "-U", "none"]
            cmd = ["pasta", "--config-net", "--no-map-gw", "--quiet", *forwards, "--", *cmd]
        elif not self.egress_hosts:
            cmd.append("--unshare-net")
        for base in self.base_paths:
            if Path(base).exists():
                cmd += ["--ro-bind", base, base]
        if self.egress_hosts or isolated:
            for dns_path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
                if Path(dns_path).exists():
                    cmd += ["--ro-bind", dns_path, dns_path]
        # Mount the root read-only, then open only declared runtime-state
        # directories. Rendered inputs inside those directories are rebound
        # read-only so a run cannot mutate the admitted composition.
        cmd += ["--ro-bind", str(root), str(root)]
        for path in writable_paths:
            cmd += ["--bind", str(path), str(path)]
        for path in readonly_paths:
            if any(path == writable or writable in path.parents for writable in writable_paths):
                cmd += ["--ro-bind", str(path), str(path)]
        cmd += ["--bind", str(workspace), str(workspace)]
        cmd += ["--chdir", str(workspace)]
        for key, value in {
            **inherited_environment(),
            **env,
            **self.env,
            "REEF_NATIVE_ENFORCE": "bwrap",
            ISOLATION_ENV: "bwrap",
        }.items():
            cmd += ["--setenv", key, value]
        cmd += ["--", *argv]
        return cmd

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
        self.preflight()
        full = self.bwrap_argv(
            argv,
            root=root,
            workspace=workspace,
            env=env,
            writable_paths=writable_paths,
            readonly_paths=readonly_paths,
        )
        return run_process(
            full,
            cwd=workspace,
            env={"PATH": os.environ.get("PATH", "")},
            timeout=timeout,
            limits=self.limits,
        )


#: Nested tool jails bind the rollout's proc filesystem instead of mounting another one.
TOOL_ISOLATION: tuple[str, ...] = (
    "--unshare-user",
    "--unshare-ipc",
    "--unshare-pid",
    "--unshare-uts",
    "--unshare-cgroup-try",
    "--die-with-parent",
    "--new-session",
    "--ro-bind",
    "/proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
    "--clearenv",
)
