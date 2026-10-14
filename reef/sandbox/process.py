"""How an episode's process runs, behind an interface.

``run_episode`` owns the executor-agnostic work: it prepares the episode root,
writes the rendered composition, builds the environment, reads the trajectory,
and collects residue. Only the launch of the harness process differs between a
development run and a hosted one, so that step is the seam:

* :class:`LocalExecutor` runs the binary as a plain subprocess in its own
  session, exactly as before. It is the default, used for development and the
  hermetic tests.
* :class:`SandboxExecutor` runs the same binary inside a bubblewrap jail: a
  fresh non-root namespace, a read-only base filesystem with only the episode
  root exposed (its workspace and declared runtime state writable, rendered
  inputs read-only), only explicitly forwarded deployment environment, resource limits,
  network disabled unless an egress allowlist is configured, and death with
  the parent.

A deployment selects the executor by config; a hosted deployment that requires
the sandbox refuses to start when the sandbox runtime is unavailable, through
:meth:`EpisodeExecutor.preflight`.
"""

from __future__ import annotations

import contextlib
import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path

from reef.core.errors import ReefError

ISOLATION_ENV = "REEF_EPISODE_ISOLATION"

EPISODE_OWNER_LEASE: ContextVar[bool] = ContextVar("episode_owner_lease", default=False)


class ProcessOwner:
    """Terminate the local process groups belonging to one environment."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process_groups: set[int] = set()
        self.stopped = False

    def register(self, process_group: int) -> None:
        with self.lock:
            if self.stopped:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGKILL)
                raise EpisodeLaunchError("the execution environment was stopped")
            self.process_groups.add(process_group)

    def release(self, process_group: int) -> None:
        with self.lock:
            self.process_groups.discard(process_group)

    def stop(self) -> None:
        with self.lock:
            self.stopped = True
            for process_group in self.process_groups:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGKILL)


PROCESS_OWNER: ContextVar[ProcessOwner | None] = ContextVar("sandbox_process_owner", default=None)


class SandboxUnavailable(ReefError):
    """A deployment requires the sandbox executor but it cannot isolate."""


@dataclass(frozen=True)
class ProcessOutcome:
    """What a launched episode process produced."""

    exit_code: int
    stdout: str
    stderr: str


class EpisodeTimeout(Exception):
    """The process exceeded its wall-clock budget; raised for run_episode to map."""


class EpisodeLaunchError(Exception):
    """The process could not be launched; raised for run_episode to map."""


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


class EpisodeExecutor(ABC):
    """Synchronous launch interface adapted by ``ExecutorEnvironment``.

    ``run_episode`` prepares ``root`` (with ``workspace`` inside it) and the
    ``env`` overlay; an executor decides how the process sees them. It must
    kill the whole process tree on timeout and raise :class:`EpisodeTimeout`
    or :class:`EpisodeLaunchError` so ``run_episode`` maps them to the public
    ``EpisodeError`` uniformly.

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

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[EpisodeExecutor, str]:
        """This executor bound to run ``program``, and the binary its processes launch.

        ``binary`` overrides the resolved binary. By default the program's binary is resolved on the Reef host and
        installed there when it is missing.
        """
        return self, binary if binary is not None else program.host_binary()

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> EpisodeExecutor:
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
    if EPISODE_OWNER_LEASE.get() and new_session:
        # Detect missing executables before the guard changes their exit status.
        executable = command[0]
        if "/" in executable and not os.path.isabs(executable):
            executable = str(cwd / executable)
        if shutil.which(executable, path=env.get("PATH", "")) is None:
            raise EpisodeLaunchError(f"harness binary {command[0]!r} not found or not executable")
        read_fd, write_fd = os.pipe()
        command = [sys.executable, str(Path(__file__).with_name("lease.py")), str(read_fd), *command]
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL if stdin_text is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=new_session,
            preexec_fn=limits.apply if limits is not None else None,
            pass_fds=() if read_fd is None else (read_fd,),
        )
    except FileNotFoundError as exc:
        if write_fd is not None:
            os.close(write_fd)
        raise EpisodeLaunchError(f"harness binary {popen_argv[0]!r} not found") from exc
    except BaseException:
        if write_fd is not None:
            os.close(write_fd)
        raise
    finally:
        if read_fd is not None:
            os.close(read_fd)
    owner = PROCESS_OWNER.get() if new_session else None
    try:
        if owner is not None:
            owner.register(process.pid)
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError):
            if new_session:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        process.wait()
        raise EpisodeTimeout(f"episode timed out after {timeout}s: {list(popen_argv)}") from exc
    except BaseException:
        # A local worker can be terminated while an episode is in flight.
        # Do not leave the harness's separate process group behind.
        with contextlib.suppress(ProcessLookupError):
            if new_session:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        process.wait()
        raise
    finally:
        if owner is not None:
            owner.release(process.pid)
        if write_fd is not None:
            os.close(write_fd)
    return ProcessOutcome(exit_code=process.returncode, stdout=stdout, stderr=stderr)


def inherited_environment() -> dict[str, str]:
    """The minimal parent environment an episode keeps: how to find binaries.

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
class LocalExecutor(EpisodeExecutor):
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


#: What every jail unshares and mounts before its binds; the episode jail and the nesting probe share it.
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


#: Resource limits every sandboxed episode gets unless the config overrides one.
@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 0  # 0 disables; else RLIMIT_CPU
    memory_bytes: int = 0  # 0 disables; else RLIMIT_AS
    processes: int = 0  # 0 disables; else RLIMIT_NPROC
    file_bytes: int = 0  # 0 disables; else RLIMIT_FSIZE

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
class SandboxExecutor(EpisodeExecutor):
    """Run the binary inside a bubblewrap jail.

    bubblewrap (``bwrap``) is an unprivileged, daemonless sandbox: it builds a
    new mount/pid/user namespace, binds a read-only base filesystem, exposes
    only the episode root (workspace writable), clears the environment, and
    dies with its parent. Network is unshared (no egress) unless
    ``egress_hosts`` lists the model endpoint the episode must reach, in which
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
        # The native loop nests one jail per tool call inside the episode's, so a host that refuses a user namespace
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

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[EpisodeExecutor, str]:
        """Bind the whole install of a binary Reef installed: npm launchers are symlinks into packages, and git
        installs need both their editable source and their venv."""
        if binary is not None:
            return self, binary
        resolved = program.host_binary()
        install_dir = program.host_install_dir()
        if install_dir is None:
            return self, resolved
        return replace(self, base_paths=(*self.base_paths, str(install_dir))), resolved

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> EpisodeExecutor:
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
        """An episode jail with a tool jail inside it around ``true``: the shape every sandboxed tool call takes."""
        bwrap = shutil.which("bwrap") or "bwrap"
        binds = [token for base in self.base_paths if Path(base).exists() for token in ("--ro-bind", base, base)]
        episode = [bwrap, *ISOLATION, "--unshare-net", *binds]
        tool = [bwrap, *TOOL_ISOLATION, "--unshare-net", *binds]
        return [*episode, "--", *tool, "--", "/bin/true"]

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


#: Nested tool jails bind the episode's proc filesystem instead of mounting another one.
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
