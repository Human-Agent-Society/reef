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
  inputs read-only), no host environment or credentials, resource limits,
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
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from reef.core.errors import ReefError
from reef.harness.runners.native.enforce import ISOLATION as TOOL_ISOLATION

EPISODE_OWNER_LEASE: ContextVar[bool] = ContextVar("episode_owner_lease", default=False)


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


class EpisodeExecutor(Protocol):
    """Launch one episode process and return its outcome.

    ``run_episode`` prepares ``root`` (with ``workspace`` inside it) and the
    ``env`` overlay; an executor decides how the process sees them. It must
    kill the whole process tree on timeout and raise :class:`EpisodeTimeout`
    or :class:`EpisodeLaunchError` so ``run_episode`` maps them to the public
    ``EpisodeError`` uniformly.
    """

    def preflight(self) -> None:
        """Raise if this executor cannot establish its isolation on this host."""

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


def _run(popen_argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float) -> ProcessOutcome:
    """Start a process in its own session and wait, killing the whole group on
    timeout. Coding agents spawn tool children, so a timeout kills the session,
    not just the direct child."""
    read_fd = write_fd = None
    command = list(popen_argv)
    if EPISODE_OWNER_LEASE.get():
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
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
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
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise EpisodeTimeout(f"episode timed out after {timeout}s: {list(popen_argv)}") from exc
    except BaseException:
        # A local worker can be terminated while an episode is in flight.
        # Do not leave the harness's separate process group behind.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    finally:
        if write_fd is not None:
            os.close(write_fd)
    return ProcessOutcome(exit_code=process.returncode, stdout=stdout, stderr=stderr)


def _inherited_env() -> dict[str, str]:
    """The minimal parent environment an episode keeps: how to find binaries."""
    return {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "SYSTEMROOT", "TMPDIR", "CUDA_VISIBLE_DEVICES")
    }


@dataclass(frozen=True)
class LocalExecutor:
    """Run the binary as a plain subprocess, as the engine always has.

    No isolation beyond the episode root, the minimal environment, and the
    process group. Suitable for development and hermetic tests, not for a
    hosted service that evaluates untrusted proposals.
    """

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
        return _run(argv, cwd=workspace, env={**_inherited_env(), **env}, timeout=timeout)


#: What every jail unshares and mounts before its binds; the episode jail and the nesting probe share it.
_ISOLATION: tuple[str, ...] = (
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


@dataclass(frozen=True)
class SandboxExecutor:
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
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    #: Base directories bound read-only so the binary and its runtime resolve.
    base_paths: tuple[str, ...] = ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ssl", "/opt")

    def preflight(self) -> None:
        if shutil.which("bwrap") is None:
            raise SandboxUnavailable(
                "the sandbox executor requires bubblewrap (bwrap) on PATH; install it or set "
                "evolution.executor: local for development"
            )
        # The native loop nests one jail per tool call inside the episode's, so a host that refuses a user namespace
        # inside another fails here instead of ending every call in SANDBOX_FAILED and tying every pairing.
        try:
            done = subprocess.run(
                self._nested_probe_argv(),
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

    def _nested_probe_argv(self) -> list[str]:
        """An episode jail with a tool jail inside it around ``true``: the shape every sandboxed tool call takes."""
        bwrap = shutil.which("bwrap") or "bwrap"
        binds = [token for base in self.base_paths if Path(base).exists() for token in ("--ro-bind", base, base)]
        episode = [bwrap, *_ISOLATION, "--unshare-net", *binds]
        tool = [bwrap, *TOOL_ISOLATION, "--unshare-net", *binds]
        return [*episode, "--", *tool, "--", "/bin/true"]

    def _bwrap_argv(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> list[str]:
        cmd = ["bwrap", *_ISOLATION]
        if not self.egress_hosts:
            cmd.append("--unshare-net")
        for base in self.base_paths:
            if Path(base).exists():
                cmd += ["--ro-bind", base, base]
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
        for key, value in {**_inherited_env(), **env, "REEF_NATIVE_ENFORCE": "bwrap"}.items():
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
        full = self._bwrap_argv(
            argv,
            root=root,
            workspace=workspace,
            env=env,
            writable_paths=writable_paths,
            readonly_paths=readonly_paths,
        )
        # bwrap clears the environment inside the jail; the outer process only
        # needs PATH to find bwrap itself. Resource limits are set on the child
        # before exec, so they apply through bwrap to the jailed process tree.
        try:
            process = subprocess.Popen(
                full,
                env={"PATH": os.environ.get("PATH", "")},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=self._apply_limits,
            )
        except FileNotFoundError as exc:
            raise EpisodeLaunchError("the sandbox runtime (bwrap) is not available") from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise EpisodeTimeout(f"episode timed out after {timeout}s: {list(argv)}") from exc
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        return ProcessOutcome(exit_code=process.returncode, stdout=stdout, stderr=stderr)

    def _apply_limits(self) -> None:
        if self.limits.cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (self.limits.cpu_seconds, self.limits.cpu_seconds))
        if self.limits.memory_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (self.limits.memory_bytes, self.limits.memory_bytes))
        if self.limits.processes:
            resource.setrlimit(resource.RLIMIT_NPROC, (self.limits.processes, self.limits.processes))
        if self.limits.file_bytes:
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.limits.file_bytes, self.limits.file_bytes))


def build_executor(config: Mapping[str, object] | None) -> EpisodeExecutor:
    """Build the executor a deployment selected.

    ``executor`` is ``local`` (default) or ``sandbox``; a sandbox reads its
    ``egress_hosts`` and ``limits`` from the same section. The executor is
    preflighted at build so a hosted deployment that requires the sandbox
    fails to start, not at the first episode.
    """
    config = config or {}
    kind = config.get("executor", "local")
    if kind == "local":
        return LocalExecutor()
    if kind != "sandbox":
        raise ReefError(f"unknown episode executor {kind!r}; use 'local' or 'sandbox'")
    sandbox_config = config.get("sandbox") or {}
    if not isinstance(sandbox_config, Mapping):
        raise ReefError("evolution.sandbox must be a mapping")
    egress = tuple(str(host) for host in (sandbox_config.get("egress_hosts") or ()))
    limits_config = sandbox_config.get("limits") or {}
    if not isinstance(limits_config, Mapping):
        raise ReefError("evolution.sandbox.limits must be a mapping")
    limits = SandboxLimits(
        cpu_seconds=int(limits_config.get("cpu_seconds", 0)),
        memory_bytes=int(limits_config.get("memory_bytes", 0)),
        processes=int(limits_config.get("processes", 0)),
        file_bytes=int(limits_config.get("file_bytes", 0)),
    )
    executor = SandboxExecutor(egress_hosts=egress, limits=limits)
    executor.preflight()
    return executor
