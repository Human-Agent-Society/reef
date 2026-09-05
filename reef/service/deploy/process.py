"""Service processes run inside an Executor worker, locally or on a remote node.

Only this boundary owns Popen, process groups and node-local readiness probes.
The deployment orchestrator exchanges serializable metadata and RPCs.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import yaml

from reef.service.deploy.config import interpolate_config

_DEFAULT_GRACE_TIMEOUT = 30
_PROCESS_REAP_TIMEOUT = 5


def _log(msg: str) -> None:
    print(f"[reef] {msg}", file=sys.stderr)


def _command_argv(config: Mapping[str, Any], command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(interpolate_config(config, command))
    return [interpolate_config(config, argument) for argument in command]


class ProcessWorker:
    """Own a set of node-local service process trees (one per deployment worker)."""

    def __init__(
        self,
        config: dict[str, Any],
        services: Sequence[dict[str, Any]],
        run_dir: Path,
        ready_timeout_default: int,
        config_path: str | Path,
    ) -> None:
        self.config = config
        self.services = services
        self.run_dir = run_dir.resolve()
        self.ready_timeout_default = ready_timeout_default
        self.config_path = Path(config_path).resolve()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        # Every POSIX service starts a fresh session, so its process-group ID
        # is the leader PID. Keep that identity after the leader exits:
        # launchers such as Ray and SGLang can leave workers alive after their
        # direct child has gone, and shutdown must still reach those workers.
        self._process_groups: dict[str, int] = {}
        self._log_fps: dict[str, TextIO] = {}
        self._names = [svc["name"] for svc in services]
        self._stopping = threading.Event()

    def _pid_file(self, name: str) -> Path:
        return self.run_dir / f"{name}.pid"

    def _log_file(self, name: str) -> Path:
        return self.run_dir / f"{name}.log"

    def _is_alive(self, name: str) -> bool:
        proc = self._procs.get(name)
        return proc is not None and proc.poll() is None

    def _service_env(self, svc: Mapping[str, Any]) -> dict[str, str]:
        """The environment a service and its ready probe run under.

        The bridge driver's readiness marker lives under ``run_dir`` with the
        stack's pid and log files, so two stacks on one machine never share
        one; the driver reads the path from ``REEF_BRIDGE_READY_FILE``.
        """
        env = dict(os.environ)
        env["REEF_CONFIG"] = str(self.config_path)
        env["REEF_BRIDGE_READY_FILE"] = str(self.run_dir / "bridge.ready")
        for k, v in (svc.get("env") or {}).items():
            env[k] = interpolate_config(self.config, str(v))
        cuda = svc.get("cuda")
        if cuda is not None:
            env["CUDA_VISIBLE_DEVICES"] = interpolate_config(self.config, str(cuda))
        return env

    def _start_service(self, svc: Mapping[str, Any]) -> None:
        name = svc["name"]
        if self._is_alive(name):
            _log(f"{name}: already running (pid {self._procs[name].pid})")
            return
        command = _command_argv(self.config, svc["command"])
        env = self._service_env(svc)
        _log(f"starting {name}: {shlex.join(command)}")
        # The handle outlives this scope (closed in stop()); a context manager
        # would close it under the running service.
        log_fp = open(self._log_file(name), "a")  # noqa: SIM115
        self._log_fps[name] = log_fp
        # Logs stay on the worker node and are tailed through Executor RPC.
        proc = self._spawn(
            command,
            env=env,
            cwd=interpolate_config(self.config, svc["cwd"]) if svc.get("cwd") else None,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        self._procs[name] = proc
        if os.name == "posix":
            self._process_groups[name] = proc.pid
        self._pid_file(name).write_text(str(proc.pid))

    def _spawn(self, command: Sequence[str], **options: Any) -> subprocess.Popen[bytes]:
        return subprocess.Popen(command, **options)

    def start(self) -> None:
        for svc in self.services:
            self._start_service(svc)
        _log(f"stack up. logs: {self.run_dir}/*.log")

    def _safe_process_group(self, name: str, proc: subprocess.Popen[bytes]) -> int | None:
        """Return the spawn-time PGID only when it is safe to signal."""
        pgid = self._process_groups.get(name)
        if os.name != "posix" or pgid is None or pgid <= 1 or pgid != proc.pid or pgid == os.getpgrp():
            return None

        # A process group may outlive its leader, but the numeric PID can be
        # reused once that group disappears.  If the leader has exited and a
        # process with its PID is present again, the stored PGID is stale and
        # must never be signalled.  ``getpgid`` raising is the expected state
        # while descendants keep the original leaderless group alive.
        leader_exited = proc.poll() is not None
        try:
            current_pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            if not leader_exited and proc.poll() is None:
                return None
        except PermissionError:
            return None
        else:
            if leader_exited or current_pgid != pgid:
                return None
        return pgid

    def _process_tree_alive(self, name: str, proc: subprocess.Popen[bytes]) -> bool:
        """Whether a managed leader or one of its process-group children lives."""
        # Reap an exited direct child before probing the group; otherwise its
        # zombie can make killpg(..., 0) report a group that has already gone.
        proc.poll()
        pgid = self._safe_process_group(name, proc)
        if pgid is None:
            return proc.returncode is None
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return proc.returncode is None
        except PermissionError:
            # The group still exists even if the current process unexpectedly
            # lost permission to signal it.
            return True
        return True

    def _signal_process_tree(self, name: str, proc: subprocess.Popen[bytes], signum: int) -> None:
        """Signal one service's whole tree, with a direct-child test fallback."""
        pgid = self._safe_process_group(name, proc)
        if pgid is not None:
            try:
                os.killpg(pgid, signum)
                return
            except ProcessLookupError:
                pass
            except PermissionError:
                _log(f"could not signal {name} process group {pgid}; falling back to leader {proc.pid}")
        if proc.poll() is None:
            if signum == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()

    def shutdown(self, grace: float = _DEFAULT_GRACE_TIMEOUT) -> None:
        """Kill service process groups in reverse order: SIGTERM → wait → SIGKILL."""
        self._stopping.set()
        deadline = time.monotonic() + max(0, grace)
        targets = []
        for name in reversed(self._names):
            proc = self._procs.get(name)
            if proc is None or not self._process_tree_alive(name, proc):
                continue
            pgid = self._safe_process_group(name, proc)
            identity = f"process group {pgid}" if pgid is not None else f"pid {proc.pid}"
            _log(f"stopping {name} ({identity})")
            self._signal_process_tree(name, proc, signal.SIGTERM)
            targets.append((name, proc))

        # Every group gets the full grace window concurrently. Waiting for
        # leaders one by one would both miss descendants and let an early
        # service consume the entire shared deadline.
        pending = targets
        while pending and time.monotonic() < deadline:
            pending = [(name, proc) for name, proc in pending if self._process_tree_alive(name, proc)]
            if pending:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))

        pending = [(name, proc) for name, proc in pending if self._process_tree_alive(name, proc)]
        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        for name, proc in pending:
            _log(f"{name}: process tree did not exit in {grace}s, SIGKILL")
            self._signal_process_tree(name, proc, kill_signal)

        reap_deadline = time.monotonic() + _PROCESS_REAP_TIMEOUT
        for name in reversed(self._names):
            proc = self._procs.get(name)
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0, reap_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    _log(f"{name}: leader pid {proc.pid} could not be reaped after SIGKILL")
            self._pid_file(name).unlink(missing_ok=True)
        for fp in self._log_fps.values():
            with contextlib.suppress(Exception):
                fp.close()

    def describe(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "pids": {name: proc.pid for name, proc in self._procs.items()},
            "run_dir": str(self.run_dir),
        }

    @property
    def host(self) -> str:
        return "127.0.0.1"

    def prepare(self, config: dict[str, Any]) -> dict[str, Any]:
        """Snapshot resolved endpoints on the execution node, before spawning."""
        self.config = config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.run_dir / "runtime.yaml"
        fd = os.open(self.config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        # Never accept a stale bridge-ready marker from a previous launch.
        (self.run_dir / "bridge.ready").unlink(missing_ok=True)
        return self.describe()

    def probe(self, name: str, timeout: float = 5.0) -> bool:
        service = next(svc for svc in self.services if svc["name"] == name)
        if not self._is_alive(name):
            raise RuntimeError(f"service {name!r} exited before ready")
        ready = service.get("ready")
        if not ready:
            return True
        with subprocess.Popen(
            interpolate_config(self.config, ready),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self._service_env(service),
            cwd=interpolate_config(self.config, service["cwd"]) if service.get("cwd") else None,
            start_new_session=os.name == "posix",
        ) as probe:
            try:
                return probe.wait(timeout=timeout) == 0 and self._is_alive(name)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(probe.pid, signal.SIGKILL)
                else:
                    probe.kill()
                probe.wait()
                return False

    def status(self) -> dict[str, Any]:
        return {name: proc.poll() for name, proc in self._procs.items()}

    def request_stop(self, force: bool = False) -> None:
        self._stopping.set()
        signum = getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM
        for name in reversed(self._names):
            proc = self._procs.get(name)
            if proc is not None and self._process_tree_alive(name, proc):
                self._signal_process_tree(name, proc, signum)

    def tree_alive(self) -> bool:
        return any(self._process_tree_alive(name, proc) for name, proc in self._procs.items())

    def read_log(self, name: str, offset: int = 0) -> tuple[str, int]:
        path = self._log_file(name)
        if not path.exists():
            return "", offset
        with path.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(65536)
            return content.decode("utf-8", errors="replace"), handle.tell()


class RayProcessWorker(ProcessWorker):
    """Ray assigns the node and CUDA visibility; child processes inherit both."""

    def __init__(
        self,
        config: dict[str, Any],
        services: Sequence[dict[str, Any]],
        run_dir: Path,
        ready_timeout_default: int,
        config_path: str | Path,
    ) -> None:
        if os.name != "posix":
            raise ValueError("Ray service process supervision requires POSIX")
        self._leases: list[int] = []
        node_dir = Path(tempfile.mkdtemp(prefix="reef-service-"))
        super().__init__(config, services, node_dir, ready_timeout_default, node_dir / "runtime.yaml")

    def _spawn(self, command: Sequence[str], **options: Any) -> subprocess.Popen[bytes]:
        read_fd, write_fd = os.pipe()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "reef.service.deploy.guard", str(read_fd), *command],
                pass_fds=(read_fd,),
                **options,
            )
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
        self._leases.append(write_fd)
        return proc

    def shutdown(self, grace: float = _DEFAULT_GRACE_TIMEOUT) -> None:
        try:
            super().shutdown(grace=grace)
        finally:
            for fd in self._leases:
                with contextlib.suppress(OSError):
                    os.close(fd)
            self._leases.clear()

    @property
    def host(self) -> str:
        from ray.util import get_node_ip_address

        return get_node_ip_address()
