"""Process orchestrator behind ``reef serve``.

Starts every service a config declares in dependency order, probes readiness,
mirrors child output, watches for unexpected exits, and tears the stack down
in reverse order on signal. Assembly of the Reef HTTP application itself
lives in :mod:`reef.service.deploy.settings` and :mod:`reef.service.assembly`.
"""

from __future__ import annotations

import contextlib
import copy
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
from types import FrameType
from typing import Any, TextIO

import yaml

from reef.service.deploy.config import (
    PROJECT_ROOT,
    config_value,
    interpolate_config,
    load_config,
    resolve_model_paths,
    validate_services,
)
from reef.service.deploy.settings import build_parser

_DEFAULT_GRACE_TIMEOUT = 30
_PROCESS_REAP_TIMEOUT = 5
_WATCHDOG_INTERVAL = 5


def _log(msg: str) -> None:
    print(f"[reef] {msg}", file=sys.stderr)


def _command_argv(config: Mapping[str, Any], command: str | Sequence[str]) -> list[str]:
    """Materialize a service command without changing its executable semantics."""
    if isinstance(command, str):
        return shlex.split(interpolate_config(config, command))
    return [interpolate_config(config, argument) for argument in command]


class InvalidOverrideError(ValueError):
    """A leftover ``reef serve`` argument is not a valid ``--key`` override."""


def _parse_overrides(extras: list[str]) -> dict[str, str]:
    """Parse leftover ``--key value`` / ``--key=value`` pairs from ``parse_known_args``.

    Anything that is not a well-formed ``--key`` override — an orphan
    positional, an unknown short option, or a ``--`` with no name — is
    rejected instead of silently discarded, so a typo fails fast at the CLI
    rather than surfacing as an unrelated missing-config error downstream.
    """
    overrides: dict[str, str] = {}
    i = 0
    while i < len(extras):
        token = extras[i]
        if not token.startswith("--"):
            raise InvalidOverrideError(f"unrecognized argument: {token!r}")
        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            if not key:
                raise InvalidOverrideError(f"override is missing a name: {token!r}")
            overrides[key] = value
            i += 1
        elif i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            if not key:
                raise InvalidOverrideError(f"override is missing a name: {token!r}")
            overrides[key] = extras[i + 1]
            i += 2
        else:
            if not key:
                raise InvalidOverrideError(f"override is missing a name: {token!r}")
            overrides[key] = "true"
            i += 1
    return overrides


def _coerce_value(raw: str) -> Any:
    """Parse a CLI string into a YAML-compatible Python value (int, bool, str, ...)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_overrides(config: dict[str, Any], overrides: dict[str, str]) -> dict[str, Any]:
    """Merge CLI overrides into a copy of the config dict.

    Bare keys (no dot) target the ``reef`` section; dotted keys traverse
    nested sections (e.g. ``training.checkpoint_dir``).
    """
    config = copy.deepcopy(config)
    for key, raw_value in overrides.items():
        if "." not in key:
            key = f"reef.{key}"
        parts = key.split(".")
        node: dict[str, Any] = config
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _coerce_value(raw_value)
    return config


def _write_override_config(config: dict[str, Any]) -> Path:
    """Write the overridden config to a temp file so child processes pick it up via ``REEF_CONFIG``."""
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="reef-override-")
    with os.fdopen(fd, "w") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)
    return Path(tmp)


class _TeeStream:
    """Pipe a child's stdout to a file and the terminal simultaneously.

    Popen needs a real fd for the child's stdout, so we give it a pipe
    (``os.pipe``) and run a background thread that reads from the read end
    and fans writes out to both the per-service log file and the orchestrator's
    own stdout. This lets the operator see live output without ``tail -f``.
    """

    def __init__(self, log_fp: TextIO, terminal: TextIO) -> None:
        self._read_fd, self._write_fd = os.pipe()
        self._log_fp = log_fp
        self._terminal = terminal
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    @property
    def write_fd(self) -> int:
        return self._write_fd

    def _pump(self) -> None:
        while True:
            try:
                chunk = os.read(self._read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            # Each sink fails independently: a broken pipe on one must not
            # stop mirroring to the other.
            for s in (self._log_fp, self._terminal):
                try:
                    s.write(text)
                    s.flush()
                except (OSError, ValueError):  # noqa: PERF203
                    pass
        with contextlib.suppress(OSError):
            os.close(self._read_fd)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self._write_fd)
        self._thread.join(timeout=5)


class _Stack:
    """Manages the lifecycle of all declared services.

    Holds Popen handles (source of truth for liveness via ``proc.poll()``),
    writes pidfiles for external consumers, probes readiness, runs a watchdog
    thread that detects unexpected exits, and tears down in reverse-dependency
    order on signal. Follows the same pattern as TGI's launcher and SGLang's
    Engine: spawn → probe readiness → block → signal-driven shutdown.
    """

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
        self.run_dir = run_dir
        self.ready_timeout_default = ready_timeout_default
        self.config_path = Path(config_path)
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        # Every POSIX service starts a fresh session, so its process-group ID
        # is the leader PID. Keep that identity after the leader exits:
        # launchers such as Ray and SGLang can leave workers alive after their
        # direct child has gone, and shutdown must still reach those workers.
        self._process_groups: dict[str, int] = {}
        self._log_fps: dict[str, TextIO] = {}
        self._tees: dict[str, _TeeStream] = {}
        self._names = [svc["name"] for svc in services]
        self._stopping = threading.Event()
        self._unexpected_exit = threading.Event()

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
        if cuda:
            env["CUDA_VISIBLE_DEVICES"] = interpolate_config(self.config, str(cuda))
        return env

    def _check_deps(self, svc: Mapping[str, Any]) -> None:
        name = svc["name"]
        for dep in svc.get("depends_on") or []:
            if dep not in self._names:
                sys.exit(f"[reef] ERROR: service {name!r} depends on unknown service {dep!r}")
            if not self._is_alive(dep):
                sys.exit(f"[reef] ERROR: service {name!r} requires {dep!r}, which is not running")

    def _wait_ready(self, svc: Mapping[str, Any], proc: subprocess.Popen[bytes]) -> None:
        name = svc["name"]
        ready = svc.get("ready")
        if not ready:
            time.sleep(2)
            return
        ready_cmd = interpolate_config(self.config, ready)
        env = self._service_env(svc)
        timeout = int(svc.get("ready_timeout", self.ready_timeout_default))
        waited = 0
        while True:
            if subprocess.run(ready_cmd, shell=True, capture_output=True, env=env).returncode == 0:
                return
            if proc.poll() is not None:
                _log(f"ERROR: {name!r} exited before ready; see {self._log_file(name)}")
                sys.exit(1)
            if waited >= timeout:
                _log(f"ERROR: {name!r} not ready after {timeout}s; see {self._log_file(name)}")
                sys.exit(1)
            time.sleep(5)
            waited += 5

    def _start_service(self, svc: Mapping[str, Any]) -> None:
        name = svc["name"]
        if self._is_alive(name):
            _log(f"{name}: already running (pid {self._procs[name].pid})")
            return
        self._check_deps(svc)
        command = _command_argv(self.config, svc["command"])
        env = self._service_env(svc)
        _log(f"starting {name}: {shlex.join(command)}")
        # The handle outlives this scope (closed in stop()); a context manager
        # would close it under the running service.
        log_fp = open(self._log_file(name), "a")  # noqa: SIM115
        self._log_fps[name] = log_fp
        # Mirror stdout/stderr to both the log file and the terminal so the
        # operator sees live output without `tail -f`.
        tee = _TeeStream(log_fp, sys.stdout)
        self._tees[name] = tee
        proc = subprocess.Popen(
            command,
            env=env,
            cwd=svc.get("cwd"),
            stdout=tee.write_fd,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        self._procs[name] = proc
        if os.name == "posix":
            self._process_groups[name] = proc.pid
        self._pid_file(name).write_text(str(proc.pid))
        self._wait_ready(svc, proc)
        _log(f"{name}: ready")

    def start(self) -> None:
        for svc in self.services:
            self._start_service(svc)
        _log(f"stack up. logs: {self.run_dir}/*.log")

    def _watchdog(self) -> None:
        """Poll Popen handles; bring down the stack if any service exits.

        GPU processes that crash leave CUDA state suspect, so — following TGI /
        SGLang / vLLM convention — we do not auto-restart. The whole stack goes
        down so the operator can restart cleanly.
        """
        while not self._stopping.is_set():
            for name, proc in list(self._procs.items()):
                if proc.poll() is not None:
                    # A signal may arrive while this polling pass is already
                    # in progress. Child exits after that stop request belong
                    # to deliberate teardown, not to the watchdog.
                    if self._stopping.is_set():
                        return
                    code = proc.returncode
                    _log(f"{name}: exited (code {code}); bringing down the stack")
                    self._unexpected_exit.set()
                    self._stopping.set()
                    return
            self._stopping.wait(_WATCHDOG_INTERVAL)

    def block(self) -> None:
        """Run the watchdog thread and block until a signal arrives or a
        service exits."""

        def _request_stop(signum: int, frame: FrameType | None) -> None:
            _log("received signal, shutting down")
            self._stopping.set()

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        watcher = threading.Thread(target=self._watchdog, daemon=True)
        watcher.start()
        self._stopping.wait()

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
        for tee in self._tees.values():
            tee.close()
        for fp in self._log_fps.values():
            with contextlib.suppress(Exception):
                fp.close()

    @property
    def exit_code(self) -> int:
        """1 when a service exited unexpectedly, else 0.

        If we got here via a signal, exit 0; if a service died, exit 1.
        """
        return int(self._unexpected_exit.is_set())


def _run_orchestrator(config_path: str, overrides: dict[str, str] | None = None) -> int:
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = PROJECT_ROOT / resolved_config_path
    config = load_config(resolved_config_path)
    if overrides:
        config = _apply_overrides(config, overrides)
    # Structure first, so a bad stack fails before a model download, a run dir, or a child process.
    services = validate_services(config, resolved_config_path)
    paths_changed = resolve_model_paths(config)
    temp_config_path: Path | None = None
    if overrides or paths_changed:
        temp_config_path = _write_override_config(config)
        resolved_config_path = temp_config_path
    run_dir = Path(config_value(config, "run_dir", default="/tmp/reef-stack") or "/tmp/reef-stack")
    run_dir.mkdir(parents=True, exist_ok=True)
    ready_timeout_default = int(config_value(config, "ready_timeout", default="3600") or "3600")

    stack = _Stack(
        config,
        services,
        run_dir,
        ready_timeout_default,
        resolved_config_path,
    )
    try:
        stack.start()
        stack.block()
    except KeyboardInterrupt:
        # SIGINT during startup (before block() registers its handler) still
        # triggers the default KeyboardInterrupt; shutdown ran via finally.
        # Swallow it so the operator sees a clean exit, not a traceback.
        stack._stopping.set()
    finally:
        stack.shutdown()
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
    return stack.exit_code


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    config_path = args.config or os.environ.get("REEF_CONFIG", "reef.yaml")
    try:
        overrides = _parse_overrides(extras)
    except InvalidOverrideError as exc:
        parser.error(str(exc))
    sys.exit(_run_orchestrator(config_path, overrides))
