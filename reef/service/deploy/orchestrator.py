"""Process orchestrator behind ``reef serve``.

Starts every service a config declares in dependency order, probes readiness,
mirrors child output, watches for unexpected exits, and tears the stack down
in reverse order on signal. Assembly of the Reef HTTP application itself
lives in :mod:`reef.service.deploy.settings` and :mod:`reef.service.assembly`.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from reef.runtime.executor import Executor
from reef.service.deploy.config import (
    PROJECT_ROOT,
    config_value,
    interpolate_config,
    load_config,
    resolve_model_paths,
    validate_services,
)
from reef.service.deploy.execution import service_executor_config, service_executor_selection
from reef.service.deploy.settings import build_parser

_DEFAULT_GRACE_TIMEOUT = 30
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
        for index, part in enumerate(parts[:-1]):
            if part not in node:
                node[part] = {}
            existing = node[part]
            if not isinstance(existing, dict):
                prefix = ".".join(parts[: index + 1])
                raise InvalidOverrideError(f"override path {prefix!r} is not a section")
            node = existing
        node[parts[-1]] = _coerce_value(raw_value)
    return config


def _write_override_config(config: dict[str, Any]) -> Path:
    """Write the overridden config to a temp file so child processes pick it up via ``REEF_CONFIG``."""
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="reef-override-")
    with os.fdopen(fd, "w") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)
    return Path(tmp)


class _Stack:
    """Dependency orchestration over Executor RPCs, independent of placement."""

    def __init__(
        self,
        config: dict[str, Any],
        services: Sequence[dict[str, Any]],
        run_dir: Path,
        ready_timeout_default: int,
        config_path: str | Path,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.services = services
        self.run_dir = run_dir
        self.ready_timeout_default = ready_timeout_default
        self.config_path = Path(config_path)
        self._executors: dict[str, Executor] = {}
        self._log_offsets: dict[str, int] = {}
        self._stopping = threading.Event()
        self._unexpected_exit = threading.Event()
        self._closed = False

    def _is_alive(self, name: str) -> bool:
        executor = self._executors.get(name)
        if executor is None:
            return False
        status = executor.rpc(0, "status", timeout=10)
        return name in status and status[name] is None

    def _drain_log(self, name: str) -> None:
        executor = self._executors[name]
        content, offset = executor.rpc(0, "read_log", args=(name, self._log_offsets.get(name, 0)), timeout=10)
        self._log_offsets[name] = offset
        if content:
            with (self.run_dir / f"{name}.log").open("a") as handle:
                handle.write(content)
            print(content, end="", flush=True)

    def _wait_ready(self, service: Mapping[str, Any], executor: Executor) -> None:
        deadline = time.monotonic() + float(service.get("ready_timeout", self.ready_timeout_default))
        while not self._stopping.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"service {service['name']!r} did not become ready")
            ready = executor.rpc(0, "probe", args=(service["name"], min(5, remaining)), timeout=min(5, remaining) + 2)
            self._drain_log(service["name"])
            if ready:
                return
            self._stopping.wait(min(0.25, max(0, deadline - time.monotonic())))
        raise InterruptedError("stack stopped during startup")

    def _start_service(self, service: Mapping[str, Any]) -> None:
        name = service["name"]
        for dependency in service.get("depends_on") or []:
            if not self._is_alive(dependency):
                raise RuntimeError(f"service {name!r} requires healthy dependency {dependency!r}")
        selection = service_executor_selection(self.config, service)
        _log(f"{name}: executor={selection.settings.backend} ({selection.reason})")
        executor = Executor.create(
            service_executor_config(
                self.config,
                service,
                self.run_dir / name,
                self.ready_timeout_default,
                self.config_path,
                selection=selection,
            )
        )
        # Register immediately: prepare/start/readiness failures must release it.
        self._executors[name] = executor
        info = executor.rpc(0, "describe", timeout=30)
        endpoint = service.get("endpoint")
        if endpoint:
            host = interpolate_config(self.config, service.get("advertise_host", info["host"]))
            host = f"[{host}]" if ":" in host and not host.startswith("[") else host
            endpoint = interpolate_config(self.config, endpoint).replace("{host}", host)
            self.config.setdefault("endpoints", {})[name] = endpoint
        executor.rpc(0, "prepare", args=(self.config,), timeout=30)
        executor.rpc(0, "start", timeout=30)
        info = executor.rpc(0, "describe", timeout=30)
        with (self.run_dir / f"{name}.worker.json").open("w") as handle:
            json.dump(info, handle)
        self._wait_ready(service, executor)
        _log(f"{name}: ready" + (f" at {endpoint}" if endpoint else ""))

    def start(self) -> None:
        try:
            for service in self.services:
                self._start_service(service)
        except BaseException:
            self.shutdown()
            raise
        _log(f"stack up. logs: {self.run_dir}/*.log")

    def _watchdog(self) -> None:
        while not self._stopping.is_set():
            for name in list(self._executors):
                try:
                    alive = self._is_alive(name)
                    self._drain_log(name)
                except Exception as exc:
                    _log(f"{name}: execution backend failed: {exc}")
                    alive = False
                if not alive:
                    if self._stopping.is_set():
                        return
                    _log(f"{name}: exited; bringing down the stack")
                    self._unexpected_exit.set()
                    self._stopping.set()
                    return
            self._stopping.wait(_WATCHDOG_INTERVAL)

    def block(self) -> None:
        def _request_stop(signum: int, frame: FrameType | None) -> None:
            _log("received signal, shutting down")
            self._stopping.set()

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        watcher = threading.Thread(target=self._watchdog, daemon=True)
        watcher.start()
        self._stopping.wait()
        watcher.join(timeout=15)

    def shutdown(self, grace: float = _DEFAULT_GRACE_TIMEOUT) -> None:
        if self._closed:
            return
        self._closed = True
        self._stopping.set()
        ordered = list(reversed(self._executors.items()))
        # Signal every dependent before its dependencies, with one grace window.
        for name, executor in ordered:
            try:
                executor.rpc(0, "request_stop", timeout=10)
            except Exception as exc:  # noqa: PERF203 -- each remote worker must be cleaned independently
                _log(f"{name}: stop RPC failed: {exc}")
        deadline = time.monotonic() + max(0, grace)
        pending = ordered
        while pending and time.monotonic() < deadline:
            living = []
            for name, executor in pending:
                try:
                    if executor.rpc(0, "tree_alive", timeout=min(2, max(0.01, deadline - time.monotonic()))):
                        living.append((name, executor))
                except Exception:  # noqa: PERF203 -- a failed node must not skip other nodes
                    living.append((name, executor))
            pending = living
            if pending:
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        for name, executor in ordered:
            try:
                executor.rpc(0, "shutdown", kwargs={"grace": 0}, timeout=15)
                self._drain_log(name)
            except Exception as exc:  # noqa: PERF203 -- continue teardown after a worker failure
                _log(f"{name}: process cleanup failed: {exc}")
            finally:
                try:
                    executor.shutdown()
                except Exception as exc:
                    _log(f"{name}: executor cleanup failed: {exc}")

    @property
    def exit_code(self) -> int:
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
        exit_code = _run_orchestrator(config_path, overrides)
    except InvalidOverrideError as exc:
        parser.error(str(exc))
    sys.exit(exit_code)
