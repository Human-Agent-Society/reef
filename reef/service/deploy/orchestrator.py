"""Process orchestrator behind ``reef serve``.

Starts independent services concurrently while honoring readiness dependencies,
mirrors child output, watches for unexpected exits, and tears the stack down
in reverse order on signal. Assembly of the Reef HTTP application itself
lives in :mod:`reef.service.deploy.settings` and :mod:`reef.service.assembly`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from reef.runtime.executor import Executor
from reef.runtime.executor.config import ExecutorSelection, role_executor_settings, select_executor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.ray_runtime import RayRuntimeLease, acquire_ray_runtime
from reef.service.deploy.config import (
    PROJECT_ROOT,
    DeployConfigError,
    config_value,
    interpolate_config,
    interpolate_environment,
    load_config,
    recipe_source_root,
    resolve_model_paths,
    validate_services,
)
from reef.service.deploy.execution import service_executor_config, service_executor_selection
from reef.service.deploy.settings import build_parser, normalize_service_config, service_override
from reef.service.profiles import PROFILES_DIR, UnknownProfileError, profile_names, profile_path

_DEFAULT_GRACE_TIMEOUT = 30
_WATCHDOG_INTERVAL = 5


def _log(msg: str) -> None:
    print(f"[reef] {msg}", file=sys.stderr)


class InvalidOverrideError(ValueError):
    """A leftover ``reef serve`` argument is not a valid ``--key`` override."""


class DeployStartupError(RuntimeError):
    """A deployment failed to start; includes its reason and local log location."""


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
        has_value = True
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        elif i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            value = extras[i + 1]
            i += 2
        else:
            value = "true"
            has_value = False
            i += 1
        if not key:
            raise InvalidOverrideError(f"override is missing a name: {token!r}")
        declared = service_override(key, value)
        if declared is not None:
            argument, _ = declared
            if not has_value and argument.kind != "bool":
                raise InvalidOverrideError(f"argument --{key} requires a value")
            if has_value and f"--{key}" in argument.negative_flags:
                raise InvalidOverrideError(f"argument --{key} does not take a value")
        # Preserve the last occurrence's position as well as its value, so
        # a repeated spelling can still override an intervening alias.
        overrides.pop(key, None)
        overrides[key] = value
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
        declared = service_override(key, raw_value)
        if declared is not None:
            argument, raw_value = declared
            key = ".".join(argument.path)
        if "." not in key and declared is None:
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
        # Public fields are converted by argparse, just like YAML values.
        # Keep generic recipe/custom-stack overrides on their legacy path.
        node[parts[-1]] = raw_value if declared is not None else _coerce_value(raw_value)
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
        source_root: Path | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.services = services
        self.run_dir = run_dir
        self.ready_timeout_default = ready_timeout_default
        self.config_path = Path(config_path)
        # Where the deployment's dotted recipe package lives (see
        # ``recipe_source_root``); every service gets it on ``PYTHONPATH``.
        self.source_root = source_root
        self._executors: dict[str, Executor] = {}
        self._log_offsets: dict[str, int] = {}
        self._stopping = threading.Event()
        self._unexpected_exit = threading.Event()
        self._closed = False
        self._ray_runtime: RayRuntimeLease | None = None

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

    def _wait_ready(self, service: Mapping[str, Any], executor: Executor, deadline: float) -> None:
        while not self._stopping.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout = service.get("ready_timeout", self.ready_timeout_default)
                raise TimeoutError(f"service {service['name']!r} did not become ready within {timeout}s")
            try:
                ready = executor.rpc(
                    0, "probe", args=(service["name"], min(5, remaining)), timeout=min(5, remaining) + 2
                )
            except Exception:
                # A failed probe can be the first observation of a child's exit.
                # Preserve its last output without replacing the startup error.
                with suppress(Exception):
                    self._drain_log(service["name"])
                raise
            self._drain_log(service["name"])
            if ready:
                return
            self._stopping.wait(min(0.25, max(0, deadline - time.monotonic())))
        raise InterruptedError("stack stopped during startup")

    def _prepare_services(self, services: Sequence[dict[str, Any]]) -> None:
        # Only prepare eligible services: an explicitly managed Ray head may
        # itself be an upstream dependency. Placement does not load models.
        selections = [service_executor_selection(self.config, service) for service in services]
        ray_services = any(
            issubclass(Executor.get_class(selection.settings.backend), RayExecutor) for selection in selections
        )
        ray_roles = any(
            select_executor(role_executor_settings(self.config, role), role=role).settings.backend == "ray"
            for role in ("training", "rollout")
            if role in self.config.get("execution", {})
        )
        if services and self._ray_runtime is None and (ray_services or ray_roles):
            address = os.environ.get("RAY_ADDRESS") or self.config.get("reef", {}).get("ray_address")
            address = interpolate_config(self.config, address) if address else None
            # Local Slime drivers connect to an explicitly supplied cluster
            # themselves, after their dependencies (possibly a Ray head) are ready.
            # Without an address, own one runtime without reserving driver GPUs.
            if ray_services or not address or address == "local":
                self._ray_runtime = acquire_ray_runtime(address)
                address = self._ray_runtime.address
            self.config.setdefault("reef", {})["ray_address"] = address
        # Publish all eligible endpoints before any of their processes start.
        for service, selection in zip(services, selections, strict=True):
            self._prepare_service(service, selection)

    def _prepare_service(self, service: Mapping[str, Any], selection: ExecutorSelection) -> None:
        name = service["name"]
        for dependency in service.get("depends_on") or []:
            if not self._is_alive(dependency):
                raise RuntimeError(f"service {name!r} requires healthy dependency {dependency!r}")
        _log(f"{name}: executor={selection.settings.backend} ({selection.reason})")
        address = self.config.get("reef", {}).get("ray_address")
        if address:
            service = {**service, "env": {"RAY_ADDRESS": address, **service.get("env", {})}}
        executor = Executor.create(
            service_executor_config(
                self.config,
                service,
                self.run_dir / name,
                self.ready_timeout_default,
                self.config_path,
                selection=selection,
                source_root=self.source_root,
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

    def _start_service(self, service: Mapping[str, Any], config: dict[str, Any]) -> None:
        name = service["name"]
        executor = self._executors[name]
        if self._stopping.is_set():
            raise InterruptedError("stack stopped during startup")
        executor.rpc(0, "prepare", args=(config,), timeout=30)
        if self._stopping.is_set():
            raise InterruptedError("stack stopped during startup")
        deadline = time.monotonic() + float(service.get("ready_timeout", self.ready_timeout_default))
        executor.rpc(0, "start", timeout=30)
        info = executor.rpc(0, "describe", timeout=30)
        with (self.run_dir / f"{name}.worker.json").open("w") as handle:
            json.dump(info, handle)
        self._wait_ready(service, executor, deadline)
        endpoint = config.get("endpoints", {}).get(name)
        _log(f"{name}: ready" + (f" at {endpoint}" if endpoint else ""))

    def start(self) -> None:
        try:
            # Launch/readiness tasks never mutate the deployment config. The
            # coordinator alone publishes addresses and unlocks dependents.
            with ThreadPoolExecutor(max_workers=max(1, len(self.services)), thread_name_prefix="reef-start") as pool:
                try:
                    pending = list(self.services)
                    running: dict[Future[None], str] = {}
                    ready: set[str] = set()
                    while pending or running:
                        if self._stopping.is_set():
                            raise InterruptedError("stack stopped during startup")
                        eligible = [svc for svc in pending if ready.issuperset(svc.get("depends_on", []))]
                        self._prepare_services(eligible)
                        for service in eligible:
                            pending.remove(service)
                            future = pool.submit(self._start_service, service, copy.deepcopy(self.config))
                            running[future] = service["name"]
                        done, _ = wait(running, timeout=0.25, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                            ready.add(running.pop(future))
                        # Readiness is not a permanent liveness guarantee. A
                        # previously ready service may die while peers load.
                        for name in ready:
                            if not self._is_alive(name):
                                raise RuntimeError(f"service {name!r} exited during startup")
                except BaseException:
                    # Wake peer readiness loops before joining launch tasks.
                    # Join first so none can create a process after cleanup.
                    self._stopping.set()
                    raise
        except BaseException:
            self.shutdown()
            raise
        _log(f"stack up. logs: {self.run_dir}/*.log")
        hint = install_hint(self.config)
        if hint is not None:
            _log(f"install the harness in another terminal: {hint}")

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
        if self._ray_runtime is not None:
            self._ray_runtime.close()

    @property
    def exit_code(self) -> int:
        return int(self._unexpected_exit.is_set())


def install_hint(config: Mapping[str, Any]) -> str | None:
    """The one line that installs a harness evolution deployment's harness, or None for a deployment without one.

    Printed when the stack is up so nobody copies it from a README: the
    address the service listens on (loopback when it binds every interface),
    the adapter the deployment evolves, and the token the config holds."""
    evolution = config.get("evolution")
    adapter = evolution.get("adapter") if isinstance(evolution, Mapping) else None
    if not isinstance(adapter, str) or not adapter:
        return None
    host = str(config_value(config, "reef", "host", default="127.0.0.1"))
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    port = config_value(config, "reef", "port", default="8900")
    token = config_value(config, "reef", "token", default=None)
    if token is None:
        tokens = config.get("reef", {}).get("tokens") if isinstance(config.get("reef"), Mapping) else None
        if isinstance(tokens, list) and tokens:
            token = str(tokens[0])
    header = f"-H 'Authorization: Bearer {token}' " if token else ""
    return f"curl -fsS {header}'http://{host}:{port}/reef/harness/install?adapter={adapter}' | bash"


def _run_orchestrator(config_path: str, overrides: dict[str, str] | None = None) -> int:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = load_config(resolved_config_path, interpolate_env=False)
    if overrides:
        config = _apply_overrides(config, overrides)
    config = interpolate_environment(config, resolved_config_path)
    cli_paths = frozenset(
        declared[0].path
        for key, value in (overrides or {}).items()
        if (declared := service_override(key, value)) is not None
    )
    try:
        normalized_config = normalize_service_config(config, cli_paths=cli_paths)
    except ValueError as exc:
        raise DeployConfigError(f"config {resolved_config_path}: {exc}") from exc
    settings_changed = normalized_config != config
    config = normalized_config
    # Structure first, so a bad stack fails before a model download, a run dir, or a child process.
    services = validate_services(config, resolved_config_path)
    # Resolved against the operator's config, before any override copy
    # relocates the path the services read.
    source_root = recipe_source_root(config, resolved_config_path)
    if source_root is not None:
        _log(f"recipe package resolves from {source_root}")
        if str(source_root) not in sys.path:
            sys.path.append(str(source_root))
    paths_changed = resolve_model_paths(config)
    temp_config_path: Path | None = None
    if overrides or paths_changed or settings_changed:
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
        source_root=source_root,
    )

    def interrupt_startup(signum: int, frame: FrameType | None) -> None:
        _log("received signal during startup, shutting down")
        raise KeyboardInterrupt

    # block() installs the steady-state handler only after every service is
    # ready. Until then, interruption must unwind start() and stop its peers.
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_startup)
    try:
        try:
            stack.start()
        except Exception as exc:
            raise DeployStartupError(f"deployment startup failed: {exc}\n  logs: {run_dir.resolve()}/*.log") from exc
        stack.block()
    except KeyboardInterrupt:
        # Startup signals unwind the launch tasks before final cleanup.
        stack._stopping.set()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        stack.shutdown()
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
    return stack.exit_code


#: ``--model <provider>/<model>``: the upstream URL and the key a provider prefix stands for. A key of None
#: comes from ``REEF_UPSTREAM_API_KEY`` and is required; ollama ignores its key, so any word will do.
_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "ollama": ("http://127.0.0.1:11434", "ollama"),
    "openai": ("https://api.openai.com", None),
}

#: The tutorial method the harness-evolve profile points at; the profile runs from the checkout that holds it.
_PROFILE_METHODS = {"harness-evolve": Path("tutorials/evolve-your-harness/harness/evolution.py")}


def _model_overrides(spec: str, environ: Mapping[str, str]) -> dict[str, str]:
    """The ``reef.*`` overrides ``--model`` stands for.

    A known provider prefix fills the URL and the key; any other spelling,
    a prefix of another kind included (``Qwen/Qwen3-8B``), is the model id
    alone and the URL comes from the config or the environment."""
    provider, _, model = spec.partition("/")
    if provider not in _PROVIDERS or not model:
        return {"upstream_model": spec}
    url, key = _PROVIDERS[provider]
    if key is None:
        key = environ.get("REEF_UPSTREAM_API_KEY", "").strip()
        if not key:
            raise DeployConfigError(f"--model {spec}: set REEF_UPSTREAM_API_KEY to the {provider} key")
    return {"upstream_url": url, "upstream_model": model, "upstream_api_key": key}


def _resolve_config(config: str | None, recipe: str | None, environ: Mapping[str, str]) -> str:
    """The config ``reef serve`` runs: ``-c`` or ``--recipe``, else ``REEF_CONFIG``, else ``reef.yaml``."""
    if config and recipe:
        raise DeployConfigError("pass -c <file> or --recipe <name>, not both")
    if config:
        return config
    if recipe:
        try:
            return str(profile_path(recipe))
        except UnknownProfileError as exc:
            raise DeployConfigError(str(exc)) from exc
    if environ.get("REEF_CONFIG"):
        return environ["REEF_CONFIG"]
    if Path("reef.yaml").is_file():
        return "reef.yaml"
    raise DeployConfigError(
        "no config: pass one with -c <file>, or start a recipe's profile with --recipe <name>; "
        f"recipes with a profile: {', '.join(profile_names())}"
    )


def _prepare_profile(recipe: str, model: str | None, environ: MutableMapping[str, str]) -> None:
    """What a profile needs from the environment before it loads: its own directory, the checkout, a model."""
    if not model and not environ.get("REEF_UPSTREAM_MODEL", "").strip():
        raise DeployConfigError(f"--recipe {recipe} needs the model: pass --model <provider>/<model>")
    method = _PROFILE_METHODS.get(recipe)
    if method is not None and not (PROJECT_ROOT / method).is_file():
        raise DeployConfigError(
            f"the {recipe} profile runs from a reef checkout: its proposer is {method}, not found under {PROJECT_ROOT}"
        )
    environ["REEF_RECIPE_CONFIG_DIR"] = str(PROFILES_DIR)
    environ["REEF_CHECKOUT"] = str(PROJECT_ROOT)


def build_serve_parser(*, service_arguments: bool = True) -> argparse.ArgumentParser:
    """``reef serve``'s own arguments: the service child's parser plus the profile and model flags.

    Only the launcher takes them; ``python -m reef.service`` still refuses ``--recipe``."""
    parser = build_parser(service_arguments=service_arguments)
    parser.add_argument(
        "--recipe",
        default=None,
        metavar="NAME",
        help="Start a built in recipe's profile instead of a config file (harness-evolve).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="[PROVIDER/]MODEL",
        help="The upstream model; a known provider prefix (ollama, openai) fills the URL and the key.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        build_serve_parser().parse_args(argv)
    # Discover the file/profile without applying defaults or converting
    # public values before YAML and environment references are available.
    parser = build_serve_parser(service_arguments=False)
    args, extras = parser.parse_known_args(argv)
    try:
        overrides = _parse_overrides(extras)
        if args.model:
            # An explicit --key beats what the provider prefix fills in.
            overrides = {**_model_overrides(args.model, os.environ), **overrides}
        config_path = _resolve_config(args.config, args.recipe, os.environ)
        if args.recipe:
            _prepare_profile(args.recipe, args.model, os.environ)
        exit_code = _run_orchestrator(config_path, overrides)
    except InvalidOverrideError as exc:
        parser.error(str(exc))
    sys.exit(exit_code)
