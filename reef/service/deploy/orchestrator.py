"""Process orchestrator behind ``reef serve``.

Starts independent services concurrently while honoring readiness dependencies,
mirrors child output, watches for unexpected exits, and tears the stack down
in reverse order on signal. Assembly of the Reef HTTP application itself
lives in :mod:`reef.service.assembly`; CLI syntax lives in :mod:`reef.service.deploy.cli`.
"""

from __future__ import annotations

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

from reef.core.config import config_arguments
from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import recipe_class_for
from reef.runtime.executor import Executor
from reef.runtime.executor.config import ExecutorSelection, role_executor_settings, select_executor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.ray_runtime import RayRuntimeLease, acquire_ray_runtime
from reef.runtime.registry import RuntimeConfigError
from reef.service.deploy.cli import (
    InvalidOverrideError,
    _apply_overrides,
    _parse_overrides,
    build_serve_parser,
    native_override,
    object_override_path,
)
from reef.service.deploy.component_config import (
    component_config_arguments,
    normalize_component_config,
    normalize_component_layout,
    translate_layout,
    translate_references,
)
from reef.service.deploy.config_utils import (
    PROJECT_ROOT,
    DeployConfigError,
    config_value,
    interpolate_config,
    interpolate_environment,
    load_config,
    recipe_source_root,
)
from reef.service.deploy.execution import service_executor_config, service_executor_selection, validate_services
from reef.service.deploy.inference import assemble_provider_services, command_line_config, resolve_model_paths
from reef.service.deploy.service_config import (
    normalize_service_config,
    service_config_arguments,
    service_config_from_mapping,
    service_override,
    service_owned_keys,
)
from reef.service.deploy.training import assemble_training_services
from reef.service.profiles import PROFILES_DIR, UnknownProfileError, profile_path

_DEFAULT_GRACE_TIMEOUT = 30
_WATCHDOG_INTERVAL = 5


def _log(msg: str) -> None:
    print(f"[reef] {msg}", file=sys.stderr)


class DeployStartupError(RuntimeError):
    """A deployment failed to start; includes its reason and local log location."""


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


def _component_selection(
    config: dict[str, Any], overrides: dict[str, str], config_path: Path
) -> tuple[dict[str, Any], Path | None]:
    """Resolve only selection inputs before loading component definitions."""
    selected = _apply_overrides(config, overrides)
    reef = selected.get("reef", {})
    if not isinstance(reef, Mapping):
        raise DeployConfigError("reef must be an object")
    selected_reference = interpolate_environment({"reef": {"recipe": reef.get("recipe")}}, config_path)
    reference = selected_reference["reef"]["recipe"]
    if reference is not None and not isinstance(reference, str):
        raise DeployConfigError("reef.recipe must be a string")
    selected.setdefault("reef", {})["recipe"] = interpolate_config(selected, reference) if reference else reference
    if isinstance(selected.get("implementation"), str):
        selected["implementation"] = interpolate_environment(
            {"implementation": selected["implementation"]}, config_path
        )["implementation"]
    runtime = (
        selected.get("runtime")
        if selected.get("implementation") and ":" not in (reference or "")
        else selected["reef"].get("runtime")
    )
    if isinstance(runtime, dict) and isinstance(runtime.get("type"), str):
        runtime["type"] = interpolate_config(
            selected, interpolate_environment({"type": runtime["type"]}, config_path)["type"]
        )
    source_root = recipe_source_root(selected, config_path)
    if source_root is not None and str(source_root) not in sys.path:
        sys.path.append(str(source_root))
    return selected, source_root


def _prepare_recipe_dependencies(config: dict[str, Any], recipe_type: type[Recipe]) -> None:
    """Prepare only the selected method's dependencies and derived client bindings."""
    training = issubclass(recipe_type, WeightTrainingRecipe)
    reef = config["reef"]
    if training:
        owned = {key: value for key, value in reef.items() if key not in service_owned_keys()}
    else:
        owned = {
            **reef.get("data", {}),
            **{key: reef[key] for key in recipe_type.config_sections if key in reef},
        }
    dependencies = list(recipe_type.prepare_deployment(owned))
    known = {argument.name for argument in config_arguments(recipe_type)} | set(recipe_type.config_sections)
    if training:
        known.add("checkpoint_every_n_versions")
    if set(owned) - known:
        raise DeployConfigError("recipe deployment hooks may bind only declared recipe settings")
    if training:
        reef.update(owned)
    else:
        for key, value in owned.items():
            target = reef if key in recipe_type.config_sections else reef.setdefault("data", {})
            target[key] = value
    if dependencies:
        services = config["services"]
        services[0].setdefault("depends_on", []).extend(item["name"] for item in dependencies)
        config["services"] = [*dependencies, *services]


def resolve_deployment_config(
    config: dict[str, Any], overrides: dict[str, str] | None, source: str | Path, *, standard: bool = False
) -> tuple[dict[str, Any], Path | None]:
    """Resolve the public layout and selected schemas without starting processes or downloading models."""
    resolved_config_path = Path(source).resolve()
    versioned = config.get("schema-version") == 2
    config = translate_layout(config)
    standard = standard or versioned
    selected, source_root = _component_selection(config, overrides or {}, resolved_config_path)
    if source_root is not None:
        _log(f"recipe package resolves from {source_root}")
    try:
        arguments = component_config_arguments(selected)
        recipe_type = recipe_class_for(config_value(selected, "reef", "recipe") or "recipe")
        training = recipe_type is not None and issubclass(recipe_type, WeightTrainingRecipe)
        if versioned:
            config = normalize_component_layout(config, arguments)
        if versioned or standard:
            for key, value in (overrides or {}).items():
                if (
                    service_override(key, value) is None
                    and native_override(key) is None
                    and object_override_path(key, (*service_config_arguments(), *arguments)) is None
                    and not any(f"--{key}" in (*argument.flags, *argument.negative_flags) for argument in arguments)
                ):
                    raise DeployConfigError(f"unknown configuration flag --{key}")
        config = _apply_overrides(config, overrides or {}, arguments=arguments)
        if versioned or standard:
            config = translate_references(config, arguments)
        config = interpolate_environment(config, resolved_config_path)
        if standard:
            if "services" in config.get("execution", {}):
                raise DeployConfigError("execution.services belongs to legacy process stacks")
            for name, flag in (("host", "reef.host"), ("model_path", "inference.model-path")):
                value = config.get("reef", {}).get(name)
                if isinstance(value, str) and not value.strip():
                    raise DeployConfigError(f"--{flag} must be non-empty")
        normalized_config = normalize_component_config(normalize_service_config(config), arguments)
        if standard:
            if training:
                assemble_training_services(normalized_config)
            else:
                assemble_provider_services(normalized_config)
            if recipe_type is not None:
                _prepare_recipe_dependencies(normalized_config, recipe_type)
    except (ValueError, RecipeConfigError, RuntimeConfigError) as exc:
        raise DeployConfigError(f"config {resolved_config_path}: {exc}") from exc
    return normalized_config, source_root


def _run_orchestrator(config_path: str | None, overrides: dict[str, str] | None = None) -> int:
    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else Path.cwd() / "<command line>"
    config = (
        load_config(resolved_config_path, interpolate_env=False) if config_path else command_line_config(os.environ)
    )
    versioned = config.get("schema-version") == 2
    _log(f"config: {resolved_config_path}" if config_path else "config: command line")
    normalized_config, source_root = resolve_deployment_config(
        config, overrides, resolved_config_path, standard=config_path is None
    )
    settings_changed = normalized_config != config
    config = normalized_config
    services = validate_services(config, resolved_config_path)
    paths_changed = resolve_model_paths(config)
    temp_config_path: Path | None = None
    try:
        if versioned or config_path is None or overrides or paths_changed or settings_changed:
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
                raise DeployStartupError(
                    f"deployment startup failed: {exc}\n  logs: {run_dir.resolve()}/*.log"
                ) from exc
            stack.block()
        except KeyboardInterrupt:
            # Startup signals unwind the launch tasks before final cleanup.
            stack._stopping.set()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            stack.shutdown()
        return stack.exit_code
    finally:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)


#: ``--model <provider>/<model>``: the upstream URL and the key a provider prefix stands for. A key of None
#: comes from ``REEF_UPSTREAM_API_KEY`` and is required; ollama ignores its key, so any word will do.
_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "ollama": ("http://127.0.0.1:11434", "ollama"),
    "openai": ("https://api.openai.com", None),
}

#: The tutorial method the harness-evolve profile points at; the profile runs from the checkout that holds it.
_PROFILE_METHODS = {"harness-evolve": Path("tutorials/evolve-your-harness/harness/evolution.py")}


def _model_overrides(
    spec: str, environ: Mapping[str, str], overrides: Mapping[str, str] | None = None
) -> dict[str, str]:
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
        for name, value in (overrides or {}).items():
            declared = service_override(name, value)
            if declared is not None and declared[0].name == "upstream_api_key":
                key = value.strip()
        if not key:
            raise DeployConfigError(
                f"--model {spec}: set REEF_UPSTREAM_API_KEY to the {provider} key or pass --inference.upstream-api-key"
            )
    return {"upstream_url": url, "upstream_model": model, "upstream_api_key": key}


def _resolve_config(config: str | None, recipe: str | None) -> str | None:
    """Select a file/profile, or return None for configuration-free provider startup."""
    if config is not None and recipe is not None:
        raise DeployConfigError("pass -c <file> or --recipe <name>, not both")
    if config is not None:
        if not config.strip():
            raise DeployConfigError("--config must name a non-empty file path")
        return config
    if recipe is not None:
        try:
            return str(profile_path(recipe))
        except UnknownProfileError as exc:
            raise DeployConfigError(str(exc)) from exc
    return None


def _prepare_profile(
    recipe: str,
    model: str | None,
    environ: MutableMapping[str, str],
    overrides: Mapping[str, str] | None = None,
) -> None:
    """What a profile needs from the environment before it loads: its own directory, the checkout, a model."""
    selected_model = model or environ.get("REEF_UPSTREAM_MODEL", "")
    for key, value in (overrides or {}).items():
        declared = service_override(key, value)
        if declared is not None and declared[0].name == "upstream_model":
            selected_model = value
    if not selected_model.strip():
        raise DeployConfigError(f"--recipe {recipe} needs the model: pass --inference.upstream-model MODEL")
    method = _PROFILE_METHODS.get(recipe)
    if method is not None and not (PROJECT_ROOT / method).is_file():
        raise DeployConfigError(
            f"the {recipe} profile runs from a reef checkout: its proposer is {method}, not found under {PROJECT_ROOT}"
        )
    environ["REEF_RECIPE_CONFIG_DIR"] = str(PROFILES_DIR)
    environ["REEF_CHECKOUT"] = str(PROJECT_ROOT)
    if method is not None:
        python_paths = [str((PROJECT_ROOT / method).parents[1]), str(PROJECT_ROOT)]
        if environ.get("PYTHONPATH"):
            python_paths.append(environ["PYTHONPATH"])
        environ["PYTHONPATH"] = os.pathsep.join(python_paths)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        bootstrap = build_serve_parser(service_arguments=False)
        selection, extras = bootstrap.parse_known_args([arg for arg in argv if arg not in ("--help", "-h")])
        help_config = None
        if (
            selection.config
            or selection.recipe
            or any(arg.startswith(("--reef.recipe", "--recipe.implementation")) for arg in extras)
        ):
            path = Path(selection.config or (profile_path(selection.recipe) if selection.recipe else "reef.yaml"))
            config = (
                translate_layout(load_config(path, interpolate_env=False))
                if selection.config or selection.recipe
                else {}
            )
            help_config, _ = _component_selection(config, _parse_overrides(extras), path.resolve())
        build_serve_parser(config=help_config).parse_args(argv)
    # Discover the file/profile without applying defaults or converting
    # public values before YAML and environment references are available.
    parser = build_serve_parser(service_arguments=False)
    args, extras = parser.parse_known_args(argv)
    try:
        overrides = _parse_overrides(extras)
        if args.model:
            # An explicit --key beats what the provider prefix fills in.
            model_defaults = _model_overrides(args.model, os.environ, overrides)
            for key, value in overrides.items():
                model_defaults.pop(key, None)
                model_defaults[key] = value
            overrides = model_defaults
        config_path = _resolve_config(args.config, args.recipe)
        if args.recipe:
            _prepare_profile(args.recipe, args.model, os.environ, overrides)
        exit_code = _run_orchestrator(config_path, overrides)
    except InvalidOverrideError as exc:
        parser.error(str(exc))
    sys.exit(exit_code)


def run_service(config_path: str | Path | None = None) -> int:
    """Run the internal Reef HTTP child from the orchestrator's config."""
    selected_config = config_path or os.environ.get("REEF_CONFIG")
    if selected_config is None:
        raise SystemExit("[reef] ERROR: internal service requires REEF_CONFIG")
    settings = service_config_from_mapping(load_config(selected_config))
    from reef.service.assembly import build_app

    app = build_app(settings)
    from aiohttp import web

    web.run_app(app, host=settings.host, port=settings.port)
    return 0
