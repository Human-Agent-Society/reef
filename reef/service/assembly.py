"""Assemble the Reef HTTP service from settings: dispatcher, registry, app.

This is the service's composition logic — a :class:`ServiceConfig` in, a
running aiohttp application out. It knows nothing about the deployment config
format; ``reef.service.deploy`` translates YAML into these settings and
orchestrates processes around the result.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.inference.http import InferenceProxyRuntime
from reef.observability import build_experiment_tracker, build_record_observer
from reef.recipe import Recipe, WeightTrainingRecipe
from reef.recipe.base import ServedEndpoint
from reef.recipe.config_fields import resolve_config_field_values
from reef.recipe.registry import build_named_recipe, build_recipe, recipe_class_for
from reef.runtime.deployment import RuntimeConnectionConfig, RuntimeRegistry, runtime_pair
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.service.app import InferenceRetryPolicy, create_app
from reef.service.deploy.service_config import ServiceConfig, service_owned_keys
from reef.service.deploy.training import training_deployment_for
from reef.storage.observer import ObservedScenarioStorage
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.records import RecordRetention
from reef.storage.scenario import ScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage


def _training_recipe_type(name: str) -> type[WeightTrainingRecipe] | None:
    """The explicit class for ``name`` when it is a weight-training recipe."""
    recipe_type = recipe_class_for(name)
    if recipe_type is not None and issubclass(recipe_type, WeightTrainingRecipe):
        return recipe_type
    return None


def _recipe_owned_settings(settings: ServiceConfig) -> dict[str, Any]:
    """The flat ``reef.*`` keys that belong to the recipe, not the service.

    The service's own vocabulary is :class:`ServiceConfig`' fields plus the
    config spellings that map onto them (``reef.token`` feeds ``tokens``), so
    it never drifts from what the settings layer consumes. Everything else
    the operator wrote under ``reef:`` is recipe configuration and must be
    consumed by a recipe config field. ``WeightTrainingRecipe.service_config`` raises for
    any key the selected recipe does not declare, instead of letting the
    deployment silently run on recipe defaults.
    """
    service_owned = service_owned_keys()
    return {key: value for key, value in settings.recipe_settings.items() if key not in service_owned}


def _repository_location(value: str) -> str | Path:
    return value if "://" in value or value.startswith("git@") else Path(value)


def _require_non_empty(value: str | None, setting: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{setting} is required")
    return value.strip()


def _connect_training_runtime(
    settings: ServiceConfig,
    *,
    model_path: str,
    max_staleness: int,
    connector: Any = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Build the selected integration's runtime, independently of its process topology."""
    RuntimeConnectionConfig(
        inference_timeout_s=settings.inference_timeout_s,
        train_timeout_s=settings.train_timeout_s,
        max_staleness=max_staleness,
    )
    backend = training_deployment_for(settings.training_backend)
    runtime_config = backend.runtime_config(asdict(settings), max_staleness=max_staleness, connector=connector)
    built = RuntimeRegistry().build(runtime_config, model_path=model_path)
    pair = runtime_pair(built)
    if pair is None:
        raise TypeError("training backend runtime factory must build a (TrainingRuntime, InferenceRuntime) pair")
    return pair


def _upstream_runtime(settings: ServiceConfig) -> InferenceRuntime | None:
    """The proxy runtime ``reef.upstream_url`` names, or None to leave recipes
    to their own resolution (a recipe-config ``runtime`` section, else the
    ``REEF_UPSTREAM_URL`` environment)."""
    if not settings.upstream_url:
        return None
    return InferenceProxyRuntime(
        model_path=settings.upstream_model or "",
        base_url=settings.upstream_url,
        api_key=settings.upstream_api_key or None,
        api=settings.upstream_api,
        inference_timeout_s=settings.inference_timeout_s,
    )


def _training_recipe(
    recipe_type: type[WeightTrainingRecipe],
    settings: ServiceConfig,
    env: Mapping[str, str],
    connector: Any,
) -> Recipe:
    """Build a weight-training recipe on the selected backend runtime."""
    model_path = _require_non_empty(settings.model_path, "reef.model_path")
    # Translate the legacy layout, then resolve the recipe fields once before
    # connecting its runtime. Construction consumes those same resolved values.
    recipe_config = recipe_type.service_config(_recipe_owned_settings(settings), model_path=model_path)
    if settings.evaluation_settings is not None:
        recipe_config["evaluation"] = dict(settings.evaluation_settings)
    # The runtime must exist before the recipe can be constructed, so
    # resolve the shared runtime-owned config field from the same inputs first.
    # WeightTrainingRecipe then verifies that both resolved the same value.
    resolved_recipe_data = resolve_config_field_values(recipe_type, recipe_config.get("data", {}), env)
    training_runtime, runtime = _connect_training_runtime(
        settings,
        model_path=model_path,
        max_staleness=resolved_recipe_data["max_staleness"],
        connector=connector,
    )
    try:
        return recipe_type.from_resolved_config(
            recipe_config, resolved_recipe_data, environ=env, runtime=runtime, training_runtime=training_runtime
        )
    except BaseException:
        for component in (training_runtime, runtime):
            with suppress(Exception):
                component.shutdown()
        raise


#: The keys a recipe that trains weights through a component reads at the top of its config.
COMPOSED_RECIPE_KEYS = frozenset({"components", "model", "artifact", "data", "execution", "executors"})


def composed_training_recipe(
    selected: str,
    recipe_type: type[Recipe],
    config: dict[str, Any],
    weight_type: type[WeightTrainingRecipe],
    weight_config: Mapping[str, Any],
    settings: ServiceConfig,
    env: Mapping[str, str],
    connector: Any,
) -> Recipe:
    """Build a recipe that trains weights through one of its components, on the selected backend runtime.

    The component's own data resolves the staleness window the runtime is
    connected with, and the pair reaches every component of the recipe. The
    deployment's model path is every component's model, and a top level
    evaluation section belongs to the component that trains.
    """
    model_path = _require_non_empty(settings.model_path, "reef.model_path")
    if config.get("runtime"):
        raise ValueError("weight training selects its runtime through training.backend; remove recipe.runtime")
    unknown = sorted(
        key for key in config if key not in COMPOSED_RECIPE_KEYS and key not in recipe_type.config_sections
    )
    if unknown:
        raise ValueError(
            f"reef.{unknown[0]} is not a setting of {recipe_type.__name__}; a component's settings go under "
            "components.<name>.data"
        )
    resolved_weight_data = resolve_config_field_values(weight_type, weight_config.get("data", {}), env)
    training_runtime, runtime = _connect_training_runtime(
        settings,
        model_path=model_path,
        max_staleness=resolved_weight_data["max_staleness"],
        connector=connector,
    )
    config = {**config, "model": {**dict(config.get("model", {})), "path": model_path}}
    if settings.evaluation_settings is not None:
        components = dict(config["components"])
        for name, component_config in components.items():
            if component_config is weight_config:
                components[name] = {**component_config, "evaluation": dict(settings.evaluation_settings)}
        config = {**config, "components": components}
    try:
        return build_recipe(selected, env, config=config, runtime=runtime, training_runtime=training_runtime)
    except BaseException:
        for component in (training_runtime, runtime):
            with suppress(Exception):
                component.shutdown()
        raise


def _serving_recipe(selected: str, settings: ServiceConfig, env: Mapping[str, str], connector: Any) -> Recipe:
    """Build the one recipe ``reef.recipe`` names.

    The spellings differ only in where config and runtime come from: a dotted
    weight-training class reads the flat ``reef.*`` section and connects the
    Ray runtime; a dotted class that trains weights through one of its
    components connects the same runtime from that component's data; another
    dotted class is built from the environment on the upstream proxy; a bare
    name is ``recipe`` or a YAML preset under ``REEF_RECIPE_CONFIG_DIR``,
    whose own ``runtime`` section wins over the upstream proxy.
    """
    training_recipe_type = _training_recipe_type(selected)
    if training_recipe_type is not None:
        return _training_recipe(training_recipe_type, settings, env, connector)
    if ":" in selected:
        config = _recipe_owned_settings(settings)
        if settings.preset_config is not None:
            config.update(
                {
                    key: settings.preset_config[key]
                    for key in ("execution", "executors")
                    if key in settings.preset_config
                }
            )
        recipe_type = recipe_class_for(selected)
        weight_training = None if recipe_type is None else recipe_type.select_weight_training(config)
        if weight_training is not None and recipe_type is not None:
            weight_type, weight_config = weight_training
            return composed_training_recipe(
                selected, recipe_type, config, weight_type, weight_config, settings, env, connector
            )
        if settings.evaluation_settings is not None:
            raise ValueError("the top-level evaluation section requires a weight-training recipe")
        runtime_config = config.get("runtime")
        runtime = (
            RuntimeRegistry().build(
                runtime_config,
                model_path=settings.model_path or settings.upstream_model or "",
                recipe_config=config,
                environ=env,
            )
            if runtime_config
            else _upstream_runtime(settings)
        )
        training_runtime = None
        if isinstance(runtime, tuple):
            training_runtime, runtime = runtime
        try:
            return build_recipe(selected, env, config=config, runtime=runtime, training_runtime=training_runtime)
        except BaseException:
            for component in (training_runtime, runtime):
                if component is not None:
                    with suppress(Exception):
                        component.shutdown()
            raise
    if settings.evaluation_settings is not None:
        raise ValueError("the top-level evaluation section requires a weight-training recipe")
    return build_named_recipe(
        selected, env, default_runtime=_upstream_runtime(settings), preset_config=settings.preset_config
    )


def default_served_url(host: str, port: int) -> str:
    """Where this service reaches itself: loopback for a wildcard bind, an IPv6 literal in brackets.

    A ``::`` bind is IPv6 only (asyncio sets IPV6_V6ONLY on it), so its loopback is ``::1``.
    """
    bound = host.strip()
    if bound in ("", "0.0.0.0"):
        return f"http://127.0.0.1:{port}"
    if bound == "::":
        return f"http://[::1]:{port}"
    try:
        literal = ipaddress.ip_address(bound)
    except ValueError:
        return f"http://{bound}:{port}"
    return f"http://[{bound}]:{port}" if literal.version == 6 else f"http://{bound}:{port}"


def build_dispatcher(
    settings: ServiceConfig,
    *,
    environ: Mapping[str, str] | None = None,
    connector: Any = None,
    hold_local_cycles: bool = False,
) -> Dispatcher:
    selected_recipe = _require_non_empty(settings.recipe, "reef.recipe")
    env = os.environ if environ is None else environ
    recipe = _serving_recipe(selected_recipe, settings, env, connector)
    # A recipe's own evaluation calls come back to this Reef, so they sample the release it serves.
    served_url = settings.served_url or default_served_url(settings.host, settings.port)
    recipe = recipe.with_served_endpoint(
        ServedEndpoint(url=served_url, token=settings.tokens[0] if settings.tokens else None)
    )
    experiment_tracker = None
    scenario_storage: ScenarioStorage | None = None
    try:
        if settings.record_backend == "postgres":
            scenario_storage = PostgresScenarioStorage(
                _require_non_empty(settings.record_database_url, "reef.record_database_url"),
                Path(settings.agent_record_dir),
                schema=settings.record_database_schema,
            )
        else:
            scenario_storage = SQLiteScenarioStorage(Path(settings.agent_record_dir))
        # Record tracing observes storage events; the storage closes the observer with itself.
        record_observer = build_record_observer(settings.tracing_config, environ=env)
        if record_observer is not None:
            scenario_storage = ObservedScenarioStorage(scenario_storage, record_observer)
        # A harness recipe's seed is the base artifact, so a fresh scenario serves a tree before any step.
        backend_factory = GitLFSRepositoryBackend.factory(
            _repository_location(settings.artifact_repository),
            work_dir=Path(settings.artifact_work_dir),
            cache_dir=Path(settings.artifact_cache_dir),
            bootstrap_files=recipe.base_artifact_files(),
            bootstrap_subdirectory=recipe.bootstrap_artifact_component(),
        )
        if not isinstance(settings.training_settings, Mapping):
            raise ValueError("training must be an object")
        experiment_tracker = build_experiment_tracker(
            settings.wandb_config,
            model=settings.model_path,
            training_config=settings.training_settings,
        )
        return Dispatcher(
            recipe,
            backend_factory,
            local_artifact_dir=Path(settings.artifact_cache_dir) / "staged",
            agent_record_dir=Path(settings.agent_record_dir),
            scenario_storage=scenario_storage,
            allow_implicit_creation=settings.allow_implicit_scenario_creation,
            experiment_tracker=experiment_tracker,
            hold_local_cycles=hold_local_cycles,
        )
    except BaseException:
        if scenario_storage is not None:
            with suppress(Exception):
                scenario_storage.close()
        for component in (recipe.training_runtime, recipe.runtime):
            if component is not None:
                with suppress(Exception):
                    component.shutdown()
        if experiment_tracker is not None:
            with suppress(Exception):
                experiment_tracker.close()
        raise


def build_app(settings: ServiceConfig, *, environ: Mapping[str, str] | None = None, connector: Any = None) -> Any:
    record_retention = RecordRetention(settings.agent_record_retention_days, settings.agent_record_retention_max_bytes)
    retry_policy = InferenceRetryPolicy(
        initial_s=settings.inference_retry_initial_s,
        max_s=settings.inference_retry_max_s,
        timeout_s=settings.inference_retry_timeout_s,
    )
    # A recipe's model calls come back to this service, which answers only once the app listens: its local
    # cycles wait for that, so a harness step started by the preload never fails its calls into a skip.
    dispatcher = build_dispatcher(settings, environ=environ, connector=connector, hold_local_cycles=True)
    # No tokens (e.g. REEF_TOKEN="" in the environment) means no auth,
    # not auth with the empty string.
    try:
        return create_app(
            dispatcher,
            tokens=settings.tokens,
            console_origins=settings.console_origins,
            inference_retry_policy=retry_policy,
            close_dispatcher=True,
            record_retention=record_retention,
            open_local_cycles_at=default_served_url(settings.host, settings.port),
        )
    except BaseException:
        with suppress(Exception):
            dispatcher.close()
        raise
