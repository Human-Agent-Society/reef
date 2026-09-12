"""Assemble the Reef HTTP service from settings: dispatcher, registry, app.

This is the service's composition logic — a :class:`ServiceConfig` in, a
running aiohttp application out. It knows nothing about the deployment config
format; ``reef.service.deploy`` translates YAML into these settings and
orchestrates processes around the result.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.observability import build_experiment_tracker
from reef.recipe import Recipe, WeightTrainingRecipe
from reef.recipe.config_fields import resolve_config_field_values
from reef.recipe.registry import build_named_recipe, build_recipe, recipe_class_for
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.base import InferenceRuntime, TrainingRuntime
from reef.runtime.registry import RuntimeRegistry
from reef.runtime.settings import TrainingRuntimeSettings
from reef.service.app import InferenceRetryPolicy, create_app
from reef.service.deploy.service_config import ServiceConfig, service_owned_keys
from reef.service.deploy.training import training_deployment_for
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
) -> TrainingRuntime:
    """Build the selected integration's runtime, independently of its process topology."""
    TrainingRuntimeSettings(
        inference_timeout_s=settings.inference_timeout_s,
        train_timeout_s=settings.train_timeout_s,
        max_staleness=max_staleness,
    )
    backend = training_deployment_for(settings.training_backend)
    runtime_config = backend.runtime_config(asdict(settings), max_staleness=max_staleness, connector=connector)
    runtime = RuntimeRegistry().build(runtime_config, model_path=model_path)
    if not isinstance(runtime, TrainingRuntime):
        with suppress(Exception):
            runtime.shutdown()
        raise TypeError("training backend runtime factory must build a TrainingRuntime")
    return runtime


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
    runtime = _connect_training_runtime(
        settings,
        model_path=model_path,
        max_staleness=resolved_recipe_data["max_staleness"],
        connector=connector,
    )
    try:
        return recipe_type.from_resolved_config(recipe_config, resolved_recipe_data, environ=env, runtime=runtime)
    except BaseException:
        with suppress(Exception):
            runtime.shutdown()
        raise


def _serving_recipe(selected: str, settings: ServiceConfig, env: Mapping[str, str], connector: Any) -> Recipe:
    """Build the one recipe ``reef.recipe`` names.

    The spellings differ only in where config and runtime come from: a dotted
    weight-training class reads the flat ``reef.*`` section and connects the
    Ray runtime; another dotted class is built from the environment on the
    upstream proxy; a bare name is ``recipe`` or a YAML preset under
    ``REEF_RECIPE_CONFIG_DIR``, whose own ``runtime`` section wins over the
    upstream proxy.
    """
    training_recipe_type = _training_recipe_type(selected)
    if training_recipe_type is not None:
        return _training_recipe(training_recipe_type, settings, env, connector)
    if settings.evaluation_settings is not None:
        raise ValueError("the top-level evaluation section requires a weight-training recipe")
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
        try:
            return build_recipe(selected, env, config=config, runtime=runtime)
        except BaseException:
            if runtime is not None:
                with suppress(Exception):
                    runtime.shutdown()
            raise
    return build_named_recipe(
        selected, env, default_runtime=_upstream_runtime(settings), preset_config=settings.preset_config
    )


def build_dispatcher(
    settings: ServiceConfig, *, environ: Mapping[str, str] | None = None, connector: Any = None
) -> Dispatcher:
    selected_recipe = _require_non_empty(settings.recipe, "reef.recipe")
    env = os.environ if environ is None else environ
    recipe = _serving_recipe(selected_recipe, settings, env, connector)
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
        # A harness recipe's seed is the base artifact, so a fresh scenario serves a tree before any step.
        backend_factory = GitLFSRepositoryBackend.factory(
            _repository_location(settings.artifact_repository),
            work_dir=Path(settings.artifact_work_dir),
            cache_dir=Path(settings.artifact_cache_dir),
            bootstrap_files=recipe.base_artifact_files(),
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
        )
    except BaseException:
        if scenario_storage is not None:
            with suppress(Exception):
                scenario_storage.close()
        if recipe.runtime is not None:
            with suppress(Exception):
                recipe.runtime.shutdown()
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
    dispatcher = build_dispatcher(settings, environ=environ, connector=connector)
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
        )
    except BaseException:
        with suppress(Exception):
            dispatcher.close()
        raise
