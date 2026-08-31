"""Assemble the Reef HTTP service from settings: dispatcher, registry, app.

This is the service's composition logic — a :class:`ServiceSettings` in, a
running aiohttp application out. It knows nothing about the deployment config
format; ``reef.service.deploy`` translates YAML into these settings and
orchestrates processes around the result.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.observability import build_experiment_tracker
from reef.recipe import Recipe, WeightTrainingRecipe
from reef.recipe.config_fields import resolve_config_field_values
from reef.recipe.registry import RecipeRegistry, build_named_recipe, build_recipe, recipe_class_for
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.base import InferenceRuntime, TrainingRuntime
from reef.runtime.inference import InferenceBackendFactory
from reef.runtime.registry import RuntimeRegistry
from reef.service.app import InferenceRetryPolicy, create_app
from reef.service.deploy.settings import ServiceSettings, service_owned_keys


def _training_recipe_type(name: str) -> type[WeightTrainingRecipe] | None:
    """The registered class for ``name`` when it is a weight-training recipe."""
    recipe_type = recipe_class_for(name)
    if recipe_type is not None and issubclass(recipe_type, WeightTrainingRecipe):
        return recipe_type
    return None


def _recipe_owned_settings(settings: ServiceSettings) -> dict[str, Any]:
    """The flat ``reef.*`` keys that belong to the recipe, not the service.

    The service's own vocabulary is :class:`ServiceSettings`' fields plus the
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


def _configured_inference_backend_factory(path: str | None) -> InferenceBackendFactory | None:
    """Load an optional backend factory selected by deployment config."""

    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        raise ValueError("reef.inference_backend_factory must be a non-empty dotted path")
    module_path, separator, attribute = path.strip().rpartition(".")
    if not separator or not module_path or not attribute:
        raise ValueError("reef.inference_backend_factory must be a dotted path")
    try:
        factory = getattr(importlib.import_module(module_path), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load reef.inference_backend_factory {path!r}") from exc
    if not callable(factory):
        raise ValueError(f"reef.inference_backend_factory {path!r} is not callable")
    return factory


def _connect_training_runtime(
    settings: ServiceSettings,
    *,
    model_path: str,
    max_staleness: int,
    connector: Any = None,
) -> TrainingRuntime:
    """Build the training runtime through the runtime registry's ``ray_training`` kind."""
    ray_address = _require_non_empty(settings.ray_address, "reef.ray_address")
    inference_url = settings.inference_url.strip() if isinstance(settings.inference_url, str) else None
    if settings.inference_timeout_s <= 0:
        raise ValueError("reef.inference_timeout_s must be positive")
    if settings.train_timeout_s is not None and settings.train_timeout_s <= 0:
        raise ValueError("reef.train_timeout_s must be positive when set")
    runtime_config: dict[str, Any] = {
        "type": "ray_training",
        "inference_url": inference_url or None,
        "actor_name": settings.ray_actor_name,
        "namespace": settings.ray_namespace,
        "ray_address": ray_address,
        "inference_timeout_s": settings.inference_timeout_s,
        "train_timeout_s": settings.train_timeout_s,
    }
    if max_staleness:
        runtime_config["max_staleness"] = max_staleness
    inference_backend_factory = _configured_inference_backend_factory(settings.inference_backend_factory)
    if inference_backend_factory is not None:
        runtime_config["inference_backend_factory"] = inference_backend_factory
    if not isinstance(settings.inference_backend_config, Mapping):
        raise ValueError("reef.inference_backend_config must be an object")
    if settings.inference_backend_config:
        if inference_backend_factory is None:
            raise ValueError("reef.inference_backend_config requires reef.inference_backend_factory")
        runtime_config["inference_backend_config"] = dict(settings.inference_backend_config)
    if connector is not None:
        runtime_config["connect"] = connector
    runtime = RuntimeRegistry().build(runtime_config, model_path=model_path)
    if not isinstance(runtime, TrainingRuntime):
        raise TypeError("ray_training runtime factory must build a TrainingRuntime")
    return runtime


def _upstream_runtime(settings: ServiceSettings) -> InferenceRuntime | None:
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
    settings: ServiceSettings,
    env: Mapping[str, str],
    connector: Any,
) -> Recipe:
    """Build a weight-training recipe on the Ray training runtime the settings name."""
    model_path = _require_non_empty(settings.model_path, "reef.model_path")
    # The recipe owns its config fields: service_config parses the recipe-owned
    # slice of the flat reef section (rejecting keys no config field consumes)
    # and from_environment resolves them, so this builder never names
    # recipe-specific keys.
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
    return recipe_type.from_environment(env, config=recipe_config, runtime=runtime)


def _serving_recipe(selected: str, settings: ServiceSettings, env: Mapping[str, str], connector: Any) -> Recipe:
    """Build the one recipe ``reef.recipe`` names.

    The three spellings differ only in where the recipe's config and runtime
    come from: a weight-training kind reads the flat ``reef.*`` section and
    connects the Ray training runtime; a dotted kind is operator configuration
    built from the environment on the upstream proxy; any other name is a
    bundled kind or a YAML preset under ``REEF_RECIPE_CONFIG_DIR``, whose own
    ``runtime`` section wins over the upstream proxy.
    """
    training_recipe_type = _training_recipe_type(selected)
    if training_recipe_type is not None:
        return _training_recipe(training_recipe_type, settings, env, connector)
    if settings.evaluation_settings is not None:
        raise ValueError("the top-level evaluation section requires a weight-training recipe")
    if ":" in selected:
        return build_recipe(selected, env, runtime=_upstream_runtime(settings))
    return build_named_recipe(selected, env, default_runtime=_upstream_runtime(settings))


def build_dispatcher(
    settings: ServiceSettings, *, environ: Mapping[str, str] | None = None, connector: Any = None
) -> Dispatcher:
    selected_recipe = _require_non_empty(settings.recipe, "reef.recipe")
    env = os.environ if environ is None else environ
    backend_factory = GitLFSRepositoryBackend.factory(
        _repository_location(settings.artifact_repository),
        work_dir=Path(settings.artifact_work_dir),
        cache_dir=Path(settings.artifact_cache_dir),
    )
    recipe = _serving_recipe(selected_recipe, settings, env, connector)
    if not isinstance(settings.training_settings, Mapping):
        raise ValueError("training must be an object")
    experiment_tracker = build_experiment_tracker(
        settings.wandb_config,
        model=settings.model_path,
        training_config=settings.training_settings,
    )
    # A deployment serves one recipe, so its registry is closed over that
    # single entry and request-time names never materialize another. Scenarios
    # bind to the operator's public name; a dotted kind is not a request name,
    # so it serves under the recipe's own.
    name = recipe.name if ":" in selected_recipe else selected_recipe
    return Dispatcher(
        RecipeRegistry({name: recipe}),
        backend_factory,
        local_artifact_dir=Path(settings.artifact_cache_dir) / "staged",
        agent_record_dir=Path(settings.agent_record_dir),
        allow_implicit_creation=settings.allow_implicit_scenario_creation,
        experiment_tracker=experiment_tracker,
    )


def build_app(settings: ServiceSettings, *, environ: Mapping[str, str] | None = None, connector: Any = None) -> Any:
    retry_policy = InferenceRetryPolicy(
        initial_s=settings.inference_retry_initial_s,
        max_s=settings.inference_retry_max_s,
        timeout_s=settings.inference_retry_timeout_s,
    )
    dispatcher = build_dispatcher(settings, environ=environ, connector=connector)
    # No tokens (e.g. REEF_TOKEN="" in the environment) means no auth,
    # not auth with the empty string.
    return create_app(
        dispatcher,
        tokens=settings.tokens,
        inference_retry_policy=retry_policy,
        close_dispatcher=True,
    )
