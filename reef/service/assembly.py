"""Assemble the Reef HTTP service from settings: dispatcher, registry, app.

This is the service's composition logic — a :class:`ServiceSettings` in, a
running aiohttp application out. It knows nothing about the deployment config
format; ``reef.service.deploy`` translates YAML into these settings and
orchestrates processes around the result.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import signal
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from aiohttp import web

from reef.artifact.repository import ConfigurableRepositoryBackendFactory, RepositoryBackendContext
from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.dispatcher.served import wait_for_control_lease
from reef.evaluation.rollout_worker import EvaluationRolloutWorker, LocalCandidateEndpoints
from reef.evaluation.trials import TrialObserver
from reef.execution.workers import SERVICE_COMPONENTS, ServiceSet
from reef.inference.http import InferenceProxyRuntime, resolve_proxy_runtime
from reef.inference.interfaces import InferenceHandler, InferenceRuntime
from reef.inference.registry import RuntimeRegistry
from reef.observability import build_experiment_tracker, build_record_observer
from reef.recipe import Recipe, WeightTrainingRecipe
from reef.recipe.composite import CompositeRecipe
from reef.recipe.config_fields import resolve_config_field_values
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.recipe.loading import build_recipe, load_recipe, named_recipe_config, weight_training_selection
from reef.recipe.registry import recipe_class_for
from reef.scenario.admission import ScenarioAdmission, ScenarioLimits
from reef.scenario.live_objects import (
    LiveObjects,
    ServedEndpoint,
    check_recipe_objects,
    close_runtimes,
    harness_models,
)
from reef.service.app import create_app
from reef.service.artifact_evaluations import ArtifactEvaluations, NewArtifacts
from reef.service.artifact_imports import ArtifactImports
from reef.service.auth import evaluation_token_for
from reef.service.data_service import DataWorker, data_jobs
from reef.service.data_transfer import ArchiveWriters
from reef.service.deploy.service_config import ServiceSettings, service_owned_keys
from reef.service.deploy.serving import serving_runtime_config
from reef.service.deploy.training import training_deployment_for, training_recipe_config
from reef.service.harness_imports import HarnessSyncWorker
from reef.service.lifecycle import LifecycleSettings
from reef.service.request_service import RequestService
from reef.service.routed_inference import InferenceRetryPolicy
from reef.service.routing import RoutingAdapter
from reef.service.routing_preview import PreviewSettings
from reef.service.task_import import DataScenarioStorage, TaskImport
from reef.storage.coordination import Coordination, instance_id
from reef.storage.data import DataStore
from reef.storage.job_data import JobDataKind
from reef.storage.observer import ObservedScenarioStorage
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.records import RecordRetention
from reef.storage.scenario import ScenarioStorage
from reef.storage.segments import LocalSegmentStore, SegmentStore
from reef.storage.sql_coordination import open_coordination
from reef.storage.sql_data import PostgresDataStore, SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore, TaskFileStore
from reef.train.component import TrainingComponent, build_training_component
from reef.train.harness_backend.component import HarnessTrainingComponent
from reef.train.manager import TrainingManager


def _training_recipe_type(name: str) -> type[WeightTrainingRecipe] | None:
    """The explicit class for ``name`` when it is a weight-training recipe."""
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
    consumed by a recipe config field. ``training_recipe_config`` raises for
    any key the selected recipe does not declare, instead of letting the
    deployment silently run on recipe defaults.
    """
    service_owned = service_owned_keys()
    return {key: value for key, value in settings.recipe_settings.items() if key not in service_owned}


#: The factory that ``storage.artifact_backend: git_lfs`` selects, over ``storage.artifact_repository``.
GIT_LFS_ARTIFACT_BACKEND = "reef.artifact.git_lfs:GitLFSRepositoryBackendFactory"


def _require_non_empty(value: str | None, setting: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{setting} is required")
    return value.strip()


def connect_training_component(
    settings: ServiceSettings, *, model_path: str, default_parameters: Mapping[JobDataKind, Mapping[str, Any]]
) -> tuple[TrainingComponent, InferenceRuntime]:
    """Build the selected training component for the recipe's ``default_parameters``, then its snapshots' serving.

    The training backend builds only the component. The serving runtime comes
    from the deployment's ``inference`` settings and the component's
    ``artifact_format`` (``reef.service.deploy.serving``).

    The dispatcher puts the component behind the jobs queue (``Dispatcher``); the training service runs it.
    """
    backend = training_deployment_for(settings.training_backend)
    component = build_training_component(
        backend.component_config(asdict(settings), default_parameters=default_parameters), model_path=model_path
    )
    try:
        serving = RuntimeRegistry().build(
            serving_runtime_config(settings, backend, component.artifact_format), model_path=model_path
        )
    except BaseException:
        component.close()
        raise
    return component, serving


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
) -> tuple[Recipe, LiveObjects]:
    """Load a weight-training recipe, and connect the selected training component and its serving runtime."""
    model_path = _require_non_empty(settings.model_path, "reef.model_path")
    # Resolve the recipe fields once before connecting its runtime.
    # Construction consumes those same resolved values.
    recipe_config = training_recipe_config(recipe_type, _recipe_owned_settings(settings), model_path=model_path)
    if settings.activation_settings is not None:
        recipe_config["activation"] = dict(settings.activation_settings)
    if settings.pipeline_settings is not None:
        recipe_config["pipeline"] = dict(settings.pipeline_settings)
    resolved_recipe_data = resolve_config_field_values(recipe_type, recipe_config.get("data", {}), env)
    training_component, runtime = connect_training_component(
        settings,
        model_path=model_path,
        default_parameters=recipe_type.configured_parameters(resolved_recipe_data.get("job_parameters")),
    )
    try:
        live = LiveObjects(runtime=runtime, training_component=training_component)
        recipe = load_recipe(recipe_type, recipe_config, env, field_values=resolved_recipe_data)
        check_recipe_objects(recipe, live)
        return recipe, live
    except BaseException:
        close_runtimes((training_component,), runtime)
        raise


#: The keys a recipe that trains weights through a component reads at the top of its config.
COMPOSED_RECIPE_KEYS = frozenset({"components", "model", "data", "execution", "executors"})


def composed_training_recipe(
    selected: str,
    recipe_type: type[Recipe],
    config: dict[str, Any],
    weight_type: type[WeightTrainingRecipe],
    weight_config: Mapping[str, Any],
    settings: ServiceSettings,
    env: Mapping[str, str],
) -> tuple[Recipe, LiveObjects]:
    """Load a recipe that trains weights through one of its components, and connect the selected training
    component and its serving runtime.

    The deployment's model path is every component's model, and a top level evaluation section
    belongs to the component that trains.
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
            "components.<name>.config"
        )
    training_component, runtime = connect_training_component(
        settings,
        model_path=model_path,
        default_parameters=weight_type.configured_parameters(weight_config.get("job_parameters")),
    )
    config = {**config, "model": {**dict(config.get("model", {})), "path": model_path}}
    if settings.pipeline_settings is not None:
        # The component that trains weights keeps the pipeline section under its own config.
        components = dict(config["components"])
        for name, section in components.items():
            if recipe_class_for(section["implementation"]) is weight_type:
                components[name] = {
                    **section,
                    "config": {**section.get("config", {}), "pipeline": dict(settings.pipeline_settings)},
                }
        config = {**config, "components": components}
    if settings.activation_settings is not None:
        # The composite activates the artifacts of all its components with one rule.
        config = {**config, "activation": dict(settings.activation_settings)}
    try:
        live = LiveObjects(runtime=runtime, training_component=training_component)
        recipe = build_recipe(selected, env, config=config)
        check_recipe_objects(recipe, live)
        return recipe, live
    except BaseException:
        close_runtimes((training_component,), runtime)
        raise


def with_harness_components(recipe: Recipe, live: LiveObjects, environ: Mapping[str, str]) -> LiveObjects:
    """``live`` with the harness training component of the harness recipe in ``recipe``.

    For a ``HarnessEvolutionRecipe``, alone or as a component of a ``CompositeRecipe``, the result holds a new
    ``HarnessTrainingComponent``, built from the recipe's settings and the models of ``live``
    (``HarnessTrainingComponent.from_settings``, ``reef.scenario.live_objects.harness_models``); ``environ`` holds
    the variables that its sandbox and its agent's isolation read. For any other recipe, the result is ``live``.
    The served endpoint must be in ``live`` before the call, because the component reads the models once.

    Raise ``ConfigError`` for an ``evolution`` section that names no usable method, scorer, collector, executor or
    agent. Raise ``ValueError`` for a composite with two harness recipes: their jobs would share one component
    name.
    """
    harness: list[tuple[str | None, HarnessEvolutionRecipe]] = []
    if isinstance(recipe, CompositeRecipe):
        harness.extend(
            (name, child) for name, child in recipe.components.items() if isinstance(child, HarnessEvolutionRecipe)
        )
    elif isinstance(recipe, HarnessEvolutionRecipe):
        harness.append((None, recipe))
    if not harness:
        return live
    if len(harness) > 1:
        raise ValueError(f"components {[name for name, _ in harness]} each evolve a harness; a composite evolves one")
    component, harness_recipe = harness[0]
    return replace(
        live,
        harness_component=HarnessTrainingComponent.from_settings(
            harness_recipe.harness_settings(),
            harness_models(harness_recipe, live, component=component),
            environ=environ,
        ),
    )


def load_named_recipe(
    name: str,
    environ: Mapping[str, str] | None = None,
    *,
    config_directory: str | Path | None = None,
    default_runtime: InferenceRuntime | None = None,
    runtime_registry: RuntimeRegistry | None = None,
) -> tuple[Recipe, InferenceRuntime | None]:
    """Load the recipe that a deployment names by its public name, and build its runtime.

    ``name`` is a YAML preset or the core name ``recipe`` (``reef.recipe.loading.named_recipe_config``). A
    preset's ``runtime`` section builds the runtime; without one the runtime is ``default_runtime``, and the preset
    may omit ``model.path`` to use that runtime's model. Without either, the runtime is the proxy that
    ``REEF_UPSTREAM_URL`` names in ``environ``, else none. Raise ``RecipeConfigError`` for a preset with no model.
    """
    values = os.environ if environ is None else environ
    settings = named_recipe_config(name, values, config_directory=config_directory)
    if settings is None:
        return build_recipe(name, values), resolve_proxy_runtime(values, default_runtime)
    runtime_config = settings["runtime"]
    if "path" not in settings["model"] and not runtime_config and default_runtime is not None:
        settings["model"]["path"] = default_runtime.model_path
    model_path = settings["model"].get("path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise RecipeConfigError(
            f"recipe {name!r} requires a non-empty model (inference.upstream-model or inference.model-path) "
            "or a deployment runtime with a model"
        )
    # A configured runtime may connect a service or launch an executor's
    # workers. Presets without one borrow the deployment's default runtime.
    runtime = (
        (runtime_registry or RuntimeRegistry()).build(runtime_config, model_path=model_path, recipe_config=settings)
        if runtime_config
        else default_runtime
    )
    try:
        recipe = build_recipe(settings["implementation"], values, config=settings)
        runtime = resolve_proxy_runtime(values, runtime)
        check_recipe_objects(recipe, LiveObjects(runtime=runtime))
        return recipe, runtime
    except BaseException:
        if runtime_config:
            # The build error stays the one raised.
            close_runtimes((), runtime)
        raise


def load_deployment(settings: ServiceSettings, env: Mapping[str, str]) -> tuple[Recipe, LiveObjects]:
    """The deployment's recipe, and the runtime and the training component that every process of the deployment
    builds for it. The runtime is the same in each process, and only the inference service opens it
    (``InferenceRuntime.open``). The result has no harness component, data store or served endpoint yet: the caller
    adds what its process needs (``with_harness_components``)."""
    return _load_deployment(_require_non_empty(settings.recipe, "reef.recipe"), settings, env)


def _load_deployment(selected: str, settings: ServiceSettings, env: Mapping[str, str]) -> tuple[Recipe, LiveObjects]:
    """Load the one recipe ``reef.recipe`` names, and build its runtime and training component.

    The spellings differ only in where config and runtime come from: a dotted
    weight-training class reads the flat ``reef.*`` section and connects the
    Ray runtime; a dotted class that trains weights through one of its
    components connects the same runtime from that component's data; another
    dotted class is loaded from the environment on the upstream proxy; a bare
    name is ``recipe`` or a YAML preset under ``REEF_RECIPE_CONFIG_DIR``,
    whose own ``runtime`` section wins over the upstream proxy.
    """
    training_recipe_type = _training_recipe_type(selected)
    if training_recipe_type is not None:
        return _training_recipe(training_recipe_type, settings, env)
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
        weight_training = None if recipe_type is None else weight_training_selection(recipe_type, config)
        if weight_training is not None and recipe_type is not None:
            return composed_training_recipe(selected, recipe_type, config, *weight_training, settings, env)
        if settings.pipeline_settings is not None:
            config["pipeline"] = dict(settings.pipeline_settings)
        if settings.activation_settings is not None:
            config["activation"] = dict(settings.activation_settings)
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
            recipe = build_recipe(selected, env, config=config)
            return recipe, LiveObjects(runtime=resolve_proxy_runtime(env, runtime))
        except BaseException:
            close_runtimes((), runtime)
            raise
    if settings.pipeline_settings is not None or settings.activation_settings is not None:
        raise ValueError(
            f"recipe {selected!r} is a preset: set its pipeline and activation sections in the preset file"
        )
    recipe, runtime = load_named_recipe(selected, env, default_runtime=_upstream_runtime(settings))
    return recipe, LiveObjects(runtime=runtime)


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


def build_segment_store(settings: ServiceSettings) -> SegmentStore | None:
    """The segment store that ``storage.segment_backend`` names, in ``<agent_record_dir>/segments``, or None."""
    directory = Path(settings.agent_record_dir) / "segments"
    if settings.segment_backend is None:
        return None
    if settings.segment_backend == "local":
        return LocalSegmentStore(directory)
    reference = SegmentStore.reference(settings.segment_backend, setting="storage.segment_backend")
    return SegmentStore.referenced_class(reference).from_config(reference.config, directory=directory)


def build_data_store(settings: ServiceSettings, task_files: TaskFileStore) -> DataStore:
    """The data store that ``storage.data_store`` names, sharing ``task_files`` without owning it, and owning the
    segment store of ``storage.segment_backend``.

    Unset follows ``storage.record_backend``: ``postgres`` uses ``storage.record_database_url`` and the schema
    ``reef_data``, and ``sqlite`` or a plugin record backend uses ``sqlite``, in ``<agent_record_dir>/data``.
    """
    directory = Path(settings.agent_record_dir)
    choice = settings.data_store
    if choice is None:
        choice = "postgres" if settings.record_backend == "postgres" else "sqlite"
    segments = build_segment_store(settings)
    # Without a segment store, the job data directories stay in a directory that the services share.
    job_directories = segments if segments is not None else LocalSegmentStore(directory / "job-directories")
    ttl_seconds = settings.job_directory_ttl_s
    try:
        if choice == "postgres":
            return PostgresDataStore(
                _require_non_empty(settings.record_database_url, "reef.record_database_url"),
                task_files,
                segments=segments,
                job_directories=job_directories,
                job_directory_ttl_seconds=ttl_seconds,
            )
        if choice == "sqlite":
            return SQLiteDataStore(directory / "data", task_files, segments, job_directories, ttl_seconds)
        reference = DataStore.reference(choice, setting="storage.data_store")
        return DataStore.referenced_class(reference).from_config(
            reference.config,
            directory=directory,
            task_files=task_files,
            segments=segments,
            job_directory_ttl_seconds=ttl_seconds,
        )
    except BaseException:
        # The data store owns the segment store after it exists. Until then, this code closes it.
        if segments is not None:
            segments.close()
        raise


def open_service_coordination(settings: ServiceSettings) -> Coordination:
    """The deployment's coordination store: on ``storage.coordination_backend``, which defaults to the built-in
    record backend, with the record database and schema of ``settings``."""
    backend = settings.coordination_backend
    if backend is None:
        if not isinstance(settings.record_backend, str) or settings.record_backend not in ("sqlite", "postgres"):
            raise ValueError("a plugin storage.record_backend needs storage.coordination_backend")
        backend = settings.record_backend
    return open_coordination(
        backend,
        agent_record_dir=settings.agent_record_dir,
        record_database_url=settings.record_database_url,
        record_database_schema=settings.record_database_schema,
        default_attempts=settings.work_max_attempts,
    )


def build_task_files(settings: ServiceSettings) -> TaskFileStore:
    """The task file store that ``storage.task_file_backend`` names, in ``storage.task_file_dir``."""
    task_file_directory = Path(settings.task_file_dir)
    if settings.task_file_backend == "local":
        return LocalTaskFileStore(task_file_directory)
    reference = TaskFileStore.reference(settings.task_file_backend, setting="storage.task_file_backend")
    return TaskFileStore.referenced_class(reference).from_config(reference.config, directory=task_file_directory)


def open_scenario_storage(settings: ServiceSettings) -> tuple[ScenarioStorage, DataStore]:
    """The deployment's scenario storage (``storage.record_backend``), which owns the task file store, and the data
    store (``build_data_store``), which keeps the scenarios' records. The caller closes the data store before the
    storage, because the data store shares the task file store."""
    record_directory = Path(settings.agent_record_dir)
    task_files = build_task_files(settings)
    data: DataStore | None = None
    try:
        data = build_data_store(settings, task_files)
        if settings.record_backend == "postgres":
            storage: ScenarioStorage = PostgresScenarioStorage(
                _require_non_empty(settings.record_database_url, "reef.record_database_url"),
                record_directory,
                schema=settings.record_database_schema,
                task_files=task_files,
                data=data,
            )
        elif settings.record_backend == "sqlite":
            storage = SQLiteScenarioStorage(record_directory, task_files=task_files, data=data)
        else:
            reference = ScenarioStorage.reference(settings.record_backend, setting="storage.record_backend")
            storage_class = ScenarioStorage.referenced_class(reference)
            storage = storage_class.from_config(
                reference.config, directory=record_directory, task_files=task_files, data=data
            )
    except BaseException:
        # The storage owns the task file store after it exists. Until then, this code closes it.
        if data is not None:
            data.close()
        task_files.close()
        raise
    return storage, data


def build_artifact_backend(settings: ServiceSettings) -> ConfigurableRepositoryBackendFactory:
    """Build the artifact backend shared by control and inference."""
    artifact_backend = (
        {"module": GIT_LFS_ARTIFACT_BACKEND, "config": {"repository": settings.artifact_repository}}
        if settings.artifact_backend == "git_lfs"
        else settings.artifact_backend
    )
    reference = ConfigurableRepositoryBackendFactory.reference(artifact_backend, setting="storage.artifact_backend")
    factory_class = ConfigurableRepositoryBackendFactory.referenced_class(reference)
    context = RepositoryBackendContext(
        work_dir=Path(settings.artifact_work_dir), cache_dir=Path(settings.artifact_cache_dir)
    )
    try:
        backend_factory = factory_class.from_config(reference.config, context=context)
    except Exception as error:
        error.add_note(f"while building storage.artifact_backend {factory_class.__name__}")
        raise
    return backend_factory


def build_dispatcher(
    settings: ServiceSettings,
    *,
    environ: Mapping[str, str] | None = None,
    hold_local_cycles: bool = False,
    evaluation_token: str | None = None,
    new_artifacts: NewArtifacts | None = None,
    components: frozenset[str] = frozenset(SERVICE_COMPONENTS),
) -> Dispatcher:
    """Build the dispatcher of a service.

    ``new_artifacts`` observes the scenario storage's commits, so the service
    evaluates each new artifact with its scenario's settings.

    A composite recipe's evaluation calls come back to this service, so a
    candidate of one component is evaluated against the snapshot the others
    serve; ``evaluation_token`` is the token they present, one the app accepts
    on the evaluation routes alone, since those calls run candidate code, and
    ``hold_local_cycles`` holds its local cycles until the service answers. A
    recipe of one component calls its runtime's endpoint directly, as it did
    before composites, and neither applies to it.

    ``components`` names the services that this process runs (``SERVICE_COMPONENTS``); the default is all of them,
    the one-process layout.
    """
    selected_recipe = _require_non_empty(settings.recipe, "reef.recipe")
    env = os.environ if environ is None else environ
    coordination = open_service_coordination(settings)
    try:
        recipe, live = _load_deployment(selected_recipe, settings, env)
    except BaseException:
        coordination.close()
        raise
    calls_service = isinstance(recipe, CompositeRecipe)
    if calls_service:
        # A composite's evaluation calls come back to this Reef, so they sample the snapshot it serves.
        served_url = settings.served_url or default_served_url(settings.host, settings.port)
        live = replace(live, served_endpoint=ServedEndpoint(url=served_url, token=evaluation_token))
    experiment_tracker = None
    scenario_storage: ScenarioStorage | None = None
    data_store: DataStore | None = None
    dispatcher: Dispatcher | None = None
    try:
        # The harness training component reads the served endpoint, so it is built after ``live`` has it.
        live = with_harness_components(recipe, live, env)
        scenario_storage, data_store = open_scenario_storage(settings)
        live = replace(live, data_store=data_store)
        task_import = (
            TaskImport(Path(settings.task_import_manifest), Path(settings.task_import_tasks_root))
            if settings.task_import_manifest is not None and settings.task_import_tasks_root is not None
            else None
        )
        # Record tracing observes storage events; the storage closes the observer with itself.
        record_observer = build_record_observer(settings.tracing_config, environ=env)
        if record_observer is not None:
            scenario_storage = ObservedScenarioStorage(scenario_storage, record_observer)
        if new_artifacts is not None:
            scenario_storage = ObservedScenarioStorage(scenario_storage, new_artifacts)
        # The inference path and the data service append and read records without opening a scenario.
        record_storage = scenario_storage
        # Each scenario's session reads its training samples from the data store.
        scenario_storage = DataScenarioStorage(scenario_storage, data_store, task_import)
        backend_factory = build_artifact_backend(settings)
        if not isinstance(settings.training_settings, Mapping):
            raise ValueError("training must be an object")
        experiment_tracker = build_experiment_tracker(
            settings.wandb_config,
            model=settings.model_path,
            training_config=settings.training_settings,
        )
        dispatcher = Dispatcher(
            recipe,
            backend_factory,
            live=live,
            local_artifact_dir=Path(settings.artifact_cache_dir) / "staged",
            scenario_storage=scenario_storage,
            record_storage=record_storage,
            allow_implicit_creation=settings.allow_implicit_scenario_creation,
            experiment_tracker=experiment_tracker,
            hold_local_cycles=hold_local_cycles and calls_service,
            scenario_idle_unload_seconds=settings.scenario_idle_unload_s,
            scenario_admission=(
                ScenarioLimits(
                    max_loaded_scenarios=settings.max_loaded_scenarios, max_scenarios=settings.max_scenarios
                )
                if settings.scenario_admission is None
                else ScenarioAdmission.from_setting(settings.scenario_admission, setting="reef.scenario_admission")
            ),
            coordination=coordination,
            # A standby waits here, before it loads or serves anything, until the active instance stops.
            control_lease=(
                wait_for_control_lease(coordination.control_lease, instance_id("control"))
                if ServiceSet.of(components).runs_control
                else None
            ),
            components=components,
            work_retention_seconds=settings.work_retention_s,
            training_max_jobs=settings.training_max_jobs,
            training_manager=(
                None
                if settings.training_manager is None
                else TrainingManager.from_setting(settings.training_manager, setting="training.manager")
            ),
        )
        if dispatcher.services.runs_evaluation:
            # The worker serves each rollout's candidate itself, from the scenario's repository.
            dispatcher.workers.add(
                EvaluationRolloutWorker(
                    dispatcher.queue,
                    LocalCandidateEndpoints(dispatcher),
                    worker_id=instance_id("evaluation"),
                    trial_observer=(
                        None
                        if settings.evaluation_trial_observer is None
                        else TrialObserver.from_setting(
                            settings.evaluation_trial_observer, setting="evaluation.trial_observer"
                        )
                    ),
                    agent_host=settings.evaluation_agent_host,
                    slots=settings.evaluation_max_sandboxes,
                )
            )
        if dispatcher.services.runs_control and not dispatcher.services.runs_harness and recipe.harness_adapter:
            # The harness service runs in another process: it sends each import here to commit.
            dispatcher.workers.add(
                HarnessSyncWorker(dispatcher.queue, RequestService(dispatcher), worker_id=instance_id("harness-sync"))
            )
        if dispatcher.services.runs_data and data_store is not None:
            dispatcher.workers.add(
                DataWorker(
                    dispatcher.queue,
                    data_jobs(dispatcher, ArtifactSources.from_setting(settings.artifact_sources), ArchiveWriters()),
                    worker_id=instance_id("data"),
                )
            )
        return dispatcher
    except BaseException:
        if dispatcher is not None:
            # The dispatcher owns these resources now; its workers must stop before their stores close.
            with suppress(Exception):
                dispatcher.close()
        else:
            # The data store first: the scenario storage closes the task file store that it shares.
            if data_store is not None:
                with suppress(Exception):
                    data_store.close()
            if scenario_storage is not None:
                with suppress(Exception):
                    scenario_storage.close()
            close_runtimes(live.job_components(), live.runtime)
            if experiment_tracker is not None:
                with suppress(Exception):
                    experiment_tracker.close()
            coordination.close()
        raise


def build_app(
    settings: ServiceSettings,
    *,
    environ: Mapping[str, str] | None = None,
    inference_handler: InferenceHandler | None = None,
    lifecycle: LifecycleSettings | None = None,
    components: frozenset[str] = frozenset(SERVICE_COMPONENTS),
) -> web.Application:
    """Build the complete HTTP service from settings: dispatcher, retention, retries, tokens and cleanup.

    The app owns the dispatcher it builds and closes it on cleanup.
    ``inference_handler`` replaces the recipe runtime's handler for every
    inference request; the caller owns it and releases it after the app stops.
    An embedding application adds its middleware and routes to the returned
    app before startup and reaches the dispatcher with ``get_dispatcher(app)``.

    With ``reef.routing_manifest``, ``lifecycle`` is required: ``reef serve`` starts this service as the
    closed child of a managed routing deployment and hands it the router's selection credential.

    ``components`` names the services that this process runs besides ``control``. With ``evaluation``, this
    process's evaluation worker serves each candidate itself; without it, the evaluation service runs in another
    process.
    """
    routing = None
    preview = None
    retry_policy = None
    if settings.routing_manifest is not None:
        if lifecycle is None or not lifecycle.selection_token or not lifecycle.control_socket:
            raise ValueError(
                "reef.routing_manifest runs only as a managed routing deployment started by reef serve, which "
                "supplies the Semantic Router selection credential and control socket"
            )
        policy = settings.routing_deployment
        if settings.port == policy.sr_management_port:
            raise ValueError("reef.port must differ from routing sr_management_port")
        routing = RoutingAdapter.from_manifest(
            settings.routing_manifest,
            router_config_hash=lifecycle.router_config_hash or None,
            environ=environ,
            timeout_s=settings.inference_timeout_s,
        )
        preview = PreviewSettings(
            f"http://127.0.0.1:{policy.sr_management_port}",
            lifecycle.selection_token,
            policy.selection_timeout_seconds,
        )
        retry_policy = InferenceRetryPolicy(
            initial_s=settings.inference_retry_initial_s,
            max_s=settings.inference_retry_max_s,
            timeout_s=settings.inference_retry_timeout_s,
        )
    record_retention = RecordRetention(max_bytes=settings.agent_record_retention_max_bytes)
    # A composite's model calls come back to this service, which answers only once the app listens: its local
    # cycles wait for that, so a harness job started by the preload never fails its calls into a skip.
    # Evaluation calls run candidate code, so they get a token of their own that opens the evaluation routes alone,
    # never a service token. Every process derives the same one from the service tokens.
    evaluation_token = evaluation_token_for(settings.tokens)
    new_artifacts = NewArtifacts()
    dispatcher = build_dispatcher(
        settings,
        environ=environ,
        hold_local_cycles=True,
        evaluation_token=evaluation_token,
        new_artifacts=new_artifacts,
        components=components,
    )
    evaluations = ArtifactEvaluations(
        dispatcher, new_artifacts=new_artifacts, sources=ArtifactSources.from_setting(settings.artifact_sources)
    )
    # No tokens (e.g. REEF_TOKEN="" in the environment) means no auth,
    # not auth with the empty string.
    try:
        app = create_app(
            dispatcher,
            tokens=settings.tokens,
            evaluation_tokens=evaluation_token,
            console_origins=settings.console_origins,
            inference_handler=inference_handler,
            close_dispatcher=True,
            record_retention=record_retention,
            open_local_cycles_at=default_served_url(settings.host, settings.port),
            routing=routing,
            lifecycle=lifecycle,
            preview=preview,
            inference_retry_policy=retry_policy,
            evaluations=evaluations,
            artifact_imports=ArtifactImports(
                ArtifactSources.from_setting(settings.artifact_sources),
                Path(settings.artifact_cache_dir) / "imports",
            ),
            harness_inference_url=settings.harness_inference_url,
        )
    except BaseException:
        with suppress(Exception):
            dispatcher.close()
        raise
    app.cleanup_ctx.append(EndOnLeaseLoss(dispatcher).run)
    return app


class EndOnLeaseLoss:
    """Ends the process of a control instance that lost the control lease, with SIGTERM, so that its supervisor
    starts it again as a standby.

    Values: ``dispatcher`` sets ``lease_lost``; ``poll_seconds`` is the time between two looks at it. Purpose: the
    app that ``build_app`` returns runs ``run`` for its life (``cleanup_ctx``); an embedded dispatcher, which no
    process owns, never ends a process.
    """

    def __init__(self, dispatcher: Dispatcher, *, poll_seconds: float = 1.0) -> None:
        self.dispatcher = dispatcher
        self.poll_seconds = poll_seconds

    async def run(self, app: web.Application) -> AsyncIterator[None]:
        task = asyncio.create_task(self.watch())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def watch(self) -> None:
        while not self.dispatcher.lease_lost.is_set():
            await asyncio.sleep(self.poll_seconds)
        signal.raise_signal(signal.SIGTERM)
