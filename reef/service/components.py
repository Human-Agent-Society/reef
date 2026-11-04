"""``reef serve --component NAME``: one service of a deployment in one process.

Each service reads and writes the shared store of the deployment (``storage.record_backend``): PostgreSQL for
processes on several hosts, SQLite for processes on one host. The services are:

- ``control``: the Reef HTTP API except the data routes, scenarios, commits and activation. It queues
  artifact startup and syncs loaded scenarios after data writes.
- ``inference``: starts selected artifacts, writes ``served_models``, answers model requests, and appends their records
  (``reef.service.inference_service``).
- ``samples``: makes the samples of record selections from the samples queue.
- ``training``: runs the jobs of the jobs queue on the training component.
- ``evaluation``: runs the evaluation rollouts of the evaluations queue; it serves each candidate itself.
- ``data``: the data routes and the reads of each scenario, record copies, imports and exports
  (``reef.service.data_service``). It needs no control lease.
- ``harness``: the routes that a harness client calls (``reef.service.routes.harness``); control commits each
  import (``reef.service.harness_imports``). It needs no control lease.

A service that only takes work from a queue answers ``/healthz`` alone. ``reef serve`` without ``--component`` runs
all of them in one process, on the same interfaces.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from aiohttp import web

from reef.dispatcher.serving import ServingWorker
from reef.execution.workers import SERVICE_COMPONENTS
from reef.observability import build_record_observer
from reef.recipe.base import ServedEndpoint, close_runtimes
from reef.recipe.composite import CompositeRecipe
from reef.service.app import create_data_app, create_harness_app
from reef.service.assembly import (
    build_app,
    build_artifact_backend,
    build_dispatcher,
    build_job_files,
    build_segment_store,
    default_served_url,
    open_scenario_storage,
    open_service_coordination,
    serving_recipe,
)
from reef.service.auth import evaluation_token_for
from reef.service.deploy.service_config import ServiceConfig
from reef.service.engine_clients import DeploymentEngineClients
from reef.service.inference_service import build_inference_app
from reef.service.web import service_application
from reef.storage.coordination import Coordination, instance_id
from reef.storage.observer import ObservedScenarioStorage
from reef.storage.segments import SegmentStore
from reef.train.job_queue import TrainingJobWorker


class ComponentApp(ABC):
    """How ``reef serve --component NAME`` builds the app of one service.

    Values: ``build(settings, environ)`` returns the app of the service, which owns what it opens and closes it at
    cleanup. Purpose: ``COMPONENT_APPS`` holds one per service; ``build_component_app`` calls it.
    """

    @abstractmethod
    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application: ...


class ControlApp(ComponentApp):
    """Control: the Reef HTTP API except the data routes."""

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        return build_app(settings, environ=environ, components=frozenset({"control"}))


class InferenceApp(ComponentApp):
    """The inference service owns runtime startup and serves requests from published routes."""

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        recipe = serving_recipe(settings, environ)
        runtime = recipe.runtime
        coordination = open_service_coordination(settings)
        try:
            backends = build_artifact_backend(settings) if runtime is not None and runtime.serves_requests else None
            storage, _ = open_scenario_storage(settings)
            # Records that this service appends are traced as the records that control appends are.
            record_observer = build_record_observer(settings.tracing_config, environ=environ)
            if record_observer is not None:
                storage = ObservedScenarioStorage(storage, record_observer)
        except BaseException:
            coordination.close()
            close_runtimes(recipe.job_components(), runtime)
            raise
        timeout_seconds = settings.inference_timeout_s if runtime is None else runtime.inference_timeout_s
        app = build_inference_app(
            coordination.served_models,
            storage,
            DeploymentEngineClients(
                runtime, provider_key=settings.upstream_api_key or None, timeout_seconds=timeout_seconds
            ),
            tokens=settings.tokens,
            evaluation_tokens=evaluation_token_for(settings.tokens),
            timeout_seconds=timeout_seconds,
            for_codex=recipe.harness_adapter == "codex",
        )

        worker = None
        if backends is not None:
            worker = ServingWorker(
                recipe,
                backends,
                coordination,
                worker_id=instance_id("inference-startup"),
                local_artifact_dir=Path(settings.artifact_cache_dir) / "staged",
            )
            worker.start()

        async def close_inference(app: web.Application) -> None:
            if worker is not None:
                await asyncio.to_thread(worker.stop)
            storage.close()
            coordination.close()
            close_runtimes(recipe.job_components(), runtime)

        app.on_cleanup.append(close_inference)
        return app


class DataApp(ComponentApp):
    """The data service: a dispatcher that loads each scenario on demand, for the checks of a write and for the
    reads, and runs the data workers."""

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        return create_data_app(
            build_dispatcher(settings, environ=environ, components=frozenset({"data"})),
            tokens=settings.tokens,
            console_origins=settings.console_origins,
        )


class HarnessApp(ComponentApp):
    """The harness service: a dispatcher that loads each scenario on demand, without activating it, and serves the
    routes that a harness client calls."""

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        dispatcher = build_dispatcher(settings, environ=environ, components=frozenset({"harness"}))
        if not dispatcher.recipe.harness_adapter:
            dispatcher.close()
            raise ValueError(f"recipe {settings.recipe!r} evolves no harness, so it has no harness service")
        return create_harness_app(
            dispatcher,
            tokens=settings.tokens,
            console_origins=settings.console_origins,
            inference_url=settings.harness_inference_url,
        )


class TrainingApp(ComponentApp):
    """Build the training service from its job components and coordination store.

    Values: this stateless builder returns an app with one worker per recipe job component, and refuses a recipe
    with no job component. The app owns these workers, their components, the runtime and the coordination store.
    Purpose: ``build_component_app("training", ...)`` runs jobs without opening scenario, artifact or data stores. It
    reads each job's data directory from the data store's job files (``build_job_files``).
    """

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        recipe = serving_recipe(settings, environ)
        if isinstance(recipe, CompositeRecipe):
            recipe = recipe.with_served_endpoint(
                ServedEndpoint(url=settings.served_url or default_served_url(settings.host, settings.port), token=None)
            )
        components = recipe.job_components()
        runtime = recipe.runtime
        coordination: Coordination | None = None
        job_files: SegmentStore | None = None
        workers: list[TrainingJobWorker] = []

        def close_training() -> None:
            errors: list[BaseException] = []
            unclosed_components = list(components[len(workers) :])
            for worker in reversed(workers):
                try:
                    worker.stop()
                except BaseException as exc:
                    errors.append(exc)
                    # A failed stop can precede component.close, for example when its thread never started.
                    unclosed_components.append(worker.component)
            try:
                errors.extend(close_runtimes(unclosed_components, runtime))
            except BaseException as exc:
                errors.append(exc)
            for store in (coordination, job_files):
                if store is not None:
                    try:
                        store.close()
                    except BaseException as exc:
                        errors.append(exc)
            if errors:
                raise errors[0]

        try:
            if not components:
                raise ValueError(f"recipe {settings.recipe!r} submits no training job, so it has no training service")
            coordination = open_service_coordination(settings)
            job_files = build_job_files(settings, build_segment_store(settings))
            workers.extend(
                TrainingJobWorker(
                    component,
                    coordination.work_queue,
                    job_files,
                    worker_id=instance_id(f"training-{component.name}"),
                )
                for component in components
            )
            for worker in workers:
                worker.start()
            app = service_application(settings.tokens)
        except BaseException:
            with suppress(Exception):
                close_training()
            raise

        async def cleanup(app: web.Application) -> None:
            await asyncio.to_thread(close_training)

        app.on_cleanup.append(cleanup)
        return app


class QueueServiceApp(ComponentApp):
    """A service that only takes work from a queue (samples, evaluation): a dispatcher that runs only the
    service's own workers on the shared store, and answers ``/healthz`` alone. The evaluation service loads each
    rollout's scenario on demand, resolves the candidate artifact and serves it itself.

    Values: ``name`` is the service.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        dispatcher = build_dispatcher(settings, environ=environ, components=frozenset({self.name}))
        app = service_application(settings.tokens)

        async def close_dispatcher(app: web.Application) -> None:
            dispatcher.close()

        app.on_cleanup.append(close_dispatcher)
        return app


#: The app builder of each service.
COMPONENT_APPS: dict[str, ComponentApp] = {
    "control": ControlApp(),
    "inference": InferenceApp(),
    "samples": QueueServiceApp("samples"),
    "training": TrainingApp(),
    "evaluation": QueueServiceApp("evaluation"),
    "data": DataApp(),
    "harness": HarnessApp(),
}


def build_component_app(
    component: str,
    settings: ServiceConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> web.Application:
    """The app of the service ``component`` (one of ``SERVICE_COMPONENTS``), which owns what it opens.

    Raise ``ValueError`` for an unknown service.
    """
    builder = COMPONENT_APPS.get(component)
    if builder is None:
        raise ValueError(f"unknown service {component!r}; the services are {SERVICE_COMPONENTS}")
    return builder.build(settings, os.environ if environ is None else environ)


__all__ = ["COMPONENT_APPS", "ComponentApp", "build_component_app"]
