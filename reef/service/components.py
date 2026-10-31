"""``reef serve --component NAME``: one service of a deployment in one process.

Each service reads and writes the shared store of the deployment (``storage.record_backend``): PostgreSQL for
processes on several hosts, SQLite for processes on one host. The services are:

- ``control``: the Reef HTTP API except the data routes, scenarios, commits and activation. It writes
  ``served_models``, adds the work of the queues, and syncs loaded scenarios after data writes.
- ``inference``: answers model requests from ``served_models`` and appends their records
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

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping

from aiohttp import web

from reef.execution.workers import SERVICE_COMPONENTS
from reef.observability import build_record_observer
from reef.runtime.deployment import close_runtimes
from reef.service.app import create_data_app, create_harness_app
from reef.service.assembly import (
    build_app,
    build_dispatcher,
    open_scenario_storage,
    open_service_coordination,
    serving_recipe,
)
from reef.service.auth import evaluation_token_for
from reef.service.deploy.service_config import ServiceConfig
from reef.service.engine_clients import DeploymentEngineClients
from reef.service.inference_service import build_inference_app
from reef.service.web import service_application
from reef.storage.observer import ObservedScenarioStorage


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
    """The inference service. It builds the same recipe as control's, whose runtime this process never opens: it
    reaches each engine with the runtime's own request handler, as control does, and changes no engine."""

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        recipe = serving_recipe(settings, environ)
        runtime = recipe.runtime
        coordination = open_service_coordination(settings)
        try:
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

        async def close_inference(app: web.Application) -> None:
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


class QueueServiceApp(ComponentApp):
    """A service that only takes work from a queue (samples, training, evaluation): a dispatcher that runs only the
    service's own workers on the shared store, and answers ``/healthz`` alone. The evaluation service loads each
    rollout's scenario on demand, resolves the candidate artifact and serves it itself.

    Values: ``name`` is the service.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def build(self, settings: ServiceConfig, environ: Mapping[str, str]) -> web.Application:
        dispatcher = build_dispatcher(settings, environ=environ, components=frozenset({self.name}))
        if dispatcher.services.runs_training and not dispatcher.recipe.job_components():
            dispatcher.close()
            raise ValueError(f"recipe {settings.recipe!r} submits no training job, so it has no training service")
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
    "training": QueueServiceApp("training"),
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
