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

A service that only takes work from a queue answers ``/healthz`` alone. ``reef serve`` without ``--component`` runs
all of them in one process, on the same interfaces.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from aiohttp import web

from reef.evaluation.rollout_worker import EvaluationRolloutWorker
from reef.execution.workers import SERVICE_COMPONENTS, ServiceWorkers
from reef.observability import build_record_observer
from reef.runtime.deployment import close_runtimes
from reef.service.app import create_data_app
from reef.service.artifact_evaluations import LOCAL_ROLLOUT_WORKERS, LocalCandidateEndpoints
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
from reef.storage.coordination import instance_id
from reef.storage.observer import ObservedScenarioStorage


def build_component_app(
    component: str,
    settings: ServiceConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> web.Application:
    """The app of the service ``component`` (one of ``SERVICE_COMPONENTS``), which owns what it opens.

    Raise ``ValueError`` for an unknown service.
    """
    env = os.environ if environ is None else environ
    if component not in SERVICE_COMPONENTS:
        raise ValueError(f"unknown service {component!r}; the services are {SERVICE_COMPONENTS}")
    if component == "control":
        return build_app(settings, environ=env, components=frozenset({"control"}))
    if component == "inference":
        # The same recipe as control's, whose runtime this process never opens: it reaches each engine with the
        # runtime's own request handler, as control does, and changes no engine.
        recipe = serving_recipe(settings, env)
        runtime = recipe.runtime
        coordination = open_service_coordination(settings)
        try:
            storage, _ = open_scenario_storage(settings)
            # Records that this service appends are traced as the records that control appends are.
            record_observer = build_record_observer(settings.tracing_config, environ=env)
            if record_observer is not None:
                storage = ObservedScenarioStorage(storage, record_observer)
        except BaseException:
            coordination.close()
            close_runtimes(recipe.training_component, runtime)
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
            close_runtimes(recipe.training_component, runtime)

        app.on_cleanup.append(close_inference)
        return app
    if component == "data":
        # The data service is a dispatcher too: it loads each scenario on demand, for the checks of a write and for
        # the reads, and runs the data workers.
        return create_data_app(
            build_dispatcher(settings, environ=env, components=frozenset({"data"})),
            tokens=settings.tokens,
            console_origins=settings.console_origins,
        )
    if component in ("samples", "training"):
        # A queue service is a dispatcher that runs only its own workers on the shared store.
        dispatcher = build_dispatcher(settings, environ=env, components=frozenset({component}))
        if component == "training" and dispatcher.recipe.training_component is None:
            dispatcher.close()
            raise ValueError(f"recipe {settings.recipe!r} trains no weights, so it has no training service")
        app = service_application(settings.tokens)

        async def close_dispatcher(app: web.Application) -> None:
            dispatcher.close()

        app.on_cleanup.append(close_dispatcher)
        return app
    # The evaluation service is a dispatcher too: it loads each rollout's scenario on demand, resolves the candidate
    # artifact and serves it to the rollout itself.
    dispatcher = build_dispatcher(settings, environ=env, components=frozenset({"evaluation"}))
    workers = ServiceWorkers()
    for _ in range(LOCAL_ROLLOUT_WORKERS):
        workers.add(
            EvaluationRolloutWorker(
                dispatcher.queue, LocalCandidateEndpoints(dispatcher), worker_id=instance_id("evaluation")
            )
        )
    workers.start()
    app = service_application(settings.tokens)

    async def close_evaluation(app: web.Application) -> None:
        workers.stop()
        dispatcher.close()

    app.on_cleanup.append(close_evaluation)
    return app


__all__ = ["build_component_app"]
