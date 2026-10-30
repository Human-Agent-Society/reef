"""``reef serve --component NAME``: one service of a deployment in one process.

Each service reads and writes the shared store of the deployment (``storage.record_backend``): PostgreSQL for
processes on several hosts, SQLite for processes on one host. The services are:

- ``control``: the Reef HTTP API, scenarios, commits and activation. It writes ``served_models``, adds the
  training jobs, and serves candidates beside the active artifact for evaluation.
- ``inference``: answers model requests from ``served_models`` and appends their records
  (``reef.service.inference_service``).
- ``samples``: makes the samples of record selections from the samples queue.
- ``training``: runs the jobs of the jobs queue on the training component.
- ``evaluation``: runs the evaluation rollouts of the evaluations queue against control's candidate route.

A service that only takes work from a queue answers ``/healthz`` alone. ``reef serve`` without ``--component`` runs
all of them in one process, on the same interfaces.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from aiohttp import web

from reef.evaluation.rollout_worker import EvaluationRolloutWorker
from reef.execution.workers import SERVICE_COMPONENTS, ServiceWorkers
from reef.service.artifact_evaluations import LOCAL_ROLLOUT_WORKERS, LocalCandidateEndpoints
from reef.service.assembly import build_app, build_dispatcher, open_scenario_storage, open_service_coordination
from reef.service.auth import create_authentication_middleware
from reef.service.deploy.service_config import ServiceConfig
from reef.service.deploy.serving import inference_handler_factory_for
from reef.service.inference_service import ConfiguredEngineClients, build_inference_app
from reef.storage.coordination import instance_id


def health_app(settings: ServiceConfig) -> web.Application:
    """An app that answers ``/healthz`` alone, behind the service tokens."""
    app = web.Application(middlewares=[create_authentication_middleware(settings.tokens)])

    async def healthz(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_get("/healthz", healthz)
    return app


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
        coordination = open_service_coordination(settings)
        try:
            storage, _ = open_scenario_storage(settings)
        except BaseException:
            coordination.close()
            raise
        try:
            # The engines are reached with the deployment's handler type, as the runtimes of control reach them.
            app = build_inference_app(
                coordination.served_models,
                storage,
                ConfiguredEngineClients(
                    inference_handler_factory_for(settings.inference_handler_factory),
                    settings.inference_handler_config,
                    model_path=settings.model_path or "",
                    timeout_seconds=settings.inference_timeout_s,
                    api_key=settings.upstream_api_key or None,
                ),
                tokens=settings.tokens,
                timeout_seconds=settings.inference_timeout_s,
            )
        except BaseException:
            storage.close()
            coordination.close()
            raise

        async def close_inference(app: web.Application) -> None:
            storage.close()
            coordination.close()

        app.on_cleanup.append(close_inference)
        return app
    if component in ("samples", "training"):
        # A queue service is a dispatcher that runs only its own workers on the shared store.
        dispatcher = build_dispatcher(settings, environ=env, components=frozenset({component}))
        if component == "training" and dispatcher.recipe.training_component is None:
            dispatcher.close()
            raise ValueError(f"recipe {settings.recipe!r} trains no weights, so it has no training service")
        app = health_app(settings)

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
    app = health_app(settings)

    async def close_evaluation(app: web.Application) -> None:
        workers.stop()
        dispatcher.close()

    app.on_cleanup.append(close_evaluation)
    return app


__all__ = ["build_component_app", "health_app"]
