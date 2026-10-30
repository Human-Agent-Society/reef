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

from reef.evaluation.rollout_worker import EvaluationRolloutWorker, ServiceCandidateEndpoints
from reef.execution.workers import SERVICE_COMPONENTS, ServiceWorkers
from reef.service.artifact_evaluations import LOCAL_ROLLOUT_WORKERS
from reef.service.assembly import build_app, build_dispatcher, build_training_service, open_scenario_storage
from reef.service.auth import create_authentication_middleware
from reef.service.deploy.service_config import ServiceConfig
from reef.service.inference_service import build_inference_app
from reef.storage.coordination import Coordination, instance_id
from reef.storage.sql_coordination import open_coordination


def open_service_coordination(settings: ServiceConfig) -> Coordination:
    """The deployment's coordination store, on the record backend of ``settings``."""
    return open_coordination(
        settings.record_backend,
        agent_record_dir=settings.agent_record_dir,
        record_database_url=settings.record_database_url,
    )


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
    control_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> web.Application:
    """The app of the service ``component`` (one of ``SERVICE_COMPONENTS``), which owns what it opens.

    ``control_url`` is the URL of the control service, which the evaluation service calls for candidates; it is
    required for ``evaluation`` alone. Raise ``ValueError`` for an unknown service or a missing ``control_url``.
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
        app = build_inference_app(
            coordination.served_models, storage, tokens=settings.tokens, timeout_seconds=settings.inference_timeout_s
        )

        async def close_inference(app: web.Application) -> None:
            storage.close()
            coordination.close()

        app.on_cleanup.append(close_inference)
        return app
    if component == "samples":
        dispatcher = build_dispatcher(settings, environ=env, components=frozenset({"samples"}))
        app = health_app(settings)

        async def close_samples(app: web.Application) -> None:
            dispatcher.close()

        app.on_cleanup.append(close_samples)
        return app
    if component == "training":
        services = build_training_service(settings, environ=env)
        services.workers.start()
        app = health_app(settings)

        async def close_training(app: web.Application) -> None:
            services.workers.stop()
            services.coordination.close()

        app.on_cleanup.append(close_training)
        return app
    if not control_url:
        raise ValueError("the evaluation service needs the control service's URL: pass --control-url")
    coordination = open_service_coordination(settings)
    token = settings.tokens[0] if settings.tokens else None
    workers = ServiceWorkers()
    for _ in range(LOCAL_ROLLOUT_WORKERS):
        workers.add(
            EvaluationRolloutWorker(
                coordination.work_queue,
                ServiceCandidateEndpoints(control_url, token),
                worker_id=instance_id("evaluation"),
            )
        )
    workers.start()
    app = health_app(settings)

    async def close_evaluation(app: web.Application) -> None:
        workers.stop()
        coordination.close()

    app.on_cleanup.append(close_evaluation)
    return app


__all__ = ["build_component_app", "health_app", "open_service_coordination"]
