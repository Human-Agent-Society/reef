from __future__ import annotations

from aiohttp import web

from reef.runtime.inference import InferenceBackend
from reef.service.request_service import RequestService
from reef.service.routes.inference import register_inference_routes
from reef.service.routes.records import register_record_routes
from reef.service.routes.scenarios import register_scenario_routes
from reef.service.routes.system import register_health_route, register_system_routes


def register_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    inference_backend: InferenceBackend | None,
) -> None:
    register_health_route(app)
    register_inference_routes(
        app,
        request_service=request_service,
        inference_backend=inference_backend,
    )
    register_scenario_routes(app, request_service=request_service)
    register_system_routes(app, request_service=request_service)
    register_record_routes(app, request_service=request_service)


__all__ = ["register_routes"]
