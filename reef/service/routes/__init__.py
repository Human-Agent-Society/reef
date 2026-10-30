from __future__ import annotations

from aiohttp import web

from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.artifact_imports import ArtifactImports
from reef.service.inference_service import InferenceReplica, add_inference_routes
from reef.service.request_service import RequestService
from reef.service.routes.artifact_imports import register_artifact_import_routes
from reef.service.routes.data import register_data_routes
from reef.service.routes.evaluations import register_evaluation_routes
from reef.service.routes.records import register_record_routes
from reef.service.routes.samples import register_sample_routes
from reef.service.routes.scenario_files import register_scenario_file_routes
from reef.service.routes.scenarios import register_scenario_routes
from reef.service.routes.system import register_health_route, register_system_routes
from reef.service.routes.tasks import register_task_routes


def register_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    replica: InferenceReplica,
    evaluations: ArtifactEvaluations,
    imports: ArtifactImports,
) -> None:
    """The routes of control's app. It serves the data routes too when its process runs the data service."""
    register_health_route(app)
    add_inference_routes(app, replica)
    register_scenario_routes(app, request_service=request_service, evaluations=evaluations)
    register_system_routes(app, request_service=request_service)
    register_record_routes(app, request_service=request_service)
    if "data" in request_service.dispatcher.components:
        register_data_routes(app, request_service=request_service)
        register_scenario_file_routes(app, request_service=request_service)
    register_task_routes(app, request_service=request_service)
    register_sample_routes(app, request_service=request_service)
    register_artifact_import_routes(app, request_service=request_service, imports=imports)
    register_evaluation_routes(app, evaluations=evaluations)


def register_data_service_routes(
    app: web.Application, *, request_service: RequestService, evaluations: ArtifactEvaluations
) -> None:
    """The routes of the data service's app: the data routes and the reads of what each scenario stores."""
    register_health_route(app)
    register_data_routes(app, request_service=request_service)
    register_scenario_file_routes(app, request_service=request_service)
    register_scenario_routes(app, request_service=request_service, evaluations=evaluations, writes=False)
    register_task_routes(app, request_service=request_service)
    register_sample_routes(app, request_service=request_service, writes=False)
    register_evaluation_routes(app, evaluations=evaluations, writes=False)


__all__ = ["register_data_service_routes", "register_routes"]
