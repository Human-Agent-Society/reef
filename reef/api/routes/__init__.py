from __future__ import annotations

from aiohttp import web

from reef.api.inference_service import InferenceReplica, add_inference_routes
from reef.api.request_service import RequestService
from reef.api.routes.artifact_imports import register_artifact_import_routes
from reef.api.routes.data import register_data_routes
from reef.api.routes.evaluations import register_evaluation_reads, register_evaluation_writes
from reef.api.routes.harness import register_harness_routes
from reef.api.routes.records import register_record_routes
from reef.api.routes.samples import register_next_job_route, register_sample_reads
from reef.api.routes.scenario_files import register_scenario_file_routes
from reef.api.routes.scenarios import register_scenario_reads, register_scenario_writes
from reef.api.routes.system import register_system_routes
from reef.api.routes.tasks import register_task_routes
from reef.control.operations.artifact_evaluations import ArtifactEvaluations
from reef.control.operations.artifact_imports import ArtifactImports
from reef.control.operations.harness_imports import DirectImports
from reef.control.operations.tasks import DEFAULT_MAX_TASK_ARCHIVE_BYTES


def register_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    replica: InferenceReplica,
    evaluations: ArtifactEvaluations,
    imports: ArtifactImports,
    max_task_archive_bytes: int = DEFAULT_MAX_TASK_ARCHIVE_BYTES,
) -> None:
    """The routes of control's app. It serves the data routes too when its process runs the data service.

    ``max_task_archive_bytes`` is ``data.max_task_archive_bytes`` (``register_data_routes``).
    """
    add_inference_routes(app, replica)
    register_scenario_reads(app, request_service=request_service)
    register_scenario_writes(app, request_service=request_service, evaluations=evaluations, imports=imports)
    register_system_routes(app, request_service=request_service)
    # Control commits a harness import itself.
    register_harness_routes(app, request_service=request_service, imports=DirectImports(request_service.dispatcher))
    register_record_routes(app, request_service=request_service)
    if request_service.dispatcher.services.runs_data:
        register_data_routes(app, request_service=request_service, max_task_archive_bytes=max_task_archive_bytes)
        register_scenario_file_routes(app, request_service=request_service)
    register_task_routes(app, request_service=request_service)
    register_sample_reads(app, request_service=request_service)
    register_next_job_route(app, request_service=request_service)
    register_artifact_import_routes(app, request_service=request_service, imports=imports)
    register_evaluation_reads(app, evaluations=evaluations)
    register_evaluation_writes(app, evaluations=evaluations)


def register_data_service_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    evaluations: ArtifactEvaluations,
    max_task_archive_bytes: int = DEFAULT_MAX_TASK_ARCHIVE_BYTES,
) -> None:
    """The routes of the data service's app: the data routes and the reads of what each scenario stores.

    ``max_task_archive_bytes`` is ``data.max_task_archive_bytes`` (``register_data_routes``).
    """
    register_data_routes(app, request_service=request_service, max_task_archive_bytes=max_task_archive_bytes)
    register_scenario_file_routes(app, request_service=request_service)
    register_scenario_reads(app, request_service=request_service)
    register_task_routes(app, request_service=request_service)
    register_sample_reads(app, request_service=request_service)
    register_evaluation_reads(app, evaluations=evaluations)


__all__ = ["register_data_service_routes", "register_routes"]
