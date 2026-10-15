"""HTTP boundaries for tenant admission and private service lifecycle control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from aiohttp import web
from aiohttp.typedefs import Handler, Middleware

from reef.core.records_types import RequestType
from reef.dispatcher import Dispatcher
from reef.harness.adapters import available_adapters
from reef.scenario.scheduling import TERMINAL_STATES, ScenarioTenantMismatch, resolve_timestamp
from reef.service.auth import EVALUATION_AUTHENTICATED, EVALUATION_ROUTES, PAGE_ROUTES
from reef.service.request_service import RequestService, normalize_request_payload
from reef.service.routes.payload import read_object
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import (
    CONTROL_PATH,
    TENANT_HEADER,
    ServiceSchedulerClient,
    integer_value,
    optional_text,
    tenant_from_headers,
    text_value,
)
from reef.service.wire import parse_request_headers
from reef.storage.model_config import read_model_config, write_model_config


def register_scheduler_routes(app: web.Application, controller: SchedulerController) -> None:
    async def register_service(request: web.Request) -> web.Response:
        payload = await read_object(request)
        return web.json_response(await asyncio.to_thread(controller.register_service, payload))

    async def create(request: web.Request) -> web.Response:
        payload = await read_object(request)
        tenant_id = (
            tenant_from_headers(request.headers)
            if TENANT_HEADER in request.headers
            else integer_value(payload, "tenant_id")
        )
        result = await asyncio.to_thread(controller.submit, payload, tenant_id)
        return web.json_response(result, status=201 if result["state"] == "admitted" else 202)

    async def scenarios(request: web.Request) -> web.Response:
        tenant_id = tenant_from_headers(request.headers)
        service_query = request.query.get("service_id")
        service_id = None if service_query is None else integer_value({"service_id": int(service_query)}, "service_id")
        return web.json_response(await asyncio.to_thread(controller.list_scenario_admissions, tenant_id, service_id))

    async def scenario(request: web.Request) -> web.Response:
        tenant_id = tenant_from_headers(request.headers)
        return web.json_response(
            await asyncio.to_thread(
                controller.get_scenario_admission, int(request.match_info["scenario_id"]), tenant_id
            )
        )

    async def wake(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(
            controller.wake,
            int(request.match_info["scenario_id"]),
            tenant_from_headers(request.headers),
            await read_object(request),
        )
        return web.json_response(result, status=200 if result["state"] == "admitted" else 202)

    async def cancel(request: web.Request) -> web.Response:
        payload = await read_object(request)
        result = await asyncio.to_thread(
            controller.cancel,
            tenant_from_headers(request.headers),
            text_value(payload, "service_name"),
            text_value(payload, "scenario_name"),
        )
        return web.json_response(result, status=200 if result["state"] in TERMINAL_STATES else 202)

    async def status(request: web.Request) -> web.Response:
        return web.json_response(await asyncio.to_thread(controller.status, tenant_from_headers(request.headers)))

    app.router.add_post("/reef/scenarios", create)
    app.router.add_get("/reef/scenarios", scenarios)
    app.router.add_get("/reef/scenarios/{scenario_id}", scenario)
    app.router.add_post("/reef/scenarios/{scenario_id}/wake", wake)
    app.router.add_post("/reef/scenarios/cancel", cancel)
    app.router.add_get("/reef/scheduler/status", status)
    app.router.add_post("/reef/services/register", register_service)


def register_scheduler_control(
    app: web.Application, dispatcher: Dispatcher, client: ServiceSchedulerClient, model_directory: Path
) -> None:
    def execute(payload: dict[str, object]) -> dict[str, object]:
        operation = text_value(payload, "operation")
        settings = client.settings
        if operation == "describe":
            return {
                "service_id": settings.service_id,
                "service_name": settings.service_name,
                "recipe": settings.recipe,
                "standard_execution_units": settings.standard_execution_units,
                "config_revision": settings.config_revision,
                "supports_idle_reclaim": dispatcher.supports_idle_reclaim,
            }
        if operation == "stage_model":
            model = payload.get("model")
            dispatcher.validate_scheduled_model(model)
            encoded = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
            staged_reference = hashlib.sha256(encoded).hexdigest()
            write_model_config(model_directory, staged_reference, model)
            return {"model_config_ref": staged_reference}
        scenario_name = text_value(payload, "scenario_name")
        scenario_id = integer_value(payload, "scenario_id")
        tenant_id = integer_value(payload, "tenant_id")
        generation = integer_value(payload, "generation")
        status = dispatcher.scheduled_scenario_status(scenario_name)
        if status is not None and status.scenario_id == scenario_id and status.tenant_id != tenant_id:
            raise ScenarioTenantMismatch("control tenant does not own the scenario")
        if operation == "status":
            return {"state": None} if status is None else asdict(status)
        if operation == "start":
            reference = optional_text(payload, "model_config_ref")
            if reference is not None and re.fullmatch(r"[a-f0-9]{64}", reference) is None:
                raise ValueError("model_config_ref must be a SHA-256 digest")
            model = None if reference is None else read_model_config(model_directory, reference)
            if reference is not None:
                encoded = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
                if hashlib.sha256(encoded).hexdigest() != reference:
                    raise ValueError("staged model configuration is missing or changed")
            result = dispatcher.start_scheduled_scenario(
                scenario_id,
                tenant_id,
                scenario_name,
                generation=generation,
                release_id=optional_text(payload, "release_id"),
                model=model,
                has_model=reference is not None,
            )
        elif operation == "cleanup":
            result = dispatcher.cleanup_scheduled_scenario(
                scenario_id, scenario_name, tenant_id=tenant_id, generation=generation
            )
        elif operation == "reclaim":
            cutoff = payload.get("inactive_before")
            if not isinstance(cutoff, int | float) or isinstance(cutoff, bool):
                raise ValueError("inactive_before must be a timestamp")
            result = dispatcher.reclaim_scheduled_scenario(
                scenario_id,
                scenario_name,
                tenant_id=tenant_id,
                generation=generation,
                inactive_before=resolve_timestamp(float(cutoff)),
            )
        else:
            raise ValueError("unknown scheduler control operation")
        return asdict(result)

    async def control(request: web.Request) -> web.Response:
        return web.json_response(await asyncio.to_thread(execute, await read_object(request)))

    app.router.add_post(CONTROL_PATH, control)


def create_tenant_admission_middleware(request_service: RequestService, client: ServiceSchedulerClient) -> Middleware:
    """Check owners for all scenario routes, including implicit creation paths."""
    dispatcher = request_service.dispatcher
    admissions = dispatcher.scenario_admissions
    if admissions is None:
        raise ValueError("tenant middleware requires scenario admission enforcement")

    @web.middleware
    async def tenant_admission(request: web.Request, handler: Handler) -> web.StreamResponse:
        if request.match_info.http_exception is not None:
            return await handler(request)
        if request.path in {"/healthz", CONTROL_PATH, "/reef/harness/adapters"}:
            return await handler(request)
        if EVALUATION_ROUTES.match(request.rel_url.raw_path):
            name = request.match_info.get("scenario", "")
            status = dispatcher.scheduled_scenario_status(name)
            if status is None or status.state != "admitted":
                raise web.HTTPConflict(text="evaluation scenario is not admitted")
            if request.get(EVALUATION_AUTHENTICATED, False):
                declared_tenant = (
                    tenant_from_headers(request.headers) if TENANT_HEADER in request.headers else status.tenant_id
                )
            else:
                declared_tenant = tenant_from_headers(request.headers)
            if declared_tenant != status.tenant_id:
                raise ScenarioTenantMismatch("evaluation tenant does not own the scenario")
            scenario_id = admissions.acquire_request(name, status.tenant_id)
            try:
                return await handler(request)
            finally:
                admissions.release_request(name, scenario_id)
        page = PAGE_ROUTES.match(request.path)
        tenant_header = request.headers.get(TENANT_HEADER, "").strip()
        if page and not tenant_header:
            tenant_header = request.query.get("tenant_id", "").strip()
        tenant_id = tenant_from_headers({TENANT_HEADER: tenant_header})
        if request.path == "/reef/status":
            return web.json_response(await asyncio.to_thread(client.list_scenario_admissions, tenant_id))
        if request.path == "/reef/scenarios":
            return await handler(request)
        name = request.match_info.get("scenario") or request.headers.get("x-reef-scenario", "").strip()
        if page and not name:
            name = request.query.get("scenario", "").strip()
        if request.path == "/reef/harness/install" and not name:
            name = f"harness-{uuid.uuid4().hex[:12]}"
            request = request.clone(headers={**request.headers, "x-reef-scenario": name})
        if not name:
            return await handler(request)
        if request.method == "DELETE" and request.match_info.get("scenario"):
            return await handler(request)
        status = dispatcher.scheduled_scenario_status(name)
        if status is not None and status.tenant_id != tenant_id and status.state not in {"deleted", "failed"}:
            raise ScenarioTenantMismatch("scenario is owned by another tenant")
        if status is not None and status.tenant_id == tenant_id:
            cold_train = request.method == "POST" and request.path == "/reef/train"
            request_query = request.method == "GET" and request.path.startswith("/reef/harness/requests/")
            if status.state != "admitted" and (cold_train or request_query):
                return await handler(request)
        if status is None or status.state in {"deleted", "failed"}:
            may_create = request.path == "/reef/harness/install" or (
                request.method == "POST"
                and (
                    request.path.startswith("/v1/")
                    or request.path in {"/reef/report", "/reef/records", "/reef/records/batch"}
                )
            )
            if not may_create:
                raise web.HTTPNotFound(text="scenario has not been admitted")
            if request.path != "/reef/harness/install" and not dispatcher.allow_implicit_creation:
                raise web.HTTPNotFound(text="implicit scenario creation is disabled")
            if request.method == "POST":
                payload = await read_object(request)
                if request.path == "/reef/records":
                    request_service.prepare_import_record(request.headers, payload)
                elif request.path == "/reef/records/batch":
                    request_service.prepare_import_records(request.headers, payload)
                else:
                    request_type = RequestType.REPORT if request.path == "/reef/report" else RequestType.INFERENCE
                    parse_request_headers(request.headers, request_type)
                    normalize_request_payload(request_type, payload)
            elif request.path == "/reef/harness/install" and request.query.get("adapter") not in available_adapters():
                raise ValueError("harness install requires a known adapter")
            admission = await asyncio.to_thread(
                client.submit,
                tenant_id,
                name,
                release_id=request.headers.get("x-reef-release-id") or request.query.get("release_id"),
            )
            if admission.get("state") != "admitted":
                return web.json_response({"error": "scenario is not ready", **admission}, status=409)
            status = dispatcher.scheduled_scenario_status(name)
        if status is None or status.state != "admitted":
            raise web.HTTPConflict(text="scenario is awaiting admission or cleanup")
        scenario_id = admissions.acquire_request(name, tenant_id)
        try:
            return await handler(request)
        finally:
            admissions.release_request(name, scenario_id)

    return tenant_admission
