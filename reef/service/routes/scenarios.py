from __future__ import annotations

import asyncio

from aiohttp import web

from reef.core.errors import UnknownScenario
from reef.core.records_types import RequestType
from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.artifact_imports import ArtifactImports
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.storage.data import check_owner
from reef.train.job_queue import cancel_scenario_job


def register_scenario_reads(app: web.Application, *, request_service: RequestService) -> None:
    """The reads of what each scenario stores, which every service with the scenarios' stores answers."""

    async def list_scenarios(request: web.Request) -> web.Response:
        return web.json_response({"scenarios": list(request_service.dispatcher.list_scenarios())})

    async def scenario_settings(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(request_service.dispatcher.scenario_settings, request.match_info["scenario"])
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def list_artifacts(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        artifacts = await asyncio.to_thread(request_service.dispatcher.list_artifacts, scenario)
        return web.json_response({"scenario": scenario, "artifacts": artifacts})

    async def record_list(request: web.Request) -> web.Response:
        request_type = request.query.get("request_type")
        result = await asyncio.to_thread(
            request_service.dispatcher.read_records,
            request.match_info["scenario"],
            after_sequence=int(request.query.get("after_sequence", "0")),
            limit=int(request.query.get("limit", "50")),
            # An unknown type is a ValueError, which answers HTTP 400.
            request_type=None if request_type is None else RequestType(request_type),
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def commit_list(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(
            request_service.dispatcher.read_commits,
            request.match_info["scenario"],
            after_step=int(request.query.get("after_step", "0")),
            limit=int(request.query.get("limit", "50")),
            record_ids=tuple(request.query.getall("record_id", [])),
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def record_detail(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(
            request_service.dispatcher.read_record,
            request.match_info["scenario"],
            request.match_info["record_id"],
        )
        if result is None:
            return web.json_response(
                {"error": "Trace body is unavailable. It may have expired or never been retained."},
                status=404,
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def scenario_contract(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        return web.json_response(
            request_service.dispatcher.scenario_contract(scenario), headers={"Cache-Control": "no-store"}
        )

    app.router.add_get("/reef/scenarios", list_scenarios)
    app.router.add_get("/reef/scenarios/{scenario}/settings", scenario_settings)
    app.router.add_get("/reef/scenarios/{scenario}/contract", scenario_contract)
    app.router.add_get("/reef/scenarios/{scenario}/records", record_list)
    app.router.add_get("/reef/scenarios/{scenario}/commits", commit_list)
    app.router.add_get("/reef/scenarios/{scenario}/records/{record_id}", record_detail)
    app.router.add_get("/reef/scenarios/{scenario}/artifacts", list_artifacts)


def register_scenario_writes(
    app: web.Application,
    *,
    request_service: RequestService,
    evaluations: ArtifactEvaluations,
    imports: ArtifactImports,
) -> None:
    """The changes of scenarios, which only control makes: create (from the base, or from an artifact item through
    ``imports``), update, delete, activate, serve, train from an artifact, and cancel a job."""

    async def create_scenario(request: web.Request) -> web.Response:
        payload = await read_object(request)
        unknown = sorted(set(payload) - {"name", "owner", "artifact_id", "component"})
        if unknown:
            raise web.HTTPBadRequest(
                text=f"unknown keys {unknown}: a scenario is created from name, owner, artifact_id and component; "
                "artifact_id names the artifact item of the owner that a new scenario starts from"
            )
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise web.HTTPBadRequest(text="name must be a non-empty string")
        name = name.strip()
        owner = payload.get("owner")
        if owner is not None:
            check_owner(owner)
        artifact_id, component = payload.get("artifact_id"), payload.get("component")
        for key, value in (("artifact_id", artifact_id), ("component", component)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise web.HTTPBadRequest(text=f"{key} must be a non-empty string")
        if component is not None and artifact_id is None:
            raise web.HTTPBadRequest(text="component names where artifact_id lands: give artifact_id too")
        dispatcher = request_service.dispatcher
        scenario, created = await asyncio.to_thread(dispatcher.create_scenario, name)
        if owner is not None:
            # The owner whose data items the scenario uses (RFC #1028); the default is the scenario's own name.
            await asyncio.to_thread(dispatcher.set_scenario_owner, name, owner)
        started: dict[str, object] | None = None
        if created and artifact_id is not None:
            # An existing scenario keeps its start: creation is idempotent.
            target = None if component is None else component.strip()
            started = await asyncio.to_thread(imports.start_scenario, dispatcher, name, target, artifact_id.strip())
        current = scenario.repository.require_current_artifact()
        answer: dict[str, object] = {
            "scenario": scenario.name,
            "owner": dispatcher.scenario_owner(name),
            "release_id": current.release_id,
            "content_id": current.content_id,
        }
        if started is not None:
            answer["artifact_id"] = started["artifact_id"]
        return web.json_response(answer, status=201 if created else 200)

    async def update_scenario(request: web.Request) -> web.Response:
        payload = await read_object(request)
        if not payload or set(payload) - {"training_mode", "artifact_evaluation", "activation"}:
            raise ValueError("expected training_mode, artifact_evaluation or activation settings")
        if "training_mode" in payload and payload["training_mode"] not in ("auto", "manual"):
            raise ValueError("expected training_mode 'auto' or 'manual'")
        name = request.match_info["scenario"]
        result: dict[str, object] = {"scenario": name}
        if "training_mode" in payload:
            result.update(
                await asyncio.to_thread(request_service.dispatcher.set_training_mode, name, payload["training_mode"])
            )
        changes: dict[str, object] = {
            key: payload[key] for key in ("artifact_evaluation", "activation") if key in payload
        }
        if changes:
            result.update(await asyncio.to_thread(evaluations.configure, name, changes))
        return web.json_response(result)

    async def delete_scenario(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(request_service.dispatcher.delete_scenario, request.match_info["scenario"])
        return web.json_response(result)

    async def read_artifact_selection(request: web.Request) -> tuple[str, str | None]:
        """The ``artifact_id`` and the optional ``component`` that an activation, a train-from or a serve names."""
        payload = await read_object(request)
        artifact_id = payload.get("artifact_id")
        component = payload.get("component")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if component is not None and (not isinstance(component, str) or not component.strip()):
            raise ValueError("component must be a non-empty string")
        return artifact_id.strip(), None if component is None else component.strip()

    async def activate_artifact(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        artifact_id, component = await read_artifact_selection(request)
        activated = await asyncio.to_thread(request_service.dispatcher.activate, scenario, artifact_id, component)
        return web.json_response(
            {
                "scenario": scenario,
                "component": activated.component,
                "artifact_id": activated.artifact_id,
                "release_id": activated.release_id,
                "content_id": activated.content_id,
            }
        )

    async def serve_artifact(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        artifact_id, component = await read_artifact_selection(request)
        served = await asyncio.to_thread(request_service.dispatcher.serve, scenario, artifact_id, component)
        return web.json_response(
            {
                "scenario": scenario,
                "component": served.component,
                "artifact_id": served.artifact_id,
                "release_id": served.release_id,
                "content_id": served.content_id,
            }
        )

    async def stop_serving_artifact(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        artifact_id, component = await read_artifact_selection(request)
        if component is not None:
            raise ValueError("stop-serving takes the artifact_id alone: it names one artifact of one component")
        stopped = await asyncio.to_thread(request_service.dispatcher.stop_serving, scenario, artifact_id)
        return web.json_response(
            {
                "scenario": scenario,
                "component": stopped.component,
                "artifact_id": stopped.artifact_id,
                "release_id": stopped.release.release_id,
            }
        )

    async def cancel_job(request: web.Request) -> web.Response:
        scenario, job_id = request.match_info["scenario"], request.match_info["job_id"]
        dispatcher = request_service.dispatcher
        if not await asyncio.to_thread(dispatcher.has_scenario, scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        job = await asyncio.to_thread(cancel_scenario_job, dispatcher.queue, scenario, job_id)
        if job is None:
            raise web.HTTPNotFound(text=f"scenario {scenario!r} has no training job {job_id!r}")
        return web.json_response({"scenario": scenario, "job": job})

    async def train_from_artifact(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        artifact_id, component = await read_artifact_selection(request)
        start = await asyncio.to_thread(request_service.dispatcher.train_from, scenario, artifact_id, component)
        return web.json_response(
            {"scenario": scenario, "component": start.component, "artifact_id": start.artifact_id}
        )

    app.router.add_post("/reef/scenarios", create_scenario)
    app.router.add_post("/reef/scenarios/{scenario}/update", update_scenario)
    app.router.add_delete("/reef/scenarios/{scenario}", delete_scenario)
    app.router.add_post("/reef/scenarios/{scenario}/activate", activate_artifact)
    app.router.add_post("/reef/scenarios/{scenario}/train-from", train_from_artifact)
    app.router.add_post("/reef/scenarios/{scenario}/serve", serve_artifact)
    app.router.add_post("/reef/scenarios/{scenario}/stop-serving", stop_serving_artifact)
    app.router.add_post("/reef/scenarios/{scenario}/jobs/{job_id}/cancel", cancel_job)


__all__ = ["register_scenario_reads", "register_scenario_writes"]
