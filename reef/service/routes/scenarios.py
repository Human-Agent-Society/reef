from __future__ import annotations

import asyncio

from aiohttp import web

from reef.core.errors import UnknownScenario
from reef.core.records_types import RequestType
from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.artifact_imports import ArtifactImports
from reef.service.inference_service import TrafficWeights
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
            after_commit_number=int(request.query.get("after_step", "0")),
            limit=int(request.query.get("limit", "50")),
            record_ids=tuple(request.query.getall("record_id", [])),
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def operation_list(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(
            request_service.dispatcher.read_commits,
            request.match_info["scenario"],
            after_commit_number=int(request.query.get("after_step", "0")),
            limit=int(request.query.get("limit", "50")),
            record_ids=tuple(request.query.getall("record_id", [])),
        )
        result["operations"] = result.pop("commits")
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
    app.router.add_get("/reef/scenarios/{scenario}/operations", operation_list)
    app.router.add_get("/reef/scenarios/{scenario}/records/{record_id}", record_detail)
    app.router.add_get("/reef/scenarios/{scenario}/artifacts", list_artifacts)


def register_scenario_writes(
    app: web.Application,
    *,
    request_service: RequestService,
    evaluations: ArtifactEvaluations,
    imports: ArtifactImports,
) -> None:
    """The changes of scenarios, which only control makes: create (from the default start, or from an artifact item
    through ``imports``), update, delete, activate, serve, train from an artifact, and cancel a job.

    The create route takes ``serves``, a boolean that is true by default. With false, the new scenario serves
    nothing: no artifact answers its inference requests until an update sets ``served`` to one or more artifacts
    (``Dispatcher.set_served``). An update with an empty ``served`` makes a scenario serve nothing again."""

    async def create_scenario(request: web.Request) -> web.Response:
        payload = await read_object(request)
        unknown = sorted(set(payload) - {"name", "owner", "artifact_id", "component", "serves"})
        if unknown:
            raise web.HTTPBadRequest(
                text=f"unknown keys {unknown}: a scenario is created from name, owner, artifact_id, component and "
                "serves; artifact_id names the artifact item of the owner that a new scenario starts from"
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
        serves = payload.get("serves", True)
        if not isinstance(serves, bool):
            raise web.HTTPBadRequest(text="serves must be a boolean: false creates a scenario that serves nothing")
        if component is not None and artifact_id is None:
            raise web.HTTPBadRequest(text="component names where artifact_id lands: give artifact_id too")
        dispatcher = request_service.dispatcher
        target = None if component is None else component.strip()
        # An existing scenario keeps its root: creation is idempotent.
        item_id = artifact_id.strip() if isinstance(artifact_id, str) and not dispatcher.has_scenario(name) else None
        if item_id is not None:
            # The root snapshot holds the item, read from the owner whose data items the scenario uses (RFC #1028).
            scenario, created = await asyncio.to_thread(
                imports.create_from_item, dispatcher, name, owner or name, target, item_id, serves=serves
            )
        else:
            scenario, created = await asyncio.to_thread(dispatcher.create_scenario, name, serves=serves)
        if owner is not None:
            # The default owner is the scenario's own name. A new owner loads the scenario again.
            await asyncio.to_thread(dispatcher.set_scenario_owner, name, owner)
            scenario = await asyncio.to_thread(dispatcher.registry.require, name)
        components = dict(scenario.artifact_selection().artifact_ids)
        answer: dict[str, object] = {
            "scenario": scenario.name,
            "owner": dispatcher.scenario_owner(name),
            "components": components,
        }
        if len(components) == 1:
            answer["artifact_id"] = next(iter(components.values()))
        return web.json_response(answer, status=201 if created else 200)

    async def update_scenario(request: web.Request) -> web.Response:
        payload = await read_object(request)
        settings = {
            "training_mode",
            "artifact_evaluation",
            "activation",
            "serving_config",
            "served",
            "component",
            "record2sample_config",
        }
        if not payload or set(payload) - settings:
            raise ValueError(
                "expected training_mode, artifact_evaluation, activation, serving_config, served or "
                "record2sample_config settings"
            )
        component = payload.get("component")
        if "component" in payload and ("served" not in payload or not isinstance(component, str) or not component):
            raise ValueError("component is the non-empty name of the component whose served artifacts change")
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
        if "serving_config" in payload:
            result.update(
                await asyncio.to_thread(request_service.dispatcher.set_serving_config, name, payload["serving_config"])
            )
        if "record2sample_config" in payload:
            result.update(
                await asyncio.to_thread(
                    request_service.dispatcher.set_record2sample_config, name, payload["record2sample_config"]
                )
            )
        if "served" in payload:
            if not isinstance(payload["served"], dict):
                raise ValueError("served must map artifact IDs to weights that sum to 100, or be empty")
            served = TrafficWeights.from_json(payload["served"]).by_artifact
            result.update(await asyncio.to_thread(request_service.dispatcher.set_served, name, served, component))
        return web.json_response(result)

    async def delete_scenario(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(request_service.dispatcher.delete_scenario, request.match_info["scenario"])
        return web.json_response(result)

    async def read_artifact_selection(request: web.Request) -> tuple[str, str | None]:
        """The ``artifact_id`` and the optional ``component`` that a train-from names."""
        payload = await read_object(request)
        artifact_id = payload.get("artifact_id")
        component = payload.get("component")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if component is not None and (not isinstance(component, str) or not component.strip()):
            raise ValueError("component must be a non-empty string")
        return artifact_id.strip(), None if component is None else component.strip()

    async def restore_selection(request: web.Request) -> web.Response:
        payload = await read_object(request)
        if set(payload) != {"step"}:
            raise ValueError("restore takes one step naming the scenario selection to restore")
        commit_number = payload["step"]
        if not isinstance(commit_number, int) or isinstance(commit_number, bool) or commit_number < 0:
            raise ValueError("restore step must be a non-negative integer")
        scenario = request.match_info["scenario"]
        components = await asyncio.to_thread(request_service.dispatcher.restore, scenario, commit_number)
        return web.json_response({"scenario": scenario, "components": dict(components)})

    async def redeploy_scenario(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        await asyncio.to_thread(request_service.dispatcher.redeploy, scenario)
        return web.json_response({"scenario": scenario})

    async def cancel_job(request: web.Request) -> web.Response:
        scenario, job_id = request.match_info["scenario"], request.match_info["job_id"]
        dispatcher = request_service.dispatcher
        if not await asyncio.to_thread(dispatcher.has_scenario, scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        job = await asyncio.to_thread(cancel_scenario_job, dispatcher.queue, scenario, job_id)
        if job is None:
            raise web.HTTPNotFound(text=f"scenario {scenario!r} has no training job {job_id!r}")
        return web.json_response({"scenario": scenario, "job": job})

    async def cancel_request(request: web.Request) -> web.Response:
        scenario, request_id = request.match_info["scenario"], request.match_info["request_id"]
        cancelled = await asyncio.to_thread(request_service.dispatcher.cancel_request, scenario, request_id)
        if cancelled is None:
            raise web.HTTPNotFound(text=f"scenario {scenario!r} has no training request {request_id!r}")
        return web.json_response(
            {"scenario": scenario, "request_id": request_id, "queued": cancelled.queued, "job": cancelled.job}
        )

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
    app.router.add_post("/reef/scenarios/{scenario}/restore", restore_selection)
    app.router.add_post("/reef/scenarios/{scenario}/train-from", train_from_artifact)
    app.router.add_post("/reef/scenarios/{scenario}/redeploy", redeploy_scenario)
    app.router.add_post("/reef/scenarios/{scenario}/jobs/{job_id}/cancel", cancel_job)
    app.router.add_post("/reef/scenarios/{scenario}/requests/{request_id}/cancel", cancel_request)


__all__ = ["register_scenario_reads", "register_scenario_writes"]
