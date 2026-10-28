from __future__ import annotations

import asyncio

from aiohttp import web

from reef.core.records_types import RequestType
from reef.scenario.scenario import Scenario
from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def register_scenario_routes(
    app: web.Application, *, request_service: RequestService, evaluations: ArtifactEvaluations
) -> None:
    async def list_scenarios(request: web.Request) -> web.Response:
        return web.json_response({"scenarios": list(request_service.dispatcher.list_scenarios())})

    async def create_scenario(request: web.Request) -> web.Response:
        payload = await read_object(request)
        name = payload.get("name")
        release_id = payload.get("release_id")
        if not isinstance(name, str) or not name.strip():
            raise web.HTTPBadRequest(text="name must be a non-empty string")
        if release_id is not None and not isinstance(release_id, str):
            raise web.HTTPBadRequest(text="release_id must be a string")
        if isinstance(release_id, str) and not release_id.strip():
            raise web.HTTPBadRequest(text="release_id must be a non-empty string")
        name = name.strip()
        if release_id is not None:
            release_id = release_id.strip()
        created = not request_service.dispatcher.has_scenario(name)
        scenario: Scenario | None
        if "model" in payload:
            scenario = await asyncio.to_thread(
                request_service.dispatcher.configure_scenario_model,
                name,
                payload["model"],
                create=True,
                release_id=release_id,
            )
        else:
            scenario = request_service.dispatcher.get_or_create_scenario(
                name,
                release_id=release_id,
                allow_implicit_creation=True,
            )
        if scenario is None:
            raise RuntimeError("scenario creation returned no scenario")
        current = scenario.repository.require_current_artifact()
        return web.json_response(
            {
                "scenario": scenario.name,
                "release_id": current.release_id,
                "content_id": current.content_id,
                "model": scenario.model_config.view(),
            },
            status=201 if created else 200,
        )

    async def update_scenario(request: web.Request) -> web.Response:
        payload = await read_object(request)
        if not payload or set(payload) - {"training_mode", "model", "artifact_evaluation"}:
            raise ValueError("expected training_mode, model or artifact_evaluation settings")
        if "training_mode" in payload and payload["training_mode"] not in ("auto", "manual", "hybrid"):
            raise ValueError("expected training_mode 'auto', 'manual' or 'hybrid'")
        name = request.match_info["scenario"]
        result: dict[str, object] = {"scenario": name}
        if "model" in payload:
            scenario = await asyncio.to_thread(
                request_service.dispatcher.configure_scenario_model, name, payload["model"]
            )
            result["model"] = scenario.model_config.view()
        if "training_mode" in payload:
            result.update(
                await asyncio.to_thread(request_service.dispatcher.set_training_mode, name, payload["training_mode"])
            )
        if "artifact_evaluation" in payload:
            result["artifact_evaluation"] = await asyncio.to_thread(
                evaluations.configure, name, payload["artifact_evaluation"]
            )
        return web.json_response(result)

    async def delete_scenario(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(request_service.dispatcher.delete_scenario, request.match_info["scenario"])
        return web.json_response(result)

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

    async def read_artifact_selection(request: web.Request) -> tuple[str, str | None]:
        """The ``artifact_id`` and the optional ``component`` an activation or a new training start names."""
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

    async def train_from_artifact(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        artifact_id, component = await read_artifact_selection(request)
        start = await asyncio.to_thread(request_service.dispatcher.train_from, scenario, artifact_id, component)
        return web.json_response(
            {"scenario": scenario, "component": start.component, "artifact_id": start.artifact_id}
        )

    app.router.add_get("/reef/scenarios", list_scenarios)
    app.router.add_post("/reef/scenarios", create_scenario)
    app.router.add_post("/reef/scenarios/{scenario}/update", update_scenario)
    app.router.add_delete("/reef/scenarios/{scenario}", delete_scenario)
    app.router.add_get("/reef/scenarios/{scenario}/contract", scenario_contract)
    app.router.add_get("/reef/scenarios/{scenario}/records", record_list)
    app.router.add_get("/reef/scenarios/{scenario}/commits", commit_list)
    app.router.add_get("/reef/scenarios/{scenario}/records/{record_id}", record_detail)
    app.router.add_get("/reef/scenarios/{scenario}/artifacts", list_artifacts)
    app.router.add_post("/reef/scenarios/{scenario}/activate", activate_artifact)
    app.router.add_post("/reef/scenarios/{scenario}/train-from", train_from_artifact)


__all__ = ["register_scenario_routes"]
