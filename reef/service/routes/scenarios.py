from __future__ import annotations

import asyncio

from aiohttp import web

from reef.scenario.scenario import Scenario
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def register_scenario_routes(app: web.Application, *, request_service: RequestService) -> None:
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
        if not payload or set(payload) - {"training_mode", "model"}:
            raise ValueError("expected training_mode or model settings")
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
        return web.json_response(result)

    async def delete_scenario(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(request_service.dispatcher.delete_scenario, request.match_info["scenario"])
        return web.json_response(result)

    async def list_releases(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        releases = request_service.dispatcher.list_releases(scenario)
        return web.json_response(
            {
                "scenario": scenario,
                "releases": releases,
            }
        )

    async def scenario_contract(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        return web.json_response(request_service.dispatcher.scenario_contract(scenario))

    async def rollback_scenario(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        payload = await read_object(request)
        release_id = payload.get("release_id")
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        published = request_service.dispatcher.rollback(scenario, release_id.strip())
        return web.json_response(
            {
                "scenario": scenario,
                "release_id": published.release_id,
                "content_id": published.content_id,
            }
        )

    async def promote_scenario(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        payload = await read_object(request)
        release_id = payload.get("release_id")
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        published = request_service.dispatcher.promote(scenario, release_id.strip())
        return web.json_response(
            {
                "scenario": scenario,
                "release_id": published.release_id,
                "content_id": published.content_id,
            }
        )

    app.router.add_get("/reef/scenarios", list_scenarios)
    app.router.add_post("/reef/scenarios", create_scenario)
    app.router.add_post("/reef/scenarios/{scenario}/update", update_scenario)
    app.router.add_delete("/reef/scenarios/{scenario}", delete_scenario)
    app.router.add_get("/reef/scenarios/{scenario}/contract", scenario_contract)
    app.router.add_get("/reef/scenarios/{scenario}/releases", list_releases)
    app.router.add_post("/reef/scenarios/{scenario}/rollback", rollback_scenario)
    app.router.add_post("/reef/scenarios/{scenario}/promote", promote_scenario)


__all__ = ["register_scenario_routes"]
