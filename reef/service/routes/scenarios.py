from __future__ import annotations

from aiohttp import web

from reef.service.request_service import RequestService


def register_scenario_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def list_scenarios(request: web.Request) -> web.Response:
        return web.json_response({"scenarios": list(request_service.dispatcher.list_scenarios())})

    async def create_scenario(request: web.Request) -> web.Response:
        payload = await request.json()
        name = payload.get("name")
        recipe = payload.get("recipe")
        artifact_version = payload.get("artifact_version")
        if not isinstance(name, str) or not name.strip():
            raise web.HTTPBadRequest(text="name must be a non-empty string")
        if not isinstance(recipe, str) or not recipe.strip():
            raise web.HTTPBadRequest(text="recipe must be a non-empty string")
        if artifact_version is not None and not isinstance(artifact_version, str):
            raise web.HTTPBadRequest(text="artifact_version must be a string")
        name = name.strip()
        recipe = recipe.strip()
        created = not request_service.dispatcher.has_scenario(name)
        scenario = request_service.dispatcher.get_or_create_scenario(
            name,
            recipe,
            artifact_version,
            allow_implicit_creation=True,
        )
        if scenario is None:
            raise RuntimeError("scenario creation returned no scenario")
        return web.json_response(
            {
                "scenario": scenario.name,
                "recipe": scenario.recipe,
                "artifact_version": scenario.repository.require_current_artifact().version,
            },
            status=201 if created else 200,
        )

    async def list_versions(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        versions = request_service.dispatcher.list_versions(scenario)
        return web.json_response(
            {
                "scenario": scenario,
                "versions": versions,
            }
        )

    async def scenario_contract(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        return web.json_response(request_service.dispatcher.scenario_contract(scenario))

    async def rollback_scenario(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        payload = await request.json()
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        artifact_version = payload.get("artifact_version")
        if not isinstance(artifact_version, str) or not artifact_version.strip():
            raise ValueError("artifact_version must be a non-empty string")
        published = request_service.dispatcher.rollback(scenario, artifact_version.strip())
        return web.json_response(
            {
                "scenario": scenario,
                "artifact_version": published.version,
            }
        )

    app.router.add_get("/reef/scenarios", list_scenarios)
    app.router.add_post("/reef/scenarios", create_scenario)
    app.router.add_get("/reef/scenarios/{scenario}/contract", scenario_contract)
    app.router.add_get("/reef/scenarios/{scenario}/versions", list_versions)
    app.router.add_post("/reef/scenarios/{scenario}/rollback", rollback_scenario)


__all__ = ["register_scenario_routes"]
