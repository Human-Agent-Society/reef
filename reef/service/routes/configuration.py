"""Read active snapshots and submit durable runtime configuration updates."""

from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def _expected_revision(request: web.Request) -> int | None:
    raw = request.headers.get("If-Match")
    if raw is None:
        return None
    value = raw.removeprefix('"').removesuffix('"')
    if not value.isascii() or not value.isdecimal():
        raise ValueError('If-Match must contain one configuration revision, e.g. "3"')
    return int(value)


def register_configuration_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def get_config(request: web.Request) -> web.Response:
        scenario = request.match_info.get("scenario")
        if scenario is None:
            status = await asyncio.to_thread(request_service.deployment_configuration)
        else:
            status = await asyncio.to_thread(request_service.dispatcher.scenario_configuration, scenario)
        return web.json_response(status, headers={"ETag": f'"{status["revision"]}"'})

    async def create_config_update(request: web.Request) -> web.Response:
        patch = await read_object(request)
        expected = _expected_revision(request)
        scenario = request.match_info.get("scenario")
        if scenario is None:
            update = await asyncio.to_thread(
                request_service.update_deployment_configuration, patch, expected_revision=expected
            )
        else:
            update = await asyncio.to_thread(
                request_service.dispatcher.update_scenario_configuration, scenario, patch, expected_revision=expected
            )
        return web.json_response(update, status=202)

    for path in ("/reef/config", "/reef/scenarios/{scenario}/config"):
        app.router.add_get(path, get_config)
        app.router.add_post(f"{path}/updates", create_config_update)
