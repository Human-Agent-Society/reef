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
        status = await asyncio.to_thread(request_service.deployment_configuration)
        return web.json_response(status, headers={"ETag": f'"{status["revision"]}"'})

    async def create_config_update(request: web.Request) -> web.Response:
        patch = await read_object(request)
        update = await asyncio.to_thread(
            request_service.update_deployment_configuration, patch, expected_revision=_expected_revision(request)
        )
        return web.json_response(update, status=202)

    app.router.add_get("/reef/config", get_config)
    app.router.add_post("/reef/config/updates", create_config_update)
