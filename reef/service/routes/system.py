from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.request_service import RequestService


def register_health_route(app: web.Application) -> None:
    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app.router.add_get("/healthz", health)


def register_system_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def harness(request: web.Request) -> web.Response:
        release_id = request.query.get("release_id") or None
        manifest = await asyncio.to_thread(request_service.harness_manifest, request.headers, release_id)
        return web.json_response(
            manifest,
            headers={"x-reef-release-id": manifest["release_id"]},
        )

    async def harness_install(request: web.Request) -> web.Response:
        release_id = request.query.get("release_id") or None
        adapter = request.query.get("adapter")
        script = await asyncio.to_thread(request_service.harness_install_script, request.headers, adapter, release_id)
        return web.Response(text=script, content_type="text/x-shellscript")

    async def harness_releases(request: web.Request) -> web.Response:
        catalog = await asyncio.to_thread(request_service.harness_releases, request.headers)
        return web.json_response(catalog)

    async def status(request: web.Request) -> web.Response:
        value = await asyncio.to_thread(lambda: request_service.dispatcher.training_status)
        return web.json_response(value)

    app.router.add_get("/reef/harness", harness)
    app.router.add_get("/reef/harness/install", harness_install)
    app.router.add_get("/reef/harness/releases", harness_releases)
    app.router.add_get("/reef/status", status)


__all__ = ["register_system_routes"]
