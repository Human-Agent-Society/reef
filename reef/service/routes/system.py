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
        artifact_version = request.query.get("version") or None
        manifest = await asyncio.to_thread(request_service.harness_manifest, request.headers, artifact_version)
        return web.json_response(
            manifest,
            headers={"x-reef-artifact-version": manifest["artifact_version"]},
        )

    async def harness_install(request: web.Request) -> web.Response:
        artifact_version = request.query.get("version") or None
        adapter = request.query.get("adapter")
        script = await asyncio.to_thread(
            request_service.harness_install_script, request.headers, adapter, artifact_version
        )
        return web.Response(text=script, content_type="text/x-shellscript")

    async def harness_versions(request: web.Request) -> web.Response:
        catalog = await asyncio.to_thread(request_service.harness_versions, request.headers)
        return web.json_response(catalog)

    async def status(request: web.Request) -> web.Response:
        value = await asyncio.to_thread(lambda: request_service.dispatcher.training_status)
        return web.json_response(value)

    app.router.add_get("/reef/harness", harness)
    app.router.add_get("/reef/harness/install", harness_install)
    app.router.add_get("/reef/harness/versions", harness_versions)
    app.router.add_get("/reef/status", status)


__all__ = ["register_system_routes"]
