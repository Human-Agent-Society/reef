from __future__ import annotations

import asyncio

from aiohttp import web

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.descriptor import DescriptorError
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


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

    async def harness_proposals(request: web.Request) -> web.Response:
        payload = await read_object(request)
        answer = await asyncio.to_thread(request_service.harness_propose, request.headers, payload)
        return web.json_response(answer)

    async def status(request: web.Request) -> web.Response:
        value = await asyncio.to_thread(lambda: request_service.dispatcher.build_training_status())
        return web.json_response(value)

    async def adapters(request: web.Request) -> web.Response:
        names = await asyncio.to_thread(available_adapters)
        entries = []
        for name in names:
            try:
                descriptor = await asyncio.to_thread(get_adapter, name)
            except DescriptorError:
                continue
            install = descriptor.install
            entries.append(
                {
                    "name": name,
                    "binary": descriptor.binary,
                    "trajectory_format": descriptor.trajectory_format,
                    "model_bindings": sorted(descriptor.model_binding),
                    "install": (
                        None
                        if install is None
                        else {"kind": install.kind, "package": install.package, "version": install.version}
                    ),
                }
            )
        return web.json_response({"adapters": entries})

    app.router.add_get("/reef/harness", harness)
    app.router.add_get("/reef/harness/install", harness_install)
    app.router.add_get("/reef/harness/releases", harness_releases)
    app.router.add_post("/reef/harness/proposals", harness_proposals)
    app.router.add_get("/reef/harness/adapters", adapters)
    app.router.add_get("/reef/status", status)


__all__ = ["register_system_routes"]
