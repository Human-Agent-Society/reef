from __future__ import annotations

import asyncio

from aiohttp import web

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.service.auth import page_query
from reef.service.harness_imports import HarnessImports
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def register_harness_routes(app: web.Application, *, request_service: RequestService, imports: HarnessImports) -> None:
    """The routes that a harness client calls: the active tree, the releases, the import of a person's files, and the
    adapters. Control's app and the harness service's app both serve them; ``imports`` commits an import."""

    async def harness(request: web.Request) -> web.Response:
        release_id = request.query.get("release_id") or None
        form = request.query.get("form", "rendered")
        service_url = request.query.get("service_url") or None
        root = request.query.get("root") or None
        manifest = await asyncio.to_thread(
            request_service.harness_manifest, request.headers, release_id, form, service_url, root
        )
        return web.json_response(
            manifest,
            headers={"x-reef-release-id": manifest["release_id"]},
        )

    async def harness_import(request: web.Request) -> web.Response:
        payload = await read_object(request)
        answer = await asyncio.to_thread(imports.import_files, request.headers, payload)
        return web.json_response(answer, headers={"x-reef-release-id": answer["release_id"]})

    async def harness_releases(request: web.Request) -> web.Response:
        catalog = await asyncio.to_thread(request_service.harness_releases, request.headers)
        # Each row names its step's page, the step being its position oldest first; the link carries a page key in
        # place of the token.
        query = page_query(request, catalog["scenario"])
        rows = [
            {**row, "page_path": f"/reef/harness/releases/{step}/page?{query}"}
            for step, row in enumerate(catalog["releases"])
        ]
        return web.json_response({**catalog, "releases": rows})

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
    app.router.add_post("/reef/harness/import", harness_import)
    app.router.add_get("/reef/harness/releases", harness_releases)
    app.router.add_get("/reef/harness/adapters", adapters)


__all__ = ["register_harness_routes"]
