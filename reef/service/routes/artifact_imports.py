from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.artifact_imports import ArtifactImports
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def register_artifact_import_routes(
    app: web.Application, *, request_service: RequestService, imports: ArtifactImports
) -> None:
    async def import_artifact(request: web.Request) -> web.Response:
        payload = await read_object(request)
        source = payload.get("source")
        if set(payload) != {"source"} or not isinstance(source, str) or not source.strip():
            raise web.HTTPBadRequest(text='an import is the object {"source": "<uri>"}')
        receipt = await asyncio.to_thread(
            imports.import_source,
            request_service.dispatcher,
            request.match_info["scenario"],
            request.query.get("component"),
            source.strip(),
        )
        return web.json_response(receipt, status=201)

    app.router.add_post("/reef/scenarios/{scenario}/artifacts", import_artifact)


__all__ = ["register_artifact_import_routes"]
