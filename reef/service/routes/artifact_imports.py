from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from reef.service.artifact_imports import MAX_ARCHIVE_BYTES, ArtifactImports
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object

#: The content types of an artifact archive.
ARTIFACT_ARCHIVE_TYPES = frozenset({"application/gzip", "application/x-gzip"})
#: The bytes read from the body at a time.
CHUNK_BYTES = 1024 * 1024


def register_artifact_import_routes(
    app: web.Application, *, request_service: RequestService, imports: ArtifactImports
) -> None:
    async def import_artifact(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        component = request.query.get("component")
        if request.content_type in ARTIFACT_ARCHIVE_TYPES:
            with imports.temporary_directory() as work:
                archive = Path(work) / "artifact.tar.gz"
                size = 0
                # The body goes to a file, not to memory; its size is bounded like the directory it unpacks to.
                with archive.open("wb") as file:
                    async for chunk in request.content.iter_chunked(CHUNK_BYTES):
                        size += len(chunk)
                        if size > MAX_ARCHIVE_BYTES:
                            raise web.HTTPRequestEntityTooLarge(max_size=MAX_ARCHIVE_BYTES, actual_size=size)
                        file.write(chunk)
                receipt = await asyncio.to_thread(
                    imports.import_archive, request_service.dispatcher, scenario, component, archive
                )
            return web.json_response(receipt, status=201)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(
                text='an import is a tar.gz archive (Content-Type application/gzip) or JSON {"source": "<uri>"}'
            )
        payload = await read_object(request)
        source = payload.get("source")
        if set(payload) != {"source"} or not isinstance(source, str) or not source.strip():
            raise web.HTTPBadRequest(text='a URI import is the object {"source": "<uri>"}')
        receipt = await asyncio.to_thread(
            imports.import_source, request_service.dispatcher, scenario, component, source.strip()
        )
        return web.json_response(receipt, status=201)

    app.router.add_post("/reef/scenarios/{scenario}/artifacts", import_artifact)


__all__ = ["register_artifact_import_routes"]
