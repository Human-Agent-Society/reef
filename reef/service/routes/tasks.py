from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from aiohttp import web

from reef.service.request_service import RequestService
from reef.service.tasks import MAX_UNPACKED_BYTES, list_imported_tasks, upload_tasks

#: The content types of a task archive.
TASK_ARCHIVE_TYPES = frozenset({"application/gzip", "application/x-gzip"})
#: The bytes read from the body at a time.
CHUNK_BYTES = 1024 * 1024


def register_task_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def add_tasks(request: web.Request) -> web.Response:
        if request.content_type not in TASK_ARCHIVE_TYPES:
            raise web.HTTPUnsupportedMediaType(
                text="a task upload is a tar.gz archive with Content-Type application/gzip"
            )
        split = request.query.get("split")
        if split is None:
            raise web.HTTPBadRequest(text="split is required: train or eval")
        with tempfile.TemporaryDirectory(prefix="reef-task-archive-") as work:
            archive = Path(work) / "tasks.tar.gz"
            size = 0
            # The body goes to a file, not to memory; its size is bounded like the tasks it unpacks to.
            with archive.open("wb") as file:
                async for chunk in request.content.iter_chunked(CHUNK_BYTES):
                    size += len(chunk)
                    if size > MAX_UNPACKED_BYTES:
                        raise web.HTTPRequestEntityTooLarge(max_size=MAX_UNPACKED_BYTES, actual_size=size)
                    file.write(chunk)
            receipt = await asyncio.to_thread(
                upload_tasks, request_service.dispatcher, request.match_info["scenario"], split, archive
            )
        return web.json_response(receipt)

    async def tasks(request: web.Request) -> web.Response:
        listing = await asyncio.to_thread(
            list_imported_tasks, request_service.dispatcher, request.match_info["scenario"]
        )
        return web.json_response(listing)

    app.router.add_post("/reef/scenarios/{scenario}/tasks", add_tasks)
    app.router.add_get("/reef/scenarios/{scenario}/tasks", tasks)


__all__ = ["register_task_routes"]
