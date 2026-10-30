"""HTTP routes of a data owner's items (RFC #1028), under ``/reef/namespaces/{owner}/data``.

``reef.service.data`` handles each route; this module reads the request and writes the response.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from aiohttp import web

from reef.service.data import (
    create_items,
    create_task_items,
    delete_item,
    parsed_metadata,
    query_items,
    read_item,
    update_items,
)
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.tasks import MAX_UNPACKED_BYTES
from reef.storage.data import MAX_QUERY_LIMIT

#: The content types of a task archive.
TASK_ARCHIVE_TYPES = frozenset({"application/gzip", "application/x-gzip"})
#: The bytes read from the body at a time.
CHUNK_BYTES = 1024 * 1024
#: The most bytes of a JSON create body: 1000 trajectories of an ATIF upload before RFC #1028.
MAX_CREATE_BYTES = 10 * 1024 * 1024


def query_int(request: web.Request, name: str, default: int) -> int:
    try:
        return int(request.query.get(name, str(default)))
    except ValueError:
        raise web.HTTPBadRequest(text=f"{name} must be an integer") from None


def register_data_routes(app: web.Application, *, request_service: RequestService) -> None:
    dispatcher = request_service.dispatcher

    async def create(request: web.Request) -> web.Response:
        body = await read_object(request.clone(client_max_size=MAX_CREATE_BYTES))
        receipt = await asyncio.to_thread(create_items, dispatcher, request.match_info["owner"], body)
        return web.json_response(receipt, status=201)

    async def create_tasks(request: web.Request) -> web.Response:
        if request.content_type not in TASK_ARCHIVE_TYPES:
            raise web.HTTPUnsupportedMediaType(
                text="task items come as a tar.gz archive with Content-Type application/gzip"
            )
        metadata = parsed_metadata(request.query.get("metadata"))
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
                create_task_items, dispatcher, request.match_info["owner"], archive, metadata
            )
        return web.json_response(receipt, status=201)

    async def query(request: web.Request) -> web.Response:
        listing = await asyncio.to_thread(
            query_items,
            dispatcher,
            request.match_info["owner"],
            kind=request.query.get("kind") or None,
            scenario=request.query.get("scenario") or None,
            after=query_int(request, "after", 0),
            limit=query_int(request, "limit", MAX_QUERY_LIMIT),
        )
        return web.json_response(listing)

    async def read(request: web.Request) -> web.Response:
        item = await asyncio.to_thread(
            read_item, dispatcher, request.match_info["owner"], request.match_info["item_id"]
        )
        return web.json_response(item)

    async def update(request: web.Request) -> web.Response:
        body = await read_object(request)
        receipt = await asyncio.to_thread(update_items, dispatcher, request.match_info["owner"], body)
        return web.json_response(receipt)

    async def delete(request: web.Request) -> web.Response:
        receipt = await asyncio.to_thread(
            delete_item, dispatcher, request.match_info["owner"], request.match_info["item_id"]
        )
        return web.json_response(receipt)

    base = "/reef/namespaces/{owner}/data"
    app.router.add_post(base, create)
    app.router.add_get(base, query)
    app.router.add_post(f"{base}/tasks", create_tasks)
    app.router.add_post(f"{base}/metadata", update)
    # An item ID can hold '/', so the route takes the rest of the path.
    app.router.add_get(base + "/{item_id:.+}", read)
    app.router.add_delete(base + "/{item_id:.+}", delete)


__all__ = ["register_data_routes"]
