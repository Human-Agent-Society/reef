"""HTTP routes of a data owner's items (RFC #1028), under ``/reef/namespaces/{owner}/data``.

``reef.service.data`` handles each route; this module reads the request and writes the response.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from aiohttp import web

from reef.service.auth import add_signed_download, signed_download_url
from reef.service.data import (
    create_items,
    create_task_items,
    create_trajectory_items,
    delete_item,
    jsonl_trajectories,
    parsed_metadata,
    query_items,
    read_item,
    update_items,
)
from reef.service.data_transfer import TRANSFER_KINDS, item_archive, start_transfer, transfer_status
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.tasks import MAX_UNPACKED_BYTES
from reef.storage.data import MAX_QUERY_LIMIT

#: The content types of a task archive.
TASK_ARCHIVE_TYPES = frozenset({"application/gzip", "application/x-gzip"})
#: The content types of a JSON Lines upload of trajectories: one ATIF document a line.
TRAJECTORY_LINES_TYPES = frozenset({"application/x-ndjson", "application/jsonl"})
#: The bytes read from the body at a time.
CHUNK_BYTES = 1024 * 1024
#: The most bytes of a create body, JSON or JSON Lines: 1000 trajectories of an ATIF upload before RFC #1028.
MAX_CREATE_BYTES = 10 * 1024 * 1024


def query_int(request: web.Request, name: str, default: int) -> int:
    try:
        return int(request.query.get(name, str(default)))
    except ValueError:
        raise web.HTTPBadRequest(text=f"{name} must be an integer") from None


def download_seconds(request: web.Request) -> int:
    """The life of a signed download URL that ``request`` asks for: ``expires_in`` seconds, one hour by default."""
    return query_int(request, "expires_in", 3600)


def register_data_routes(app: web.Application, *, request_service: RequestService) -> None:
    dispatcher = request_service.dispatcher

    async def start(request: web.Request) -> web.Response:
        body = await read_object(request)
        segment = request.match_info["transfer"]
        status = await asyncio.to_thread(start_transfer, dispatcher, request.match_info["owner"], segment, body)
        return web.json_response(status, status=202)

    async def transfer(request: web.Request) -> web.Response:
        status = await asyncio.to_thread(
            transfer_status,
            dispatcher,
            request.match_info["owner"],
            request.match_info["transfer"],
            request.match_info["transfer_id"],
        )
        return web.json_response(status, headers={"Cache-Control": "no-store"})

    async def archive(request: web.Request) -> web.StreamResponse:
        """A task's files as tar.gz, or another item's content as JSON; with ``signed=1``, a signed URL of it."""
        if request.query.get("signed") == "1":
            url, expires_at = signed_download_url(request, download_seconds(request))
            return web.json_response({"url": url, "expires_at": expires_at})
        with tempfile.TemporaryDirectory(prefix="reef-item-archive-") as work:
            file, media_type = await asyncio.to_thread(
                item_archive, dispatcher, request.match_info["owner"], request.match_info["item_id"], Path(work)
            )
            response = web.StreamResponse(
                headers={"Content-Type": media_type, "Content-Disposition": f'attachment; filename="{file.name}"'}
            )
            response.content_length = file.stat().st_size
            await response.prepare(request)
            # The file goes out in chunks, then the directory is removed.
            with file.open("rb") as source:
                while chunk := await asyncio.to_thread(source.read, CHUNK_BYTES):
                    await response.write(chunk)
            await response.write_eof()
        return response

    async def create(request: web.Request) -> web.Response:
        body = await read_object(request.clone(client_max_size=MAX_CREATE_BYTES))
        receipt = await asyncio.to_thread(create_items, dispatcher, request.match_info["owner"], body)
        return web.json_response(receipt, status=201)

    async def create_trajectories(request: web.Request) -> web.Response:
        if request.content_type not in TRAJECTORY_LINES_TYPES:
            raise web.HTTPUnsupportedMediaType(
                text="trajectory items come as JSON Lines with Content-Type application/x-ndjson"
            )
        metadata = parsed_metadata(request.query.get("metadata"))
        body = await request.clone(client_max_size=MAX_CREATE_BYTES).read()
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise web.HTTPBadRequest(text=f"the JSON Lines body is not UTF-8: {exc}") from None
        owner = request.match_info["owner"]

        def create() -> dict[str, Any]:
            return create_trajectory_items(
                dispatcher, owner, jsonl_trajectories(text.splitlines(), "the body"), metadata
            )

        return web.json_response(await asyncio.to_thread(create), status=201)

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
    transfers = "{transfer:" + "|".join(TRANSFER_KINDS) + "}"
    app.router.add_post(base, create)
    app.router.add_get(base, query)
    app.router.add_post(f"{base}/trajectories", create_trajectories)
    app.router.add_post(f"{base}/tasks", create_tasks)
    app.router.add_post(f"{base}/metadata", update)
    # These come before the item routes, whose ID takes the rest of the path.
    app.router.add_post(f"{base}/{transfers}", start)
    app.router.add_get(f"{base}/{transfers}/{{transfer_id}}", transfer)
    add_signed_download(app, base + "/{item_id:.+}/archive", archive)
    # An item ID can hold '/', so the route takes the rest of the path.
    app.router.add_get(base + "/{item_id:.+}", read)
    app.router.add_delete(base + "/{item_id:.+}", delete)


__all__ = ["download_seconds", "register_data_routes"]
