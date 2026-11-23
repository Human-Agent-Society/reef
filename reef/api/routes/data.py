"""HTTP routes of a data owner's items (RFC #1028), under ``/reef/namespaces/{owner}/data``, of its deleted items,
under ``/reef/namespaces/{owner}/deleted-items``, and of its collections (RFC #1312), under
``/reef/namespaces/{owner}/collections``.

``reef.control.operations.data`` and ``reef.control.operations.collections`` handle each route; this module reads the request and writes the response.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from aiohttp import web

from reef.api.auth import add_signed_download, signed_download_url
from reef.api.request_handling import RequestHandling
from reef.api.routes.payload import read_object
from reef.control.operations.collections import (
    HOLD_ROUTES,
    add_to_scenario,
    delete_collection,
    hold_set,
    list_collections,
    list_holds,
    read_collection,
    release_set,
    save_collection,
)
from reef.control.operations.data import (
    DataAccess,
    create_items,
    create_task_items,
    create_task_references,
    create_trajectory_items,
    delete_item,
    deleted_listing,
    group_items,
    jsonl_trajectories,
    parsed_metadata,
    purge_item,
    query_items,
    read_item,
    replace_task_item,
    replace_task_reference,
    restore_items,
    set_listing,
    update_items,
)
from reef.control.operations.data_transfer import TRANSFER_KINDS, item_archive, start_transfer, transfer_status
from reef.control.operations.tasks import DEFAULT_MAX_TASK_ARCHIVE_BYTES
from reef.storage.data.store import MAX_QUERY_LIMIT

#: The content types of a task archive.
TASK_ARCHIVE_TYPES = frozenset({"application/gzip", "application/x-gzip"})
#: The content type of a create or a replace of task items by reference to the files in the task file store.
TASK_REFERENCE_TYPE = "application/json"
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


async def write_body_file(request: web.Request, path: Path, max_bytes: int) -> None:
    """Write the body of ``request`` to the file ``path``, in chunks; HTTP 413 for a body of more than ``max_bytes``.

    The body goes to a file, not to memory, so ``max_bytes`` bounds disk use.
    """
    size = 0
    with path.open("wb") as file:
        async for chunk in request.content.iter_chunked(CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise web.HTTPRequestEntityTooLarge(max_size=max_bytes, actual_size=size)
            file.write(chunk)


def register_data_routes(
    app: web.Application,
    *,
    request_handling: RequestHandling,
    max_task_archive_bytes: int = DEFAULT_MAX_TASK_ARCHIVE_BYTES,
) -> None:
    """Register the data routes. ``max_task_archive_bytes`` is ``data.max_task_archive_bytes``: the most bytes of the
    body of a task archive."""
    data = DataAccess.of(request_handling.dispatcher)

    async def start(request: web.Request) -> web.Response:
        body = await read_object(request)
        segment = request.match_info["transfer"]
        status = await asyncio.to_thread(start_transfer, data, request.match_info["owner"], segment, body)
        return web.json_response(status, status=202)

    async def transfer(request: web.Request) -> web.Response:
        status = await asyncio.to_thread(
            transfer_status,
            data,
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
                item_archive, data, request.match_info["owner"], request.match_info["item_id"], Path(work)
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
        receipt = await asyncio.to_thread(create_items, data, request.match_info["owner"], body)
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
            documents = [
                (item_id, document)
                for item_id, document, characters in jsonl_trajectories([("the body", text.splitlines())])
            ]
            return create_trajectory_items(data, owner, documents, metadata)

        return web.json_response(await asyncio.to_thread(create), status=201)

    async def create_tasks(request: web.Request) -> web.Response:
        """Create task items from a tar.gz of task directories, or by reference to the files in the task file store."""
        owner = request.match_info["owner"]
        if request.content_type == TASK_REFERENCE_TYPE:
            body = await read_object(request.clone(client_max_size=MAX_CREATE_BYTES))
            return web.json_response(await asyncio.to_thread(create_task_references, data, owner, body), status=201)
        if request.content_type not in TASK_ARCHIVE_TYPES:
            raise web.HTTPUnsupportedMediaType(
                text="task items come as a tar.gz archive with Content-Type application/gzip, or as references with"
                " Content-Type application/json"
            )
        metadata = parsed_metadata(request.query.get("metadata"))
        with tempfile.TemporaryDirectory(prefix="reef-task-archive-") as work:
            archive = Path(work) / "tasks.tar.gz"
            await write_body_file(request, archive, max_task_archive_bytes)
            receipt = await asyncio.to_thread(create_task_items, data, owner, archive, metadata)
        return web.json_response(receipt, status=201)

    async def replace_task(request: web.Request) -> web.Response:
        """Replace a task item's files with the task directory of a tar.gz body (RFC #1359), or by reference to the
        files in the task file store."""
        owner, item_id = request.match_info["owner"], request.match_info["item_id"]
        if request.content_type == TASK_REFERENCE_TYPE:
            body = await read_object(request)
            return web.json_response(await asyncio.to_thread(replace_task_reference, data, owner, item_id, body))
        if request.content_type not in TASK_ARCHIVE_TYPES:
            raise web.HTTPUnsupportedMediaType(
                text="a task's files come as a tar.gz archive with Content-Type application/gzip, or as a reference"
                " with Content-Type application/json"
            )
        with tempfile.TemporaryDirectory(prefix="reef-task-archive-") as work:
            archive = Path(work) / "task.tar.gz"
            await write_body_file(request, archive, max_task_archive_bytes)
            receipt = await asyncio.to_thread(replace_task_item, data, owner, item_id, archive)
        return web.json_response(receipt)

    async def query(request: web.Request) -> web.Response:
        content = query_int(request, "content", 0)
        if content not in (0, 1):
            raise web.HTTPBadRequest(text="content must be 0 or 1")
        group_key = request.query.get("group_key")
        if group_key is None and "group_value" in request.query:
            raise web.HTTPBadRequest(text="group_value needs group_key")
        listing = await asyncio.to_thread(
            query_items,
            data,
            request.match_info["owner"],
            kind=request.query.get("kind") or None,
            scenario=request.query.get("scenario") or None,
            after=query_int(request, "after", 0),
            limit=query_int(request, "limit", MAX_QUERY_LIMIT),
            content=content == 1,
            group=None if group_key is None else (group_key, request.query.get("group_value")),
            benchmark=request.query.get("benchmark") or None,
            trajectory_set=request.query.get("trajectory_set") or None,
            id_prefix=request.query.get("id_prefix") or None,
            before=query_int(request, "before", 0) or None,
        )
        return web.json_response(listing)

    async def sets(request: web.Request) -> web.Response:
        before = query_int(request, "before", 0)
        listing = await asyncio.to_thread(
            set_listing,
            data,
            request.match_info["owner"],
            kind=request.match_info["kind"],
            scenario=request.query.get("scenario") or None,
            before=before or None,
            limit=query_int(request, "limit", MAX_QUERY_LIMIT),
        )
        return web.json_response(listing)

    async def groups(request: web.Request) -> web.Response:
        before = query_int(request, "before", 0)
        listing = await asyncio.to_thread(
            group_items,
            data,
            request.match_info["owner"],
            kind=request.query.get("kind") or None,
            key=request.match_info["key"],
            scenario=request.query.get("scenario") or None,
            before=before or None,
            limit=query_int(request, "limit", MAX_QUERY_LIMIT),
        )
        return web.json_response(listing)

    async def read(request: web.Request) -> web.Response:
        item = await asyncio.to_thread(read_item, data, request.match_info["owner"], request.match_info["item_id"])
        return web.json_response(item)

    async def update(request: web.Request) -> web.Response:
        body = await read_object(request)
        receipt = await asyncio.to_thread(update_items, data, request.match_info["owner"], body)
        return web.json_response(receipt)

    async def delete(request: web.Request) -> web.Response:
        receipt = await asyncio.to_thread(
            delete_item, data, request.match_info["owner"], request.match_info["item_id"]
        )
        return web.json_response(receipt)

    async def deleted(request: web.Request) -> web.Response:
        listing = await asyncio.to_thread(
            deleted_listing,
            data,
            request.match_info["owner"],
            kind=request.query.get("kind") or None,
            before=query_int(request, "before", 0) or None,
            limit=query_int(request, "limit", MAX_QUERY_LIMIT),
        )
        return web.json_response(listing)

    async def restore(request: web.Request) -> web.Response:
        body = await read_object(request)
        return web.json_response(await asyncio.to_thread(restore_items, data, request.match_info["owner"], body))

    async def purge(request: web.Request) -> web.Response:
        receipt = await asyncio.to_thread(purge_item, data, request.match_info["owner"], request.match_info["item_id"])
        return web.json_response(receipt)

    async def collections(request: web.Request) -> web.Response:
        return web.json_response(
            await asyncio.to_thread(
                list_collections, data, request.match_info["owner"], counts=request.query.get("counts") == "true"
            )
        )

    async def collection(request: web.Request) -> web.Response:
        """Store, read or delete one collection, by the method."""
        owner, name = request.match_info["owner"], request.match_info["name"]
        if request.method == "PUT":
            body = await read_object(request)
            return web.json_response(await asyncio.to_thread(save_collection, data, owner, name, body))
        handler = read_collection if request.method == "GET" else delete_collection
        return web.json_response(await asyncio.to_thread(handler, data, owner, name))

    async def collection_scenarios(request: web.Request) -> web.Response:
        body = await read_object(request)
        receipt = await asyncio.to_thread(
            add_to_scenario, data, request.match_info["owner"], request.match_info["name"], body
        )
        return web.json_response(receipt)

    async def holds(request: web.Request) -> web.Response:
        owner, scenario = request.match_info["owner"], request.match_info["scenario"]
        kind = HOLD_ROUTES[request.match_info["sets"]]
        return web.json_response(await asyncio.to_thread(list_holds, data, owner, scenario, kind))

    async def hold(request: web.Request) -> web.Response:
        """Hold one benchmark or trajectory set, or end its hold, by the method."""
        names = (
            request.match_info["owner"],
            request.match_info["scenario"],
            HOLD_ROUTES[request.match_info["sets"]],
            request.match_info["name"],
        )
        if request.method == "PUT":
            body = await read_object(request)
            return web.json_response(await asyncio.to_thread(hold_set, data, *names, body))
        split = request.query.get("split")
        return web.json_response(await asyncio.to_thread(release_set, data, *names, split))

    base = "/reef/namespaces/{owner}/data"
    transfers = "{transfer:" + "|".join(TRANSFER_KINDS) + "}"
    app.router.add_post(base, create)
    app.router.add_get(base, query)
    app.router.add_post(f"{base}/trajectories", create_trajectories)
    app.router.add_post(f"{base}/tasks", create_tasks)
    app.router.add_put(f"{base}/tasks/{{item_id}}", replace_task)
    app.router.add_post(f"{base}/metadata", update)
    # These come before the item routes, whose ID takes the rest of the path.
    app.router.add_post(f"{base}/{transfers}", start)
    app.router.add_get(f"{base}/{transfers}/{{transfer_id}}", transfer)
    app.router.add_get(f"{base}/groups/{{key}}", groups)
    app.router.add_get(f"{base}/sets/{{kind}}", sets)
    add_signed_download(app, base + "/{item_id:.+}/archive", archive)
    # An item ID can hold '/', so the route takes the rest of the path.
    app.router.add_get(base + "/{item_id:.+}", read)
    app.router.add_delete(base + "/{item_id:.+}", delete)
    deleted_base = "/reef/namespaces/{owner}/deleted-items"
    app.router.add_get(deleted_base, deleted)
    app.router.add_post(f"{deleted_base}/restore", restore)
    app.router.add_delete(deleted_base + "/{item_id:.+}", purge)
    stored = "/reef/namespaces/{owner}/collections"
    app.router.add_get(stored, collections)
    for method in ("PUT", "GET", "DELETE"):
        app.router.add_route(method, stored + "/{name}", collection)
    app.router.add_post(stored + "/{name}/scenarios", collection_scenarios)
    held = "/reef/namespaces/{owner}/scenarios/{scenario}/{sets:" + "|".join(HOLD_ROUTES) + "}"
    app.router.add_get(held, holds)
    for method in ("PUT", "DELETE"):
        app.router.add_route(method, held + "/{name}", hold)


__all__ = ["download_seconds", "register_data_routes"]
