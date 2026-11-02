from __future__ import annotations

import asyncio

from aiohttp import web

from reef.record2sample.service import SAMPLES_QUEUE
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.sample_processing import list_processing, processing_item
from reef.service.samples import MAX_SAMPLES_PER_PAGE, change_buffer, list_samples, read_buffer


def register_sample_reads(app: web.Application, *, request_service: RequestService) -> None:
    """The list of a scenario's sample store."""
    dispatcher = request_service.dispatcher

    async def samples(request: web.Request) -> web.Response:
        try:
            after = int(request.query.get("after", "0"))
            limit = int(request.query.get("limit", str(MAX_SAMPLES_PER_PAGE)))
        except ValueError:
            raise web.HTTPBadRequest(text="after and limit must be integers") from None
        listing = await asyncio.to_thread(
            list_samples, dispatcher, request.match_info["scenario"], after_sequence=after, limit=limit
        )
        return web.json_response(listing)

    async def processing(request: web.Request) -> web.Response:
        try:
            before = int(request.query["before"]) if "before" in request.query else None
            limit = int(request.query.get("limit", "50"))
        except ValueError:
            raise web.HTTPBadRequest(text="before and limit must be integers") from None
        listing = await asyncio.to_thread(
            list_processing,
            dispatcher,
            request.match_info["scenario"],
            state=request.query.get("state"),
            before=before,
            limit=limit,
        )
        return web.json_response(listing)

    async def processing_detail(request: web.Request) -> web.Response:
        item = await asyncio.to_thread(dispatcher.queue.get, SAMPLES_QUEUE, request.match_info["task_id"])
        if item is None or item.scenario != request.match_info["scenario"]:
            raise web.HTTPNotFound(text="sample processing task not found")
        return web.json_response(processing_item(item))

    app.router.add_get("/reef/scenarios/{scenario}/samples", samples)
    app.router.add_get("/reef/scenarios/{scenario}/sample-processing", processing)
    app.router.add_get("/reef/scenarios/{scenario}/sample-processing/{task_id}", processing_detail)


def register_buffer_routes(app: web.Application, *, request_service: RequestService) -> None:
    """The read and the changes of a trainer's buffer, which is the trainer's state that control holds."""
    dispatcher = request_service.dispatcher

    async def buffer(request: web.Request) -> web.Response:
        listing = await asyncio.to_thread(
            read_buffer, dispatcher, request.match_info["scenario"], request.query.get("component") or None
        )
        return web.json_response(listing)

    async def insert(request: web.Request) -> web.Response:
        body = await read_object(request)
        listing = await asyncio.to_thread(change_buffer, dispatcher, request.match_info["scenario"], body, insert=True)
        return web.json_response(listing)

    async def remove(request: web.Request) -> web.Response:
        body = await read_object(request)
        listing = await asyncio.to_thread(
            change_buffer, dispatcher, request.match_info["scenario"], body, insert=False
        )
        return web.json_response(listing)

    app.router.add_get("/reef/scenarios/{scenario}/buffer", buffer)
    app.router.add_post("/reef/scenarios/{scenario}/buffer/insert", insert)
    app.router.add_post("/reef/scenarios/{scenario}/buffer/remove", remove)


__all__ = ["register_buffer_routes", "register_sample_reads"]
