from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
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

    app.router.add_get("/reef/scenarios/{scenario}/samples", samples)


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
