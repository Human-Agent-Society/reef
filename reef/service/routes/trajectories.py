from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.trajectories import upload_trajectories

#: The most bytes that one trajectory upload request can hold, the same as one record import.
MAX_TRAJECTORY_UPLOAD_BYTES = 10 * 1024 * 1024


def register_trajectory_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def trajectories(request: web.Request) -> web.Response:
        body = await read_object(request.clone(client_max_size=MAX_TRAJECTORY_UPLOAD_BYTES))
        receipt = await asyncio.to_thread(upload_trajectories, request_service.dispatcher, request.headers, body)
        return web.json_response(receipt)

    app.router.add_post("/reef/trajectories", trajectories)
