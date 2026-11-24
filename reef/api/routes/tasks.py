from __future__ import annotations

import asyncio

from aiohttp import web

from reef.api.request_handling import RequestHandling
from reef.control.operations.tasks import list_imported_tasks


def register_task_routes(app: web.Application, *, request_handling: RequestHandling) -> None:
    async def tasks(request: web.Request) -> web.Response:
        listing = await asyncio.to_thread(
            list_imported_tasks, request_handling.dispatcher, request.match_info["scenario"]
        )
        return web.json_response(listing)

    app.router.add_get("/reef/scenarios/{scenario}/tasks", tasks)


__all__ = ["register_task_routes"]
