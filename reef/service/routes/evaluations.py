from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.routes.payload import read_object


def register_evaluation_routes(app: web.Application, *, evaluations: ArtifactEvaluations) -> None:
    async def start(request: web.Request) -> web.Response:
        body = await read_object(request)
        stored = await asyncio.to_thread(evaluations.start, request.match_info["scenario"], body)
        return web.json_response(stored.to_json(), status=202)

    async def evaluation(request: web.Request) -> web.Response:
        evaluation_id = request.match_info["evaluation_id"]
        stored = await asyncio.to_thread(evaluations.get, request.match_info["scenario"], evaluation_id)
        if stored is None:
            raise web.HTTPNotFound(text=f"no evaluation {evaluation_id!r}")
        return web.json_response(stored.to_json(), headers={"Cache-Control": "no-store"})

    async def newest(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        # A limit that is not a positive integer is a ValueError, which answers HTTP 400.
        listed = await asyncio.to_thread(evaluations.newest, scenario, limit=int(request.query.get("limit", "50")))
        return web.json_response(
            {"scenario": scenario, "evaluations": [stored.to_json() for stored in listed]},
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_post("/reef/scenarios/{scenario}/evaluations", start)
    app.router.add_get("/reef/scenarios/{scenario}/evaluations", newest)
    app.router.add_get("/reef/scenarios/{scenario}/evaluations/{evaluation_id}", evaluation)


__all__ = ["register_evaluation_routes"]
