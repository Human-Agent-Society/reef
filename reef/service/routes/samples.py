from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.request_service import RequestService
from reef.service.sample_processing import get_processing, list_processing, read_processing_log
from reef.service.samples import MAX_SAMPLES_PER_PAGE, list_samples, read_next_job


def register_sample_reads(app: web.Application, *, request_service: RequestService) -> None:
    """The list of a scenario's samples."""
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
        changed = request.query.get("changed", "false")
        if changed not in ("true", "false"):
            raise web.HTTPBadRequest(text="changed must be true or false")
        listing = await asyncio.to_thread(
            list_processing,
            dispatcher,
            request.match_info["scenario"],
            state=request.query.get("state"),
            before=before,
            limit=limit,
            changed=changed == "true",
        )
        return web.json_response(listing)

    async def processing_detail(request: web.Request) -> web.Response:
        task = await asyncio.to_thread(
            get_processing, dispatcher, request.match_info["scenario"], request.match_info["task_id"]
        )
        if task is None:
            raise web.HTTPNotFound(text="sample processing task not found")
        return web.json_response(task)

    async def processing_log(request: web.Request) -> web.Response:
        try:
            attempt = int(request.query["attempt"]) if "attempt" in request.query else None
            after = int(request.query.get("after", "-1"))
            limit = int(request.query.get("limit", "500"))
        except ValueError:
            raise web.HTTPBadRequest(text="attempt, after and limit must be integers") from None
        log = await asyncio.to_thread(
            read_processing_log,
            dispatcher,
            request.match_info["scenario"],
            request.match_info["task_id"],
            attempt=attempt,
            after=after,
            limit=limit,
        )
        if log is None:
            raise web.HTTPNotFound(text="sample processing task not found")
        return web.json_response(log)

    app.router.add_get("/reef/scenarios/{scenario}/samples", samples)
    app.router.add_get("/reef/scenarios/{scenario}/sample-processing", processing)
    app.router.add_get("/reef/scenarios/{scenario}/sample-processing/{task_id}", processing_detail)
    app.router.add_get("/reef/scenarios/{scenario}/sample-processing/{task_id}/logs", processing_log)


def register_next_job_route(app: web.Application, *, request_service: RequestService) -> None:
    """The read of a trainer's next automatic job, which is the trainer's state that control holds."""
    dispatcher = request_service.dispatcher

    async def next_job(request: web.Request) -> web.Response:
        listing = await asyncio.to_thread(
            read_next_job, dispatcher, request.match_info["scenario"], request.query.get("component") or None
        )
        return web.json_response(listing)

    app.router.add_get("/reef/scenarios/{scenario}/next-job", next_job)


__all__ = ["register_next_job_route", "register_sample_reads"]
