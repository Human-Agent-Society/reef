from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.request_service import RequestService, page_headers


#: How long the install route renders before it sends the script's first lines with a spinner.
INSTALL_PREAMBLE_DELAY_SECONDS = 0.5


def register_system_routes(app: web.Application, *, request_service: RequestService) -> None:
    """Control's own harness routes, the pages and the step records, and ``/reef/status``."""

    async def harness_release_page(request: web.Request) -> web.Response:
        step = int(request.match_info["step"])
        headers = page_headers(request.headers, request.query)
        # The chain's links open the way this page was opened: the query parameters travel with them.
        link_query = {key: request.query[key] for key in ("scenario", "key") if key in request.query}
        page = await asyncio.to_thread(request_service.harness_release_page, headers, step, link_query)
        return web.Response(text=page, content_type="text/html")

    async def harness_request_page(request: web.Request) -> web.Response:
        headers = page_headers(request.headers, request.query)
        # The version page link opens the way this page was opened: the query parameters travel with it.
        link_query = {key: request.query[key] for key in ("scenario", "key") if key in request.query}
        page = await asyncio.to_thread(
            request_service.harness_request_page, headers, request.match_info["record_id"], link_query
        )
        # The page changes every few seconds while the step runs; nothing should serve a stale copy.
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def harness_request_progress(request: web.Request) -> web.Response:
        progress = await asyncio.to_thread(
            request_service.harness_request_progress, request.headers, request.match_info["record_id"]
        )
        # The state changes every few seconds while the step runs; nothing should serve a stale copy.
        return web.json_response(progress, headers={"Cache-Control": "no-store"})

    async def harness_step_records(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(
            request_service.harness_step_records,
            request.headers,
            int(request.match_info["step"]),
            request.query.get("path"),
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def status(request: web.Request) -> web.Response:
        value = await asyncio.to_thread(lambda: request_service.dispatcher.build_training_status())
        return web.json_response(value)

    # One to nine digits: a step that is no number, or longer than any catalog, is no row; the router's own 404 answers it.
    app.router.add_get(r"/reef/harness/releases/{step:\d{1,9}}/page", harness_release_page)
    app.router.add_get(r"/reef/harness/releases/{step:\d{1,9}}/records", harness_step_records)
    app.router.add_get("/reef/harness/requests/{record_id}/page", harness_request_page)
    app.router.add_get("/reef/harness/requests/{record_id}/progress", harness_request_progress)
    app.router.add_get("/reef/status", status)


__all__ = ["register_system_routes"]
