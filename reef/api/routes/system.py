from __future__ import annotations

import asyncio

from collections.abc import Mapping

from aiohttp import web

from reef.api.artifact_page import harness_artifact_page
from reef.api.request_page import harness_request_page
from reef.api.request_service import RequestService
from reef.api.wire import scenario_header
from reef.control.operations.harness import harness_request_progress, harness_step_records
from reef.core.records_types import SCENARIO_HEADER


def page_headers(headers: Mapping[str, str], query: Mapping[str, str]) -> dict[str, str]:
    """The headers a page route reads, ``?scenario=`` standing in for ``x-reef-scenario`` when that header is absent.

    A page is a link a person opens in a browser, which sends no ``x-reef-*``
    header; the header wins when both are present. The page key stands in for
    the token in :mod:`reef.api.auth`, for the page routes alone.
    """
    merged = dict(headers.items())
    scenario = query.get("scenario", "").strip()
    if scenario and not any(key.lower() == SCENARIO_HEADER for key in merged):
        merged[SCENARIO_HEADER] = scenario
    return merged


def register_system_routes(app: web.Application, *, request_service: RequestService) -> None:
    """Control's own harness routes, the pages and the step records, and ``/reef/status``."""

    dispatcher = request_service.dispatcher

    async def artifact_page(request: web.Request) -> web.Response:
        step = int(request.match_info["step"])
        scenario = scenario_header(page_headers(request.headers, request.query))
        # The chain's links open the way this page was opened: the query parameters travel with them.
        link_query = {key: request.query[key] for key in ("scenario", "key") if key in request.query}
        page = await asyncio.to_thread(harness_artifact_page, dispatcher, scenario, step, link_query)
        return web.Response(text=page, content_type="text/html")

    async def request_page(request: web.Request) -> web.Response:
        scenario = scenario_header(page_headers(request.headers, request.query))
        # The version page link opens the way this page was opened: the query parameters travel with it.
        link_query = {key: request.query[key] for key in ("scenario", "key") if key in request.query}
        page = await asyncio.to_thread(
            harness_request_page, dispatcher, scenario, request.match_info["record_id"], link_query
        )
        # The page changes every few seconds while the step runs; nothing should serve a stale copy.
        return web.Response(text=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def request_progress(request: web.Request) -> web.Response:
        progress = await asyncio.to_thread(
            harness_request_progress, dispatcher, scenario_header(request.headers), request.match_info["record_id"]
        )
        # The state changes every few seconds while the step runs; nothing should serve a stale copy.
        return web.json_response(progress, headers={"Cache-Control": "no-store"})

    async def step_records(request: web.Request) -> web.Response:
        result = await asyncio.to_thread(
            harness_step_records,
            dispatcher,
            scenario_header(request.headers),
            int(request.match_info["step"]),
            request.query.get("path"),
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def status(request: web.Request) -> web.Response:
        value = await asyncio.to_thread(lambda: dispatcher.build_training_status())
        return web.json_response(value)

    # One to nine digits: a step that is no number, or longer than any catalog, is no row; the router's own 404 answers it.
    app.router.add_get(r"/reef/harness/artifacts/{step:\d{1,9}}/page", artifact_page)
    app.router.add_get(r"/reef/harness/artifacts/{step:\d{1,9}}/records", step_records)
    app.router.add_get("/reef/harness/requests/{record_id}/page", request_page)
    app.router.add_get("/reef/harness/requests/{record_id}/progress", request_progress)
    app.router.add_get("/reef/status", status)


__all__ = ["register_system_routes"]
