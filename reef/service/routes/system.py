from __future__ import annotations

import asyncio

from aiohttp import web

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.service.auth import page_query
from reef.service.errors import translate_error
from reef.service.install_script import render_install_failure, render_install_preamble, render_streamed_install
from reef.service.request_service import RequestService, page_headers
from reef.service.routes.payload import read_object


#: How long the install route renders before it sends the script's first lines with a spinner.
INSTALL_PREAMBLE_DELAY_SECONDS = 0.5


def register_health_route(app: web.Application) -> None:
    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app.router.add_get("/healthz", health)


def register_system_routes(app: web.Application, *, request_service: RequestService) -> None:
    async def harness(request: web.Request) -> web.Response:
        release_id = request.query.get("release_id") or None
        manifest = await asyncio.to_thread(request_service.harness_manifest, request.headers, release_id)
        return web.json_response(
            manifest,
            headers={"x-reef-release-id": manifest["release_id"]},
        )

    async def harness_install(request: web.Request) -> web.StreamResponse:
        release_id = request.query.get("release_id") or None
        adapter = request.query.get("adapter")
        rendering = asyncio.ensure_future(
            asyncio.to_thread(request_service.harness_install_script, request.headers, adapter, release_id)
        )
        done, _ = await asyncio.wait({rendering}, timeout=INSTALL_PREAMBLE_DELAY_SECONDS)
        if done:
            # A quick render answers as a whole, so a failure keeps its HTTP status.
            return web.Response(text=rendering.result(), content_type="text/x-shellscript")
        # A slow render (a new scenario is created and recovered first) starts the script with a spinner, so the
        # person piping it into a shell sees progress. A failure after that point is the script's own exit.
        response = web.StreamResponse()
        response.content_type = "text/x-shellscript"
        response.charset = "utf-8"
        await response.prepare(request)
        await response.write(render_install_preamble().encode())
        try:
            body = await rendering
        except Exception as exc:
            translated = translate_error(exc)
            if translated is None:
                raise
            body = render_install_failure(translated.status, translated.text or str(exc))
        await response.write(render_streamed_install(body).encode())
        await response.write_eof()
        return response

    async def harness_releases(request: web.Request) -> web.Response:
        catalog = await asyncio.to_thread(request_service.harness_releases, request.headers)
        # Each row names its step's page, the step being its position oldest first; the link carries a page key in
        # place of the token.
        query = page_query(request, catalog["scenario"])
        rows = [
            {**row, "page_path": f"/reef/harness/releases/{step}/page?{query}"}
            for step, row in enumerate(catalog["releases"])
        ]
        return web.json_response({**catalog, "releases": rows})

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

    async def harness_proposals(request: web.Request) -> web.Response:
        payload = await read_object(request)
        answer = await asyncio.to_thread(request_service.harness_propose, request.headers, payload)
        return web.json_response(answer)

    async def status(request: web.Request) -> web.Response:
        value = await asyncio.to_thread(lambda: request_service.dispatcher.build_training_status())
        return web.json_response(value)

    async def adapters(request: web.Request) -> web.Response:
        names = await asyncio.to_thread(available_adapters)
        entries = []
        for name in names:
            try:
                descriptor = await asyncio.to_thread(get_adapter, name)
            except DescriptorError:
                continue
            install = descriptor.install
            entries.append(
                {
                    "name": name,
                    "binary": descriptor.binary,
                    "trajectory_format": descriptor.trajectory_format,
                    "model_bindings": sorted(descriptor.model_binding),
                    "install": (
                        None
                        if install is None
                        else {"kind": install.kind, "package": install.package, "version": install.version}
                    ),
                }
            )
        return web.json_response({"adapters": entries})

    app.router.add_get("/reef/harness", harness)
    app.router.add_get("/reef/harness/install", harness_install)
    app.router.add_get("/reef/harness/releases", harness_releases)
    # One to nine digits: a step that is no number, or longer than any catalog, is no row; the router's own 404 answers it.
    app.router.add_get(r"/reef/harness/releases/{step:\d{1,9}}/page", harness_release_page)
    app.router.add_get(r"/reef/harness/releases/{step:\d{1,9}}/records", harness_step_records)
    app.router.add_get("/reef/harness/requests/{record_id}/page", harness_request_page)
    app.router.add_get("/reef/harness/requests/{record_id}/progress", harness_request_progress)
    app.router.add_post("/reef/harness/proposals", harness_proposals)
    app.router.add_get("/reef/harness/adapters", adapters)
    app.router.add_get("/reef/status", status)


__all__ = ["register_system_routes"]
