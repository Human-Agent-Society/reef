"""HTTP routes of the data service's scenario reads: an artifact's files and the scenario's training jobs.

``reef.service.scenario_files`` handles each route; this module reads the request and writes the response.
"""

from __future__ import annotations

import asyncio

from aiohttp import web

from reef.core.errors import UnknownScenario
from reef.service.auth import signed_download_url
from reef.service.request_service import RequestService
from reef.service.routes.data import download_seconds
from reef.service.scenario_files import artifact_file, artifact_files
from reef.train.job_queue import scenario_jobs


def register_scenario_file_routes(app: web.Application, *, request_service: RequestService) -> None:
    dispatcher = request_service.dispatcher

    async def files(request: web.Request) -> web.Response | web.FileResponse:
        """The artifact's file list, with the backend's URL of each file when it signs them; with ``path``, that file;
        with ``signed=1``, a signed URL of either. ``expires_in`` sets the life of the URLs."""
        scenario, artifact_id = request.match_info["scenario"], request.match_info["artifact_id"]
        component = request.query.get("component") or None
        path = request.query.get("path", "")
        if request.query.get("signed") == "1":
            url, expires_at = signed_download_url(request, download_seconds(request))
            return web.json_response({"url": url, "expires_at": expires_at})
        if not path:
            listing = await asyncio.to_thread(
                artifact_files, dispatcher, scenario, artifact_id, component, download_seconds(request)
            )
            return web.json_response(listing)
        file = await asyncio.to_thread(artifact_file, dispatcher, scenario, artifact_id, component, path)
        return web.FileResponse(file, headers={"Content-Disposition": f'attachment; filename="{file.name}"'})

    async def jobs(request: web.Request) -> web.Response:
        scenario = request.match_info["scenario"]
        if not await asyncio.to_thread(dispatcher.has_scenario, scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        listing = await asyncio.to_thread(scenario_jobs, dispatcher.queue, scenario)
        return web.json_response({"scenario": scenario, "jobs": listing}, headers={"Cache-Control": "no-store"})

    app.router.add_get("/reef/scenarios/{scenario}/artifacts/{artifact_id}/files", files)
    app.router.add_get("/reef/scenarios/{scenario}/jobs", jobs)


__all__ = ["register_scenario_file_routes"]
