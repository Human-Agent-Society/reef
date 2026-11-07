from __future__ import annotations

import asyncio

from aiohttp import web

from reef.service.artifact_imports import ArtifactImports
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object


def register_artifact_import_routes(
    app: web.Application, *, request_service: RequestService, imports: ArtifactImports
) -> None:
    async def import_artifact(request: web.Request) -> web.Response:
        payload = await read_object(request)
        refused = web.HTTPBadRequest(
            text='an import is the object {"source": "<uri>"} or {"item": "<artifact item ID>"}, with an optional '
            '"serving_profile"'
        )
        keys = set(payload) - {"serving_profile"}
        if keys == {"source"}:
            key = "source"
        elif keys == {"item"}:
            key = "item"
        else:
            raise refused
        value, profile = payload[key], payload.get("serving_profile")
        if not isinstance(value, str) or not value.strip():
            raise refused
        if profile is not None and (not isinstance(profile, str) or not profile.strip()):
            raise web.HTTPBadRequest(text="serving_profile must be a non-empty string")
        profile = None if profile is None else profile.strip()
        scenario, component = request.match_info["scenario"], request.query.get("component")
        if key == "source":
            receipt = await asyncio.to_thread(
                imports.import_source,
                request_service.dispatcher,
                scenario,
                component,
                value.strip(),
                serving_profile=profile,
            )
        else:
            receipt = await asyncio.to_thread(
                imports.import_item, request_service.dispatcher, scenario, component, value, profile
            )
        return web.json_response(receipt, status=201)

    app.router.add_post("/reef/scenarios/{scenario}/artifacts", import_artifact)


__all__ = ["register_artifact_import_routes"]
