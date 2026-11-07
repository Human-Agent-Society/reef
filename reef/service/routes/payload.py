"""Shared request body checks for the JSON routes."""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web

from reef.service.lifecycle import REQUEST_ACCEPTANCE
from reef.service.wire import strict_json_object


async def read_object(request: web.Request, *, strict: bool = False) -> dict[str, Any]:
    """The decoded JSON body, which must be an object; anything else is a 400, never a traceback.

    The body is read and decoded once. ``strict`` also refuses repeated keys and non-finite numbers, as
    routed requests require. In a managed deployment the upload has a deadline, and a request whose upload
    ends after the deployment began stopping is refused before it reaches any model.
    """
    acceptance = request.app.get(REQUEST_ACCEPTANCE)
    if acceptance is None:
        data = await request.read()
    else:
        try:
            data = await asyncio.wait_for(request.read(), acceptance.settings.body_timeout_seconds)
        except TimeoutError as exc:
            raise web.HTTPRequestTimeout(text="request body deadline exceeded") from exc
        if acceptance.stopping:
            raise web.HTTPServiceUnavailable(text="routing deployment is stopping")
    if strict:
        try:
            return strict_json_object(data)
        except ValueError as exc:
            raise web.HTTPBadRequest(
                text="request body must be one JSON object without repeated keys or non-finite numbers"
            ) from exc
    # The body is buffered now, so this decodes it with its declared charset without reading again.
    payload = await request.json()
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be an object")
    return payload
