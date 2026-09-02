"""Shared request body checks for the JSON routes."""

from __future__ import annotations

from typing import Any

from aiohttp import web


async def read_object(request: web.Request) -> dict[str, Any]:
    """The decoded JSON body, which must be an object; anything else is a 400, never a traceback."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be an object")
    return payload
