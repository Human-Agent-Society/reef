"""Opt-in browser access to a Reef service from explicitly trusted consoles."""

from collections.abc import Iterable
from urllib.parse import urlsplit

from aiohttp import web

_METHODS = frozenset({"GET", "POST", "DELETE"})
_HEADERS = frozenset({"authorization", "content-type", "x-reef-scenario"})
_EXPOSE = "x-reef-agent-record-id, x-reef-release-id, x-reef-artifact-version"


def console_origins(values: Iterable[str]) -> tuple[str, ...]:
    """Validate exact HTTP(S) origins; wildcards and URL paths are not origins."""
    if isinstance(values, str):
        raise ValueError("reef.console_origins must be a list of origins")
    origins: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("reef.console_origins must contain only origins")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or "*" in value
            or any(character.isspace() for character in value)
            or value != f"{parsed.scheme}://{parsed.netloc}"
        ):
            raise ValueError("reef.console_origins must contain exact HTTP(S) origins without paths or wildcards")
        # Force validation of an explicitly supplied port.
        _ = parsed.port
        origins.append(value)
    return tuple(dict.fromkeys(origins))


def configure_browser_access(app: web.Application, origins: Iterable[str]) -> None:
    """Allow CORS preflights before authentication, retaining auth for actual calls.

    Streamed responses prepare headers inside their handler, so CORS headers
    are attached with on_response_prepare, including for authentication errors.
    Unlisted browser origins are refused before a handler can change state.
    Non-browser clients without an Origin header retain their existing behavior.
    """
    allowed = frozenset(console_origins(origins))
    if not allowed:
        return

    @web.middleware
    async def browser_access(request: web.Request, handler):
        origin = request.headers.get("Origin")
        if origin is None:
            return await handler(request)
        if origin not in allowed:
            raise web.HTTPForbidden(text="console origin is not allowed")
        if request.method == "OPTIONS" and "Access-Control-Request-Method" in request.headers:
            method = request.headers["Access-Control-Request-Method"]
            headers = {
                value.strip().lower()
                for value in request.headers.get("Access-Control-Request-Headers", "").split(",")
                if value.strip()
            }
            if method not in _METHODS or not headers.issubset(_HEADERS):
                raise web.HTTPForbidden(text="console request method or headers are not allowed")
            response = web.Response(status=204)
            response.headers["Access-Control-Allow-Methods"] = ", ".join(sorted(_METHODS))
            response.headers["Access-Control-Allow-Headers"] = ", ".join(sorted(_HEADERS))
            response.headers["Access-Control-Max-Age"] = "600"
            if request.headers.get("Access-Control-Request-Private-Network") == "true":
                response.headers["Access-Control-Allow-Private-Network"] = "true"
            return response
        return await handler(request)

    async def prepare(request: web.Request, response: web.StreamResponse) -> None:
        response.headers.add("Vary", "Origin")
        if request.headers.get("Origin") in allowed:
            response.headers["Access-Control-Allow-Origin"] = request.headers["Origin"]
            response.headers["Access-Control-Expose-Headers"] = _EXPOSE

    app.middlewares.insert(0, browser_access)
    app.on_response_prepare.append(prepare)
