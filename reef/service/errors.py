"""The single Reef-error-to-HTTP-status table, applied as middleware.

Every route raises domain errors and lets this middleware translate them;
routes never carry their own try/except tables. The table is ordered most
specific first because several entries are subclasses of later ones
(``ArtifactNotFound`` is an ``ArtifactError``, everything is a ``ReefError``).
"""

from __future__ import annotations

import logging

from aiohttp import web

from reef.artifact.artifact import ArtifactConflict, ArtifactError, ArtifactNotFound
from reef.artifact.version_chain import VersionNotRestorable
from reef.core.errors import ReefError, UnknownScenario
from reef.recipe.errors import ScenarioRecipeConflict
from reef.records import RecordConflict
from reef.runtime.inference import UpstreamStatusError
from reef.service.request_service import InferenceRetryTimeout
from reef.surface.weights import WeightVersionMismatch

logger = logging.getLogger(__name__)

#: Ordered most-specific first; the first isinstance match wins.
ERROR_STATUS_TABLE: tuple[tuple[type[Exception], type[web.HTTPError]], ...] = (
    (ArtifactNotFound, web.HTTPNotFound),
    (UnknownScenario, web.HTTPNotFound),
    (ArtifactConflict, web.HTTPConflict),
    (VersionNotRestorable, web.HTTPConflict),
    (RecordConflict, web.HTTPConflict),
    (ScenarioRecipeConflict, web.HTTPConflict),
    (WeightVersionMismatch, web.HTTPConflict),
    (InferenceRetryTimeout, web.HTTPServiceUnavailable),
    (ArtifactError, web.HTTPServiceUnavailable),
    (ReefError, web.HTTPBadRequest),
    (ValueError, web.HTTPBadRequest),
)


#: Upstream 4xx worth relaying verbatim; anything else 4xx becomes 400. The
#: statuses whose aiohttp classes require extra constructor arguments (413's
#: max_size, for one) are deliberately absent — they fall back to 400 rather
#: than crash the translation.
_UPSTREAM_CLIENT_ERRORS: dict[int, type[web.HTTPError]] = {
    400: web.HTTPBadRequest,
    401: web.HTTPUnauthorized,
    403: web.HTTPForbidden,
    404: web.HTTPNotFound,
    408: web.HTTPRequestTimeout,
    409: web.HTTPConflict,
    422: web.HTTPUnprocessableEntity,
    429: web.HTTPTooManyRequests,
}


def _translate_upstream(exc: UpstreamStatusError) -> web.HTTPError:
    """Relay an upstream failure without flattening it into a 500.

    A 4xx means the caller's request was rejected, so it is passed through
    with the upstream's own message — agents parse those to repair the next
    attempt. A 5xx is the upstream's own fault, not the caller's, so it
    becomes 502 rather than blaming the request.
    """
    if 400 <= exc.status < 500:
        return _UPSTREAM_CLIENT_ERRORS.get(exc.status, web.HTTPBadRequest)(text=str(exc))
    return web.HTTPBadGateway(text=str(exc))


def translate_error(exc: Exception) -> web.HTTPError | None:
    """The HTTP error for a domain exception, or None when it is not mapped."""
    if isinstance(exc, UpstreamStatusError):  # before the table: status is dynamic
        return _translate_upstream(exc)
    for error_type, http_error in ERROR_STATUS_TABLE:
        if isinstance(exc, error_type):
            return http_error(text=str(exc))
    return None


@web.middleware
async def translate_errors(request: web.Request, handler):
    """Translate domain errors into HTTP responses via ERROR_STATUS_TABLE."""
    try:
        return await handler(request)
    except tuple(error_type for error_type, _ in ERROR_STATUS_TABLE) as exc:
        translated = translate_error(exc)
        if translated is None:
            raise RuntimeError("mapped service error did not translate") from exc
        logger.warning(
            "%s %s failed with %s: %s (HTTP %d)",
            request.method,
            request.path,
            type(exc).__name__,
            exc,
            translated.status,
        )
        raise translated from exc


__all__ = [
    "ERROR_STATUS_TABLE",
    "translate_error",
    "translate_errors",
]
