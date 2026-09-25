from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Iterable

from aiohttp import web


def normalize_tokens(tokens: str | Iterable[str] | None) -> frozenset[str]:
    """Coerce a configured token value into the set of accepted Bearer tokens.

    Accepts one token, an iterable of tokens, or ``None``. Empty strings are
    dropped, so an unset ``${REEF_TOKEN}`` never becomes a valid credential. An
    empty result disables authentication.
    """

    if tokens is None:
        return frozenset()
    if isinstance(tokens, str):
        tokens = (tokens,)
    normalized = set()
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError(f"reef tokens must be strings, got {type(token).__name__}")
        if token:
            normalized.add(token)
    return frozenset(normalized)


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


#: The harness pages a person opens by a link: the only routes that read the token from the query string.
PAGE_ROUTES = re.compile(r"^/reef/harness/(requests/[^/]+|releases/\d{1,9})/page$")

#: The routes a recipe's own evaluation calls reach, matched on the raw path so an escaped scenario name stays one
#: segment: the only routes an evaluation token opens.
EVALUATION_ROUTES = re.compile(r"^/reef/scenarios/[^/]+(/components/[^/]+)?/evaluation/v1/.+$")


def create_authentication_middleware(
    tokens: str | Iterable[str] | None, *, evaluation_tokens: str | Iterable[str] | None = None
):
    """Bearer-token authentication against the accepted token set.

    The token is the service boundary: whoever presents an accepted token is
    trusted. Several tokens may be accepted at once so a caller (typically a
    gateway) can rotate its credential without downtime. Per-user
    authorization is the gateway's job, not Reef's. Error translation lives in
    :mod:`reef.service.errors`.

    The two harness pages (``PAGE_ROUTES``) also accept the token as
    ``?token=`` on a GET that carries no Authorization header: they are links
    a person opens in a browser, which cannot send the header. The token then
    sits in the URL, in the browser's history and in whatever logs request
    lines, so the deployment that hands out such links is a local one, as the
    profiles that print them are. Every other route, and any request that
    carries the header, is judged by the header alone. A request with no
    Authorization header may present the token in ``x-api-key`` instead, the
    header the Anthropic dialect's clients send.

    ``evaluation_tokens`` open the evaluation routes (``EVALUATION_ROUTES``)
    only, on POST. The service hands one to the recipe for its evaluation
    episodes and proposer, which run candidate code: that code can sample the
    served release, and nothing else of the service. With no ``tokens``,
    authentication is off and these tokens change nothing.
    """

    # Compare digests in constant time so the response time leaks nothing
    # about how much of a token matched, and keep no plaintext tokens around.
    accepted = tuple(_digest(token) for token in normalize_tokens(tokens))
    evaluation_accepted = tuple(_digest(token) for token in normalize_tokens(evaluation_tokens))

    def _presented(request: web.Request) -> str | None:
        """The credential to judge: the Bearer header's, else the Anthropic dialect's ``x-api-key``, else ``?token=`` on a page route a browser opens."""
        authorization = request.headers.get("Authorization")
        if isinstance(authorization, str):
            # The auth-scheme is case-insensitive per RFC 9110 §11.1; only the
            # credential itself stays case-sensitive.
            scheme, separator, credential = authorization.partition(" ")
            if not separator or scheme.lower() != "bearer" or not credential:
                return None
            return credential
        api_key = request.headers.get("x-api-key")
        if isinstance(api_key, str) and api_key:
            # A client speaking the Anthropic dialect (an evaluation episode, a
            # proposer bound to this service) presents its key in this header.
            return api_key
        if request.method == "GET" and PAGE_ROUTES.match(request.path):
            return request.query.get("token") or None
        return None

    def _authorized(request: web.Request) -> bool:
        credential = _presented(request)
        if credential is None:
            return False
        presented = _digest(credential)
        matched = False
        for digest in accepted:
            matched |= secrets.compare_digest(presented, digest)
        if request.method == "POST" and EVALUATION_ROUTES.match(request.rel_url.raw_path):
            for digest in evaluation_accepted:
                matched |= secrets.compare_digest(presented, digest)
        return matched

    @web.middleware
    async def authenticate(request: web.Request, handler):
        # /healthz stays reachable without credentials: liveness probes (the
        # bundled configs' ready checks, orchestrators) cannot authenticate.
        if accepted and request.path != "/healthz" and not _authorized(request):
            raise web.HTTPUnauthorized(text="invalid service token")
        return await handler(request)

    return authenticate


__all__ = ["EVALUATION_ROUTES", "PAGE_ROUTES", "create_authentication_middleware", "normalize_tokens"]
