from __future__ import annotations

import hashlib
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


def create_authentication_middleware(tokens: str | Iterable[str] | None):
    """Bearer-token authentication against the accepted token set.

    The token is the service boundary: whoever presents an accepted token is
    trusted. Several tokens may be accepted at once so a caller (typically a
    gateway) can rotate its credential without downtime. Per-user
    authorization is the gateway's job, not Reef's. Error translation lives in
    :mod:`reef.service.errors`.
    """

    # Compare digests in constant time so the response time leaks nothing
    # about how much of a token matched, and keep no plaintext tokens around.
    accepted = tuple(_digest(token) for token in normalize_tokens(tokens))

    def _authorized(headers) -> bool:
        authorization = headers.get("Authorization")
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            return False
        presented = _digest(authorization.removeprefix("Bearer "))
        matched = False
        for digest in accepted:
            matched |= secrets.compare_digest(presented, digest)
        return matched

    @web.middleware
    async def authenticate(request: web.Request, handler):
        # /healthz stays reachable without credentials: liveness probes (the
        # bundled configs' ready checks, orchestrators) cannot authenticate.
        if accepted and request.path != "/healthz" and not _authorized(request.headers):
            raise web.HTTPUnauthorized(text="invalid service token")
        return await handler(request)

    return authenticate


__all__ = ["create_authentication_middleware", "normalize_tokens"]
