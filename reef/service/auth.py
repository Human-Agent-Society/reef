from __future__ import annotations

import hashlib
import hmac
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


#: What a page key's HMAC covers before the scenario name, so the key is good for nothing but a page link.
PAGE_KEY_CONTEXT = b"reef-page\n"

#: Where the middleware keeps the digest of the service token a request presented, for ``page_key_of``.
TOKEN_DIGEST = web.RequestKey("reef_token_digest", bytes)


def page_key_for_digest(digest: bytes, scenario: str) -> str:
    """The page key of ``scenario`` for the token whose sha256 is ``digest``, as hex.

    A request's page and a step's page are links a person opens in a browser,
    which sends no Authorization header, so the link carries a credential in
    its query; the token there would reach the model of the session that
    prints the link and the provider behind it. The page key is an HMAC of the
    scenario keyed by the token's digest (the service keeps only digests): it
    opens those two pages of that one scenario, and the token cannot be read
    back from it. Only the service derives it; clients read it from
    ``page_key_of``'s responses."""
    return hmac.new(digest, PAGE_KEY_CONTEXT + scenario.encode("utf-8"), hashlib.sha256).hexdigest()


def page_key_of(request: web.Request, scenario: str) -> str | None:
    """The page key of ``scenario`` for the service token ``request`` presented; ``None`` when it presented none
    (authentication off, or a page opened by its key)."""
    digest = request.get(TOKEN_DIGEST)
    return None if digest is None else page_key_for_digest(digest, scenario)


#: The harness pages a person opens by a link: the only routes that read a credential from the query string.
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

    The two harness pages (``PAGE_ROUTES``) also accept a credential in the
    query on a GET that carries no Authorization header: they are links a
    person opens in a browser, which cannot send the header. ``?key=`` is the
    page key of the query's ``scenario`` (``page_key_for_digest``) for an
    accepted token: it opens those two pages of that scenario alone, a
    request whose ``x-reef-scenario`` header names another scenario is
    refused, and the token cannot be read back from it, so the links the
    harness wrapper and pi's extension print, which a session's model reads,
    carry it. ``?token=`` still opens them too; that token sits in the URL,
    in the browser's history and in whatever logs request lines. Every other
    route, and any request that carries the header, is judged by the header
    alone. A request with no Authorization header may present the token in
    ``x-api-key`` instead, the header the Anthropic dialect's clients send.

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

    def _page_key_opens(request: web.Request) -> bool:
        """Whether ``?key=`` on a page GET with no Authorization header is the page key of the query's scenario."""
        if "Authorization" in request.headers or request.method != "GET" or not PAGE_ROUTES.match(request.path):
            return False
        key = request.query.get("key") or ""
        scenario = request.query.get("scenario", "").strip()
        if not key or not scenario:
            return False
        # The page renders the header's scenario when both are present: a key opens only the scenario it names.
        header = request.headers.get("x-reef-scenario")
        if header is not None and header.strip() != scenario:
            return False
        presented = key.encode("utf-8")
        matched = False
        for digest in accepted:
            matched |= secrets.compare_digest(presented, page_key_for_digest(digest, scenario).encode("utf-8"))
        return matched

    def _authorized(request: web.Request) -> bool:
        if _page_key_opens(request):
            return True
        credential = _presented(request)
        if credential is None:
            return False
        presented = _digest(credential)
        matched = False
        for digest in accepted:
            matched |= secrets.compare_digest(presented, digest)
        if matched:
            # A service token, not an evaluation one: the routes that hand out page keys derive them from it.
            request[TOKEN_DIGEST] = presented
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


__all__ = [
    "EVALUATION_ROUTES",
    "PAGE_KEY_CONTEXT",
    "PAGE_ROUTES",
    "create_authentication_middleware",
    "normalize_tokens",
    "page_key_for_digest",
    "page_key_of",
]
