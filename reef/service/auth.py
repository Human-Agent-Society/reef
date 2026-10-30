from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from collections.abc import Iterable, Mapping
from urllib.parse import urlencode

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

#: Where the middleware keeps the digest of the service token a request presented, for ``page_query``.
TOKEN_DIGEST = web.RequestKey("reef_token_digest", bytes)


def page_key_for_digest(digest: bytes, scenario: str) -> str:
    """The page key of ``scenario`` for the token whose sha256 is ``digest``, as hex.

    A request's page and a step's page are links a person opens in a browser,
    which sends no Authorization header, so the link carries a credential in
    its query; the token there would reach the model of the session that
    prints the link and the provider behind it. The page key is an HMAC of the
    scenario keyed by the token's digest (the service keeps only digests): it
    opens those two pages of that one scenario, and the token cannot be read
    back from it. Only the service derives it, inside the page paths its
    responses carry (``page_query``); clients never see it apart from a link."""
    return hmac.new(digest, PAGE_KEY_CONTEXT + scenario.encode("utf-8"), hashlib.sha256).hexdigest()


def page_query(request: web.Request, scenario: str) -> str:
    """The query a harness page link carries for ``scenario``: the scenario and, when ``request`` presented a service
    token, that scenario's page key, so a browser opens the page without the headers."""
    digest = request.get(TOKEN_DIGEST)
    key = {} if digest is None else {"key": page_key_for_digest(digest, scenario)}
    return urlencode({"scenario": scenario, **key})


#: What a download signature's HMAC covers before the path, so a signature is good for nothing but one download.
DOWNLOAD_SIGNATURE_CONTEXT = b"reef-download\n"
#: The most seconds that a signed download URL stays valid.
MAX_DOWNLOAD_URL_SECONDS = 24 * 3600

#: The downloads that a signed URL opens, matched on the raw path: a data item's archive and an artifact's files.
DOWNLOAD_ROUTES = re.compile(r"^/reef/(namespaces/[^/]+/data/.+/archive|scenarios/[^/]+/artifacts/[^/]+/files)$")


#: The query parameters of a download that its signature covers: what the download reads.
DOWNLOAD_PARAMETERS = ("path", "component")


def download_signature(digest: bytes, raw_path: str, parameters: Mapping[str, str], expires_at: int) -> str:
    """The signature of a download of ``raw_path`` with the query ``parameters`` (``DOWNLOAD_PARAMETERS``) until
    ``expires_at``, keyed by the digest of an accepted token, as hex. It opens that one download until then, and the
    token cannot be read back from it."""
    query = urlencode(sorted((name, parameters[name]) for name in DOWNLOAD_PARAMETERS if parameters.get(name)))
    message = f"{raw_path}\n{query}\n{expires_at}".encode()
    return hmac.new(digest, DOWNLOAD_SIGNATURE_CONTEXT + message, hashlib.sha256).hexdigest()


def signed_download_url(request: web.Request, expires_in_seconds: int) -> tuple[str, int]:
    """The signed URL of the download that ``request`` asks for, and when it expires, for a request that
    presented a service token. A service without tokens needs no signature, so its URL carries none.

    The URL is the request's raw path with its ``DOWNLOAD_PARAMETERS``, ``expires`` and ``signature``. Raise
    ``ValueError`` for an ``expires_in_seconds`` outside 1 to ``MAX_DOWNLOAD_URL_SECONDS``.
    """
    if not 1 <= expires_in_seconds <= MAX_DOWNLOAD_URL_SECONDS:
        raise ValueError(f"expires_in must be 1 to {MAX_DOWNLOAD_URL_SECONDS} seconds")
    expires_at = int(time.time()) + expires_in_seconds
    raw_path = request.rel_url.raw_path
    parameters = {name: request.query[name] for name in DOWNLOAD_PARAMETERS if request.query.get(name)}
    query = dict(parameters)
    digest = request.get(TOKEN_DIGEST)
    if digest is not None:
        query.update(expires=str(expires_at), signature=download_signature(digest, raw_path, parameters, expires_at))
    return (f"{raw_path}?{urlencode(query)}" if query else raw_path), expires_at


#: The harness pages a person opens by a link: the only routes that read a credential from the query string.
PAGE_ROUTES = re.compile(r"^/reef/harness/(requests/[^/]+|releases/\d{1,9})/page$")

#: The routes a recipe's own evaluation calls reach, matched on the raw path so an escaped scenario name stays one
#: segment: the only routes an evaluation token opens.
EVALUATION_ROUTES = re.compile(r"^/reef/scenarios/[^/]+(/components/[^/]+|/artifacts/[^/]+)?/evaluation/v1/.+$")


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
    carry it. The token itself is never read from a query. Every other route,
    and any request that carries the header, is judged by the header alone. A
    request with no Authorization header may present the token in
    ``x-api-key`` instead, the header the Anthropic dialect's clients send.

    A download (``DOWNLOAD_ROUTES``) also opens on a GET with no Authorization
    header that carries ``expires`` in the future and the ``signature`` of
    its raw path, its ``DOWNLOAD_PARAMETERS`` and ``expires`` for an accepted token
    (``download_signature``): a signed URL that a browser or another service
    fetches without the token (``signed_download_url``).

    ``evaluation_tokens`` open the evaluation routes (``EVALUATION_ROUTES``)
    only, on POST. The service hands one to the recipe for its evaluation
    rollouts and proposer, which run candidate code: that code can sample the
    served release, and nothing else of the service. With no ``tokens``,
    authentication is off and these tokens change nothing.
    """

    # Compare digests in constant time so the response time leaks nothing
    # about how much of a token matched, and keep no plaintext tokens around.
    accepted = tuple(_digest(token) for token in normalize_tokens(tokens))
    evaluation_accepted = tuple(_digest(token) for token in normalize_tokens(evaluation_tokens))

    def _presented(request: web.Request) -> str | None:
        """The credential to judge: the Bearer header's, else the Anthropic dialect's ``x-api-key``."""
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
            # A client speaking the Anthropic dialect (an evaluation rollout, a
            # proposer bound to this service) presents its key in this header.
            return api_key
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

    def _signature_opens(request: web.Request) -> bool:
        """Whether a download GET with no Authorization header carries a valid, unexpired signature."""
        raw_path = request.rel_url.raw_path
        if "Authorization" in request.headers or request.method != "GET" or not DOWNLOAD_ROUTES.match(raw_path):
            return False
        signature = request.query.get("signature") or ""
        try:
            expires_at = int(request.query.get("expires", ""))
        except ValueError:
            return False
        if not signature or expires_at < time.time():
            return False
        presented = signature.encode("utf-8")
        matched = False
        for digest in accepted:
            expected = download_signature(digest, raw_path, request.query, expires_at)
            matched |= secrets.compare_digest(presented, expected.encode("utf-8"))
        return matched

    def _authorized(request: web.Request) -> bool:
        if _page_key_opens(request) or _signature_opens(request):
            return True
        credential = _presented(request)
        if credential is None:
            return False
        presented = _digest(credential)
        matched = False
        for digest in accepted:
            matched |= secrets.compare_digest(presented, digest)
        if matched:
            # A service token, not an evaluation one: the routes that hand out page links derive their key from it.
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
    "DOWNLOAD_ROUTES",
    "EVALUATION_ROUTES",
    "MAX_DOWNLOAD_URL_SECONDS",
    "PAGE_KEY_CONTEXT",
    "PAGE_ROUTES",
    "create_authentication_middleware",
    "download_signature",
    "normalize_tokens",
    "page_key_for_digest",
    "page_query",
    "signed_download_url",
]
