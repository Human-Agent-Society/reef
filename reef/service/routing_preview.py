"""Automatic selection through the Semantic Router's routing preview.

``POST /api/v1/routing/preview`` asks an unmodified router which target a
Scenario's private entrypoint selects for one request, without sending the
request to any model. Reef makes one such call for each automatic request and
checks the reply against the routing policy the request captured: its
manifest and the router runtime hash it was published with. The checked reply
becomes the request's ``RoutingDecision``; the provider call that follows
never asks the router again.

Each call goes to the router's loopback management listener with the
selection-only bearer token and the expected runtime hash. The body holds
only the Scenario's entrypoint and the request's messages, reduced to the
fields the router's strict ``IntentRequest`` decoder accepts. Reef's keyword
rules read only message text, so tools, generation parameters and response
formats are left out. The caller's credentials and provider secrets are never
sent. Nothing is retried, proxied or redirected, router error text is never
relayed, and no log line carries a token or request text.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from aiohttp import ClientConnectionError, ClientError, ClientResponse, ClientSession, ClientTimeout, DummyCookieJar

from reef.service.routing import (
    DEFAULT_RULE,
    RoutingAdapter,
    RoutingDecision,
    RoutingRequestError,
    RoutingScenario,
    scenario_entrypoint,
)
from reef.service.wire import strict_json_object

PREVIEW_PATH = "/api/v1/routing/preview"
#: The runtime hash a preview must run against, and the one it actually used (pkg/headers/config_hash.go).
EXPECTED_HASH_HEADER = "x-sr-bench-expected-config-hash"
RETURNED_HASH_HEADER = "x-vsr-config-hash"
#: The router accepts a preview body of at most this many bytes. A larger one fails in its outer
#: ``http.MaxBytesReader`` (apiserver/middleware.go:36), which the pinned native router reports as 400
#: INVALID_INPUT rather than 413. Reef refuses such a body locally with 413 and never sends it.
PREVIEW_BODY_LIMIT_BYTES = 10 * 1024 * 1024
#: The router may still be writing a reply this long after its request deadline (config/api_routing_preview.go:12).
RESPONSE_WRITE_ALLOWANCE_SECONDS = 5.0
#: A reply echoes the evaluated prompt as ``original_text``; Go's JSON encoder writes one byte as at most six.
RESPONSE_EXPANSION = 6
#: Room for every reply field besides the echoed prompt.
RESPONSE_ENVELOPE_BYTES = 1024 * 1024
#: The fields the router's IntentMessage declares; strict decoding rejects any other (intent_request_messages.go:22).
MESSAGE_FIELDS = frozenset(
    {"role", "name", "content", "tool_calls", "tool_call_id", "function_call", "refusal", "reasoning_content"}
)
#: IntentMessage fields decoded as Go strings, where JSON null means empty.
STRING_MESSAGE_FIELDS = ("name", "tool_call_id")
#: The only methods whose selection the router reports as stateless (extproc/router_learning_preview.go:219).
STATELESS_METHODS = frozenset({"static", "single"})
#: EvalResponse diagnostics that must be absent or empty: a signal that failed or fell back to an
#: unknown-signal policy may have turned a rule match into a default decision.
EMPTY_DIAGNOSTICS = {
    "decision_error": "",
    "signal_errors": {},
    "signal_error_matches": {},
    "applied_unknown_policies": {},
}
#: Far deeper than any chat message; bounds the copy's recursion.
MAX_JSON_DEPTH = 256
ROUTER_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
READ_CHUNK_BYTES = 64 * 1024
#: The message of a readiness preview. Any selection valid for the Scenario passes, whichever rule it names.
READINESS_TEXT = "Reef routing readiness check."


@dataclass(frozen=True)
class PreviewSettings:
    """The router's loopback management listener, its selection-only token, and its preview timeout.

    Values: ``base_url`` is exactly ``http://127.0.0.1:<port>`` with a port of at least 1 and no
    credentials, path, query or fragment. ``token`` is a non-empty string of printable ASCII other than
    space (code points 33 to 126). ``request_timeout_seconds`` is an int from 1 to 300 and must equal the
    router's ``global.services.api.routing_preview.request_timeout_seconds``.
    Purpose: service assembly makes it from the managed child's settings; ``RoutingPreviewClient`` reads it.
    ``preview_url`` is ``base_url + PREVIEW_PATH``. ``http_timeout_seconds`` is
    ``request_timeout_seconds + RESPONSE_WRITE_ALLOWANCE_SECONDS``: how long the client waits for one reply.
    """

    base_url: str
    token: str = field(repr=False)
    request_timeout_seconds: int = 10

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        try:
            port = parts.port
        except ValueError as exc:
            raise ValueError("routing preview URL has an invalid port") from exc
        if (
            parts.scheme != "http"
            or parts.hostname != "127.0.0.1"
            or port is None
            or port < 1
            or parts.username is not None
            or parts.password is not None
            or parts.path
            or parts.query
            or parts.fragment
        ):
            raise ValueError("routing preview URL must be http://127.0.0.1:<port>")
        if not isinstance(self.token, str) or not self.token or any(not 33 <= ord(char) <= 126 for char in self.token):
            raise ValueError("routing preview token must be nonempty printable ASCII without spaces")
        if type(self.request_timeout_seconds) is not int or not 1 <= self.request_timeout_seconds <= 300:
            raise ValueError("routing preview request timeout must be a whole number of seconds from 1 to 300")

    @property
    def preview_url(self) -> str:
        return self.base_url + PREVIEW_PATH

    @property
    def http_timeout_seconds(self) -> float:
        return self.request_timeout_seconds + RESPONSE_WRITE_ALLOWANCE_SECONDS


def unrepresentable(message: str) -> RoutingRequestError:
    """Request content the preview cannot carry unchanged; the caller's request is at fault."""
    return RoutingRequestError(message, code="routing_input_unrepresentable", status=400)


def router_failure(message: str, code: str = "routing_protocol") -> RoutingRequestError:
    """A router reply Reef cannot use; the router, not the caller, is at fault."""
    return RoutingRequestError(message, code=code, status=502)


def json_copy(value: object, depth: int = 0) -> object:
    """A new copy of a JSON value that encoding would not change.

    Raises:
        ValueError: The value holds something JSON cannot carry exactly, or nests too deeply.
    """
    if depth > MAX_JSON_DEPTH:
        raise ValueError("value nests too deeply")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("number is not finite")
        return value
    if isinstance(value, list):
        return [json_copy(item, depth + 1) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("object key is not a string")
            copied[key] = json_copy(item, depth + 1)
        return copied
    raise ValueError(f"{type(value).__name__} is not a JSON value")


def preview_messages(payload: Mapping[str, object]) -> list[object]:
    """Copies of the request's messages with only IntentMessage fields; the payload itself is never changed."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise unrepresentable("routed requests need a non-empty messages list")
    projected: list[object] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise unrepresentable("every message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise unrepresentable("every message needs a role string")
        for name in STRING_MESSAGE_FIELDS:
            value = message.get(name)
            if value is not None and not isinstance(value, str):
                raise unrepresentable(f"message {name} must be a string")
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise unrepresentable("message tool_calls must be a list")
        kept = {name: value for name, value in message.items() if name in MESSAGE_FIELDS}
        try:
            projected.append(json_copy(kept))
        except ValueError as exc:
            raise unrepresentable(f"message content cannot be sent as JSON: {exc}") from exc
    return projected


def preview_body(scenario: RoutingScenario, payload: Mapping[str, object]) -> bytes:
    """The exact preview body: the Scenario's private entrypoint and the projected messages."""
    request = {"model": scenario_entrypoint(scenario.name), "messages": preview_messages(payload)}
    try:
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise unrepresentable("message text is not valid Unicode") from exc
    if len(body) > PREVIEW_BODY_LIMIT_BYTES:
        raise RoutingRequestError(
            "routed messages exceed the router's preview size limit", code="routing_input_too_large", status=413
        )
    return body


def router_error_code(data: bytes) -> str | None:
    """The router's error code, when its reply is a well-formed error; never its message."""
    try:
        error = strict_json_object(data).get("error")
    except ValueError:
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and ROUTER_ERROR_CODE.fullmatch(code) else None


def router_rejection(status: int, data: bytes) -> RoutingRequestError:
    """Reef's error for a router reply other than 200; only the router's error code is kept."""
    code = router_error_code(data)
    label = f" ({code})" if code else ""
    # The pinned router also answers an oversized body this way; Reef's own 413 check runs first.
    if status == 400 and code == "INVALID_INPUT":
        return RoutingRequestError("the router rejected the routed messages" + label, code="routing_input_rejected")
    if status == 412 and code == "CONFIG_HASH_MISMATCH":
        return RoutingRequestError(
            "the router no longer runs the policy Reef selected with" + label,
            code="routing_runtime_changed",
            status=503,
        )
    if status == 413:  # Not observed from the pinned router; kept for a router that reports its size limit.
        return RoutingRequestError(
            "routed messages exceed the router's preview size limit" + label,
            code="routing_input_too_large",
            status=413,
        )
    if status == 429:
        return RoutingRequestError(
            "the router is at its preview concurrency limit" + label, code="routing_overloaded", status=429
        )
    if status == 503:
        return RoutingRequestError(
            "the router could not complete the preview" + label, code="routing_unavailable", status=503
        )
    if status == 504:
        return RoutingRequestError("the router's preview timed out" + label, code="routing_timeout", status=504)
    if status in (401, 403):
        return router_failure("the router refused Reef's selection credentials" + label, "routing_credentials")
    return router_failure(f"the router answered the preview with HTTP {status}{label}")


def checked_selection(
    routing: RoutingAdapter, scenario: RoutingScenario, reply: Mapping[str, object], returned_hashes: list[str]
) -> RoutingDecision:
    """The decision a 200 reply makes, after checking every field Reef relies on against ``routing``."""

    def unexpected(name: str) -> RoutingRequestError:
        return router_failure(f"the router's preview reply has an unexpected {name}", "routing_selection")

    expected_hash = routing.router_config_hash
    if expected_hash is None:
        raise RoutingRequestError("routing has no verified router runtime yet", code="routing_unpublished", status=503)
    decision = reply.get("decision_result")
    if not isinstance(decision, Mapping):
        raise unexpected("decision_result")
    selection_mode = reply.get("selection_provenance")
    if not isinstance(selection_mode, Mapping):
        raise unexpected("selection_provenance")
    if returned_hashes != [expected_hash]:
        raise unexpected(RETURNED_HASH_HEADER + " header")
    required = (
        ("config_hash", reply.get("config_hash"), expected_hash),
        ("selection_provenance.config_hash", selection_mode.get("config_hash"), expected_hash),
        ("selection_status", reply.get("selection_status"), "selected"),
        ("recipe", reply.get("recipe"), scenario.name),
        ("requested_model", reply.get("requested_model"), scenario_entrypoint(scenario.name)),
        ("decision_result.algorithm", decision.get("algorithm"), "static"),
        ("selection_provenance.mode", selection_mode.get("mode"), "stateless"),
    )
    for name, actual, wanted in required:
        if not isinstance(actual, str) or actual != wanted:
            raise unexpected(name)
    # Both flags are always serialized; only an explicit false shows an unsampled, state-free choice.
    for flag in ("state_dependent", "sampled"):
        if selection_mode.get(flag) is not False:
            raise unexpected(f"selection_provenance.{flag}")
    method = reply.get("selection_method")
    if not isinstance(method, str) or method not in STATELESS_METHODS:
        raise unexpected("selection_method")
    if decision.get("plugins") not in (None, []):
        raise unexpected("decision_result.plugins")
    for name, empty in EMPTY_DIAGNOSTICS.items():
        if reply.get(name) not in (None, empty):
            raise unexpected(name)
    rule = decision.get("decision_name")
    if not isinstance(rule, str) or not rule or reply.get("routing_decision") != rule:
        raise unexpected("decision")
    target_id = scenario.target_for_rule(rule)
    if target_id is None:
        raise unexpected("decision outside the Scenario")
    if reply.get("selected_model") != target_id:
        raise unexpected("selected_model")
    target = routing.snapshot.targets.get(target_id)
    if target is None or not target.enabled or target_id not in scenario.target_ids:
        raise unexpected("selected_model")
    return RoutingDecision(
        target,
        routing.snapshot.version,
        scenario.name,
        routing.snapshot.automatic_model,
        "default" if rule == DEFAULT_RULE else "rule",
        rule,
        expected_hash,
    )


async def read_bounded(response: ClientResponse, limit: int) -> bytes:
    """The whole reply body, refusing one larger than ``limit`` before it is buffered."""
    too_large = router_failure("the router's preview reply exceeds its size limit")
    if response.content_length is not None and response.content_length > limit:
        raise too_large
    data = bytearray()
    while chunk := await response.content.read(READ_CHUNK_BYTES):
        data += chunk
        if len(data) > limit:
            raise too_large
    return bytes(data)


class RoutingPreviewClient:
    """One HTTP session to the router's preview route, opened by ``start`` and released by ``close``.

    Values: ``settings`` is the ``PreviewSettings`` the client was made with.
    State: ``session`` is None (not started, or closed) or one open ``ClientSession``. ``start`` opens it
    and is refused while it is open; ``close`` releases it, does nothing when there is none, and allows a
    later ``start``. ``select`` and ``check_ready`` need an open session.
    Operations: ``select(routing, scenario, payload)`` sends exactly one preview for an automatic request
    of ``scenario``, a Scenario of ``routing.snapshot``, and returns a decision with ``target_id ∈
    target_ids(scenario)`` or raises ``RoutingRequestError``. It starts no background task; a cancelled
    ``select`` closes its connection and raises ``CancelledError``. ``check_ready`` sends one such preview
    per Scenario. One service owns one client; requests share its session.
    """

    def __init__(self, settings: PreviewSettings) -> None:
        if not isinstance(settings, PreviewSettings):
            raise ValueError("a routing preview client needs preview settings")
        self.settings = settings
        self.session: ClientSession | None = None

    async def start(self) -> None:
        if self.session is not None:
            raise RuntimeError("routing preview client is already started")
        # No environment proxies, netrc credentials, cookies or transparent decompression.
        self.session = ClientSession(cookie_jar=DummyCookieJar(), trust_env=False, auto_decompress=False)

    async def close(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            await session.close()

    async def select(
        self, routing: RoutingAdapter, scenario: RoutingScenario, payload: Mapping[str, object]
    ) -> RoutingDecision:
        """The decision the router makes for an automatic request in ``scenario`` under ``routing``'s runtime.

        Raises:
            RoutingRequestError: The request cannot be previewed, or the router's answer is not a usable
                selection; ``status`` and ``code`` say which.
            ValueError: The caller broke this method's contract.
            RuntimeError: The client is not started.
        """
        session = self.session
        if session is None:
            raise RuntimeError("routing preview client is not started")
        expected_hash = routing.router_config_hash
        if expected_hash is None:
            # A selection is valid only under a runtime the owner verified; never under a guessed hash.
            raise RoutingRequestError(
                "routing has no verified router runtime yet", code="routing_unpublished", status=503
            )
        if routing.snapshot.scenarios.get(scenario.name) != scenario:
            raise ValueError("Scenario is not part of the routing snapshot")
        if payload.get("model") != routing.snapshot.automatic_model:
            raise ValueError("routing preview selects only automatic requests; explicit targets resolve locally")
        body = preview_body(scenario, payload)
        headers = {
            "Authorization": "Bearer " + self.settings.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            EXPECTED_HASH_HEADER: expected_hash,
        }
        try:
            async with session.post(
                self.settings.preview_url,
                data=body,
                headers=headers,
                allow_redirects=False,
                timeout=ClientTimeout(total=self.settings.http_timeout_seconds),
            ) as response:
                status = response.status
                content_type = response.content_type
                encoding = response.headers.get("Content-Encoding", "identity")
                returned_hashes = response.headers.getall(RETURNED_HASH_HEADER, [])
                data = await read_bounded(response, RESPONSE_EXPANSION * len(body) + RESPONSE_ENVELOPE_BYTES)
        except TimeoutError as exc:
            raise RoutingRequestError(
                "the router did not answer the preview in time", code="routing_timeout", status=504
            ) from exc
        except ClientConnectionError as exc:
            # Refused, reset or closed before any reply arrived, as when the router stops or restarts.
            raise RoutingRequestError(
                "the router's management listener is unavailable", code="routing_unavailable", status=503
            ) from exc
        except ClientError as exc:
            # A reply did start: a malformed status line or headers, or a body that broke off before its end.
            raise router_failure("the router's preview reply was malformed or incomplete") from exc
        if status != 200:
            raise router_rejection(status, data)
        if content_type != "application/json" or encoding.strip().lower() != "identity":
            raise router_failure("the router's preview reply is not plain JSON")
        try:
            reply = strict_json_object(data)
        except ValueError as exc:
            raise router_failure("the router's preview reply is not one JSON object") from exc
        return checked_selection(routing, scenario, reply, list(returned_hashes))

    async def check_ready(self, routing: RoutingAdapter) -> None:
        """Preview one fixed message per Scenario; any selection valid for that Scenario passes.

        Raises:
            RoutingRequestError: A Scenario's preview failed or returned an unusable selection.
        """
        payload = {
            "model": routing.snapshot.automatic_model,
            "messages": [{"role": "user", "content": READINESS_TEXT}],
        }
        for scenario in routing.snapshot.scenarios.values():
            await self.select(routing, scenario, payload)


__all__ = [
    "EXPECTED_HASH_HEADER",
    "PREVIEW_BODY_LIMIT_BYTES",
    "PREVIEW_PATH",
    "READINESS_TEXT",
    "RETURNED_HASH_HEADER",
    "PreviewSettings",
    "RoutingPreviewClient",
    "checked_selection",
    "preview_body",
]
