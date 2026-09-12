"""ASGI middleware stamping Reef correlation headers onto CORAL gateway traffic.

Maps CORAL identity headers to ``x-reef-scenario`` + ``x-reef-tag-*`` (headers
only; provider-native bodies are mirrored into ``extra_headers`` for proxy
hops), captures Reef receipts from the response, and journals one
:class:`CallRecord` per call. One discovery problem = one scenario; agent and
worktree identity stay in tags.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Protocol

from recipes.beta.coral.journal import CallJournal, CallRecord

logger = logging.getLogger(__name__)

SCENARIO_HEADER = b"x-reef-scenario"
RECORD_ID_RESPONSE_HEADER = b"x-reef-agent-record-id"
RELEASE_ID_RESPONSE_HEADER = b"x-reef-release-id"

#: CORAL identity headers (stamped upstream by CoralGatewayMiddleware) and the
#: reef tag names they map to. Tag values are opaque to reef; these names are
#: this example's contract with its reporter.
CORAL_TO_REEF_TAGS: dict[bytes, str] = {
    b"x-coral-agent-id": "coral-agent",
    b"x-coral-session-id": "coral-commit",
}

_API_PREFIXES = (
    "/v1/messages",
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
    "/completions",
)


def _is_api_path(path: str) -> bool:
    return any(path.startswith(p) for p in _API_PREFIXES)


class AsgiApp(Protocol):
    """The downstream ASGI application (LiteLLM, or another middleware)."""

    async def __call__(self, scope: dict, receive: Any, send: Any) -> Any: ...


class ReefGatewayMiddleware:
    """See module docstring.

    ``extra_tags`` lets the launcher attach run-level parent links (e.g.
    ``{"coral-run": run_id}``) once, instead of per request.
    """

    def __init__(
        self,
        app: AsgiApp,
        *,
        scenario: str,
        journal: CallJournal,
        extra_tags: Mapping[str, str] | None = None,
        body_extra_headers: bool = True,
    ) -> None:
        if not scenario or not scenario.strip():
            raise ValueError("scenario must be a non-empty string")
        self.app = app
        self.scenario = scenario.strip()
        self.journal = journal
        self.extra_tags = dict(extra_tags or {})
        #: Also mirror the stamped headers into the JSON body's
        #: ``extra_headers`` — LiteLLM's own per-request mechanism for
        #: carrying headers to the upstream. A proxy hop builds a fresh
        #: upstream request and drops inbound headers, so headers alone
        #: only reach reef on direct deployments.
        self.body_extra_headers = body_extra_headers

    async def __call__(self, scope: dict, receive: Any, send: Any) -> Any:
        if scope.get("type") != "http" or not _is_api_path(scope.get("path", "")):
            return await self.app(scope, receive, send)

        request_id = uuid.uuid4().hex[:12]

        # -- request side: read CORAL identity, stamp reef headers ----------
        coral: dict[str, str] = {}
        new_headers: list[tuple[bytes, bytes]] = []
        for raw_name, raw_value in scope.get("headers", []):
            name = bytes(raw_name).lower()
            tag = CORAL_TO_REEF_TAGS.get(name)
            if tag is not None:
                coral[tag] = bytes(raw_value).decode("latin-1")
            # Drop any inbound reef headers: the adapter owns this channel,
            # and a client must not be able to redirect its own scenario.
            if name == SCENARIO_HEADER or name.startswith(b"x-reef-tag-"):
                continue
            new_headers.append((raw_name, raw_value))

        tags = {**self.extra_tags, **coral}
        new_headers.append((SCENARIO_HEADER, self.scenario.encode("latin-1")))
        for tag_name, tag_value in tags.items():
            new_headers.append((b"x-reef-tag-" + tag_name.encode("latin-1"), tag_value.encode("latin-1")))
        scope = dict(scope)
        scope["headers"] = new_headers

        reef_headers = {SCENARIO_HEADER.decode("latin-1"): self.scenario}
        for tag_name, tag_value in tags.items():
            reef_headers[f"x-reef-tag-{tag_name}"] = tag_value
        if self.body_extra_headers:
            receive = _extra_headers_receive(receive, reef_headers, scope)

        # -- response side: capture the reef receipt ------------------------
        status_code = 0
        record_id: str | None = None
        release_id: str | None = None
        body_tail = bytearray()  # receipt + usage extraction after the stream ends

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code, record_id, release_id
            if message.get("type") == "http.response.start":
                status_code = message.get("status", 0)
                for raw_name, raw_value in message.get("headers", []):
                    lowered = bytes(raw_name).lower()
                    # Suffix match: a proxy hop that does forward provider
                    # headers usually prefixes them (e.g. LiteLLM's
                    # llm_provider-x-reef-agent-record-id).
                    if lowered.endswith(RECORD_ID_RESPONSE_HEADER):
                        record_id = bytes(raw_value).decode("latin-1")
                    elif lowered.endswith(RELEASE_ID_RESPONSE_HEADER):
                        release_id = bytes(raw_value).decode("latin-1")
            elif message.get("type") == "http.response.body":
                body_tail.extend(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            body = bytes(body_tail)
            if record_id is None:
                record_id = _receipt_from_body(body)
            usage = _usage_from_body(body)
            entry = CallRecord(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                scenario=self.scenario,
                agent_id=coral.get("coral-agent", "unknown"),
                commit_hash=coral.get("coral-commit", "unknown"),
                path=scope.get("path", ""),
                status_code=status_code,
                agent_record_id=record_id,
                release_id=release_id,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                tags=tags,
            )
            try:
                self.journal.append(entry)
            except OSError as exc:  # never take the data path down with us
                logger.warning("call journal append failed: %s", exc)


def _receipt_from_body(body: bytes) -> str | None:
    """Extract ``agent_record_id`` from a response body, if reef's receipt survived.

    Handles the streaming shape (an SSE frame whose JSON payload carries a
    top-level ``"reef"`` object) and the JSON-body shape. Returns ``None``
    when the provider hop stripped the receipt — correlation then rests on
    the tags reef stored with the INFERENCE record.
    """
    if not body:
        return None
    text = body.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        reef = payload.get("reef") if isinstance(payload, dict) else None
        if isinstance(reef, dict) and isinstance(reef.get("agent_record_id"), str):
            return reef["agent_record_id"]
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        reef = payload.get("reef")
        if isinstance(reef, dict) and isinstance(reef.get("agent_record_id"), str):
            return reef["agent_record_id"]
    return None


def _usage_from_body(body: bytes) -> dict:
    """Token accounting from a JSON body or the last SSE usage chunk.

    OpenAI-shaped ``usage`` objects only ({prompt,completion}_tokens ints);
    anything else yields {} — accounting is best-effort by design.
    """
    if not body:
        return {}
    text = body.decode("utf-8", errors="replace")
    candidates = []
    if text.lstrip().startswith("{"):
        candidates.append(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                candidates.append(line[5:].strip())
    usage: dict = {}
    for raw in candidates:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            found = payload["usage"]
            cleaned = {
                key: value
                for key, value in found.items()
                if key in ("prompt_tokens", "completion_tokens") and isinstance(value, int)
            }
            if cleaned:
                usage = cleaned  # last usage wins (final SSE chunk is authoritative)
    return usage


def _extra_headers_receive(receive: Any, reef_headers: Mapping[str, str], scope: dict) -> Any:
    """Wrap ``receive`` so the JSON body carries ``extra_headers``.

    Buffers the request body, merges the reef headers into the body's
    ``extra_headers`` (caller-provided entries win nothing — the adapter
    owns this channel, same rule as the header path), and replays it as a
    single message with a corrected ``content-length``. Non-JSON bodies
    pass through untouched.
    """

    async def buffered() -> dict:
        parts: list[bytes] = []
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                return {"body": b"".join(parts), "passthrough": message}
            parts.append(message.get("body", b""))
            if not message.get("more_body"):
                return {"body": b"".join(parts), "passthrough": None}

    state = {"done": False}

    async def replay() -> dict:
        if state["done"]:
            return {"type": "http.request", "body": b"", "more_body": False}
        result = await buffered()
        state["done"] = True
        if result["passthrough"] is not None:
            return result["passthrough"]
        body = result["body"]
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            merged = dict(payload.get("extra_headers") or {})
            merged.update(reef_headers)
            payload["extra_headers"] = merged
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            new_headers = []
            for raw_name, raw_value in scope.get("headers", []):
                if bytes(raw_name).lower() == b"content-length":
                    new_headers.append((raw_name, str(len(body)).encode("latin-1")))
                else:
                    new_headers.append((raw_name, raw_value))
            scope["headers"] = new_headers
        return {"type": "http.request", "body": body, "more_body": False}

    return replay
