"""Body ``extra_headers`` mirroring: how the stamp survives a LiteLLM hop."""

from __future__ import annotations

import asyncio
import json

from recipes.beta.coral.journal import CallJournal
from recipes.beta.coral.middleware import ReefGatewayMiddleware


class CaptureDownstream:
    def __init__(self):
        self.body = None
        self.content_length = None

    async def __call__(self, scope, receive, send):
        parts = []
        while True:
            message = await receive()
            parts.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        self.body = b"".join(parts)
        for name, value in scope["headers"]:
            if bytes(name).lower() == b"content-length":
                self.content_length = int(value)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _run(mw, body: bytes, headers):
    async def go():
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        await mw(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": list(headers)},
            receive,
            send,
        )

    asyncio.run(go())


CORAL_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", b"2"),
    (b"x-coral-agent-id", b"agent-1"),
    (b"x-coral-session-id", b"commit-a"),
]


def test_body_carries_extra_headers_for_the_proxy_hop(tmp_path):
    downstream = CaptureDownstream()
    mw = ReefGatewayMiddleware(
        downstream,
        scenario="coral-demo",
        journal=CallJournal(tmp_path / "j.jsonl"),
        extra_tags={"coral-run": "run-7"},
    )
    _run(mw, json.dumps({"messages": []}).encode(), CORAL_HEADERS)

    payload = json.loads(downstream.body)
    extra = payload["extra_headers"]
    assert extra["x-reef-scenario"] == "coral-demo"
    assert extra["x-reef-tag-coral-agent"] == "agent-1"
    assert extra["x-reef-tag-coral-commit"] == "commit-a"
    assert extra["x-reef-tag-coral-run"] == "run-7"
    assert downstream.content_length == len(downstream.body)


def test_client_supplied_extra_headers_cannot_override(tmp_path):
    downstream = CaptureDownstream()
    mw = ReefGatewayMiddleware(downstream, scenario="coral-demo", journal=CallJournal(tmp_path / "j.jsonl"))
    smuggle = {"messages": [], "extra_headers": {"x-reef-scenario": "evil", "x-other": "kept"}}
    _run(mw, json.dumps(smuggle).encode(), CORAL_HEADERS)
    extra = json.loads(downstream.body)["extra_headers"]
    assert extra["x-reef-scenario"] == "coral-demo"
    assert extra["x-other"] == "kept"


def test_non_json_body_passes_through_untouched(tmp_path):
    downstream = CaptureDownstream()
    mw = ReefGatewayMiddleware(downstream, scenario="coral-demo", journal=CallJournal(tmp_path / "j.jsonl"))
    _run(mw, b"not-json", CORAL_HEADERS)
    assert downstream.body == b"not-json"


def test_body_mirroring_can_be_disabled(tmp_path):
    downstream = CaptureDownstream()
    mw = ReefGatewayMiddleware(
        downstream,
        scenario="coral-demo",
        journal=CallJournal(tmp_path / "j.jsonl"),
        body_extra_headers=False,
    )
    _run(mw, json.dumps({"messages": []}).encode(), CORAL_HEADERS)
    assert "extra_headers" not in json.loads(downstream.body)
