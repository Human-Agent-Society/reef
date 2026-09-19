"""Smoke of the real hermes binary: render, episode, and trajectory against the
real thing. The hermetic suites drive scripted fakes, so every format and
environment assumption about the real Hermes Agent lives here: a pinned
hermes runs one episode against a stdlib OpenAI-compatible stub and must read
our rendered config.yaml, reach the stub with the SOUL.md text and the bound
key, write a session snapshot our reader parses, make no extra model call,
and leave no unexplained residue. Skipped unless REEF_REAL_HERMES_BINARY
points at an installed hermes (.github/workflows/harness-smoke.yml provides
one)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.tree.render import render_composition

REAL_HERMES = os.environ.get("REEF_REAL_HERMES_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_HERMES, reason="REEF_REAL_HERMES_BINARY does not name a real hermes binary")

MODEL = "smoke-model"
RULES_SENTENCE = "The reef smoke marker phrase is TIDEPOOL-STONE-41."


class StubOpenAI(ThreadingHTTPServer):
    """A canned /v1 endpoint that records every request it is sent."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.requests: list[tuple[str, str | None, str]] = []


class StubHandler(BaseHTTPRequestHandler):
    server: StubOpenAI

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the harness's stderr out of pytest output

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.requests.append((self.path, self.headers.get("Authorization"), body))
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)  # hermes probes the base URL as a local endpoint first
            self.end_headers()
            return
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if json.loads(body).get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "READY"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            for chunk in chunks:
                event = {"id": "chatcmpl-smoke", "object": "chat.completion.chunk", "model": MODEL, **chunk}
                self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "READY"}, "finish_reason": "stop"}
                ],
                "usage": usage,
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


def test_real_hermes_episode_renders_runs_and_cleans_up() -> None:
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("hermes")
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key-1234"
        )
        nodes = [("rules", {"text": RULES_SENTENCE}), *binding.compose_nodes(descriptor)]
        files = render_composition(nodes, descriptor)
        # The task starts with a dash on purpose: it must pass through -q as the query.
        result = run_episode(
            descriptor, files, "- Reply with exactly the word READY", binary=REAL_HERMES, timeout=300.0
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    completions = [(auth, body) for path, auth, body in server.requests if path.endswith("/chat/completions")]
    # (a) Exactly one model call: the title call is off. It carried the rules text and the bound key.
    assert len(completions) == 1, [path for path, _, _ in server.requests]
    assert RULES_SENTENCE in completions[0][1] and "- Reply with exactly the word READY" in completions[0][1]
    assert completions[0][0] == "Bearer smoke-key-1234"
    # (b) The snapshot the real binary wrote parses through our reader and carries the exchange.
    kinds = [event["type"] for event in result.trajectory]
    assert kinds == ["session", "message", "message"], kinds
    assert [event["role"] for event in result.trajectory[1:]] == ["user", "assistant"]
    assert result.trajectory[2]["content"] == "READY"
    # (c) The cleanup audit found nothing outside the declared whitelist.
    assert result.residue == ()
