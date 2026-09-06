"""Smoke of the real dsh binary: render, episode, and trajectory against the
real thing. The hermetic suites drive scripted fakes, so every format and
environment assumption about the real DeepSeek Harness lives here: a pinned
dsh runs one episode against a stdlib OpenAI-compatible stub and must read
our rendered patch layer, reach the stub with the rules text and the bound
key, write a session our reader parses, and leave no unexplained residue.
Skipped unless REEF_REAL_DSH_BINARY points at an installed dsh
(.github/workflows/harness-smoke.yml provides one)."""

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

REAL_DSH = os.environ.get("REEF_REAL_DSH_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_DSH, reason="REEF_REAL_DSH_BINARY does not name a real dsh binary")

MODEL = "smoke-model"
RULES_SENTENCE = "The reef smoke marker phrase is TIDEPOOL-STONE-41."


class StubOpenAI(ThreadingHTTPServer):
    """A canned /v1 endpoint that records every request it is sent."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.bodies: list[str] = []
        self.auth: list[str | None] = []


class StubHandler(BaseHTTPRequestHandler):
    server: StubOpenAI

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the harness's stderr out of pytest output

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.bodies.append(body)
        self.server.auth.append(self.headers.get("Authorization"))
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
            message = {"role": "assistant", "content": "READY"}
            payload = {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": usage,
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


def test_real_dsh_episode_renders_runs_and_cleans_up() -> None:
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("dsh")
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key"
        )
        nodes = [("rules", {"text": RULES_SENTENCE}), *binding.compose_nodes(descriptor)]
        files = render_composition(nodes, descriptor)
        # The task starts with a dash on purpose: it must pass both commander layers as a positional.
        result = run_episode(descriptor, files, "- Reply with exactly the word READY", binary=REAL_DSH, timeout=180.0)
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    # (a) The rendered rules text and the bound key reached the model: the
    # whole chain (patch layer, .env, relocation env, real binary, request) held.
    assert any(RULES_SENTENCE in body for body in server.bodies), server.bodies
    assert any("- Reply with exactly the word READY" in body for body in server.bodies), server.bodies
    assert server.auth and server.auth[0] == "Bearer smoke-key", server.auth
    # (b) The session the real binary wrote parses through our reader and
    # carries the assistant message and the turn end.
    kinds = [event.get("type") for event in result.trajectory]
    assert kinds and kinds[0] == "session" and "assistant/message" in kinds and kinds[-1] == "turn/end", kinds
    # (c) The cleanup audit found nothing outside the declared whitelist.
    assert result.residue == ()
