"""Smoke of the real pi binary: render, episode, and trajectory against the
real thing. The hermetic suites drive a scripted fake, so every format and
environment assumption about the real binary lives here: a pinned pi runs one
episode against a stdlib OpenAI-compatible stub and must read our rendered
composition, reach the stub with the rules text, write a session our reader
parses, and leave no unexplained residue. Skipped unless REEF_REAL_PI_BINARY
points at an installed pi (.github/workflows/harness-smoke.yml provides one)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episode import run_episode
from reef.harness.render import render_composition

REAL_PI = os.environ.get("REEF_REAL_PI_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_PI, reason="REEF_REAL_PI_BINARY does not name a real pi binary")

MODEL = "smoke-model"
RULES_SENTENCE = "The reef smoke marker phrase is TIDEPOOL-STONE-41."


class StubOpenAI(ThreadingHTTPServer):
    """A canned /v1 endpoint that records every request body it is sent."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.bodies: list[str] = []


class StubHandler(BaseHTTPRequestHandler):
    server: StubOpenAI

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the harness's stderr out of pytest output

    def _json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._json({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "stub"}]})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.bodies.append(body)
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
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            self._json(
                {
                    "id": "chatcmpl-smoke",
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [choice],
                    "usage": usage,
                }
            )


def capturing_wrapper(tmp_path: Path, capture: Path) -> str:
    """Wrap the real pi so the raw session file survives the root teardown.

    ``run_episode`` deletes the episode root before returning, so the raw
    bytes pi wrote are copied out by the wrapper first: they are the artifact
    the workflow uploads and the refresh source of the hermetic fixture
    (tests/reef_service/data/pi_session_real/README.md).
    """
    wrapper = tmp_path / "pi-capturing"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'"{REAL_PI}" "$@"\n'
        "status=$?\n"
        f'find "$PI_CODING_AGENT_SESSION_DIR" -name "*.jsonl" -exec cp {{}} "{capture}" \\;\n'
        "exit $status\n"
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def test_real_pi_episode_renders_runs_and_cleans_up(tmp_path: Path) -> None:
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        nodes = [
            ("config", {"data": {"defaultProvider": "stub", "defaultModel": MODEL}}),
            (
                "config",
                {
                    "target": "models",
                    "data": {
                        "providers": {
                            "stub": {
                                "name": "stub",
                                "api": "openai-completions",
                                "apiKey": "dummy",
                                "baseUrl": base_url,
                                "models": [{"id": MODEL, "name": MODEL}],
                            }
                        }
                    },
                },
            ),
            ("rules", {"text": RULES_SENTENCE}),
        ]
        files = render_composition(nodes, get_adapter("pi"))
        capture = Path(os.environ.get("REEF_REAL_PI_SESSION_OUT", tmp_path / "real-pi-session.jsonl"))
        binary = capturing_wrapper(tmp_path, capture)
        result = run_episode(
            get_adapter("pi"), files, "Reply with exactly the word READY", binary=binary, timeout=120.0
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    # (a) The rendered rules text reached the model: the whole render chain
    # (composition -> files -> relocation env -> real binary -> request) held.
    assert any(RULES_SENTENCE in body for body in server.bodies), server.bodies
    # (b) The session the real binary wrote parses through our reader and
    # carries an assistant message.
    assert result.trajectory
    roles = [event["message"]["role"] for event in result.trajectory if event.get("type") == "message"]
    assert "assistant" in roles, result.trajectory
    # (c) The cleanup audit found nothing outside the declared whitelist.
    assert result.residue == ()
    # The wrapper captured the raw session file for the workflow's artifact.
    assert capture.is_file() and capture.stat().st_size > 0
