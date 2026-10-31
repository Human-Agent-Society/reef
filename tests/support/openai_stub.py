"""A canned OpenAI-compatible endpoint for the real-binary smoke tests.

``StubOpenAI`` answers ``GET /v1/models`` with ``MODEL`` and every chat
completion, streamed or not, with the text ``READY``, and records each request
body it is sent.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The model name the stub lists and answers as.
MODEL = "smoke-model"


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
