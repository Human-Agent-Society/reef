"""A canned OpenAI-compatible endpoint for the real-binary smoke tests.

``StubOpenAI`` answers ``GET /v1/models`` with ``MODEL`` and every chat
completion, streamed or not, with the text ``READY``, and records each request
it is sent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The model name the stub lists and answers as.
MODEL = "smoke-model"


class StubOpenAI(ThreadingHTTPServer):
    """A canned /v1 endpoint on a free local port that records every POST request it is sent.

    ``requests`` holds one (path, ``Authorization`` header or None, body text) tuple per POST, in arrival order.

    - ``tool_calls`` are (tool name, arguments) pairs. A streamed answer asks for the next pair until the request
      holds one tool result per pair; then it is ``READY``. An answer that is not streamed is always ``READY``.
    - With ``lists_models`` false, a GET gets the status 501, as from a server without a GET method.
    - With ``chat_completions_only``, a POST to another path than ``/chat/completions`` is recorded and gets 404.
    """

    def __init__(
        self,
        *,
        tool_calls: Sequence[tuple[str, dict[str, str]]] = (),
        lists_models: bool = True,
        chat_completions_only: bool = False,
    ) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.requests: list[tuple[str, str | None, str]] = []
        self.tool_calls = list(tool_calls)
        self.lists_models = lists_models
        self.chat_completions_only = chat_completions_only

    @property
    def bodies(self) -> list[str]:
        """The body text of each recorded request."""
        return [body for _, _, body in self.requests]

    @property
    def auth(self) -> list[str | None]:
        """The ``Authorization`` header of each recorded request."""
        return [authorization for _, authorization, _ in self.requests]


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
        if not self.server.lists_models:
            self.send_error(501, "Unsupported method ('GET')")
            return
        self._json({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "stub"}]})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.requests.append((self.path, self.headers.get("Authorization"), body))
        if self.server.chat_completions_only and not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        request = json.loads(body)
        if request.get("stream"):
            answered = sum(1 for message in request.get("messages", []) if message.get("role") == "tool")
            if answered < len(self.server.tool_calls):
                name, arguments = self.server.tool_calls[answered]
                call = {"name": name, "arguments": json.dumps(arguments)}
                tool_call = {"index": 0, "id": f"call-{answered}", "type": "function", "function": call}
                delta: dict[str, Any] = {"role": "assistant", "tool_calls": [tool_call]}
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "READY"}
                finish = "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"index": 0, "delta": delta}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
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
