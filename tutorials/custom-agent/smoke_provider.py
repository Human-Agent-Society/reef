"""Scripted Chat Completions replies for the calculator smoke run, not a model.

The first reply requests multiply(17, 23). The second requires its tool result
and returns that result as the answer. No network model or API key is used.
"""

import json
from http.server import BaseHTTPRequestHandler


class SmokeHandler(BaseHTTPRequestHandler):
    """Stateless HTTP handler for the calculator's two model calls.

    Values: standard BaseHTTPRequestHandler instances attached to a local
    HTTP server. POST /v1/chat/completions accepts a non-empty messages list.
    Purpose: run.py supplies this handler to its smoke server; each accepted
    request returns a tool call, or the last tool message as the final text.
    Other paths return 404 and invalid messages return 400.
    """

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except ValueError:
            self.send_error(400, "expected a JSON request")
            return
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages or not all(isinstance(item, dict) for item in messages):
            self.send_error(400, "expected messages")
            return
        if messages[-1].get("role") == "tool":
            message = {"role": "assistant", "content": messages[-1].get("content")}
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "multiply-1",
                        "type": "function",
                        "function": {"name": "multiply", "arguments": '{"left":17,"right":23}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        response = {
            "id": "chatcmpl-calculator",
            "object": "chat.completion",
            "created": 0,
            "model": "langgraph-smoke",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the scripted provider quiet; Reef keeps the model call records."""
