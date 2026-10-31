"""Loopback HTTP transport for a Designer's task checks and result polls."""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reef.recipe.reefine.agent_gateway import WorkspaceTools
from reef.recipe.reefine.trial import trial_script


class CheckGateway:
    """Serve WorkspaceTools on a token-prefixed loopback URL for one Designer run.

    Values: ``tools`` implements WorkspaceTools, ``token`` is a random URL token,
    and ``server`` is either None (stopped) or the running loopback HTTP server.
    Purpose: ClaudeCodeDesigner starts the server for task_check.py and stops it
    after the run. Only POST /check and /trial reach tools. All other routes
    return 404. The gateway has no model binding, provider, or model-call budget.
    """

    def __init__(self, tools: WorkspaceTools) -> None:
        self.tools = tools
        self.token = secrets.token_urlsafe(24)
        self.server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        """Return the listening port; raise RuntimeError when stopped."""
        if self.server is None:
            raise RuntimeError("the check gateway is not running")
        return int(self.server.server_address[1])

    @property
    def base_url(self) -> str:
        """Return the running server's token-prefixed URL, without a trailing slash."""
        return f"http://127.0.0.1:{self.port}/t/{self.token}"

    def start(self) -> None:
        """Start listening on a free port; reject a second start until stopped."""
        if self.server is not None:
            raise RuntimeError("the check gateway is already running")
        server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler_class())
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="autoenvscaling-check-gateway", daemon=True).start()
        self.server = server

    def stop(self) -> None:
        """Close the listening socket if running; otherwise do nothing."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        """Bind HTTP parsing to this gateway's workspace tools and URL token."""
        gateway = self
        prefix = f"/t/{self.token}"

        class Handler(BaseHTTPRequestHandler):
            """One HTTP request, using BaseHTTPRequestHandler's stream and response lifecycle.

            The server constructs each instance with its request socket, client address,
            and server. POST dispatches check requests; GET always returns 404.
            """

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                route = path[len(prefix) :]
                if not secrets.compare_digest(path[: len(prefix)], prefix) or route not in ("/check", "/trial"):
                    self.answer(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length < 0:
                        raise ValueError("negative content length")
                    payload = json.loads(self.rfile.read(length) if length else b"{}")
                except (ValueError, UnicodeDecodeError):
                    self.answer(400, {"error": "the body must be JSON with a valid Content-Length"})
                    return
                if not isinstance(payload, dict):
                    self.answer(400, {"error": "the body must be a JSON object"})
                    return
                if route == "/check":
                    self.answer(200, gateway.tools.check())
                    return
                task = payload.get("task")
                script = payload.get("script")
                if script is not None:
                    if task is not None:
                        self.answer(400, {"error": "choose task or script, not both"})
                        return
                    try:
                        validated = trial_script(script)
                    except ValueError as error:
                        self.answer(400, {"error": str(error)})
                        return
                    self.answer(200, gateway.tools.trial("", validated))
                elif isinstance(task, str) and task.strip():
                    self.answer(200, gateway.tools.trial(task.strip()))
                else:
                    self.answer(400, {"error": "a trial needs a task or script"})

            def do_GET(self) -> None:
                self.answer(404, {"error": "not found"})

            def answer(self, status: int, body: Mapping[str, object]) -> None:
                encoded = json.dumps(body, default=str).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                # Requests contain the gateway token; do not log their URLs.
                return None

        return Handler
