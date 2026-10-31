"""Loopback HTTP transport for an author's task checks and result polls."""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reef.harness.gateway import AgentTool


class CheckGateway:
    """Serve an author's ``check`` tool on a token-prefixed loopback URL for one author run.

    Values: ``tools`` is the ``AgentTool`` named ``check`` (``author.AgentTools``), ``token`` is a random URL
    token, and ``server`` is either None (stopped) or the running loopback HTTP server.
    Purpose: ClaudeCodeAuthor starts the server for task_check.py and stops it
    after the run. Only POST /check reaches the tool, with the request body as its arguments: ``{}`` starts a
    check, and ``{"check_id": n}`` polls one; a ``check_id`` of no check of this run answers ``ran: false``. All
    other routes return 404. The gateway has no model binding,
    provider, or model-call budget.
    """

    def __init__(self, tools: AgentTool) -> None:
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
        """Bind HTTP parsing to this gateway's check tool and URL token."""
        gateway = self
        prefix = f"/t/{self.token}"

        class Handler(BaseHTTPRequestHandler):
            """One HTTP request, using BaseHTTPRequestHandler's stream and response lifecycle.

            The server constructs each instance with its request socket, client address,
            and server. POST /check calls the check tool; GET always returns 404.
            """

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                route = path[len(prefix) :]
                if not secrets.compare_digest(path[: len(prefix)], prefix) or route != "/check":
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
                try:
                    answer = gateway.tools.call(payload)
                except ValueError as error:
                    # A check_id that names no check of this run starts nothing; the agent reads why.
                    answer = {"ran": False, "error": str(error)}
                self.answer(200, answer)

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
