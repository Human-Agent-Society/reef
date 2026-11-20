"""Bounded serial transport for the BBH experiment, not a general serving proxy."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from reef.scenario.evaluation import atomic_json


class BudgetServer(HTTPServer):
    def __init__(self, api_key: str, model: str, max_calls: int, output: Path) -> None:
        self.api_key = api_key
        self.model = model
        self.max_calls = max_calls
        self.output = output
        self.phase = "initialization"
        self.calls: list[dict[str, object]] = []
        super().__init__(("127.0.0.1", 0), BudgetHandler)


class BudgetHandler(BaseHTTPRequestHandler):
    server: BudgetServer

    def log_message(self, format: str, *args: object) -> None:
        # Never log request headers, credentials or provider error bodies.
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if len(self.server.calls) >= self.server.max_calls:
            self.send_error(429, "evaluation request budget exhausted")
            return
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 100000:
            self.send_error(413, "evaluation request size limit")
            return
        body = json.loads(self.rfile.read(size))
        body.update(
            model=self.server.model,
            max_tokens=min(int(body.get("max_tokens", 1536)), 1536),
            temperature=0,
            thinking={"type": "disabled"},
            stream=False,
        )
        record: dict[str, object] = {
            "ordinal": len(self.server.calls),
            "status": "started",
            "phase": self.server.phase,
        }
        self.server.calls.append(record)
        atomic_json(self.server.output, self.server.calls)
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.server.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                value = json.load(response)
        except (urllib.error.URLError, TimeoutError):
            record["status"] = "provider_error"
            atomic_json(self.server.output, self.server.calls)
            self.send_error(502, "evaluation provider call failed")
            return
        record.update(status="completed", response_model=value.get("model"), usage=value.get("usage"))
        atomic_json(self.server.output, self.server.calls)
        encoded = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
