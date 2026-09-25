"""Small asynchronous HTTP surface shared by task-specific judge services."""

from __future__ import annotations

import json
import math
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
from uuid import uuid4


class Judge(ABC):
    @abstractmethod
    def __call__(self, pid: str, language: str, code: str) -> dict: ...


def parse_submission(content_type: str, body: bytes) -> tuple[str, str, str]:
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    fields: dict[str, str] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = part.get_content()
    return fields.get("pid", ""), fields.get("lang", ""), fields.get("code", "")


def serve(judge: Judge, *, host: str = "127.0.0.1", port: int, max_workers: int = 1) -> None:
    results: dict[str, dict] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=max_workers)

    def evaluate(sid: str, pid: str, language: str, code: str) -> None:
        try:
            payload = judge(pid, language, code)
            score = float(payload.get("score", 0.0))
            if not math.isfinite(score) or score < 0:
                raise ValueError("judge reward must be finite and non-negative")
            candidate_valid = bool(payload.get("valid", False))
            training_reward_on_invalid = bool(payload.get("training_reward_on_invalid", False))
            failure_domain = payload.get("failure_domain")
            result = {
                "status": "done" if candidate_valid or training_reward_on_invalid else "error",
                "valid": candidate_valid,
                "trainingRewardOnInvalid": training_reward_on_invalid,
                "score": float(payload.get("score", 0.0)),
                "scoreUnbounded": payload.get("score_unbounded", payload.get("score", 0.0)),
                "message": str(payload.get("message") or ("accepted" if candidate_valid else "rejected")),
                "artifacts": payload.get("artifacts") or {},
                "failureDomain": failure_domain,
            }
        except Exception as exc:
            result = {
                "status": "environment_error",
                "valid": False,
                "score": 0.0,
                "message": f"judge infrastructure failure: {exc!r}",
                "failureDomain": "infrastructure",
            }
        with lock:
            results[sid] = result

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply(200, {"ok": True})
                return
            if self.path.startswith("/result/"):
                sid = unquote(self.path.removeprefix("/result/"))
                with lock:
                    payload = results.get(sid)
                self.reply(200 if payload is not None else 404, payload or {"status": "pending"})
                return
            self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            if self.path != "/submit":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                pid, language, code = parse_submission(self.headers.get("Content-Type", ""), self.rfile.read(length))
            except Exception as exc:
                self.reply(400, {"error": f"invalid multipart submission: {exc}"})
                return
            if not code.strip():
                self.reply(400, {"error": "code is empty"})
                return
            sid = uuid4().hex
            pool.submit(evaluate, sid, pid, language, code)
            self.reply(200, {"sid": sid})

        def log_message(self, *_args) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


__all__ = ["parse_submission", "serve"]
