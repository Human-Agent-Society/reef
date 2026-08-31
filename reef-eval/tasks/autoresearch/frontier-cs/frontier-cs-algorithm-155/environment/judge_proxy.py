#!/usr/bin/env python3
"""Black-box Harbor judge proxy for Frontier-CS algorithmic tasks.

The upstream go-judge service keeps detailed checker/interactor output for
score aggregation and artifacts. This proxy is the only judge endpoint exposed
to the agent container; it submits code to the upstream engine and returns a
small, sanitized score result.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, parse, request

ENGINE_URL = os.environ.get("JUDGE_ENGINE_URL", "http://judge-engine:8081").rstrip("/")
PORT = int(os.environ.get("PORT", "8082"))
PROBLEM_ID = os.environ.get("PROBLEM_ID", "")
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
MAX_POLL_TIME_SECONDS = float(os.environ.get("MAX_POLL_TIME", "600"))
MAX_SUBMISSION_BYTES = int(os.environ.get("MAX_SUBMISSION_BYTES", "10000000"))
SUBMISSIONS_LOG = Path("/logs/judge/submissions.jsonl")


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def log_submission(record: dict[str, Any]) -> None:
    SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), **record}, ensure_ascii=False) + "\n")


def read_submissions() -> list[dict[str, Any]]:
    if not SUBMISSIONS_LOG.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in SUBMISSIONS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def get_json(path: str, timeout: float = 10) -> dict[str, Any]:
    with request.urlopen(f"{ENGINE_URL}{path}", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected JSON payload from {path}")
    return payload


def post_form(path: str, fields: dict[str, str], timeout: float = 30) -> dict[str, Any]:
    body = parse.urlencode(fields).encode("utf-8")
    req = request.Request(
        f"{ENGINE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected JSON payload from {path}")
    return payload


def wait_for_engine() -> dict[str, Any]:
    return get_json("/problems", timeout=5)


def submit_to_engine(problem_id: str, code: str) -> int:
    payload = post_form(
        "/submit",
        {"pid": problem_id, "lang": "cpp", "code": code},
        timeout=30,
    )
    sid = payload.get("sid")
    if not isinstance(sid, int):
        raise RuntimeError(f"judge engine did not return a submission id: {payload}")
    return sid


def poll_engine_result(sid: int) -> dict[str, Any]:
    start = time.time()
    while time.time() - start < MAX_POLL_TIME_SECONDS:
        try:
            result = get_json(f"/result/{sid}", timeout=10)
        except error.HTTPError as exc:
            if exc.code != 404:
                raise
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if result.get("status") in ("done", "error"):
            return result
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"evaluation timed out after {MAX_POLL_TIME_SECONDS}s")


def sanitize_result(raw: dict[str, Any], sid: int) -> dict[str, Any]:
    score = float(raw.get("score") or 0.0)
    score_unbounded = raw.get("scoreUnbounded", score)
    try:
        score_unbounded = float(score_unbounded)
    except (TypeError, ValueError):
        score_unbounded = score

    cases = raw.get("cases")
    total_cases = len(cases) if isinstance(cases, list) else None
    perfect_cases = None
    if isinstance(cases, list):
        perfect_cases = 0
        for case in cases:
            if isinstance(case, dict) and float(case.get("scoreRatio") or 0.0) == 1.0:
                perfect_cases += 1

    metrics: dict[str, Any] = {"sid": sid}
    if total_cases is not None:
        metrics["total_cases"] = total_cases
    if perfect_cases is not None:
        metrics["perfect_cases"] = perfect_cases

    if raw.get("status") == "error":
        return {
            "status": "error",
            "score": 0.0,
            "score_unbounded": 0.0,
            "message": "judge engine error",
            "metrics": metrics,
        }

    passed = bool(raw.get("passed", False))
    result = str(raw.get("result") or ("Correct Answer" if passed else "Wrong Answer"))
    return {
        "status": "done",
        "score": score,
        "score_unbounded": score_unbounded,
        "passed": passed,
        "message": result,
        "metrics": metrics,
    }


def evaluate_code(payload: dict[str, Any]) -> dict[str, Any]:
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("request must include non-empty string field 'code'")
    problem_id = str(payload.get("problem_id") or PROBLEM_ID)
    if not problem_id:
        raise ValueError("problem_id is required")
    sid = submit_to_engine(problem_id, code)
    raw_result = poll_engine_result(sid)
    return sanitize_result(raw_result, sid)


class JudgeProxyHandler(BaseHTTPRequestHandler):
    server_version = "FrontierCSAlgorithmJudgeProxy/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            try:
                wait_for_engine()
                self._write_json(200, {"status": "ok", "ready": True})
            except Exception as exc:
                self._write_json(
                    503,
                    {"status": "error", "ready": False, "error": str(exc)},
                )
            return
        if self.path == "/submissions":
            self._write_json(200, {"status": "ok", "submissions": read_submissions()})
            return
        self._write_json(404, {"status": "error", "error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/evaluate":
            self._write_json(404, {"status": "error", "error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(400, {"status": "error", "error": "invalid content length"})
            return
        if content_length <= 0:
            self._write_json(400, {"status": "error", "error": "empty request body"})
            return
        if content_length > MAX_SUBMISSION_BYTES:
            self._write_json(413, {"status": "error", "error": "submission too large"})
            return

        submission_uuid = ""
        submission_role = "agent"
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            submission_uuid = str(payload.get("submission_uuid") or "")
            submission_role = str(payload.get("submission_role") or "agent")
            result = evaluate_code(payload)
            log_submission(
                {
                    "submission_uuid": submission_uuid,
                    "submission_role": submission_role,
                    "problem_id": str(payload.get("problem_id") or PROBLEM_ID),
                    **result,
                }
            )
            self._write_json(200, result)
        except Exception as exc:
            detail = traceback.format_exc()
            result = {
                "status": "error",
                "score": 0.0,
                "score_unbounded": 0.0,
                "message": str(exc),
                "metrics": {},
            }
            log_submission(
                {
                    "submission_uuid": submission_uuid,
                    "submission_role": submission_role,
                    "problem_id": PROBLEM_ID,
                    **result,
                    "detail": detail,
                }
            )
            self._write_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"[{now_iso()}] {self.address_string()} {format % args}",
            flush=True,
        )


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), JudgeProxyHandler)
    print(
        f"[frontier algorithm judge proxy] listening on {PORT}, engine={ENGINE_URL}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
