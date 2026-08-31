#!/usr/bin/env python3
"""Black-box Frontier-CS 2.0 judge service for Harbor agent submissions."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import base64
import multiprocessing
import signal
import tarfile
import tempfile
import io
import time
import traceback
import threading
import secrets
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROBLEM_EVALUATOR_PATH = Path("/judge/problem_evaluator.py")
TASK_CONFIG_PATH = Path("/judge/task_config.json")
JUDGE_READY_LOG = Path("/logs/judge/judge_ready.json")
JUDGE_SUBMISSIONS_LOG = Path("/logs/judge/submissions.jsonl")
BEST_SUBMISSION_PAYLOAD = Path("/logs/judge/best_submission_payload.json")
BEST_SUBMISSION_META = Path("/logs/judge/best_submission_meta.json")
MAX_SUBMISSION_BYTES = 30_000_000
MAX_ARCHIVE_BYTES = 20_000_000
FINAL_ROLE_TOKEN = "x0zebunXe7ubG26JKyxd4cZXbwN5gbWUG0uhfpZc6mA"
DEFAULT_MAX_QUEUE_SIZE = 3
EVALUATION_LOCK_TIMEOUT_SECONDS = int(
    os.environ.get("JUDGE_TIMEOUT_SECONDS", "10800")
)


def load_task_config() -> dict[str, Any]:
    if not TASK_CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def configured_max_queue_size() -> int:
    env_value = os.environ.get("FRONTIER_MAX_ASYNC_SUBMISSIONS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    config = load_task_config()
    submission = config.get("submission", {})
    if isinstance(submission, dict):
        for key in ("max_queue_size", "queue_size"):
            value = submission.get(key)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    return DEFAULT_MAX_QUEUE_SIZE


def configured_allow_empty_submission() -> bool:
    config = load_task_config()
    submission = config.get("submission", {})
    return bool(isinstance(submission, dict) and submission.get("allow_empty"))


def configured_async_start_method() -> str:
    config = load_task_config()
    evaluation = config.get("evaluation", {})
    configured = None
    if isinstance(evaluation, dict):
        configured = evaluation.get("async_start_method")
    method = str(
        os.environ.get("FRONTIER_ASYNC_EVAL_START_METHOD")
        or configured
        or "fork"
    )
    if method not in multiprocessing.get_all_start_methods():
        return "fork"
    return method


MAX_QUEUE_SIZE = configured_max_queue_size()


def load_problem_evaluator():
    spec = importlib.util.spec_from_file_location(
        "frontier_cs_2_0_problem_evaluator", PROBLEM_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load evaluator from {PROBLEM_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = None
READY = False
READY_PAYLOAD: dict[str, Any] = {"status": "starting"}
SUBMISSION_LOCK = threading.RLock()
SUBMISSION_CONDITION = threading.Condition(SUBMISSION_LOCK)
SUBMISSION_QUEUE: deque[str] = deque()
SUBMISSION_PAYLOADS: dict[str, dict[str, Any]] = {}
SUBMISSION_RECORDS: dict[str, dict[str, Any]] = {}
SUBMISSION_ORDER: list[str] = []
EVALUATION_LOCK = threading.Lock()


@dataclass
class RunningSubmission:
    process: multiprocessing.Process
    started_at: float


class SubmissionCancelled(Exception):
    pass


RUNNING_SUBMISSIONS: dict[str, RunningSubmission] = {}


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def write_judge_ready(payload: dict[str, Any]) -> None:
    try:
        JUDGE_READY_LOG.parent.mkdir(parents=True, exist_ok=True)
        JUDGE_READY_LOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"WARN: failed to write judge_ready.json: {exc}", flush=True)


def log_submission(record: dict[str, Any]) -> None:
    JUDGE_SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with JUDGE_SUBMISSIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), **record}, ensure_ascii=False) + "\n")


def update_submission_record(submission_uuid: str, record: dict[str, Any]) -> dict[str, Any]:
    with SUBMISSION_LOCK:
        previous = SUBMISSION_RECORDS.get(submission_uuid, {})
        merged = {**previous, **record, "submission_uuid": submission_uuid}
        SUBMISSION_RECORDS[submission_uuid] = merged
        if submission_uuid not in SUBMISSION_ORDER:
            SUBMISSION_ORDER.append(submission_uuid)
    log_submission(merged)
    return merged


def latest_submissions() -> list[dict[str, Any]]:
    with SUBMISSION_LOCK:
        return [
            dict(SUBMISSION_RECORDS[submission_uuid])
            for submission_uuid in SUBMISSION_ORDER
            if submission_uuid in SUBMISSION_RECORDS
        ]


def read_submissions() -> list[dict[str, Any]]:
    if not JUDGE_SUBMISSIONS_LOG.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in JUDGE_SUBMISSIONS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def best_score_key_from_meta(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        score = float(metadata.get("score_raw", 0.0))
        return (score, float(metadata.get("score_unbounded", score)))
    except Exception:
        return None


def save_best_submission(payload: dict[str, Any], record: dict[str, Any]) -> None:
    if record.get("status") != "done":
        return
    try:
        score = float(record.get("score", 0.0))
        score_unbounded = float(record.get("score_unbounded", score))
    except (TypeError, ValueError):
        return

    score_key = (score, score_unbounded)
    previous_key = best_score_key_from_meta(BEST_SUBMISSION_META)
    if previous_key is not None and score_key <= previous_key:
        return

    metadata = {
        "submission_uuid": record.get("submission_uuid"),
        "ts": record.get("ts") or now_iso(),
        "score_raw": score,
        "score_unbounded": score_unbounded,
        "detail": record.get("message", ""),
        "metrics": record.get("metrics", {}),
    }
    BEST_SUBMISSION_PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    BEST_SUBMISSION_PAYLOAD.write_text(json.dumps(payload), encoding="utf-8")
    BEST_SUBMISSION_META.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_evaluator() -> None:
    global EVALUATOR, READY, READY_PAYLOAD
    start = time.time()
    try:
        EVALUATOR = load_problem_evaluator()
        prepare = getattr(EVALUATOR, "prepare", None)
        if callable(prepare):
            payload = prepare()
        else:
            ensure = getattr(EVALUATOR, "_ensure_benchmark", None)
            payload = ensure() if callable(ensure) else {}
        elapsed = time.time() - start
        if not isinstance(payload, dict):
            payload = {}
        READY_PAYLOAD = {
            "status": "ok",
            "ready": True,
            "prepare_seconds": elapsed,
            **payload,
        }
        write_judge_ready(READY_PAYLOAD)
        READY = True
        print(
            "[frontier judge] ready "
            + " ".join(f"{key}={value}" for key, value in READY_PAYLOAD.items()),
            flush=True,
        )
    except Exception as exc:
        READY_PAYLOAD = {
            "status": "error",
            "ready": False,
            "error": str(exc),
        }
        print(traceback.format_exc(), flush=True)


def normalize_result(result: Any) -> tuple[float, float, str, dict[str, Any]]:
    if not isinstance(result, tuple) or len(result) not in (3, 4):
        raise TypeError("evaluator must return (score, score_unbounded, message[, metrics])")
    score = float(result[0])
    score_unbounded = float(result[1])
    message = str(result[2])
    metrics = result[3] if len(result) == 4 else {}
    if not isinstance(metrics, dict):
        raise TypeError("evaluator metrics must be a dict")
    return score, score_unbounded, message, metrics


def evaluate_path(solution_path: Path, *, submission_role: str = "agent") -> dict[str, Any]:
    if EVALUATOR is None:
        raise RuntimeError("problem evaluator is not loaded")
    previous_role = os.environ.get("FRONTIER_SUBMISSION_ROLE")
    os.environ["FRONTIER_SUBMISSION_ROLE"] = submission_role
    try:
        score, score_unbounded, message, metrics = normalize_result(
            EVALUATOR.evaluate(str(solution_path))
        )
    finally:
        if previous_role is None:
            os.environ.pop("FRONTIER_SUBMISSION_ROLE", None)
        else:
            os.environ["FRONTIER_SUBMISSION_ROLE"] = previous_role
    return {
        "status": "done",
        "score": float(score),
        "score_unbounded": float(score_unbounded),
        "message": message,
        "metrics": metrics,
    }


def evaluate_code(code: str, *, submission_role: str = "agent") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="frontier_cs_2_0_submission_") as tmp:
        solution_path = Path(tmp) / "solution.py"
        solution_path.write_text(code, encoding="utf-8")
        return evaluate_path(solution_path, submission_role=submission_role)


def is_safe_tar_member(member: tarfile.TarInfo) -> bool:
    path = Path(member.name)
    return not path.is_absolute() and ".." not in path.parts


def evaluate_archive(archive_b64: str, *, submission_role: str = "agent") -> dict[str, Any]:
    archive = base64.b64decode(archive_b64.encode("ascii"), validate=True)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("submission archive too large")
    with tempfile.TemporaryDirectory(prefix="frontier_cs_2_0_project_") as tmp:
        root = Path(tmp) / "submission"
        root.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            members = tar.getmembers()
            if not all(is_safe_tar_member(member) for member in members):
                raise ValueError("unsafe path in submission archive")
            tar.extractall(root)
        return evaluate_path(root, submission_role=submission_role)


def load_best_submission_payload(submission_uuid: str) -> tuple[dict[str, Any], str]:
    if not BEST_SUBMISSION_PAYLOAD.exists():
        raise FileNotFoundError("best submission payload not found")
    payload = json.loads(BEST_SUBMISSION_PAYLOAD.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("best submission payload is invalid")
    payload = dict(payload)
    payload["submission_role"] = "final"
    payload["submission_uuid"] = submission_uuid
    submission_kind = str(payload.get("submission_kind", "file"))
    return payload, submission_kind


def validate_payload(payload: dict[str, Any], *, allow_final: bool, role_token: str = "") -> tuple[str, str, str]:
    submission_uuid = str(payload.get("submission_uuid") or "")
    if not submission_uuid:
        raise ValueError("submission_uuid is required")

    requested_role = str(payload.get("submission_role") or "agent")
    if requested_role == "final":
        if not allow_final or not secrets.compare_digest(role_token, FINAL_ROLE_TOKEN):
            raise PermissionError("final evaluation role is verifier-only")
        submission_role = "final"
    else:
        submission_role = "agent"

    submission_kind = str(payload.get("submission_kind", "file"))
    if submission_kind == "directory":
        archive_b64 = payload.get("archive_b64")
        if not isinstance(archive_b64, str) or not archive_b64:
            raise ValueError("directory submission must include archive_b64")
    else:
        code = payload.get("code")
        if not isinstance(code, str) or (not configured_allow_empty_submission() and not code.strip()):
            raise ValueError("file submission must include non-empty string field 'code'")
        submission_kind = "file"
    return submission_uuid, submission_role, submission_kind


def evaluate_payload_direct(payload: dict[str, Any], *, submission_role: str) -> dict[str, Any]:
    submission_kind = str(payload.get("submission_kind", "file"))
    if submission_kind == "directory":
        return evaluate_archive(str(payload.get("archive_b64")), submission_role=submission_role)
    return evaluate_code(str(payload.get("code")), submission_role=submission_role)


def run_payload(payload: dict[str, Any], *, submission_role: str) -> dict[str, Any]:
    acquired = EVALUATION_LOCK.acquire(timeout=EVALUATION_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        raise TimeoutError("timed out waiting for evaluator lock")
    try:
        ctx = multiprocessing.get_context(configured_async_start_method())
        with tempfile.TemporaryDirectory(prefix="frontier_cs_2_0_final_result_") as tmp:
            result_path = Path(tmp) / "result.json"
            process = ctx.Process(
                target=_async_evaluate_child,
                args=(payload, submission_role, str(result_path)),
            )
            process.start()
            deadline = time.time() + EVALUATION_LOCK_TIMEOUT_SECONDS
            while process.is_alive():
                process.join(timeout=0.2)
                if time.time() >= deadline:
                    terminate_process_group(process)
                    raise TimeoutError("final evaluation timed out")

            if not result_path.exists():
                raise RuntimeError(
                    f"evaluation process exited without a result (exitcode={process.exitcode})"
                )
            output = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(output, dict):
                raise RuntimeError("evaluation process returned invalid output")
            if not output.get("ok"):
                raise RuntimeError(str(output.get("detail") or "evaluation failed"))
            result = output.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("evaluation process returned invalid result")
            return result
    finally:
        EVALUATION_LOCK.release()


def _async_evaluate_child(
    payload: dict[str, Any],
    submission_role: str,
    result_path: str,
) -> None:
    global EVALUATOR
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            pass
    try:
        if EVALUATOR is None:
            EVALUATOR = load_problem_evaluator()
        result = evaluate_payload_direct(payload, submission_role=submission_role)
        output = {"ok": True, "result": result}
    except BaseException:
        output = {"ok": False, "detail": traceback.format_exc()}
    Path(result_path).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


def terminate_process_group(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        return

    if os.name == "posix" and process.pid:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.terminate()
        except OSError:
            process.terminate()
    else:
        process.terminate()

    process.join(timeout=5)
    if not process.is_alive():
        return

    if os.name == "posix" and process.pid:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.kill()
        except OSError:
            process.kill()
    else:
        process.kill()
    process.join(timeout=5)


def submission_is_cancelling(submission_uuid: str) -> bool:
    with SUBMISSION_LOCK:
        status = SUBMISSION_RECORDS.get(submission_uuid, {}).get("status")
    return status in {"cancelling", "cancelled"}


def run_async_payload(
    submission_uuid: str,
    payload: dict[str, Any],
    *,
    submission_role: str,
) -> dict[str, Any]:
    acquired = EVALUATION_LOCK.acquire(timeout=EVALUATION_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        raise TimeoutError("timed out waiting for evaluator lock")
    try:
        if submission_is_cancelling(submission_uuid):
            raise SubmissionCancelled()

        ctx = multiprocessing.get_context(configured_async_start_method())
        with tempfile.TemporaryDirectory(prefix="frontier_cs_2_0_async_result_") as tmp:
            result_path = Path(tmp) / "result.json"
            process = ctx.Process(
                target=_async_evaluate_child,
                args=(payload, submission_role, str(result_path)),
            )
            process.start()
            should_cancel = False
            with SUBMISSION_CONDITION:
                RUNNING_SUBMISSIONS[submission_uuid] = RunningSubmission(
                    process=process,
                    started_at=time.time(),
                )
                if submission_is_cancelling(submission_uuid):
                    should_cancel = True
                else:
                    update_submission_record(submission_uuid, {"status": "running"})
                SUBMISSION_CONDITION.notify()
            if should_cancel:
                terminate_process_group(process)
                raise SubmissionCancelled()

            try:
                while process.is_alive():
                    process.join(timeout=0.2)
                    if submission_is_cancelling(submission_uuid):
                        terminate_process_group(process)
                        raise SubmissionCancelled()

                if submission_is_cancelling(submission_uuid):
                    raise SubmissionCancelled()
                if not result_path.exists():
                    raise RuntimeError(
                        f"evaluation process exited without a result (exitcode={process.exitcode})"
                    )
                output = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(output, dict):
                    raise RuntimeError("evaluation process returned invalid output")
                if not output.get("ok"):
                    raise RuntimeError(str(output.get("detail") or "evaluation failed"))
                result = output.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("evaluation process returned invalid result")
                return result
            finally:
                with SUBMISSION_LOCK:
                    RUNNING_SUBMISSIONS.pop(submission_uuid, None)
    finally:
        EVALUATION_LOCK.release()


def queue_size() -> int:
    return sum(
        1
        for record in SUBMISSION_RECORDS.values()
        if record.get("status") in {"queued", "running", "cancelling"}
    )


def submission_worker() -> None:
    while True:
        with SUBMISSION_CONDITION:
            while True:
                while SUBMISSION_QUEUE:
                    submission_uuid = SUBMISSION_QUEUE.popleft()
                    record = SUBMISSION_RECORDS.get(submission_uuid, {})
                    if record.get("status") == "queued":
                        break
                else:
                    SUBMISSION_CONDITION.wait()
                    continue
                break

        payload = SUBMISSION_PAYLOADS.get(submission_uuid)
        if payload is None:
            update_submission_record(
                submission_uuid,
                {"status": "error", "message": "queued payload is missing"},
            )
            continue

        started = time.time()
        try:
            result = run_async_payload(submission_uuid, payload, submission_role="agent")
            elapsed = time.time() - started
            record = update_submission_record(
                submission_uuid,
                {
                    "status": result.get("status", "done"),
                    "score": float(result.get("score", 0.0)),
                    "score_unbounded": float(result.get("score_unbounded", 0.0)),
                    "message": str(result.get("message", "")),
                    "metrics": result.get("metrics", {}),
                    "elapsed_seconds": elapsed,
                },
            )
            save_best_submission(payload, record)
        except SubmissionCancelled:
            elapsed = time.time() - started
            with SUBMISSION_LOCK:
                record = SUBMISSION_RECORDS.get(submission_uuid, {})
                already_cancelled = record.get("status") == "cancelled"
            if not already_cancelled:
                update_submission_record(
                    submission_uuid,
                    {
                        "status": "cancelled",
                        "score": 0.0,
                        "score_unbounded": 0.0,
                        "message": "evaluation cancelled",
                        "elapsed_seconds": elapsed,
                    },
                )
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            update_submission_record(
                submission_uuid,
                {
                    "status": "error",
                    "score": 0.0,
                    "score_unbounded": 0.0,
                    "message": "evaluation failed",
                    "detail": detail,
                    "elapsed_seconds": time.time() - started,
                },
            )


class JudgeHandler(BaseHTTPRequestHandler):
    server_version = "FrontierCS20Judge/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(200 if READY else 503, READY_PAYLOAD)
            return
        if parsed.path == "/submissions":
            self._write_json(200, {"status": "ok", "submissions": latest_submissions()})
            return
        if parsed.path.startswith("/submission/"):
            submission_uuid = parsed.path.removeprefix("/submission/").strip("/")
            with SUBMISSION_LOCK:
                record = SUBMISSION_RECORDS.get(submission_uuid)
            if record is None:
                self._write_json(404, {"status": "error", "error": "submission not found"})
            else:
                self._write_json(200, {"status": "ok", "submission": record})
            return
        self._write_json(404, {"status": "error", "error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/submit":
            self.handle_submit()
            return
        if parsed.path.startswith("/submission/") and parsed.path.endswith("/cancel"):
            submission_uuid = parsed.path.removeprefix("/submission/").removesuffix("/cancel").strip("/")
            self.handle_cancel(submission_uuid)
            return
        if parsed.path == "/evaluate_best":
            self.handle_evaluate_best()
            return
        if parsed.path != "/evaluate":
            self._write_json(404, {"status": "error", "error": "not found"})
            return
        if not READY:
            self._write_json(
                503,
                {
                    "status": "error",
                    "score": 0.0,
                    "score_unbounded": 0.0,
                    "message": "judge is not ready",
                    "health": READY_PAYLOAD,
                },
            )
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
        submission_kind = "file"
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            submission_uuid, submission_role, submission_kind = validate_payload(
                payload,
                allow_final=True,
                role_token=self.headers.get("X-Frontier-CS-Role-Token", ""),
            )
            result = run_payload(payload, submission_role=submission_role)
            log_submission(
                {
                    "submission_uuid": submission_uuid,
                    "submission_role": submission_role,
                    "submission_kind": submission_kind,
                    **result,
                }
            )
            self._write_json(200, result)
        except Exception:
            print(traceback.format_exc(), flush=True)
            result = {
                "status": "error",
                "score": 0.0,
                "score_unbounded": 0.0,
                "message": "evaluation failed",
            }
            log_submission(
                {
                    "submission_uuid": submission_uuid,
                    "submission_role": submission_role,
                    "submission_kind": submission_kind,
                    **result,
                }
            )
            self._write_json(200, result)

    def read_json_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if content_length <= 0:
            raise ValueError("empty request body")
        if content_length > MAX_SUBMISSION_BYTES:
            raise ValueError("submission too large")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def handle_evaluate_best(self) -> None:
        if not READY:
            self._write_json(
                503,
                {
                    "status": "error",
                    "score": 0.0,
                    "score_unbounded": 0.0,
                    "message": "judge is not ready",
                    "health": READY_PAYLOAD,
                },
            )
            return

        submission_uuid = ""
        submission_kind = "file"
        try:
            request_payload = self.read_json_body()
            if request_payload.get("submission_role") != "final":
                raise PermissionError("best-submission rerun must use final role")
            if not secrets.compare_digest(
                self.headers.get("X-Frontier-CS-Role-Token", ""),
                FINAL_ROLE_TOKEN,
            ):
                raise PermissionError("final evaluation role is verifier-only")
            submission_uuid = str(request_payload.get("submission_uuid") or uuid.uuid4())
            payload, submission_kind = load_best_submission_payload(submission_uuid)
            result = run_payload(payload, submission_role="final")
            log_submission(
                {
                    "submission_uuid": submission_uuid,
                    "submission_role": "final",
                    "submission_kind": submission_kind,
                    **result,
                }
            )
            self._write_json(200, result)
        except Exception:
            print(traceback.format_exc(), flush=True)
            result = {
                "status": "error",
                "score": 0.0,
                "score_unbounded": 0.0,
                "message": "best iterative evaluation failed",
            }
            log_submission(
                {
                    "submission_uuid": submission_uuid,
                    "submission_role": "final",
                    "submission_kind": submission_kind,
                    **result,
                }
            )
            self._write_json(200, result)

    def handle_submit(self) -> None:
        if not READY:
            self._write_json(503, {"status": "error", "error": "judge is not ready", "health": READY_PAYLOAD})
            return
        try:
            payload = self.read_json_body()
            submission_uuid, submission_role, submission_kind = validate_payload(
                payload,
                allow_final=False,
            )
            if submission_role != "agent":
                raise PermissionError("async submissions are agent-only")
            with SUBMISSION_CONDITION:
                if queue_size() >= MAX_QUEUE_SIZE:
                    self._write_json(
                        429,
                        {
                            "status": "error",
                            "error": f"too many queued/running submissions; limit={MAX_QUEUE_SIZE}",
                        },
                    )
                    return
                if submission_uuid in SUBMISSION_RECORDS:
                    raise ValueError("duplicate submission_uuid")
                SUBMISSION_PAYLOADS[submission_uuid] = dict(payload)
                SUBMISSION_QUEUE.append(submission_uuid)
                position = len(SUBMISSION_QUEUE)
                update_submission_record(
                    submission_uuid,
                    {
                        "submission_role": "agent",
                        "submission_kind": submission_kind,
                        "status": "queued",
                        "code_chars": payload.get("code_chars"),
                        "file_count": payload.get("file_count"),
                        "solution_path": payload.get("solution_path"),
                        "queue_position": position,
                    },
                )
                SUBMISSION_CONDITION.notify()
            self._write_json(
                202,
                {
                    "status": "queued",
                    "submission_uuid": submission_uuid,
                    "queue_position": position,
                },
            )
        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            self._write_json(400, {"status": "error", "error": str(exc)})

    def handle_cancel(self, submission_uuid: str) -> None:
        running: RunningSubmission | None = None
        with SUBMISSION_CONDITION:
            record = SUBMISSION_RECORDS.get(submission_uuid)
            if record is None:
                self._write_json(404, {"status": "error", "error": "submission not found"})
                return
            current_status = record.get("status")
            if current_status == "cancelled":
                self._write_json(200, {"status": "cancelled", "submission_uuid": submission_uuid})
                return
            if current_status == "queued":
                update_submission_record(submission_uuid, {"status": "cancelled"})
                SUBMISSION_PAYLOADS.pop(submission_uuid, None)
                SUBMISSION_CONDITION.notify()
                self._write_json(200, {"status": "cancelled", "submission_uuid": submission_uuid})
                return
            if current_status in {"running", "cancelling"}:
                running = RUNNING_SUBMISSIONS.get(submission_uuid)
                if running is None:
                    self._write_json(
                        409,
                        {
                            "status": "error",
                            "error": "running submission process is no longer cancellable",
                            "submission": record,
                        },
                    )
                    return
                if current_status != "cancelling":
                    update_submission_record(
                        submission_uuid,
                        {"status": "cancelling", "message": "cancellation requested"},
                    )
                SUBMISSION_CONDITION.notify()
            else:
                self._write_json(
                    409,
                    {
                        "status": "error",
                        "error": "only queued or running submissions can be cancelled",
                        "submission": record,
                    },
                )
                return

        if running is not None:
            terminate_process_group(running.process)
            with SUBMISSION_CONDITION:
                update_submission_record(
                    submission_uuid,
                    {
                        "status": "cancelled",
                        "score": 0.0,
                        "score_unbounded": 0.0,
                        "message": "running evaluation cancelled",
                        "elapsed_seconds": time.time() - running.started_at,
                    },
                )
                SUBMISSION_PAYLOADS.pop(submission_uuid, None)
                SUBMISSION_CONDITION.notify()
        self._write_json(200, {"status": "cancelled", "submission_uuid": submission_uuid})

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8082"))
    server = ThreadingHTTPServer(("0.0.0.0", port), JudgeHandler)
    threading.Thread(target=prepare_evaluator, daemon=True).start()
    threading.Thread(target=submission_worker, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
