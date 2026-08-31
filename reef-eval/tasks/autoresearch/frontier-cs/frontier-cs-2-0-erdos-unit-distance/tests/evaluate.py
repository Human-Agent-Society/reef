#!/usr/bin/env python3
"""Harbor verifier for Frontier-CS 2.0 problems."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tarfile
import time
import traceback
import uuid
from pathlib import Path
from urllib import error, request

SOLUTION_PATH = Path("/app/solution.py")
APP_PATH = Path("/app")
SUBMISSION_CONFIG_PATH = Path("/app/submission_config.json")
REWARD_TXT = Path("/logs/verifier/reward.txt")
REWARD_JSON = Path("/logs/verifier/reward.json")
JUDGE_SUBMISSIONS_LOG = Path("/logs/judge/submissions.jsonl")
JUDGE_READY_LOG = Path("/logs/judge/judge_ready.json")
VERIFIER_SUBMISSIONS_LOG = Path("/logs/verifier/submissions.jsonl")
VERIFIER_JUDGE_READY_LOG = Path("/logs/verifier/judge_ready.json")
EVALUATION_JSON = Path("/logs/verifier/evaluation_result.json")
BEST_AGENT_PAYLOAD = Path("/logs/agent/best_submission_payload.json")
BEST_AGENT_META = Path("/logs/agent/best_submission_meta.json")
BEST_JUDGE_PAYLOAD = Path("/logs/judge/best_submission_payload.json")
BEST_JUDGE_META = Path("/logs/judge/best_submission_meta.json")
JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8082").rstrip("/")
JUDGE_TIMEOUT_SECONDS = int(os.environ.get("JUDGE_TIMEOUT_SECONDS", "10800"))
FINAL_ROLE_TOKEN = "9OIyUeTfgSV9hS3FnDhzoZSA6vIArMx4b_6-bj61Xww"


def result_score_key(record: dict) -> tuple[float, float]:
    score = float(record.get("score", 0.0))
    return (score, float(record.get("score_unbounded", score)))


def write_reward(reward: float, detail: str = "", extra: dict | None = None) -> None:
    REWARD_TXT.parent.mkdir(parents=True, exist_ok=True)
    reward = max(0.0, min(1.0, float(reward)))
    REWARD_TXT.write_text(str(reward), encoding="utf-8")
    payload = {"reward": reward}
    sidecar = {"reward": reward, "detail": detail}
    if extra:
        for key, value in extra.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[key] = value
            sidecar[key] = value
    REWARD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    EVALUATION_JSON.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def copy_judge_artifacts() -> None:
    records: list[dict] = []
    try:
        with request.urlopen(f"{JUDGE_URL}/submissions", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        http_records = payload.get("submissions", [])
        if isinstance(http_records, list):
            records.extend(record for record in http_records if isinstance(record, dict))
    except Exception as exc:
        print(f"WARN: failed to fetch judge submissions: {exc}")

    if JUDGE_SUBMISSIONS_LOG.exists():
        try:
            with JUDGE_SUBMISSIONS_LOG.open("r", encoding="utf-8") as src:
                for line in src:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError as exc:
            print(f"WARN: failed to copy submissions.jsonl: {exc}")

    by_uuid: dict[str, dict] = {}
    VERIFIER_SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    status_rank = {"queued": 0, "running": 1, "cancelled": 2, "error": 3, "done": 4}
    for record in records:
        if record.get("submission_role", "agent") != "agent":
            continue
        key = str(record.get("submission_uuid") or json.dumps(record, sort_keys=True))
        previous = by_uuid.get(key)
        if previous is None or status_rank.get(str(record.get("status")), -1) >= status_rank.get(str(previous.get("status")), -1):
            by_uuid[key] = record
    with VERIFIER_SUBMISSIONS_LOG.open("w", encoding="utf-8") as dst:
        for record in by_uuid.values():
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    if JUDGE_READY_LOG.exists():
        VERIFIER_JUDGE_READY_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(JUDGE_READY_LOG, VERIFIER_JUDGE_READY_LOG)
        except OSError as exc:
            print(f"WARN: failed to copy judge_ready.json: {exc}")


def load_submission_config() -> dict:
    if not SUBMISSION_CONFIG_PATH.exists():
        return {"kind": "file", "path": str(SOLUTION_PATH), "exclude": []}
    return json.loads(SUBMISSION_CONFIG_PATH.read_text(encoding="utf-8"))


def should_exclude(path: Path, root: Path, exclude: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    parts = set(path.relative_to(root).parts)
    for pattern in exclude:
        pattern = str(pattern).strip("/")
        if not pattern:
            continue
        if rel == pattern or rel.startswith(pattern + "/") or pattern in parts:
            return True
    return False


def make_directory_archive(root: Path, exclude: list[str]) -> tuple[str, int]:
    buf = io.BytesIO()
    file_count = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if should_exclude(path, root, exclude):
                continue
            if path.is_file():
                tar.add(path, arcname=path.relative_to(root).as_posix())
                file_count += 1
    return base64.b64encode(buf.getvalue()).decode("ascii"), file_count


def wait_for_judge() -> None:
    deadline = time.time() + JUDGE_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with request.urlopen(f"{JUDGE_URL}/health", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "error":
                raise RuntimeError(f"judge setup failed: {payload.get('error') or payload}")
            return
        except error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = {}
            if payload.get("status") == "error":
                raise RuntimeError(f"judge setup failed: {payload.get('error') or payload}")
            last_error = exc
            time.sleep(1)
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"judge service is not ready at {JUDGE_URL}: {last_error}")


def get_judge_submissions() -> list[dict]:
    try:
        with request.urlopen(f"{JUDGE_URL}/submissions", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    records = payload.get("submissions", [])
    return [record for record in records if isinstance(record, dict)]


def wait_for_async_submissions() -> None:
    deadline = time.time() + JUDGE_TIMEOUT_SECONDS
    printed = False
    while True:
        records = get_judge_submissions()
        pending = [
            record
            for record in records
            if record.get("submission_role", "agent") == "agent"
            and record.get("status") in {"queued", "running", "cancelling"}
        ]
        if not pending:
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"timed out waiting for {len(pending)} async submission(s) to finish"
            )
        if not printed:
            print(
                f"Waiting for {len(pending)} async submission(s) before final verification"
            )
            printed = True
        time.sleep(1)


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if payload.get("submission_role") == "final":
        headers["X-Frontier-CS-Role-Token"] = FINAL_ROLE_TOKEN
    req = request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=JUDGE_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc


def build_judge_payload(solution_path: Path, config: dict) -> dict:
    submission_kind = str(config.get("kind", "file"))
    exclude = list(config.get("exclude", []) or [])
    submission_uuid = str(uuid.uuid4())

    if submission_kind == "directory":
        archive_b64, _ = make_directory_archive(solution_path, exclude)
        return {
            "submission_kind": "directory",
            "submission_uuid": submission_uuid,
            "submission_role": "final",
            "archive_b64": archive_b64,
        }

    return {
        "submission_kind": "file",
        "submission_uuid": submission_uuid,
        "submission_role": "final",
        "code": solution_path.read_text(encoding="utf-8"),
    }


def load_best_agent_payload() -> dict | None:
    payload_path = BEST_JUDGE_PAYLOAD if BEST_JUDGE_PAYLOAD.exists() else BEST_AGENT_PAYLOAD
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN: failed to read best agent payload: {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    payload["submission_role"] = "final"
    payload["submission_uuid"] = str(uuid.uuid4())
    return payload


def describe_best_agent_payload() -> str:
    meta_path = BEST_JUDGE_META if BEST_JUDGE_META.exists() else BEST_AGENT_META
    if not meta_path.exists():
        return "best iterative artifact"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "best iterative artifact"
    if not isinstance(meta, dict):
        return "best iterative artifact"
    return f"best iterative artifact {meta.get('submission_uuid', '')}".strip()


def evaluate_with_judge(payload: dict) -> dict:
    wait_for_judge()
    result = post_json(f"{JUDGE_URL}/evaluate", payload)
    if result.get("status") != "done":
        raise RuntimeError(str(result.get("message") or result.get("error") or result))
    return result


def evaluate_best_with_judge() -> dict | None:
    wait_for_judge()
    payload = {
        "submission_kind": "file",
        "submission_uuid": str(uuid.uuid4()),
        "submission_role": "final",
    }
    result = post_json(f"{JUDGE_URL}/evaluate_best", payload)
    if result.get("status") != "done":
        return None
    return result


def load_submission_path(config: dict) -> Path:
    if config.get("kind") == "directory":
        return Path(config.get("path") or APP_PATH)
    return Path(config.get("path") or SOLUTION_PATH)


def write_result(result: dict) -> None:
    score = float(result.get("score", 0.0))
    score_unbounded = float(result.get("score_unbounded", score))
    message = str(result.get("message", ""))
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    reward = score / 100.0
    print(message)
    print(f"Score: {score}/100 (reward: {reward:.4f})")
    if score_unbounded != score:
        print(f"Unbounded score: {score_unbounded}")
    write_reward(
        reward,
        message,
        {
            "score": score,
            "score_unbounded": score_unbounded,
            "raw_result": result,
            **metrics,
        },
    )


def main() -> None:
    def write_best_final_result(best_result: dict, reason: str) -> None:
        metrics = dict(best_result.get("metrics", {}) or {})
        metrics["used_best_agent_artifact"] = 1
        best_result["metrics"] = metrics
        best_result["message"] = (
            f"Using {describe_best_agent_payload()} after {reason}: "
            f"{best_result.get('message', '')}"
        )
        write_result(best_result)

    def try_write_best_final_result(reason: str, final_key: tuple[float, float] | None = None) -> bool:
        try:
            best_result = evaluate_best_with_judge()
            copy_judge_artifacts()
        except Exception as exc:
            print(f"WARN: failed to rerun best iterative artifact: {exc}")
            best_result = None
        if best_result is None:
            return False
        if final_key is not None and result_score_key(best_result) <= final_key:
            return False
        write_best_final_result(best_result, reason)
        return True

    try:
        wait_for_judge()
        wait_for_async_submissions()
        copy_judge_artifacts()
        config = load_submission_config()
        solution_path = load_submission_path(config)
        if not solution_path.exists():
            print(f"ERROR: {solution_path} not found")
            if try_write_best_final_result(f"{solution_path} not found"):
                return
            write_reward(0.0, f"{solution_path} not found")
            return
        allow_empty = bool(config.get("allow_empty", False))
        if solution_path.is_file() and not allow_empty and not solution_path.read_text(encoding="utf-8").strip():
            print(f"ERROR: {solution_path} is empty")
            if try_write_best_final_result(f"{solution_path} is empty"):
                return
            write_reward(0.0, f"{solution_path} is empty")
            return

        final_result = evaluate_with_judge(build_judge_payload(solution_path, config))
        copy_judge_artifacts()
        final_key = result_score_key(final_result)
        if try_write_best_final_result("full-suite rerun", final_key):
            return
        write_result(final_result)
    except Exception as exc:
        print(traceback.format_exc())
        copy_judge_artifacts()
        if try_write_best_final_result("final evaluation failed"):
            return
        write_reward(0.0, f"Evaluation failed: {exc}")


if __name__ == "__main__":
    main()
