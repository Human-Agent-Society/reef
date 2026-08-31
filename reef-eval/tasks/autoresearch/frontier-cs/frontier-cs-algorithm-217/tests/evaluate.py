#!/usr/bin/env python3
"""
Harbor verifier for Frontier-CS algorithmic problems.
Submits the agent's C++ code to the black-box judge proxy and retrieves the score.
"""

import json
import os
import time
import uuid
from pathlib import Path

import requests

JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8082").rstrip("/")
PROBLEM_ID = os.environ.get("PROBLEM_ID", "0")
SOLUTION_PATH = Path("/app/solution.cpp")
REWARD_TXT = Path("/logs/verifier/reward.txt")
REWARD_JSON = Path("/logs/verifier/reward.json")
VERIFIER_SUBMISSIONS_LOG = Path("/logs/verifier/submissions.jsonl")
JUDGE_RESULT_JSON = Path("/logs/verifier/judge_result.json")
MAX_POLL_TIME = int(os.environ.get("MAX_POLL_TIME", "600"))  # seconds


def judge_submission_records() -> list[dict]:
    """Fetch sanitized iterative submission records from the judge proxy."""
    try:
        response = requests.get(f"{JUDGE_URL}/submissions", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"WARN: failed to fetch judge submissions: {exc}")
        return []

    records = payload.get("submissions", [])
    if not isinstance(records, list):
        return []

    normalized: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        score_raw = record.get("score") or 0.0
        try:
            reward = float(score_raw) / 100.0
        except (TypeError, ValueError):
            reward = 0.0
        normalized.append(
            {
                "ts": record.get("ts"),
                "status": record.get("status", "unknown"),
                "submission_role": record.get("submission_role", "agent"),
                "submission_uuid": record.get("submission_uuid"),
                "problem_id": record.get("problem_id") or PROBLEM_ID,
                "score": reward,
                "score_raw": score_raw,
                "score_unbounded": record.get("score_unbounded", score_raw),
                "detail": record.get("message") or "",
                "metrics": record.get("metrics") or {},
            }
        )
    return normalized


def write_submissions_log(records: list[dict]) -> None:
    VERIFIER_SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VERIFIER_SUBMISSIONS_LOG.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def best_submission(records: list[dict]) -> dict | None:
    best: dict | None = None
    for record in records:
        try:
            reward = float(record.get("score", 0.0))
        except (TypeError, ValueError):
            reward = 0.0
        if record.get("submission_role", "agent") != "agent":
            continue
        if record.get("status") != "done":
            continue
        if best is None or reward > float(best.get("score", 0.0)):
            best = record
    return best


def write_reward(
    score: float, detail: str = "", extra: dict[str, object] | None = None
):
    """Write Harbor's reward files (strictly numeric) plus a sidecar
    `judge_result.json` for any qualitative judge metadata.

    Harbor's VerifierResult.rewards is dict[str, float|int]; non-numeric
    fields here would fail Pydantic validation, so we route status / detail /
    raw_result into the sidecar file that the analysis pipeline reads.
    """
    REWARD_TXT.parent.mkdir(parents=True, exist_ok=True)
    REWARD_TXT.write_text(str(score))
    numeric_rewards: dict[str, float | int] = {"reward": score}

    sidecar: dict[str, object] = {"reward": score, "detail": detail}
    if extra:
        for key, value in extra.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_rewards[key] = value
            sidecar[key] = value
    REWARD_JSON.write_text(json.dumps(numeric_rewards, indent=2))
    JUDGE_RESULT_JSON.write_text(json.dumps(sidecar, indent=2))


def wait_for_judge() -> bool:
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            response = requests.get(f"{JUDGE_URL}/health", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def evaluate_with_judge(code: str) -> dict:
    response = requests.post(
        f"{JUDGE_URL}/evaluate",
        json={
            "submission_uuid": str(uuid.uuid4()),
            "submission_role": "final",
            "problem_id": PROBLEM_ID,
            "code": code,
        },
        timeout=MAX_POLL_TIME,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") not in ("done", "error"):
        raise RuntimeError(str(result.get("message") or result))
    return result


def main():
    print(f"Frontier-CS Problem {PROBLEM_ID}")
    print(f"Judge: {JUDGE_URL}")

    records = judge_submission_records()
    write_submissions_log(records)
    best = best_submission(records)

    def write_best_submission_reward(reason: str) -> bool:
        if best is None:
            return False
        reward = float(best.get("score", 0.0))
        score_raw = best.get("score_raw", reward * 100.0)
        print(f"Using best iterative submission after {reason}: reward={reward:.4f}")
        write_reward(
            reward,
            f"Using best iterative submission after {reason}: {best.get('detail', '')}",
            extra={
                "score": score_raw,
                "best_submission_reward": reward,
                "used_best_submission": 1,
                "raw_result": best.get("raw_result"),
            },
        )
        return True

    # 1. Check solution file
    if not SOLUTION_PATH.exists():
        print("ERROR: /app/solution.cpp not found")
        if write_best_submission_reward("solution.cpp not found"):
            return
        write_reward(0.0, "solution.cpp not found")
        return

    code = SOLUTION_PATH.read_text()
    if not code.strip():
        print("ERROR: solution.cpp is empty")
        if write_best_submission_reward("solution.cpp is empty"):
            return
        write_reward(0.0, "solution.cpp is empty")
        return

    # 2. Wait for judge availability (may take time to start)
    print("Waiting for judge...")
    if not wait_for_judge():
        print(f"ERROR: Judge not available at {JUDGE_URL} after 60s")
        write_reward(0.0, "Judge not available")
        return

    print("Judge is ready.")

    # 3. Evaluate
    print("Submitting...")
    try:
        result = evaluate_with_judge(code)
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        records = judge_submission_records()
        write_submissions_log(records)
        best = best_submission(records)
        if write_best_submission_reward(f"final evaluation failed: {e}"):
            return
        write_reward(0.0, f"Evaluation failed: {e}")
        return

    records = judge_submission_records()
    write_submissions_log(records)
    best = best_submission(records)

    # 4. Parse result
    if result.get("status") == "error":
        msg = result.get("message") or result.get("error") or "Unknown error"
        print(f"ERROR: {msg}")
        if write_best_submission_reward(f"final evaluation error: {msg}"):
            return
        write_reward(
            0.0,
            msg,
            extra={
                "status": result.get("status"),
                "raw_result": result,
            },
        )
        return

    score = result.get("score") or 0.0  # 0-100
    reward = float(score) / 100.0  # normalize to 0-1
    unbounded = result.get("score_unbounded")
    if best is not None and float(best.get("score", 0.0)) > reward:
        write_best_submission_reward("final solution scored below best submission")
        return

    print(f"Score: {score}/100 (reward: {reward:.4f})")
    if unbounded is not None and unbounded != score:
        print(f"Unbounded score: {unbounded}")

    REWARD_TXT.parent.mkdir(parents=True, exist_ok=True)
    REWARD_TXT.write_text(str(reward))
    # Harbor's VerifierResult demands strictly numeric reward dict keys; we
    # write score and the raw score variants here, then push the rest to
    # the sidecar file the analysis pipeline reads.
    numeric_rewards: dict[str, float | int] = {"reward": reward, "score": float(score)}
    if isinstance(unbounded, (int, float)):
        numeric_rewards["score_unbounded"] = float(unbounded)
    REWARD_JSON.write_text(json.dumps(numeric_rewards, indent=2))
    JUDGE_RESULT_JSON.write_text(
        json.dumps(
            {
                "reward": reward,
                "score": score,
                "score_unbounded": unbounded,
                "status": result.get("status"),
                "detail": result.get("message") or "",
                "metrics": result.get("metrics") or {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
