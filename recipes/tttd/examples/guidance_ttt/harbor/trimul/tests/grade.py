"""The verifier: re-submit the trial's candidate to task judge and record it.

The agent's own search already scored thousands of candidates through the same
judge, but those scores reached Reef as training reports. This runs once, in
the task environment, on whatever ``/workspace/solution.py`` the trial left
behind, and writes the trusted trial reward:

- ``/logs/verifier/reward.json`` — ``{"reward": <task judge score>}``
- ``/logs/verifier/reason.txt``  — the judge's message

An unreachable judge, a rejected program, or a missing candidate is reward
zero with the reason recorded, never a crash.
"""

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

SOLUTION_PATH = Path("/workspace/solution.py")
VERIFIER_DIR = Path("/logs/verifier")
POLL_INTERVAL_S = 2.0
DEADLINE_S = 1200.0


def submit(judge_url: str, problem_id: str, code: str) -> str:
    boundary = f"reef-guidance-{uuid.uuid4().hex}"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="pid"\r\n\r\n{problem_id}\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="lang"\r\n\r\npython\r\n'.encode(),
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="code"; filename="solution.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
        ).encode()
        + code.encode()
        + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    request = urllib.request.Request(
        f"{judge_url}/submit",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    submission_id = payload.get("sid")
    if isinstance(submission_id, int) and not isinstance(submission_id, bool):
        submission_id = str(submission_id)
    if not isinstance(submission_id, str) or not submission_id.strip():
        raise RuntimeError(f"task judge submission response has no sid: {payload!r}")
    return submission_id


def poll(judge_url: str, submission_id: str) -> dict:
    deadline = time.monotonic() + DEADLINE_S
    quoted = urllib.parse.quote(submission_id, safe="")
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{judge_url}/result/{quoted}", timeout=10) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            result = None
        except (OSError, ValueError):
            result = None
        if isinstance(result, dict) and result.get("status") in {"done", "error", "environment_error"}:
            return result
        time.sleep(POLL_INTERVAL_S)
    return {"status": "timeout", "message": f"judge did not finish within {DEADLINE_S:g}s"}


def grade() -> dict:
    judge_url = os.environ.get("GUIDANCE_JUDGE_CONTAINER_URL", "http://host.docker.internal:8082").rstrip("/")
    problem_id = "trimul"
    if not SOLUTION_PATH.is_file():
        return {"reward": 0.0, "reason": f"{SOLUTION_PATH} is missing"}
    code = SOLUTION_PATH.read_text(errors="replace")
    if not code.strip():
        return {"reward": 0.0, "reason": f"{SOLUTION_PATH} is empty"}
    try:
        result = poll(judge_url, submit(judge_url, problem_id, code))
    except Exception as error:  # An unreachable judge is a zero, not a crash.
        return {"reward": 0.0, "reason": f"judge unreachable: {error!r}"}
    if result.get("status") != "done" or result.get("valid", True) is not True:
        return {"reward": 0.0, "reason": str(result.get("message") or result.get("status") or "rejected")}
    try:
        reward = float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return {"reward": 0.0, "reason": f"judge returned a non-numeric score: {result!r}"}
    if not math.isfinite(reward) or reward < 0:
        return {"reward": 0.0, "reason": "judge returned a non-finite or negative reward"}
    return {"reward": reward, "reason": str(result.get("message") or "accepted")}


if __name__ == "__main__":
    outcome = grade()
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.json").write_text(json.dumps({"reward": outcome["reward"]}))
    (VERIFIER_DIR / "reason.txt").write_text(outcome["reason"])
    print(json.dumps(outcome))
