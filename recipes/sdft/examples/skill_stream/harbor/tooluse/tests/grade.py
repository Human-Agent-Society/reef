"""The verifier: ask the judge for the stage's final result and write it down.

reef-eval's template verifier, extended for a stage whose submissions score
the served model: the judge's final result is the last submission, and its
``data`` (the step and every skill's accuracy) becomes extra reward keys and
extra fields on the submission log, so a stream's rows carry every skill's
accuracy at every position. Runs inside the task environment after the
agent's budget ends; the judge exposes finalization on a separate verifier
port behind a token.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VERIFIER_DIR = Path("/logs/verifier")
SKILLS = ("tooluse", "science")


def verifier_url(judge_url: str) -> str:
    explicit = os.environ.get("VERIFIER_JUDGE_URL")
    if explicit:
        return explicit.rstrip("/")
    parsed = urllib.parse.urlparse(judge_url)
    return f"http://{parsed.hostname or 'localhost'}:{(parsed.port or 8082) + 1}"


def finalize(judge_url: str) -> dict:
    url = verifier_url(judge_url)
    try:
        with urllib.request.urlopen(f"{url}/token", timeout=10) as response:
            token = json.loads(response.read())["token"]
        request = urllib.request.Request(f"{url}/final", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        return {"reward": 0.0, "reason": f"judge unreachable: {error!r}", "submissions": []}


if __name__ == "__main__":
    result = finalize(os.environ.get("JUDGE_URL", "http://judge:8082"))
    submissions = result.get("submissions", [])
    last = submissions[-1] if submissions else {}
    rewards = {"reward": result["reward"], **{skill: last[skill] for skill in SKILLS if skill in last}}
    if "step" in last:
        rewards["step"] = last["step"]
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(rewards))
    (VERIFIER_DIR / "reason.txt").write_text(result.get("reason") or "ok")
    with (VERIFIER_DIR / "submissions.jsonl").open("w") as handle:
        for entry in submissions:
            handle.write(json.dumps({key: value for key, value in entry.items() if key != "reason"}) + "\n")
    print(json.dumps(rewards), result.get("reason", ""))
