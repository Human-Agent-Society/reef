"""The verifier's whole job under the judge protocol: ask the judge for the
final result and write it down.

Generic — task authors never edit this file. It runs inside the task
environment after the agent's budget ends, calls ``GET $JUDGE_URL/final``
(final.py on the best submission if the task has one, else the best
session score), and writes:

- ``/logs/verifier/reward.json``  — ``{"reward": <float>}``
- ``/logs/verifier/reason.txt``   — the human-readable reason
- ``/logs/verifier/submissions.jsonl`` — one ``{"t", "score"}`` line per
  submission; tide ingests these as the trusted score-over-time curve.
"""

import json
import os
import urllib.request
from pathlib import Path

VERIFIER_DIR = Path("/logs/verifier")


def finalize(judge_url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{judge_url}/final", timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"reward": 0.0, "reason": f"judge unreachable: {e!r}", "submissions": []}


if __name__ == "__main__":
    result = finalize(os.environ.get("JUDGE_URL", "http://judge:8082"))
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.json").write_text(json.dumps({"reward": result["reward"]}))
    (VERIFIER_DIR / "reason.txt").write_text(result.get("reason") or "ok")
    with (VERIFIER_DIR / "submissions.jsonl").open("w") as f:
        for entry in result.get("submissions", []):
            f.write(json.dumps({"t": entry["t"], "score": entry["score"]}) + "\n")
    print(json.dumps({"reward": result["reward"]}), result.get("reason", ""))
