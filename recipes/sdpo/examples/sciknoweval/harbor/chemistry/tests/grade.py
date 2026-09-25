"""The verifier: read the run's curve and report its last avg@16 as the reward.

The runner writes the curve as it goes, so a run that ends early still leaves
the evaluations it finished. A missing record means the runner never completed
an evaluation, which is a zero, not an error.
"""

import json
import os
from pathlib import Path

VERIFIER_DIR = Path("/logs/verifier")
RESULT_PATH = Path(os.environ.get("SDPO_RESULT_PATH", "/workspace/chemistry.json"))

if __name__ == "__main__":
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    if RESULT_PATH.is_file():
        record = json.loads(RESULT_PATH.read_text())
        evaluations = record.get("evaluations", [])
        last = evaluations[-1] if evaluations else {}
        rewards = {
            "reward": last.get("avg_at_n", 0.0),
            "step": last.get("step", 0),
            "initial": evaluations[0]["avg_at_n"] if evaluations else 0.0,
            "evaluations": len(evaluations),
        }
        reason = f"avg@{last.get('n', 0)} after {last.get('step', 0)} steps"
    else:
        rewards = {"reward": 0.0, "step": 0, "initial": 0.0, "evaluations": 0}
        reason = f"{RESULT_PATH} is missing: the runner completed no evaluation"
        evaluations = []
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(rewards))
    (VERIFIER_DIR / "reason.txt").write_text(reason)
    with (VERIFIER_DIR / "curve.jsonl").open("w") as handle:
        for entry in evaluations:
            handle.write(json.dumps(entry) + "\n")
    print(json.dumps(rewards), reason)
