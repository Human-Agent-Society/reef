"""The verifier: read the run's record and write the last grid's accuracy as the reward.

The runner in the agent's container writes ``/workspace/grid.json`` as it goes,
so a run that ends early still leaves the grids it finished. A missing record
means the runner never completed a grid, which is a zero, not an error.
"""

import json
import os
from pathlib import Path

VERIFIER_DIR = Path("/logs/verifier")
RESULT_PATH = Path(os.environ.get("SDPO_RESULT_PATH", "/workspace/grid.json"))

if __name__ == "__main__":
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    if RESULT_PATH.is_file():
        record = json.loads(RESULT_PATH.read_text())
        steps = record.get("steps", [])
        rewards = {"reward": record["accuracy"], "steps": len(steps)}
        reason = f"accuracy of the last of {len(steps)} grids"
    else:
        rewards = {"reward": 0.0, "steps": 0}
        reason = f"{RESULT_PATH} is missing: the runner completed no grid"
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(rewards))
    (VERIFIER_DIR / "reason.txt").write_text(reason)
    print(json.dumps(rewards), reason)
