"""Score the answer and require an actual multiply call in the run's ATIF."""

import json
from pathlib import Path


def grade(answer_path: Path, trajectory_path: Path) -> float:
    """Return 1 for answer 391 with a recorded multiply call, otherwise 0."""
    if not answer_path.is_file() or answer_path.read_text(encoding="utf-8").strip() != "391":
        return 0.0
    if not trajectory_path.is_file():
        return 0.0
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls", []):
            if call.get("function_name") == "multiply" and call.get("arguments") == {"left": 17, "right": 23}:
                return 1.0
    return 0.0


if __name__ == "__main__":
    reward = grade(Path("/workspace/answer.txt"), Path("/logs/agent/trajectory.json"))
    Path("/logs/verifier/reward.txt").write_text(f"{reward}\n", encoding="utf-8")
