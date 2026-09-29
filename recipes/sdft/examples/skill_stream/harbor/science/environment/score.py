"""The judge's scoring rule: score the served model on both skills' test splits.

A submission is ``{"task": <this stage's task>, "step": n}``; the score it
earns is the accuracy on this task's test split, and both accuracies ride
along as ``data`` so the trace rows carry the whole picture of the stream
at that step. The judge reaches the Reef service the stage trains through
(``REEF_SERVICE_URL``/``REEF_TOKEN``/``REEF_SCENARIO`` in its environment)
and decodes greedily, the reference's evaluation setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import skills

#: The task this judge scores as its reward; the other skill is reported beside it.
TASK = json.loads((Path(__file__).parent / "task.json").read_text())["task"]


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no submission"}
    try:
        submission = json.loads(Path(artifact).read_text())
        step = int(submission["step"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return {"reward": 0.0, "reason": f"malformed submission: {error!r}"}
    client = skills.make_client()
    results = {task: skills.evaluate(client, task) for task in skills.TASKS}
    reason = f"step {step}: " + ", ".join(
        f"{task} {result['accuracy']:.4f} ({result['num_correct']}/{result['num_total']})"
        for task, result in results.items()
    )
    return {
        "reward": results[TASK]["accuracy"],
        "reason": reason,
        "data": {"step": step, **{task: result["accuracy"] for task, result in results.items()}},
    }
