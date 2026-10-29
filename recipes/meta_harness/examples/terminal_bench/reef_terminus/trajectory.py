"""Read the terminus runner's trials back as a rollout's trajectory.

The adapter names :func:`read_trials` as its ``sessions`` format, a
``module:function`` reference that Reef's trajectory reader resolves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reef.harness.rollouts.trajectory import TrajectoryError


def read_trials(path: Path) -> tuple[dict[str, Any], ...]:
    """Every ``*.json`` under ``path``, each a trial the runner wrote: one ``verifier`` event, then its steps.

    The runner writes one file per trial: an ATIF trajectory whose ``steps``
    carry the agent's turns, beside the verifier's outcome. Each file becomes
    a ``verifier`` event holding the reward and the task it scored, then one
    ``step`` event per ATIF step, so a scorer reads the reward and a method
    reads the turns from the same trajectory.
    """
    events: list[dict[str, Any]] = []
    for file in sorted(Path(path).rglob("*.json")):
        try:
            trial = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TrajectoryError(f"terminus trial {file} is not valid JSON") from exc
        if not isinstance(trial, dict) or not isinstance(trial.get("steps"), list):
            raise TrajectoryError(f"terminus trial {file} is not an ATIF trajectory with a steps list")
        events.append({"type": "verifier", **{key: value for key, value in trial.items() if key != "steps"}})
        events.extend({"type": "step", **step} for step in trial["steps"] if isinstance(step, dict))
    return tuple(events)
