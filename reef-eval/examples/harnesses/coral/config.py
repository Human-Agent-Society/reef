"""CORAL task configuration builder."""

from __future__ import annotations

from typing import Any


def coral_config(
    instruction: str,
    model: str,
    *,
    agents: int = 2,
) -> dict[str, Any]:
    """Return a CORAL task whose only grader is the reef-eval judge."""
    description = f"""{instruction}

Work on solution.json. It must contain exactly the JSON solution described above.
Run `coral eval` whenever a candidate is worth spending one reef-eval submission on.
The returned score and feedback come directly from the reef-eval judge. Keep the best
candidate in solution.json and commit useful improvements so the other agents can
build on them.
"""
    return {
        "task": {
            "name": "reef-eval benchmark",
            "description": description,
            "tips": (
                "Submissions are limited. Use local checks for cheap filtering and "
                "reserve `coral eval` for promising candidates."
            ),
        },
        "grader": {
            "entrypoint": "reef_eval_coral_grader.grader:Grader",
            "setup": ["uv pip install -e ./grader"],
            "timeout": 60,
            "direction": "maximize",
            "args": {"solution_file": "solution.json"},
        },
        "agents": {
            "count": agents,
            "runtime": "codex",
            "model": model,
            "max_turns": 200,
        },
        "workspace": {
            "results_dir": "./results",
            "repo_path": "./seed",
        },
        "run": {"verbose": False, "ui": False, "session": "local"},
    }
