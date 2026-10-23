"""The task-shaped inputs the harness takes, without knowing the task.

The search, the archive, the prompts, and the Reef adapter are the same for
every discovery problem. What changes per task is a handful of strings — the
language a candidate is written in, the sentence that constrains proposable
mechanisms, the label of the judge's raw score — plus the problem statement
itself.

The problem statement comes from the Harbor task's ``instruction.md``. The
remaining strings come from the task's ``contract.json``, so adding a task
means adding a Harbor task directory, not editing the harness. A task whose
candidate is one file of a larger repository also passes the code around that
file as ``execution_context``, which only the executor reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCORE_DIRECTIONS = ("max", "min")


@dataclass(frozen=True)
class TaskContract:
    """One task's prompt and scoring vocabulary."""

    problem_prompt: str
    solution_language: str
    solution_contract: str
    mechanism_constraint: str
    objective: str
    raw_score_label: str
    score_direction: str = "max"
    judge_problem_id: str = "0"
    execution_context: str = ""

    def __post_init__(self) -> None:
        for name in (
            "problem_prompt",
            "solution_language",
            "solution_contract",
            "mechanism_constraint",
            "objective",
            "raw_score_label",
            "judge_problem_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"the task contract needs a non-empty {name}")
        if self.score_direction not in SCORE_DIRECTIONS:
            raise ValueError(f"score_direction must be one of {SCORE_DIRECTIONS}")

    @classmethod
    def load(cls, path: str | Path, *, problem_prompt: str, execution_context: str = "") -> TaskContract:
        """Read a Harbor task's ``contract.json`` and bind its instruction."""
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object in {path}")
        bound = {"problem_prompt", "execution_context"}
        unknown = set(payload) - {field for field in cls.__dataclass_fields__ if field not in bound}
        if unknown:
            raise ValueError(f"unknown task contract fields in {path}: {sorted(unknown)}")
        return cls(problem_prompt=problem_prompt, execution_context=execution_context, **payload)


__all__ = ["SCORE_DIRECTIONS", "TaskContract"]
