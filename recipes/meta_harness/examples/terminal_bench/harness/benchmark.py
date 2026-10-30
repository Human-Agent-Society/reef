"""Pinned task paths and the shared Terminus verifier score."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from reef.evaluation.harness import EvaluationTask, TaskScorer
from reef.harness import TaskResult

MANIFEST = json.loads(Path(__file__).with_name("tasks.json").read_text(encoding="utf-8"))


def task_paths(root: Path, names: list[str] | None = None) -> tuple[str, ...]:
    """Refuse dataset drift before running any paid run."""
    root = root.resolve()
    selected = MANIFEST["tasks"] if names is None else names
    if not selected or len(set(selected)) != len(selected) or any(name not in MANIFEST["tasks"] for name in selected):
        raise ValueError("tasks must be unique names from the pinned Terminal-Bench manifest")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != MANIFEST["revision"]:
        raise ValueError(f"Terminal-Bench revision must be {MANIFEST['revision']}; found {revision}")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--ignored", "--", *selected],
        text=True,
    )
    if changed:
        raise ValueError("the selected Terminal-Bench task directories must be clean, including untracked files")
    paths = []
    for name in selected:
        path = root / name
        if path.resolve().parent != root or not (path / "task.toml").is_file():
            raise ValueError(f"missing or unsafe Harbor task directory: {path}")
        paths.append(str(path))
    return tuple(paths)


def evaluate(task: str, result: TaskResult) -> float:
    """Read one binary verifier reward; failed runs count as zero."""
    if result.exit_code:
        return 0.0
    rows = [event for event in result.trajectory if event.get("type") == "verifier"]
    if len(rows) != 1 or rows[0].get("task") != task:
        raise ValueError("expected exactly one verifier record for this task")
    row = rows[0]
    if row.get("failed") or row.get("reward") is None:
        return 0.0
    reward = row["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError("Terminal-Bench verifier reward must be finite and numeric")
    if reward not in (0, 1):
        raise ValueError("Terminal-Bench verifier reward must be binary")
    return float(reward)


class VerifierScorer(TaskScorer):
    """:func:`evaluate` as the ``evolution.evaluation.scorer`` plugin terminal_bench.yaml names."""

    def score(self, task: EvaluationTask, result: TaskResult) -> float:
        return evaluate(task.text, result)
