"""Evaluation values retained in the Meta-Harness workspace."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

SEARCH_SPLITS = ("train", "dev")
ALL_SPLITS = (*SEARCH_SPLITS, "test")


@dataclass(frozen=True)
class TrialEvidence:
    """One verifier-owned task trial and its auditable execution evidence."""

    task_id: str
    trial: int
    reward: float
    trajectory: tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    status: str = "completed"
    error: str | None = None
    verifier: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("trial evidence requires a task id")
        if self.trial < 0:
            raise ValueError("trial index must be non-negative")
        if not 0.0 <= float(self.reward) <= 1.0:
            raise ValueError("trial reward must be between zero and one")
        if self.estimated_cost_usd < 0 or self.wall_time_s < 0:
            raise ValueError("trial cost and wall time must be non-negative")
        if not self.status:
            raise ValueError("trial evidence requires a status")

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """One candidate's results on exactly one named task split."""

    split: str
    trials: tuple[TrialEvidence, ...]

    def __post_init__(self) -> None:
        if self.split not in ALL_SPLITS:
            raise ValueError(f"evaluation split must be one of {ALL_SPLITS}")
        if not self.trials:
            raise ValueError("evaluation result requires at least one trial")
        identities = [(trial.task_id, trial.trial) for trial in self.trials]
        if len(set(identities)) != len(identities):
            raise ValueError("evaluation result contains a duplicate task trial")

    @property
    def score(self) -> float:
        return statistics.fmean(trial.reward for trial in self.trials)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(trial.task_id for trial in self.trials))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "score": self.score,
            "trials": [trial.to_jsonable() for trial in self.trials],
            "usage": aggregate_usage(trial.usage for trial in self.trials),
            "estimated_cost_usd": sum(trial.estimated_cost_usd for trial in self.trials),
            "wall_time_s": sum(trial.wall_time_s for trial in self.trials),
        }


def aggregate_usage(items: Iterable[Mapping[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals
