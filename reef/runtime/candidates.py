"""Runtime boundary for producing and activating model candidates.

Selection-aware runtimes implement this structural protocol so candidate
evaluation happens after checkpoint export and before serving mutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.train.evaluation.contracts import UpdateCandidate


class CandidateTrainingDeferred(Exception):
    """A candidate was not produced, but retrying the same batch is safe."""

    def __init__(self, storage: Mapping[str, Any]) -> None:
        super().__init__("candidate training is waiting for checkpoint storage")
        self.storage = dict(storage)


class StaleCandidate(Exception):
    """The reserved batch became stale before it could produce a candidate."""

    def __init__(self, metrics: Mapping[str, Any] | None = None) -> None:
        super().__init__("candidate training rejected a stale batch")
        self.metrics = dict(metrics or {})


@dataclass(frozen=True, kw_only=True)
class ModelCandidate(UpdateCandidate):
    """A checkpointed model update that has not changed serving weights."""

    training_job_id: str
    checkpoint_path: str
    current_runtime_load_id: str | None
    training_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.training_job_id, str) or not self.training_job_id:
            raise ValueError("training_job_id must be a non-empty string")
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
            raise ValueError("checkpoint_path must be a non-empty string")
        if self.current_runtime_load_id is not None and (
            not isinstance(self.current_runtime_load_id, str) or not self.current_runtime_load_id
        ):
            raise ValueError("current_runtime_load_id must be a non-empty string or None")


@dataclass(frozen=True)
class ActivatedModel:
    """Serving identity returned after a selected candidate is activated."""

    candidate_id: str
    runtime_load_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.runtime_load_id, str) or not self.runtime_load_id:
            raise ValueError("runtime_load_id must be a non-empty string")


__all__ = [
    "ActivatedModel",
    "CandidateTrainingDeferred",
    "ModelCandidate",
    "StaleCandidate",
]
