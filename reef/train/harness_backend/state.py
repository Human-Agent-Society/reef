"""The harness backend's algorithm state, the part of a commit record the backend owns.

The tree itself is not here: it is the git release the commit recorded. The
state keeps the bookkeeping that crosses steps: the counters the budgets
read, the grown evaluation tasks, the failure manifest, the rejected
proposals, the recheck's rollback target, the latest outcome, and the
method's own state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from reef.train.harness_backend.contract import RejectedProposal, StepOutcome
from reef.train.harness_backend.manifest import FailureManifest


@dataclass(frozen=True)
class HarnessState:
    """The harness backend's state after one step.

    Values: ``steps`` ≥ 0 counts the steps the backend prepared.
    ``failure_streak`` ≥ 0 counts the proposals rejected since the latest
    selected one. ``manifest`` is the failure manifest of the tree the latest
    evaluated step committed, ``None`` before the first. ``promoted_tasks``
    are the prompts promoted into the evaluation tasks, in promotion order,
    without duplicates. ``promoted_clients`` counts, per tagged client, the
    prompts promoted from its traffic; every count is positive.
    ``rejected`` holds the latest rejected proposals, oldest first.
    ``rollback_release_id`` names the release the latest publish replaced,
    which a recheck evaluates again; ``rollback_context`` is the evaluation
    context of that publish, and the two are both ``None`` or both set.
    ``last_outcome`` is the outcome of the latest step that evaluated the
    method's own proposal, ``None`` before the first. ``method_state`` is the
    method's JSON state.
    Purpose: ``HarnessBackend`` reads it from the committed algorithm state
    with ``from_json`` and returns ``to_json`` of the next one in each
    ``TrainStepResult``.
    """

    steps: int = 0
    failure_streak: int = 0
    manifest: FailureManifest | None = None
    promoted_tasks: tuple[str, ...] = ()
    promoted_clients: Mapping[str, int] = field(default_factory=dict)
    rejected: tuple[RejectedProposal, ...] = ()
    rollback_release_id: str | None = None
    rollback_context: Mapping[str, Any] | None = None
    last_outcome: StepOutcome | None = None
    method_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (("steps", self.steps), ("failure_streak", self.failure_streak)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"HarnessState.{label} must be an integer of at least 0")
        if len(set(self.promoted_tasks)) != len(self.promoted_tasks):
            raise ValueError("HarnessState.promoted_tasks must not repeat a prompt")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in self.promoted_clients.values()
        ):
            raise ValueError("HarnessState.promoted_clients counts must be positive integers")
        if (self.rollback_release_id is None) != (self.rollback_context is None):
            raise ValueError("HarnessState rollback_release_id and rollback_context are both set or both None")
        object.__setattr__(self, "promoted_clients", MappingProxyType(dict(self.promoted_clients)))
        object.__setattr__(self, "method_state", MappingProxyType(dict(self.method_state)))

    def to_json(self) -> dict[str, Any]:
        """The commit record's form of this state; ``from_json`` reads it back as an equal value."""
        return {
            "steps": self.steps,
            "failure_streak": self.failure_streak,
            "failure_manifest": None if self.manifest is None else self.manifest.to_state(),
            "promoted_tasks": list(self.promoted_tasks),
            "promoted_clients": dict(self.promoted_clients),
            "rejected_proposals": [proposal.to_json() for proposal in self.rejected],
            "rollback_release_id": self.rollback_release_id,
            "rollback_context": None if self.rollback_context is None else dict(self.rollback_context),
            "last_outcome": None if self.last_outcome is None else self.last_outcome.to_json(),
            "method_state": dict(self.method_state),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> HarnessState:
        """The state a commit record holds; a key ``to_json`` writes and ``value`` lacks takes its default."""
        manifest = value.get("failure_manifest")
        outcome = value.get("last_outcome")
        return cls(
            steps=int(value.get("steps", 0)),
            failure_streak=int(value.get("failure_streak", 0)),
            manifest=None if manifest is None else FailureManifest.from_state(manifest),
            promoted_tasks=tuple(str(task) for task in value.get("promoted_tasks", ())),
            promoted_clients={
                str(name): int(count) for name, count in dict(value.get("promoted_clients", {})).items()
            },
            rejected=tuple(RejectedProposal.from_json(item) for item in value.get("rejected_proposals", ())),
            rollback_release_id=value.get("rollback_release_id"),
            rollback_context=value.get("rollback_context"),
            last_outcome=None if outcome is None else StepOutcome.from_json(outcome),
            method_state=dict(value.get("method_state", {})),
        )


__all__ = ["HarnessState"]
