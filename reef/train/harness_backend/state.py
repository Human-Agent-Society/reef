"""The harness backend's algorithm state, the part of a commit record the backend owns.

The tree itself is not here: it is the git release the commit recorded, and
the trees a step tried are in the scenario's history. The state keeps what
crosses steps and must commit with them: the failure streak the budget reads, the
collected evaluation tasks, the recheck's rollback target, and the method's
own files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class HarnessState:
    """The harness backend's state after one step.

    Values: ``failure_streak`` ≥ 0 counts the candidates rejected since the latest
    selected one. ``collected_tasks`` are the prompts collected into the evaluation tasks, in collection order,
    without duplicates. ``collected_clients`` counts, per tagged client, the
    prompts collected from its traffic; every count is positive.
    ``rollback_release_id`` names the release the latest publish replaced,
    which a recheck evaluates again; ``rollback_context`` is the evaluation
    context key the latest publish or the latest recheck that kept the head
    ran in. Each is non-empty text or ``None``, and the two are both ``None``
    or both set.
    ``method_files`` maps each relative path of the method's ``state/``
    directory to its text.
    Purpose: ``HarnessBackend`` reads it from the committed algorithm state
    with ``from_json`` and returns ``to_json`` of the next one in each
    ``TrainStepResult``.
    """

    failure_streak: int = 0
    collected_tasks: tuple[str, ...] = ()
    collected_clients: Mapping[str, int] = field(default_factory=dict)
    rollback_release_id: str | None = None
    rollback_context: str | None = None
    method_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        streak = self.failure_streak
        if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
            raise ValueError("HarnessState.failure_streak must be an integer of at least 0")
        if len(set(self.collected_tasks)) != len(self.collected_tasks):
            raise ValueError("HarnessState.collected_tasks must not repeat a prompt")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in self.collected_clients.values()
        ):
            raise ValueError("HarnessState.collected_clients counts must be positive integers")
        rollback = (("rollback_release_id", self.rollback_release_id), ("rollback_context", self.rollback_context))
        for label, text in rollback:
            if text is not None and (not isinstance(text, str) or not text):
                raise ValueError(f"HarnessState.{label} must be non-empty text or None")
        if (self.rollback_release_id is None) != (self.rollback_context is None):
            raise ValueError("HarnessState rollback_release_id and rollback_context are both set or both None")
        if any(not isinstance(text, str) for text in self.method_files.values()):
            raise ValueError("HarnessState.method_files must map paths to text")
        object.__setattr__(self, "collected_clients", MappingProxyType(dict(self.collected_clients)))
        object.__setattr__(self, "method_files", MappingProxyType(dict(self.method_files)))

    def to_json(self) -> dict[str, Any]:
        """The commit record's form of this state; ``from_json`` reads it back as an equal value."""
        return {
            "failure_streak": self.failure_streak,
            "collected_tasks": list(self.collected_tasks),
            "collected_clients": dict(self.collected_clients),
            "rollback_release_id": self.rollback_release_id,
            "rollback_context": self.rollback_context,
            "method_files": dict(self.method_files),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> HarnessState:
        """The state a commit record holds; a key ``to_json`` writes and ``value`` lacks takes its default."""
        return cls(
            failure_streak=int(value.get("failure_streak", 0)),
            collected_tasks=tuple(str(task) for task in value.get("collected_tasks", ())),
            collected_clients={
                str(name): int(count) for name, count in dict(value.get("collected_clients", {})).items()
            },
            rollback_release_id=value.get("rollback_release_id"),
            rollback_context=value.get("rollback_context"),
            method_files={str(path): str(text) for path, text in dict(value.get("method_files", {})).items()},
        )


__all__ = ["HarnessState"]
