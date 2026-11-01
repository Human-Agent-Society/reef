"""The harness backend's algorithm state, the part of a commit record the backend owns.

The tree itself is not here: it is the git release the commit recorded, and
the trees a step tried are in the scenario's history. The state keeps what
crosses steps and must commit with them: the failure streak the budget reads, the
collected evaluation tasks, and the method's own files.
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
    ``method_files`` maps each relative path of the method's ``state/``
    directory to its text.
    Purpose: ``HarnessBackend`` reads it from the committed algorithm state
    with ``from_json`` and returns ``to_json`` of the next one in each
    ``TrainStepResult``.
    """

    failure_streak: int = 0
    collected_tasks: tuple[str, ...] = ()
    collected_clients: Mapping[str, int] = field(default_factory=dict)
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
            method_files={str(path): str(text) for path, text in dict(value.get("method_files", {})).items()},
        )


__all__ = ["HarnessState"]
