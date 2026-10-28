"""The activation rule: whether Reef activates an artifact that training made (RFC #965).

Training makes artifacts and never activates one. A scenario's ``ActivationRule`` decides for each new
artifact: activate it at once, after its evaluation, or never. The committer asks the rule when a training step
publishes an artifact, and the training record keeps the result (``CommitRecord.activated``). The artifact
evaluation service asks the rule when an evaluation of an artifact finishes. A deployment names the rule in its
``activation`` section.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from reef.core.evaluation import PolicyEvaluation
from reef.core.plugins import ConfiguredPlugin


@dataclass(frozen=True)
class MadeArtifact:
    """An artifact that a training step makes, as the activation rule sees it.

    Values: ``component`` is the non-empty name of the component whose trainer makes it, or None for the one
    trainer of a scenario whose surface names no component. ``changed_kinds`` holds the harness node kinds that
    the artifact changes from the tree it was trained from, and is empty for weights. Purpose: the committer
    makes it when a training step publishes an artifact, and the scenario's ``ActivationRule`` reads it.
    """

    component: str | None
    changed_kinds: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.component is not None and (not isinstance(self.component, str) or not self.component):
            raise ValueError("a made artifact's component is None or a non-empty name")
        if not isinstance(self.changed_kinds, frozenset) or any(
            not isinstance(kind, str) or not kind for kind in self.changed_kinds
        ):
            raise ValueError("a made artifact's changed_kinds is a frozenset of non-empty kind names")


class ActivationRule(ConfiguredPlugin, ABC):
    """Decides whether Reef activates an artifact that training made.

    Operations: ``activates(made)`` is true when Reef activates ``made`` as soon as a training step publishes it.
    ``activates_evaluated(artifact_id, evaluation)`` is true when Reef activates the artifact ``artifact_id``
    once ``evaluation``, an evaluation of it, finished; the default is false. Both decide from their arguments
    alone. Purpose: the scenario's committer and the artifact evaluation service ask the rule of the scenario's
    recipe, and activate the artifact when it says so.
    """

    @abstractmethod
    def activates(self, made: MadeArtifact) -> bool:
        """Whether Reef activates ``made`` as soon as a training step publishes it."""

    def activates_evaluated(self, artifact_id: str, evaluation: PolicyEvaluation) -> bool:
        """Whether Reef activates ``artifact_id`` once ``evaluation`` of it finished."""
        return False


__all__ = ["ActivationRule", "MadeArtifact"]
