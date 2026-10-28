"""The built-in activation rules, and the ``activation`` section that names one.

An ``activation`` section has this shape::

    activation:
      module: always            # or never, evaluated, or a package.module:ClassName
      config:
        review_kinds: [skill]

- ``always`` (``AlwaysActivate``) activates every new artifact, except a harness tree that changes a reviewed
  kind: a kind in ``review_kinds``, or one that Reef always reviews (``ALWAYS_REVIEWED_KINDS``). It is the
  default.
- ``never`` (``NeverActivate``) activates none; a person activates each one
  (``POST /reef/scenarios/{scenario}/activate``).
- ``evaluated`` (``EvaluatedActivation``) activates an artifact once an evaluation of it finishes with a
  ``mean_score`` of at least ``min_mean_score``. The scenario's ``artifact_evaluation`` settings evaluate each
  new artifact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

from reef.core.activation import ActivationRule, MadeArtifact
from reef.core.evaluation import PolicyEvaluation
from reef.core.plugins import PluginConfigError, PluginReference
from reef.harness.tree.nodes import ALWAYS_REVIEWED_KINDS


class AlwaysActivate(ActivationRule):
    """Activate every new artifact, except a harness tree that changes a reviewed kind.

    Values: ``review_kinds`` is a frozenset of non-empty node kind names. A tree that changes one of them, or one
    of ``ALWAYS_REVIEWED_KINDS``, waits for a person to activate it.
    """

    def __init__(self, review_kinds: frozenset[str] = frozenset()) -> None:
        if not isinstance(review_kinds, frozenset) or any(
            not isinstance(kind, str) or not kind for kind in review_kinds
        ):
            raise ValueError("review_kinds is a set of non-empty node kind names")
        self.review_kinds = review_kinds

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        unknown = sorted(set(config) - {"review_kinds"})
        if unknown:
            raise PluginConfigError(f"the always activation takes only review_kinds, not {unknown}")
        kinds = config.get("review_kinds", [])
        if isinstance(kinds, str) or not isinstance(kinds, list):
            raise PluginConfigError("activation.config.review_kinds must be a list of node kind names")
        try:
            return cls(frozenset(kinds))
        except ValueError as exc:
            raise PluginConfigError(f"activation.config.{exc}") from exc

    def activates(self, made: MadeArtifact) -> bool:
        return not (made.changed_kinds & (self.review_kinds | ALWAYS_REVIEWED_KINDS))


class NeverActivate(ActivationRule):
    """Activate no artifact: a person activates each one."""

    def activates(self, made: MadeArtifact) -> bool:
        return False


class EvaluatedActivation(ActivationRule):
    """Activate an artifact once an evaluation of it finishes with a ``mean_score`` of at least ``min_mean_score``.

    Values: ``min_mean_score`` is a finite number. An evaluation whose rollouts all failed has no mean score and
    activates nothing.
    """

    def __init__(self, min_mean_score: float) -> None:
        if (
            isinstance(min_mean_score, bool)
            or not isinstance(min_mean_score, (int, float))
            or not math.isfinite(min_mean_score)
        ):
            raise ValueError("min_mean_score is a finite number")
        self.min_mean_score = float(min_mean_score)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        if set(config) != {"min_mean_score"}:
            raise PluginConfigError("the evaluated activation takes exactly min_mean_score")
        try:
            return cls(config["min_mean_score"])
        except ValueError as exc:
            raise PluginConfigError(f"activation.config.{exc}") from exc

    def activates(self, made: MadeArtifact) -> bool:
        return False

    def activates_evaluated(self, artifact_id: str, evaluation: PolicyEvaluation) -> bool:
        score = evaluation.mean_score
        return score is not None and score >= self.min_mean_score


#: The built-in rules by the name an ``activation`` section gives them.
BUILT_IN_RULES: dict[str, type[ActivationRule]] = {
    "always": AlwaysActivate,
    "never": NeverActivate,
    "evaluated": EvaluatedActivation,
}


def activation_reference(value: object) -> PluginReference:
    """The rule that an ``activation`` section names: a built-in name, or a ``package.module:ClassName``, with its
    config. Raise ``PluginConfigError`` for any other value."""
    if isinstance(value, Mapping) and isinstance(value.get("module"), str) and value["module"] in BUILT_IN_RULES:
        unknown = sorted(str(key) for key in value if key not in ("module", "config"))
        if unknown:
            raise PluginConfigError(f"activation has unknown key(s) {', '.join(unknown)}; expected module and config")
        config = value.get("config", {})
        if not isinstance(config, Mapping):
            raise PluginConfigError("activation.config must be a mapping")
        return PluginReference(BUILT_IN_RULES[value["module"]], dict(config))
    if isinstance(value, str) and value in BUILT_IN_RULES:
        return PluginReference(BUILT_IN_RULES[value], {})
    return ActivationRule.reference(value, setting="activation")


def build_activation_rule(reference: PluginReference | None) -> ActivationRule:
    """The rule that ``reference`` names, built from its config; ``AlwaysActivate`` when it is None."""
    if reference is None:
        return AlwaysActivate()
    return ActivationRule.referenced_class(reference).from_config(reference.config)


__all__ = [
    "BUILT_IN_RULES",
    "AlwaysActivate",
    "EvaluatedActivation",
    "NeverActivate",
    "activation_reference",
    "build_activation_rule",
]
