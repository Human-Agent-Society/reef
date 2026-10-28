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

A scenario can select one of the built-in rules above for itself, in the mapping form
(``ScenarioSettings.activation``). This rule replaces the deployment's rule for that scenario. The setting comes
from an HTTP client. Thus it cannot name a ``package.module:ClassName``, because a client must not choose the code
that the service imports.
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

    Values: ``review_kinds`` is a frozenset of non-empty node kind names. Operations: ``activates(made)`` is true
    exactly when ``made.changed_kinds`` has no kind in ``review_kinds`` or in ``ALWAYS_REVIEWED_KINDS``;
    ``activates_evaluated`` is false. Purpose: the ``always`` activation section, and the rule when there is no
    section.
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
    """Activate no artifact: a person activates each one.

    Operations: ``activates`` and ``activates_evaluated`` are false. Purpose: the ``never`` activation section.
    """

    def activates(self, made: MadeArtifact) -> bool:
        return False


class EvaluatedActivation(ActivationRule):
    """Activate an artifact once an evaluation of it finishes with a ``mean_score`` of at least ``min_mean_score``.

    Values: ``min_mean_score`` is a finite number. Operations: ``activates`` is false.
    ``activates_evaluated(artifact_id, evaluation)`` is true exactly when ``evaluation.mean_score`` is not None
    and at least ``min_mean_score``. An evaluation whose rollouts all failed has no mean score, so it activates
    nothing. Purpose: the ``evaluated`` activation section.
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
    config.

    The rule is built once from its config, so a bad config fails when the deployment loads. Raise
    ``PluginConfigError`` for any other value, and for a config that the rule refuses.
    """
    if isinstance(value, Mapping) and isinstance(value.get("module"), str) and value["module"] in BUILT_IN_RULES:
        unknown = sorted(str(key) for key in value if key not in ("module", "config"))
        if unknown:
            raise PluginConfigError(f"activation has unknown key(s) {', '.join(unknown)}; expected module and config")
        config = value.get("config", {})
        if not isinstance(config, Mapping):
            raise PluginConfigError("activation.config must be a mapping")
        reference = PluginReference(BUILT_IN_RULES[value["module"]], dict(config))
    elif isinstance(value, str) and value in BUILT_IN_RULES:
        reference = PluginReference(BUILT_IN_RULES[value], {})
    else:
        reference = ActivationRule.reference(value, setting="activation")
    build_activation_rule(reference)
    return reference


def scenario_activation_reference(value: object) -> PluginReference:
    """The built-in rule that a scenario's ``activation`` setting names, with its config.

    ``value`` is a mapping ``{"module": <built-in name>, "config": {...}}``, and ``config`` is optional. Raise
    ``PluginConfigError`` for any other value, also for a ``package.module:ClassName``. Raise it also when the rule
    refuses the config.
    """
    if not isinstance(value, Mapping) or value.get("module") not in BUILT_IN_RULES:
        raise PluginConfigError(f"activation must be an object whose module is one of {', '.join(BUILT_IN_RULES)}")
    return activation_reference(value)


def activation_setting(reference: PluginReference | None) -> dict[str, object]:
    """The JSON form ``{"module", "config"}`` of the rule that ``reference`` names.

    ``module`` is the name of a built-in rule, or the ``package.module:ClassName`` of a rule class of your own.
    A None ``reference`` is ``always``, the default rule.
    """
    if reference is None:
        return {"module": "always", "config": {}}
    plugin_class = ActivationRule.referenced_class(reference)
    names = {rule_class: name for name, rule_class in BUILT_IN_RULES.items()}
    module = names.get(plugin_class, f"{plugin_class.__module__}:{plugin_class.__qualname__}")
    return {"module": module, "config": dict(reference.config)}


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
    "activation_setting",
    "build_activation_rule",
    "scenario_activation_reference",
]
