"""Candidate evaluation and the activation rules: contracts, built-ins, and plugin loading.

A harness recipe builds a :class:`SelectionPlugin` from ``evolution.selection``,
and :class:`reef.train.Trainer` runs it between prepare and settle to decide
which candidate tree the search keeps. Whether an artifact that training makes
serves is the activation rule's decision (``activation``): the always, never
and evaluated rules, which the ``activation`` section names. The shared
contracts live in ``reef.core.evaluation`` and ``reef.core.activation``;
this package owns the built-in evaluators, selectors and rules.

``harness`` measures harness candidates by running evaluation rollouts. It
holds ``HarnessCandidate``, its evaluator ``HarnessRolloutEvaluator``, the
``RolloutScorer`` contract that a method supplies, and the ``score_comparison``
and ``floor`` selection policies. ``artifact_endpoint`` answers the model calls
of an artifact evaluation's rollouts from the release that it evaluates.
"""

from reef.core.activation import ActivationRule, MadeArtifact
from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.evaluation.activation import AlwaysActivate, EvaluatedActivation, NeverActivate
from reef.evaluation.evaluators import (
    AlwaysSelectMixin,
    BackendAlwaysSelectPlugin,
    BackendEvaluateMixin,
    RegressionCheckMixin,
    SelectionPlugin,
)

__all__ = [
    "ActivationRule",
    "AlwaysActivate",
    "AlwaysSelectMixin",
    "BackendAlwaysSelectPlugin",
    "BackendEvaluateMixin",
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "EvaluatedActivation",
    "EvaluationResult",
    "MadeArtifact",
    "NeverActivate",
    "RegressionCheckMixin",
    "SelectionDecision",
    "SelectionPlugin",
    "UpdateCandidate",
]
