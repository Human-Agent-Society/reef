"""Candidate evaluation and the activation rules: contracts, built-ins, and plugin loading.

A harness recipe builds a publication policy from ``evolution.evaluation``, a
:class:`CandidateEvaluationPlugin`. :class:`reef.train.Trainer` runs it between
prepare and settle to decide which candidate tree the search keeps. The activation rule decides whether an
artifact that training makes serves; ``activation`` holds the built-in rules
(``always``, ``never`` and ``evaluated``). The shared contracts live in
``reef.core.evaluation`` and ``reef.core.activation``. This package owns the
built-in evaluators, selectors and rules.

``harness`` measures harness candidates by running evaluation rollouts. It
holds ``HarnessCandidate``, its evaluator ``HarnessRolloutEvaluator``, the
``RolloutScorer`` contract that a deployment names, the history that is the
score cache, and the ``compare``, ``floor`` and ``none`` publication policies.
``artifact_endpoint`` answers the model calls of an artifact evaluation's
rollouts from the release that it evaluates.
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
    "UpdateCandidate",
]
