"""Candidate evaluation and the activation rules: contracts, built-ins, and plugin loading.

A harness recipe builds a publication policy from ``evolution.evaluation``, a
:class:`CandidateEvaluationPlugin`. :class:`reef.train.Trainer` runs it between
prepare and settle to decide which candidate tree the search keeps. The activation rule decides whether an
artifact that training makes serves; ``activation`` holds the built-in rules
(``always``, ``never`` and ``evaluated``). The shared contracts live in
``reef.core.evaluation`` and ``reef.core.activation``. This package owns the
built-in evaluators, selectors and rules.

``harness`` measures harness candidates by running evaluation rollouts. It
holds ``HarnessCandidate``, its evaluator ``HarnessTaskEvaluator``, the
``TaskScorer`` contract that a deployment names, the history that is the
score cache, and the ``compare``, ``floor`` and ``none`` publication policies.
``rollout_worker`` runs the rollouts of an artifact evaluation,
``artifact_endpoint`` answers their model calls from the evaluated artifact, and
``trials`` holds the ``TrialObserver`` contract that reads each trial before the
worker removes it (``evaluation.trial_observer``).
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

__all__ = [
    "ActivationRule",
    "AlwaysActivate",
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "EvaluatedActivation",
    "EvaluationResult",
    "MadeArtifact",
    "NeverActivate",
    "SelectionDecision",
    "UpdateCandidate",
]
