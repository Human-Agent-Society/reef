"""The recipe's candidate evaluation: contracts, built-ins, and plugin loading.

Every :class:`reef.recipe.Recipe` carries a ``candidate_evaluation`` plugin —
a :class:`SelectionPlugin` that harness recipes build from ``evolution.selection``,
or a :class:`ConfiguredEvaluationPlugin` that weight recipes build from the
top-level ``evaluation`` config section — and :class:`reef.train.Trainer` runs it between
prepare and settle to decide whether a produced candidate is published. The
shared contracts live in ``reef.core.evaluation`` and are re-exported here.
Runtimes import those contracts directly; this package owns the built-in
evaluators, selectors, and configuration machinery.

``harness`` measures harness candidates by running evaluation episodes. It
holds ``HarnessCandidate``, its evaluator ``HarnessEpisodeEvaluator``, the
``EpisodeScorer`` contract that a method supplies, and the ``score_comparison``
and ``floor`` selection policies.
"""

from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.evaluation.config import (
    CandidateEvaluationConfig,
    CandidateEvaluationConfigError,
    ConfiguredEvaluationPlugin,
    build_candidate_evaluation,
)
from reef.evaluation.evaluators import (
    AlwaysSelectMixin,
    BackendAlwaysSelectPlugin,
    BackendEvaluateMixin,
    RegressionCheckMixin,
    SelectionPlugin,
)

# Compatibility aliases for existing imports.
RegressionGateMixin = RegressionCheckMixin

__all__ = [
    "AlwaysSelectMixin",
    "BackendAlwaysSelectPlugin",
    "BackendEvaluateMixin",
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "ConfiguredEvaluationPlugin",
    "EvaluationResult",
    "RegressionCheckMixin",
    "RegressionGateMixin",
    "SelectionDecision",
    "SelectionPlugin",
    "UpdateCandidate",
    "build_candidate_evaluation",
]
