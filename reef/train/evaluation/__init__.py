"""The recipe's candidate evaluation: contracts, built-ins, and factory loading.

Every :class:`reef.recipe.Recipe` carries a ``candidate_evaluation`` plugin —
built in code by harness recipes, or from the top-level ``evaluation`` config
section by weight recipes — and :class:`reef.train.Trainer` runs it between
prepare and settle to decide whether a produced candidate is published. The
shared contracts live in ``reef.core.evaluation`` and are re-exported here.
Runtimes import those contracts directly; this package owns the built-in
evaluators, selectors, and configuration machinery.
"""

from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.train.evaluation.config import (
    CandidateEvaluationConfig,
    CandidateEvaluationConfigError,
    CandidateEvaluationPluginFactory,
    build_candidate_evaluation,
)
from reef.train.evaluation.evaluators import (
    AlwaysSelectMixin,
    BackendAlwaysSelectPlugin,
    BackendEvaluateMixin,
    RegressionGateMixin,
)

__all__ = [
    "AlwaysSelectMixin",
    "BackendAlwaysSelectPlugin",
    "BackendEvaluateMixin",
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPlugin",
    "CandidateEvaluationPluginFactory",
    "CandidateEvaluator",
    "CandidateSelector",
    "EvaluationResult",
    "RegressionGateMixin",
    "SelectionDecision",
    "UpdateCandidate",
    "build_candidate_evaluation",
]
