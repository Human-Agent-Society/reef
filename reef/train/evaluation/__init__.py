"""The recipe's candidate evaluation: contracts, built-ins, and factory loading.

Every :class:`reef.recipe.Recipe` carries a ``candidate_evaluation`` plugin —
built in code by harness recipes, or from the top-level ``evaluation`` config
section by weight recipes — and :class:`reef.train.Trainer` runs it between
prepare and settle to decide whether a produced candidate is published. The
types live here rather than under ``reef.recipe`` only because the trainer,
backends, and runtime bind to them as well.
"""

from reef.train.evaluation.config import (
    CandidateEvaluationConfig,
    CandidateEvaluationConfigError,
    CandidateEvaluationPluginFactory,
    build_candidate_evaluation,
)
from reef.train.evaluation.contracts import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.train.evaluation.evaluators import AlwaysSelect, DefaultCandidateEvaluationPlugin

__all__ = [
    "AlwaysSelect",
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPlugin",
    "CandidateEvaluationPluginFactory",
    "CandidateEvaluator",
    "CandidateSelector",
    "DefaultCandidateEvaluationPlugin",
    "EvaluationResult",
    "SelectionDecision",
    "UpdateCandidate",
    "build_candidate_evaluation",
]
