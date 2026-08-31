"""Reef: continual learning infra for self-improving agents.

``reef`` holds every shared mechanism
(records, service, runtime, scenario, artifacts, surfaces, the training
loop with the candidate evaluation a recipe configures, and the recipe
contract a method implements). Learning methods are separate packages selected
through dotted class references; importing ``reef`` never imports method code.
The repository's sibling ``recipes`` tree is a cookbook and is not part of the
installed Reef package.
"""

# isort: skip_file
# Written by setuptools-scm at install time from the git tag (see
# [tool.setuptools_scm] in pyproject.toml); absent only on an uninstalled
# checkout imported straight off the tree.
try:
    from reef._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from reef.core import ReefError, RequestType, AgentRecord, ReportBase, ReportValidationError
from reef.service.wire import ReportPayload, RequestHeaders, parse_request_headers
from reef.records import RecordStore
from reef.train.evaluation import (
    AlwaysSelect,
    CandidateEvaluationConfig,
    CandidateEvaluationConfigError,
    CandidateEvaluationPlugin,
    CandidateEvaluationPluginFactory,
    CandidateEvaluator,
    CandidateSelector,
    DefaultCandidateEvaluationPlugin,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
    build_candidate_evaluation,
)
from reef.scenario import (
    SCENARIO_SNAPSHOT_METADATA_KEY,
    CheckpointStrategy,
    EveryNVersions,
    Scenario,
)
from reef.recipe import (
    RecipeConfigError,
    RecipeRegistry,
    ScenarioRecipeConflict,
    ScenarioRecipeError,
    Recipe,
    UnknownScenarioRecipe,
)
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.train import DataProcessor, Trainer
from reef.runtime import ActivatedModel, InferenceRuntime, ModelCandidate, TrainingRuntime

__all__ = [
    "SCENARIO_SNAPSHOT_METADATA_KEY",
    "ActivatedModel",
    "AgentRecord",
    "AlwaysSelect",
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPlugin",
    "CandidateEvaluationPluginFactory",
    "CandidateEvaluator",
    "CandidateSelector",
    "CheckpointStrategy",
    "DataProcessor",
    "DefaultCandidateEvaluationPlugin",
    "Dispatcher",
    "EvaluationResult",
    "EveryNVersions",
    "InferenceRuntime",
    "ModelCandidate",
    "Recipe",
    "RecipeConfigError",
    "RecipeRegistry",
    "RecordStore",
    "ReefError",
    "ReportBase",
    "ReportPayload",
    "ReportValidationError",
    "RequestHeaders",
    "RequestType",
    "Scenario",
    "ScenarioRecipeConflict",
    "ScenarioRecipeError",
    "SelectionDecision",
    "Trainer",
    "TrainingRuntime",
    "UnknownScenarioRecipe",
    "UpdateCandidate",
    "build_candidate_evaluation",
    "build_default_dispatcher",
    "parse_request_headers",
]
