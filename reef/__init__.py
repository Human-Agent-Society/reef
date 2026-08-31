"""Reef: continual learning infra for self-improving agents.

Laid out like stable-baselines3: ``reef`` holds every shared mechanism
(records, service, runtime, scenario, artifacts, surfaces, the training
loop with the candidate evaluation a recipe configures, and the recipe
contract a method implements); each bundled
method is one package under the sibling ``recipes`` tree —
``recipes.sao``, ``recipes.tttd``, ``recipes.openclawrl``,
``recipes.harness_evolve`` — holding its recipe, processor, preparer, (for
weight methods) a ``slime/`` subpackage that only the training process
imports, and its runnable ``examples/``. Importing ``reef`` imports the
methods, which registers their kinds.
"""

# isort: skip_file
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

# The bundled methods, imported last: they depend on everything above and
# register their recipe kinds and step preparers on import.
from recipes.harness_evolve import HarnessEvolveRecipe
from recipes.openclawrl import OpenClawRLRecipe
from recipes.sao import SAORecipe
from recipes.tttd import TTTDRecipe

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
    "HarnessEvolveRecipe",
    "InferenceRuntime",
    "ModelCandidate",
    "OpenClawRLRecipe",
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
    "SAORecipe",
    "Scenario",
    "ScenarioRecipeConflict",
    "ScenarioRecipeError",
    "SelectionDecision",
    "TTTDRecipe",
    "Trainer",
    "TrainingRuntime",
    "UnknownScenarioRecipe",
    "UpdateCandidate",
    "build_candidate_evaluation",
    "build_default_dispatcher",
    "parse_request_headers",
]
