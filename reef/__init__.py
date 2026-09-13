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
from reef.core.version import __version__

from reef.core import ReefError, RequestType, AgentRecord, ReportBase, ReportValidationError
from reef.service.wire import ReportPayload, RequestHeaders, parse_request_headers
from reef.storage.records import RecordStore
from reef.train.evaluation import (
    AlwaysSelectMixin,
    BackendAlwaysSelectPlugin,
    BackendEvaluateMixin,
    CandidateEvaluationConfig,
    CandidateEvaluationConfigError,
    CandidateEvaluationPlugin,
    CandidateEvaluationPluginFactory,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    RegressionGateMixin,
    SelectionDecision,
    UpdateCandidate,
    build_candidate_evaluation,
)
from reef.storage.commits import SCENARIO_METADATA_KEY
from reef.recipe.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario import Scenario
from reef.recipe import (
    RecipeConfigError,
    Recipe,
)
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.train import DataProcessor, Trainer
from reef.runtime import ActivatedModel, InferenceRuntime, ModelCandidate, ModelRuntime, TrainingRuntime

__all__ = [
    "SCENARIO_METADATA_KEY",
    "ActivatedModel",
    "AgentRecord",
    "AlwaysSelectMixin",
    "BackendAlwaysSelectPlugin",
    "BackendEvaluateMixin",
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPlugin",
    "CandidateEvaluationPluginFactory",
    "CandidateEvaluator",
    "CandidateSelector",
    "CheckpointStrategy",
    "DataProcessor",
    "Dispatcher",
    "EvaluationResult",
    "EveryNVersions",
    "InferenceRuntime",
    "ModelCandidate",
    "ModelRuntime",
    "PostgresRecordStore",
    "Recipe",
    "RecipeConfigError",
    "RecordStore",
    "ReefError",
    "RegressionGateMixin",
    "ReportBase",
    "ReportPayload",
    "ReportValidationError",
    "RequestHeaders",
    "RequestType",
    "SQLiteRecordStore",
    "Scenario",
    "SelectionDecision",
    "Trainer",
    "TrainingRuntime",
    "UpdateCandidate",
    "__version__",
    "build_candidate_evaluation",
    "build_default_dispatcher",
    "parse_request_headers",
]


def __getattr__(name: str) -> type:
    # Preserve convenient root imports without loading databases for interface users.
    if name == "SQLiteRecordStore":
        from reef.storage.sqlite import SQLiteRecordStore

        return SQLiteRecordStore
    if name == "PostgresRecordStore":
        from reef.storage.postgres import PostgresRecordStore

        return PostgresRecordStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
