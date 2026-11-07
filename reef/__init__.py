"""Reef: continual learning infra for self-improving agents.

``reef`` holds every shared mechanism
(records, service, runtime, scenario, artifacts and their components, the training
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
from reef.evaluation import (
    ActivationRule,
    AlwaysSelectMixin,
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    MadeArtifact,
    RegressionCheckMixin,
    SelectionDecision,
    UpdateCandidate,
)
from reef.storage.commits import SCENARIO_METADATA_KEY
from reef.scenario import Scenario
from reef.recipe import (
    RecipeConfigError,
    Recipe,
)
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.train import Sample2Job, Trainer, TrainingPipeline
from reef.inference.interfaces import InferenceRuntime
from reef.train.component import TrainingComponent, TrainingJob

__all__ = [
    "SCENARIO_METADATA_KEY",
    "ActivationRule",
    "AgentRecord",
    "AlwaysSelectMixin",
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "Dispatcher",
    "EvaluationResult",
    "InferenceRuntime",
    "MadeArtifact",
    "Recipe",
    "RecipeConfigError",
    "RecordStore",
    "ReefError",
    "RegressionCheckMixin",
    "ReportBase",
    "ReportPayload",
    "ReportValidationError",
    "RequestHeaders",
    "RequestType",
    "Sample2Job",
    "Scenario",
    "SelectionDecision",
    "Trainer",
    "TrainingComponent",
    "TrainingJob",
    "TrainingPipeline",
    "UpdateCandidate",
    "__version__",
    "build_default_dispatcher",
    "parse_request_headers",
]
