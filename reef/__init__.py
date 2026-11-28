"""Reef: continual learning infra for self-improving agents.

``reef`` holds every shared mechanism
(records and their storage, the services and their deployment, scenario control,
inference, artifacts and their components, the training
loop with the candidate evaluation a recipe configures, and the recipe
contract a method implements). Learning methods are separate packages selected
through dotted class references; importing ``reef`` never imports method code.
The repository's sibling ``recipes`` tree is a cookbook and is not part of the
installed Reef package.
"""

from importlib import import_module

from reef.core.version import __version__

export_modules = {
    "ActivationRule": "reef.evaluation",
    "AgentRecord": "reef.core",
    "CandidateEvaluationPlugin": "reef.evaluation",
    "CandidateEvaluator": "reef.evaluation",
    "CandidateSelector": "reef.evaluation",
    "Dispatcher": "reef.control.dispatcher",
    "EvaluationResult": "reef.evaluation",
    "InferenceRuntime": "reef.inference.interfaces",
    "MadeArtifact": "reef.evaluation",
    "Recipe": "reef.recipe",
    "RecipeConfigError": "reef.recipe",
    "RecordStore": "reef.storage.scenario.records",
    "ReefError": "reef.core",
    "ReportBase": "reef.core",
    "ReportPayload": "reef.api.wire",
    "ReportValidationError": "reef.core",
    "RequestHeaders": "reef.api.wire",
    "RequestType": "reef.core",
    "SCENARIO_METADATA_KEY": "reef.storage.scenario.commits",
    "Sample2Job": "reef.train",
    "Scenario": "reef.control.scenario",
    "SelectionDecision": "reef.evaluation",
    "Trainer": "reef.train",
    "TrainingComponent": "reef.train.component",
    "TrainingJob": "reef.train.component",
    "TrainingPipeline": "reef.train",
    "UpdateCandidate": "reef.evaluation",
    "build_default_dispatcher": "reef.control.dispatcher",
    "parse_request_headers": "reef.api.wire",
}


def __getattr__(name: str) -> object:
    """Resolve a public name on first access and retain the original object."""
    if name not in export_modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = vars(import_module(export_modules[name]))[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


__all__ = [*export_modules, "__version__"]
