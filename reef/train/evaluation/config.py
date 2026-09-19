"""Deployment configuration and loading for external candidate evaluators.

An evaluation section has this shape::

    evaluation:
      module: my_package.evaluation:EvaluationFactory
      config:                              # opaque to Reef
        benchmark: gsm8k
        threshold: 0.8

The dotted reference names a ``CandidateEvaluationPluginFactory`` subclass or
instance. Its ``build(config, runtime=..., training_runtime=..., scenario=..., environ=...)`` returns one
scenario-local object implementing both ``evaluate(candidate)`` and
``decide(candidate, evaluation)``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.errors import ReefError
from reef.core.evaluation import CandidateEvaluationPlugin
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime


class CandidateEvaluationConfigError(ReefError):
    """A candidate evaluation plugin declaration cannot be loaded or built."""


class CandidateEvaluationPluginFactory(ABC):
    """Build one scenario-local candidate evaluation plugin from opaque config."""

    @abstractmethod
    def build(
        self,
        config: Mapping[str, Any],
        *,
        runtime: InferenceRuntime,
        training_runtime: TrainingRuntime,
        scenario: str,
        environ: Mapping[str, str],
    ) -> CandidateEvaluationPlugin: ...


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    """A dotted candidate evaluation plugin factory and its opaque configuration."""

    module: str
    config: Mapping[str, Any] = field(default_factory=dict)
    environ: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module:
            raise CandidateEvaluationConfigError("evaluation.module must be a non-empty dotted reference")
        if not isinstance(self.config, Mapping):
            raise CandidateEvaluationConfigError("evaluation.config must be an object")
        if not isinstance(self.environ, Mapping):
            raise CandidateEvaluationConfigError("candidate evaluation environment must be a mapping")
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(self, "environ", dict(self.environ))
        # Validate imports when the recipe is constructed, not on its first
        # training step. The evaluator still instantiates per scenario below.
        _dotted_factory(self.module, "candidate evaluation plugin factory")

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        environ: Mapping[str, str],
    ) -> CandidateEvaluationConfig:
        unknown = sorted(set(data) - {"module", "config"})
        if unknown:
            raise CandidateEvaluationConfigError(
                f"evaluation contains unknown key(s) {', '.join(map(repr, unknown))}; expected module and config"
            )
        module = data.get("module")
        if not isinstance(module, str) or not module:
            raise CandidateEvaluationConfigError("evaluation.module must be a non-empty dotted reference")
        return cls(
            module=module,
            config=data.get("config", {}),
            environ=dict(environ),
        )


def _dotted_factory(reference: str, what: str) -> CandidateEvaluationPluginFactory:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise CandidateEvaluationConfigError(f"{what} {reference!r} must be 'package.module:factory_name'")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise CandidateEvaluationConfigError(f"cannot import {what} {reference!r}: {exc}") from exc
    if isinstance(factory, type) and issubclass(factory, CandidateEvaluationPluginFactory):
        try:
            factory = factory()
        except TypeError as exc:
            raise CandidateEvaluationConfigError(f"cannot construct {what} {reference!r}: {exc}") from exc
    if not isinstance(factory, CandidateEvaluationPluginFactory):
        raise CandidateEvaluationConfigError(
            f"{what} {reference!r} must name a CandidateEvaluationPluginFactory subclass or instance"
        )
    return factory


def build_candidate_evaluation(
    config: CandidateEvaluationConfig,
    *,
    runtime: InferenceRuntime,
    training_runtime: TrainingRuntime,
    scenario: str,
) -> CandidateEvaluationPlugin:
    """Resolve and build one scenario's external candidate evaluation plugin."""

    factory: CandidateEvaluationPluginFactory = _dotted_factory(config.module, "candidate evaluation plugin factory")
    evaluator = factory.build(
        config.config, runtime=runtime, training_runtime=training_runtime, scenario=scenario, environ=config.environ
    )
    if not isinstance(evaluator, CandidateEvaluationPlugin):
        raise CandidateEvaluationConfigError(
            f"candidate evaluation plugin factory {config.module!r} returned {type(evaluator).__name__}, "
            "which must inherit CandidateEvaluationPlugin and implement evaluate and decide"
        )
    return evaluator


__all__ = [
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPluginFactory",
    "build_candidate_evaluation",
]
