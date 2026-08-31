"""Deployment configuration and loading for external candidate evaluators.

An evaluation section has this shape::

    evaluation:
      module: my_package.evaluation:build
      config:                              # opaque to Reef
        benchmark: gsm8k
        threshold: 0.8

The dotted reference names a factory called as
``factory(config, runtime=..., scenario=..., environ=...)``. It returns one
scenario-local object implementing both ``evaluate(candidate)`` and
``decide(candidate, evaluation)``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from reef.core.errors import ReefError
from reef.runtime.base import TrainingRuntime
from reef.train.evaluation.contracts import CandidateEvaluationPlugin


class CandidateEvaluationConfigError(ReefError):
    """A candidate evaluation plugin declaration cannot be loaded or built."""


class CandidateEvaluationPluginFactory(Protocol):
    """Build one scenario-local candidate evaluation plugin from opaque config."""

    def __call__(
        self,
        config: Mapping[str, Any],
        *,
        runtime: TrainingRuntime,
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


def _dotted_factory(reference: str, what: str) -> Any:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise CandidateEvaluationConfigError(f"{what} {reference!r} must be 'package.module:factory_name'")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise CandidateEvaluationConfigError(f"cannot import {what} {reference!r}: {exc}") from exc
    if not callable(factory):
        raise CandidateEvaluationConfigError(f"{what} {reference!r} is not callable")
    return factory


def build_candidate_evaluation(
    config: CandidateEvaluationConfig,
    *,
    runtime: TrainingRuntime,
    scenario: str,
) -> CandidateEvaluationPlugin:
    """Resolve and build one scenario's external candidate evaluation plugin."""

    factory: CandidateEvaluationPluginFactory = _dotted_factory(config.module, "candidate evaluation plugin factory")
    evaluator = factory(config.config, runtime=runtime, scenario=scenario, environ=config.environ)
    if not callable(getattr(evaluator, "evaluate", None)):
        raise CandidateEvaluationConfigError(
            f"candidate evaluation plugin factory {config.module!r} returned {type(evaluator).__name__}, "
            "which does not provide evaluate(candidate)"
        )
    if not callable(getattr(evaluator, "decide", None)):
        raise CandidateEvaluationConfigError(
            f"candidate evaluation plugin factory {config.module!r} returned {type(evaluator).__name__}, "
            "which does not provide decide(candidate, evaluation)"
        )
    return evaluator


__all__ = [
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "CandidateEvaluationPluginFactory",
    "build_candidate_evaluation",
]
