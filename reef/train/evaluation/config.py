"""Deployment configuration and loading for external candidate evaluators.

An evaluation section has this shape::

    evaluation:
      module: my_package.evaluation:BenchmarkEvaluator
      config:                              # opaque to Reef
        benchmark: gsm8k
        threshold: 0.8

The reference names a :class:`ConfiguredEvaluationPlugin` subclass. Loading
the section imports and checks the class only. Its
``from_config(config, runtime=..., training_runtime=..., scenario=..., environ=...)``
builds one scenario-local plugin implementing both ``evaluate(candidate)`` and
``decide(candidate, evaluation)``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from reef.core.errors import ReefError
from reef.core.evaluation import CandidateEvaluationPlugin
from reef.core.plugins import Plugin, PluginConfigError, PluginReference
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime


class CandidateEvaluationConfigError(ReefError):
    """A candidate evaluation plugin declaration cannot be loaded or built."""


class ConfiguredEvaluationPlugin(CandidateEvaluationPlugin, Plugin):
    """A candidate evaluation plugin that a weight-training deployment's ``evaluation`` section names."""

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        runtime: InferenceRuntime,
        training_runtime: TrainingRuntime,
        scenario: str,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the plugin for one scenario; allocate model resources here, not when the class is loaded."""


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    """The ``evaluation`` section: a checked plugin class and its opaque configuration."""

    plugin: PluginReference
    environ: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, environ: Mapping[str, str]) -> CandidateEvaluationConfig:
        # Validate the import when the recipe is constructed, not on its first
        # training step. The plugin itself is built for each scenario.
        if "module" not in data:
            raise CandidateEvaluationConfigError("evaluation.module must name a ConfiguredEvaluationPlugin class")
        try:
            plugin = ConfiguredEvaluationPlugin.reference(data, setting="evaluation")
        except PluginConfigError as exc:
            raise CandidateEvaluationConfigError(str(exc)) from exc
        return cls(plugin=plugin, environ=dict(environ))


def build_candidate_evaluation(
    config: CandidateEvaluationConfig,
    *,
    runtime: InferenceRuntime,
    training_runtime: TrainingRuntime,
    scenario: str,
) -> CandidateEvaluationPlugin:
    """Build one scenario's external candidate evaluation plugin."""
    return ConfiguredEvaluationPlugin.referenced_class(config.plugin).from_config(
        config.plugin.config,
        runtime=runtime,
        training_runtime=training_runtime,
        scenario=scenario,
        environ=config.environ,
    )


__all__ = [
    "CandidateEvaluationConfig",
    "CandidateEvaluationConfigError",
    "ConfiguredEvaluationPlugin",
    "build_candidate_evaluation",
]
