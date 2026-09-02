"""Capabilities bound immutably to one scenario."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from reef.artifact import Artifact
from reef.core.reports import ReportBase
from reef.runtime.base import InferenceRuntime
from reef.runtime.inference import InferenceBackend
from reef.surface.base import Surface


class ArtifactValidator(Protocol):
    """Artifact admission checks selected for one scenario."""

    def validate(self, artifact: Artifact) -> None: ...


@dataclass(frozen=True)
class AcceptAnyArtifact:
    """Default admission policy for shape-agnostic scenarios."""

    def validate(self, artifact: Artifact) -> None:
        return None


@dataclass(frozen=True)
class ScenarioBinding:
    """Runtime-facing capabilities selected by a recipe at construction."""

    surface: Surface
    runtime: InferenceRuntime | None
    inference_backend: InferenceBackend | None
    artifact_validator: ArtifactValidator
    #: The report contract selected by the recipe while building its trainer;
    #: when set, every report on this scenario is parsed through it at
    #: ingress. ``None`` keeps open ingress.
    report_type: type[ReportBase] | None = None
