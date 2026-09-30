"""Capabilities bound immutably to one scenario."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, TrainingRuntime
from reef.surface.base import Surface


@dataclass(frozen=True)
class ScenarioBinding:
    """Runtime-facing capabilities selected by a recipe at construction."""

    #: The surface also carries artifact admission: each component's validator
    #: runs before that component is published or restored.
    surface: Surface
    runtime: InferenceRuntime | None
    inference_handler: InferenceHandler | None
    #: The report contract selected by the recipe while building its trainer;
    #: when set, every report on this scenario is parsed through it at
    #: ingress. ``None`` keeps open ingress.
    report_type: type[ReportBase] | None = None
    training_runtime: TrainingRuntime | None = None
