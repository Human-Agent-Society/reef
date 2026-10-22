"""Capabilities bound immutably to one scenario."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, TrainingRuntime
from reef.surface.base import Surface


@dataclass(frozen=True)
class ScenarioBinding:
    """Runtime-facing capabilities selected by a recipe at construction.

    Values: ``surface`` is a ``Surface``; ``runtime``,
    ``inference_handler`` and ``training_runtime`` are an instance of their
    interface or ``None``; ``report_type`` is a ``ReportBase`` subclass or
    ``None``; and ``refuses_image_content`` is a bool. Construction raises
    ``TypeError`` for any value outside these domains. There are no
    additional conditions between fields.
    Purpose: ``ScenarioFactory`` makes one binding for each scenario from the
    scenario's recipe (``refuses_image_content`` is the recipe's
    ``refuses_image_content``). ``Scenario`` holds it. The service, record
    ingress and training read the fields through the scenario's properties.
    """

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
    #: Whether the service refuses recorded inference requests, and record ingress refuses records, with image
    #: content (the recipe's ``refuses_image_content``).
    refuses_image_content: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.surface, Surface):
            raise TypeError("surface must be a Surface")
        if self.runtime is not None and not isinstance(self.runtime, InferenceRuntime):
            raise TypeError("runtime must be an InferenceRuntime or None")
        if self.inference_handler is not None and not isinstance(self.inference_handler, InferenceHandler):
            raise TypeError("inference_handler must be an InferenceHandler or None")
        if self.training_runtime is not None and not isinstance(self.training_runtime, TrainingRuntime):
            raise TypeError("training_runtime must be a TrainingRuntime or None")
        if self.report_type is not None and (
            not isinstance(self.report_type, type) or not issubclass(self.report_type, ReportBase)
        ):
            raise TypeError("report_type must be a ReportBase subclass or None")
        if not isinstance(self.refuses_image_content, bool):
            raise TypeError("refuses_image_content must be a bool")
