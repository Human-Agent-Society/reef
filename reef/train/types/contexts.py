from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger, NullExperimentLogger


@dataclass(frozen=True)
class ProcessorContext:
    scenario: str
    config: Mapping[str, Any] = field(default_factory=dict)
    report_type: type[ReportBase] | None = None
    experiment_logger: ExperimentLogger = field(default_factory=NullExperimentLogger)
    training_mode: str = "auto"
    max_pending_requests: int = 8

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        if (
            isinstance(self.max_pending_requests, bool)
            or not isinstance(self.max_pending_requests, int)
            or self.max_pending_requests < 1
        ):
            raise ValueError("max_pending_requests must be an integer of at least 1")

    def with_config(self, config: Mapping[str, Any]) -> ProcessorContext:
        return replace(self, config=config)
