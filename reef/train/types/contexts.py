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

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")

    def with_config(self, config: Mapping[str, Any]) -> ProcessorContext:
        return replace(self, config=config)
