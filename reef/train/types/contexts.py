from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger, NullExperimentLogger


@dataclass(frozen=True)
class ProcessorContext:
    scenario: str
    config: Mapping[str, Any] = field(default_factory=dict)
    report_type: type[ReportBase] | None = None
    experiment_logger: ExperimentLogger = field(default_factory=NullExperimentLogger)

    def with_config(self, config: Mapping[str, Any]) -> ProcessorContext:
        return ProcessorContext(self.scenario, config, self.report_type, self.experiment_logger)
