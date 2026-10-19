"""The context that the training pipeline and its three stages read for one scenario.

The trainer builds it. It sits below the stage packages so that each of them
can read it. A stage's own config is not here: each stage parses its config
into its settings dataclass in ``from_config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.storage.tasks import TaskStorage


@dataclass(frozen=True)
class PipelineContext:
    """What the training pipeline and its stages read for one scenario: its report contract, logger, and mode."""

    scenario: str
    report_type: type[ReportBase] | None = None
    experiment_logger: ExperimentLogger = field(default_factory=NullExperimentLogger)
    training_mode: str = "auto"
    #: The contract the scenario's ingress admits, when it is wider than ``report_type``: a report it
    #: accepts and ``report_type`` refuses is another component's, and this pipeline releases it.
    admitted_report_type: type[ReportBase] | None = None
    #: How many versions a sample may lag behind and still train; the recipe's ``max_staleness``.
    max_staleness: int = 0
    #: Opens the generated task stores of the scenario. ``None`` if the trainer has no scenario session.
    task_storage: TaskStorage | None = None
    #: The most unreserved sets the record stage holds, or None for no limit. The pipeline bounds the
    #: held sets in manual mode, where batches follow instructions instead of sets.
    held_set_limit: int | None = None
    #: Whether the pipeline takes training instructions; a recipe whose proposer reads none turns it off.
    manual_enabled: bool = True

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        if not isinstance(self.max_staleness, int) or isinstance(self.max_staleness, bool) or self.max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        if self.held_set_limit is not None and (
            not isinstance(self.held_set_limit, int)
            or isinstance(self.held_set_limit, bool)
            or self.held_set_limit < 0
        ):
            raise ValueError("held_set_limit must be None or a non-negative integer")
