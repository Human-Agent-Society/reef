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
    """What the training pipeline and its stages read for one scenario: its report contract, logger, and mode.

    Values: ``training_mode`` is ``auto``, ``manual`` or ``hybrid``;
    ``default_job_samples`` is None or a positive int; ``Trainer.build`` gives
    None to a trainer that trains nothing. The constructor refuses other values.
    Purpose: ``Trainer.build`` makes it. The pipeline and each stage's
    ``from_config`` read it.
    """

    scenario: str
    report_type: type[ReportBase] | None = None
    experiment_logger: ExperimentLogger = field(default_factory=NullExperimentLogger)
    training_mode: str = "auto"
    #: The contract the scenario's ingress admits, when it is wider than ``report_type``: a report it
    #: accepts and ``report_type`` refuses is another component's, and this pipeline releases it.
    admitted_report_type: type[ReportBase] | None = None
    #: Opens the generated task stores of the scenario. ``None`` if the trainer has no scenario session.
    task_storage: TaskStorage | None = None
    #: The fewest samples of one job when the sample-to-job stage sets no ``job_samples``:
    #: ``evolution.batch_size`` for harness evolution. None otherwise: a weight trainer's stage sets its own
    #: ``job_samples``, and a trainer that trains nothing keeps every sample.
    default_job_samples: int | None = None
    #: Whether the pipeline takes training instructions; a recipe whose proposer reads none turns it off.
    manual_enabled: bool = True

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        if self.default_job_samples is not None and (
            not isinstance(self.default_job_samples, int)
            or isinstance(self.default_job_samples, bool)
            or self.default_job_samples <= 0
        ):
            raise ValueError("default_job_samples must be None or a positive integer")
