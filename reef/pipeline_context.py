"""The context that the training pipeline and its two stages read for one scenario.

The trainer builds it. It sits below the stage packages so that each of them
can read it. A stage's own config is not here: each stage parses its config
into its settings dataclass in ``from_config``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.storage.samples import SAMPLE_KINDS, SampleKind, TrainerSamples
from reef.storage.tasks import TaskStorage


@dataclass(frozen=True)
class PipelineContext:
    """What the training pipeline and its stages read for one scenario: its report contract, logger, and mode.

    Values: ``training_mode`` is ``auto`` or ``manual``;
    ``default_job_samples`` is None or a positive int; ``Trainer.build`` gives
    None to a trainer that trains nothing. ``job_parameters`` maps a subset of
    the sample kinds to mappings that encode as JSON; the value keeps its own
    copies. The constructor refuses other values.
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
    #: The training samples of the scenario as this trainer reads them, such as its uploads, its imported tasks and
    #: the samples that its record-to-sample stage makes. ``None`` if the trainer has no scenario session: the pipeline then
    #: keeps its samples in memory.
    samples: TrainerSamples | None = None
    #: The fewest samples of one job when the sample-to-job stage sets no ``job_samples``:
    #: ``evolution.batch_size`` for harness evolution. None otherwise: a weight trainer's stage sets its own
    #: ``job_samples``, and a trainer that trains nothing keeps every sample.
    default_job_samples: int | None = None
    #: The parameters of each training job, by the kind of the job's samples (``trajectory`` or ``task``), each a
    #: mapping that encodes as JSON; the recipe gives them, with its ``method``. Empty for a trainer without a job
    #: loop. ``Sample2Job.job_parameters`` reads them.
    job_parameters: Mapping[SampleKind, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        if self.default_job_samples is not None and (
            not isinstance(self.default_job_samples, int)
            or isinstance(self.default_job_samples, bool)
            or self.default_job_samples <= 0
        ):
            raise ValueError("default_job_samples must be None or a positive integer")
        if not isinstance(self.job_parameters, Mapping) or set(self.job_parameters) - set(SAMPLE_KINDS):
            raise ValueError(f"job_parameters maps sample kinds {list(SAMPLE_KINDS)} to parameters")
        for parameters in self.job_parameters.values():
            if not isinstance(parameters, Mapping):
                raise TypeError("the job parameters of a sample kind must be a mapping")
            json.dumps(dict(parameters), allow_nan=False)
        object.__setattr__(
            self, "job_parameters", {kind: dict(parameters) for kind, parameters in self.job_parameters.items()}
        )
