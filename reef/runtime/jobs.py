"""Bounded training jobs and the training components that run them.

Reef submits a ``TrainingJob`` to a ``TrainingComponent``. The component runs
its framework's own training loop over the job's data, starting from the job's
start weights, and saves checkpoints. Reef reads the job's ``JobStatus``, turns
each saved checkpoint into an artifact and evaluates it. The job's
``parameters`` belong to the backend: Reef reads none of them, and they
override the backend's own defaults.

``reef.train.job_runtime.JobTrainingRuntime`` maps Reef's step lifecycle onto
one job for each step.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from reef.core.batches import TrainingBatch

JobState = Literal["running", "finished", "failed"]


@dataclass(frozen=True)
class TrainingJob:
    """One bounded training job: a data snapshot, the weights it starts from, and the backend's parameters.

    A value has a non-empty ``job_id``, a non-empty ``backend`` name, ``start``
    (the directory of the start weights as the component reads them, or
    ``None`` for the backend's base model), ``data`` (the batch the job trains)
    and ``parameters``, a mapping that encodes as JSON.

    Reef makes the value and names the job with ``job_id_for``, so a retry of
    the same request has the same ID. A component reads it in ``submit``.
    """

    job_id: str
    backend: str
    start: str | None
    data: TrainingBatch
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("TrainingJob.job_id must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("TrainingJob.backend must be a non-empty string")
        if self.start is not None and (not isinstance(self.start, str) or not self.start):
            raise ValueError("TrainingJob.start must be a non-empty string or None")
        if not isinstance(self.data, TrainingBatch):
            raise TypeError("TrainingJob.data must be a TrainingBatch")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("TrainingJob.parameters must be a mapping")
        json.dumps(dict(self.parameters), allow_nan=False)


def job_id_for(backend: str, start: str | None, data_identity: object, parameters: Mapping[str, Any]) -> str:
    """The ID of a job: a hash of its backend, start, data and parameters, so a retry gets the same ID.

    ``data_identity`` is a JSON value that names the data, not the batch's
    process-local number: a reload numbers the same rows again.
    """
    encoded = json.dumps(
        {"backend": backend, "start": start, "data": data_identity, "parameters": dict(parameters)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SavedCheckpoint:
    """A checkpoint a job saved: the optimizer step that saved it and the directory that holds it.

    ``step`` is a non-negative int, counted within the job; ``path`` is a
    non-empty directory path as the component wrote it.
    """

    step: int
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("SavedCheckpoint.step must be a non-negative int")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("SavedCheckpoint.path must be a non-empty string")


@dataclass(frozen=True)
class JobStatus:
    """What a component reports about one job.

    ``state`` is ``running``, ``finished`` or ``failed``. ``checkpoints`` are
    the checkpoints saved so far, in step order. ``metrics`` is the backend's
    telemetry, which Reef records without reading. ``algorithm_state`` is the
    state the job proposes for Reef to commit, set only when the job finished.
    ``error`` is set exactly when the job failed. A finished job without
    checkpoints trained nothing, for example because its objective skipped the
    batch.
    """

    job_id: str
    state: JobState
    checkpoints: tuple[SavedCheckpoint, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    algorithm_state: Mapping[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.state not in ("running", "finished", "failed"):
            raise ValueError(f"JobStatus.state must be running, finished or failed, got {self.state!r}")
        if (self.error is not None) != (self.state == "failed"):
            raise ValueError("JobStatus.error must be set exactly when the job failed")
        if self.algorithm_state is not None and self.state != "finished":
            raise ValueError("JobStatus.algorithm_state is set only when the job finished")
        if not all(isinstance(checkpoint, SavedCheckpoint) for checkpoint in self.checkpoints):
            raise TypeError("JobStatus.checkpoints must hold SavedCheckpoint values")
        steps = [checkpoint.step for checkpoint in self.checkpoints]
        if steps != sorted(set(steps)):
            raise ValueError("JobStatus.checkpoints must be in increasing step order")


class TrainingComponent(ABC):
    """A training backend that runs whole jobs with its framework's own training loop.

    ``submit`` starts a job; submitting a job ID again does nothing. ``status``
    reports a submitted job, also after the component restarts, as long as its
    files remain. ``cancel`` stops a running job. A component may run a job
    inside ``submit``; it then raises from ``submit`` when the job fails,
    instead of reporting the job as failed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The backend name that jobs for this component carry."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """The samples in one optimizer step, by the backend's own configuration."""

    @abstractmethod
    def check_start(self, path: Path) -> None:
        """Raise ``ValueError`` when a job cannot start from the weights in ``path``."""

    @abstractmethod
    def submit(self, job: TrainingJob) -> None: ...

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Report the job; raise ``KeyError`` for a job this component never saw."""

    @abstractmethod
    def cancel(self, job_id: str) -> None: ...

    def close(self) -> None:
        """Release the component's resources; safe to call repeatedly."""
        return
