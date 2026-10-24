"""Tinker as a training component: each job trains its batch remotely, branched from the job's start checkpoint.

A job starts from ``job.start``, a directory that holds a Tinker checkpoint
manifest, or from the base checkpoint when ``start`` is ``None``. Each attempt
branches from those weights and their optimizer state, so repeating an attempt
can consume API resources but cannot apply its gradient twice. The inference
side, ``reef.inference.tinker``, reads each checkpoint's manifest from its
artifact; nothing is shared in process.
"""

from __future__ import annotations

import fcntl
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any

from reef.core.batches import StepScheduling
from reef.runtime.interfaces import TrainingMethod, learning_rate_metrics
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.runtime.recovery import read_json, write_json
from reef.train.tinker_backend.checkpoint import MANIFEST, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import job_learning_rates, prepare_tinker_step


class TinkerTrainingComponent(TrainingComponent):
    """Train each job as one remote Tinker session from the job's start checkpoint.

    A job's parameters are ``method`` (a ``TrainingMethod`` as a dict),
    ``scheduling`` (a ``StepScheduling`` as a dict), ``algorithm_state``, and
    optionally ``batch_size``, ``kl_coef`` and ``learning_rate``, which override
    the component's ``TinkerConfig``. ``submit`` runs the job to its end and
    raises when it fails, so a job is never reported as failed.

    ``config.state_dir`` holds ``base/`` (the base checkpoint), and for every
    job that trained, ``jobs/<job_id>/`` (its checkpoint manifest) and
    ``jobs/<job_id>.json`` (its metrics and proposed algorithm state).
    """

    def __init__(self, base_model: str, config: TinkerConfig, client: TinkerClient) -> None:
        self._model = base_model
        self._config = config
        self._client = client
        self._lock = RLock()
        self._root = Path(config.state_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._state_lock = (self._root / ".lock").open("a")
        try:
            fcntl.flock(self._state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._state_lock.close()
            raise ValueError("Tinker state_dir is already owned by another runtime") from None
        # Jobs that ended without a checkpoint; they leave no files.
        self._skipped: dict[str, JobStatus] = {}
        self._closed = False
        try:
            base = self._root / "base"
            if (base / MANIFEST).exists():
                self._base = TinkerCheckpoint.read(base)
            else:
                self._base = client.initialize()
                self._base.write(base)
            self._base.validate_model(base_model, config.lora_rank)
        except BaseException:
            self._state_lock.close()
            raise

    @property
    def name(self) -> str:
        return "tinker"

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

    @property
    def state_dir(self) -> Path:
        return self._root

    def _validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.validate_model(self._model, self._config.lora_rank)
        return checkpoint

    def check_start(self, path: Path) -> None:
        if not (path / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        self._validated(TinkerCheckpoint.read(path))

    def submit(self, job: TrainingJob) -> None:
        with self._lock:
            if job.job_id in self._skipped or (self._root / "jobs" / f"{job.job_id}.json").exists():
                return
        defaults: dict[str, Any] = {
            "batch_size": self._config.batch_size,
            "kl_coef": self._config.kl_coef,
            "learning_rate": self._config.learning_rate,
        }
        parameters = {**defaults, **job.parameters}
        method = TrainingMethod.from_dict(parameters["method"])
        prepared = prepare_tinker_step(
            job.data,
            method,
            parameters["algorithm_state"],
            StepScheduling(**parameters["scheduling"]),
            batch_size=parameters["batch_size"],
            kl_coef=parameters["kl_coef"],
        )
        if prepared.payload is None:
            with self._lock:
                self._skipped[job.job_id] = JobStatus(
                    job.job_id, "finished", metrics=prepared.metrics, algorithm_state=prepared.next_algorithm_state
                )
            return
        start = self._base if job.start is None else self._validated(TinkerCheckpoint.read(Path(job.start)))
        batches = [[row_from_payload(row) for row in batch] for batch in prepared.payload["batches"]]
        learning_rates, schedule_state = job_learning_rates(
            start.learning_rate_schedule_state,
            method.learning_rate_schedule,
            len(batches),
            parameters["learning_rate"],
        )
        checkpoint, metrics = self._client.train(
            start, batches, resolve_tinker_loss(prepared.payload["loss"]), learning_rates
        )
        checkpoint = replace(self._validated(checkpoint), learning_rate_schedule_state=schedule_state)
        checkpoint.write(self._root / "jobs" / job.job_id)
        # The result is written last: a job with a result file has a complete checkpoint.
        write_json(
            self._root / "jobs" / f"{job.job_id}.json",
            {
                "steps": len(batches),
                "metrics": {**prepared.metrics, **metrics, **learning_rate_metrics(learning_rates, schedule_state)},
                "algorithm_state": dict(prepared.next_algorithm_state),
            },
        )

    def status(self, job_id: str) -> JobStatus:
        with self._lock:
            skipped = self._skipped.get(job_id)
        if skipped is not None:
            return skipped
        result: Mapping[str, Any] | None = read_json(self._root / "jobs" / f"{job_id}.json")
        if result is None:
            raise KeyError(job_id)
        return JobStatus(
            job_id,
            "finished",
            (SavedCheckpoint(result["steps"], str(self._root / "jobs" / job_id)),),
            metrics=result["metrics"],
            algorithm_state=result["algorithm_state"],
        )

    def cancel(self, job_id: str) -> None:
        """A job runs inside ``submit``, so no job is running when ``cancel`` is called."""
        return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._client.close()
        finally:
            self._state_lock.close()
