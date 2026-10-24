"""Run Reef's training steps as one job each on a training component.

``JobTrainingRuntime`` is the ``TrainingRuntime`` that Reef's step lifecycle
calls. For each step it submits one ``TrainingJob`` with the step's batch to a
``TrainingComponent`` and waits for it to end. The job starts from the weights
of the last committed job, or of the release a rollback restored; the runtime
keeps that start, so the component keeps no state about commits.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.batches import StepScheduling, TrainingBatch
from reef.runtime.interfaces import (
    ModelCandidate,
    PreparedTrainingStep,
    StaleCandidate,
    TrainingMethod,
    TrainingRuntime,
    TrainingRuntimeError,
)
from reef.runtime.jobs import JobStatus, TrainingComponent, TrainingJob, job_id_for
from reef.runtime.recovery import read_json, write_json

START = "start.json"
#: Files a repository adds to every artifact; an artifact with nothing else holds no weights.
REPOSITORY_FILES = frozenset({".git", ".gitattributes", "reef-artifact.json"})


class JobTrainingRuntime(TrainingRuntime):
    """Train each step as one job on ``component``, starting from the committed weights.

    ``state_dir`` keeps the start of the next job (``start.json``) and the
    weights of restored releases (``restored/``), so a restart starts from the
    same weights. ``poll_interval_seconds`` is the wait between two status
    reads of a running job.
    """

    def __init__(self, component: TrainingComponent, state_dir: Path, *, poll_interval_seconds: float = 1.0) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._component = component
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._poll_interval_seconds = poll_interval_seconds
        self._lock = threading.RLock()
        value = read_json(self._state_dir / START)
        self._start: str | None = None if value is None else value["start"]
        # The checkpoint of every trained job that Reef has not committed or rejected yet.
        self._candidates: dict[str, str] = {}

    @property
    def component(self) -> TrainingComponent:
        return self._component

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def start(self) -> str | None:
        """The weights the next job starts from; ``None`` is the backend's base model."""
        with self._lock:
            return self._start

    @property
    def batch_size(self) -> int:
        return self._component.batch_size

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
        *,
        parent_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        """Run the step's job to its end; a job that saved no checkpoint is a state-only skip."""
        parameters = {
            "method": method.to_dict(),
            "scheduling": asdict(scheduling),
            "algorithm_state": dict(algorithm_state),
        }
        start = self.start
        name = self._component.name
        # The rows name the data; the batch's number is process-local, and a reload numbers the same rows again.
        rows = [json.dumps(asdict(item), default=str, sort_keys=True) for item in batch.items]
        job_id = job_id_for(name, start, rows, parameters)
        self._component.submit(TrainingJob(job_id, name, start, batch, parameters))
        status = self._wait(job_id)
        if status.algorithm_state is None:
            raise TrainingRuntimeError(f"training job {job_id} finished without an algorithm state")
        if not status.checkpoints:
            return PreparedTrainingStep("skip", status.algorithm_state, status.metrics)
        payload: dict[str, Any] = {"job_id": job_id, "start": start, "scenario_step": scenario_step}
        if parent_runtime_load_id is not None:
            payload["source_runtime_load_id"] = parent_runtime_load_id
        return PreparedTrainingStep("train", status.algorithm_state, {}, payload)

    def _wait(self, job_id: str) -> JobStatus:
        while True:
            status = self._component.status(job_id)
            if status.state == "failed":
                raise TrainingRuntimeError(f"training job {job_id} failed: {status.error}")
            if status.state == "finished":
                return status
            time.sleep(self._poll_interval_seconds)

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """The finished job's last checkpoint, unless a commit or a rollback moved the start meanwhile."""
        job_id = payload["job_id"]
        status = self._component.status(job_id)
        if status.state != "finished" or not status.checkpoints:
            raise TrainingRuntimeError(f"training job {job_id} has no saved checkpoint")
        checkpoint = status.checkpoints[-1].path
        with self._lock:
            if self._start != payload["start"]:
                raise StaleCandidate({"job_start_changed": 1})
            self._candidates[job_id] = checkpoint
        return ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=checkpoint,
            current_runtime_load_id=payload.get("source_runtime_load_id"),
            training_metrics=dict(status.metrics),
            metadata={"scenario_step": payload["scenario_step"]},
        )

    def reject_candidate(self, candidate: ModelCandidate) -> None:
        """The start stays: the next job trains from the same weights again."""
        with self._lock:
            if self._candidates.pop(candidate.candidate_id, None) is None:
                raise ValueError(f"unknown training candidate {candidate.candidate_id!r}")

    def commit_candidate(self, training_job_id: str) -> None:
        """The committed job's checkpoint becomes the start, also after a restart; an unknown job changes nothing."""
        with self._lock:
            checkpoint = self._candidates.pop(training_job_id, None)
            if checkpoint is None:
                try:
                    status = self._component.status(training_job_id)
                except KeyError:
                    return
                if not status.checkpoints:
                    return
                checkpoint = status.checkpoints[-1].path
            self._set_start(checkpoint)

    @property
    def supports_checkpoint_restore(self) -> bool:
        return True

    def restore_checkpoint(self, artifact: Artifact) -> None:
        """The restored release becomes the start; a release without weights restores the base model."""
        path = artifact.materialize().local_path
        if path is None:
            raise ValueError("restoring training weights requires a materialized artifact")
        if not {entry.name for entry in path.iterdir()} - REPOSITORY_FILES:
            with self._lock:
                self._set_start(None)
            return
        self._component.check_start(path)
        # The materialized artifact can be discarded later; the start must outlive it.
        restored = self._state_dir / "restored" / artifact.ref.content_id
        if not restored.exists():
            shutil.copytree(path, restored, ignore=shutil.ignore_patterns(*REPOSITORY_FILES))
        with self._lock:
            self._set_start(str(restored))

    def _set_start(self, start: str | None) -> None:
        self._start = start
        write_json(self._state_dir / START, {"start": start})

    def shutdown(self) -> None:
        self._component.close()
