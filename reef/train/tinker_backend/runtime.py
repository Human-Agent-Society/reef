"""Tinker's in-process training runtime: one optimizer step per candidate, branched from the incumbent.

The incumbent is the checkpoint Reef last committed, remembered on disk so a
restart branches from the same weights and optimizer state; Reef reports
commits through ``commit_candidate`` and rollbacks through
``restore_checkpoint``. The inference side, ``reef.inference.tinker``, reads
each candidate's manifest from its artifact; nothing is shared in process.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.batches import StepScheduling, TrainingBatch
from reef.core.evaluation import SelectionDecision
from reef.runtime.interfaces import ModelCandidate, PreparedTrainingStep, StaleCandidate, TrainingRuntime
from reef.runtime.recovery import read_json, write_json
from reef.train.tinker_backend.checkpoint import MANIFEST, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import prepare_tinker_step

INCUMBENT = "incumbent.json"


class TinkerTrainingRuntime(TrainingRuntime):
    """Train candidates from the committed incumbent in separate remote sessions.

    Each attempt branches from the incumbent's weights AND optimizer.
    Repeating a failed attempt can consume API resources, but cannot apply its
    gradient twice to the incumbent. Reef's commit log remains authoritative.
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
        self._candidates: dict[str, tuple[ModelCandidate, TinkerCheckpoint]] = {}
        self._closed = False
        try:
            base = self._root / "base"
            if (base / MANIFEST).exists():
                self._base = TinkerCheckpoint.read(base)
            else:
                self._base = client.initialize()
                self._base.write(base)
            self._base.validate_model(base_model, config.lora_rank)
            value = read_json(self._root / INCUMBENT)
            self._incumbent = self._base if value is None else self._validated(TinkerCheckpoint(**value))
        except BaseException:
            self._state_lock.close()
            raise

    @property
    def incumbent(self) -> TinkerCheckpoint:
        with self._lock:
            return self._incumbent

    def _validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.validate_model(self._model, self._config.lora_rank)
        return checkpoint

    def _set_incumbent(self, checkpoint: TinkerCheckpoint) -> None:
        self._incumbent = self._validated(checkpoint)
        write_json(self._root / INCUMBENT, asdict(checkpoint))

    def _candidate_directory(self, identity: str) -> Path:
        return self._root / "candidates" / identity

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        prepared = prepare_tinker_step(
            batch,
            objective,
            algorithm_state,
            scheduling,
            batch_size=self._config.batch_size,
            runtime_load_id=serving_runtime_load_id,
        )
        if prepared.payload is None:
            return prepared
        return PreparedTrainingStep(
            prepared.action,
            prepared.next_algorithm_state,
            prepared.metrics,
            {**prepared.payload, "scenario_step": scenario_step},
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        if payload.get("stale"):
            raise StaleCandidate({"tinker_stale_samples": 1})
        identity = hashlib.sha256(json.dumps(dict(payload), sort_keys=True, allow_nan=False).encode()).hexdigest()
        with self._lock:
            known = self._candidates.get(identity)
            if known is not None:
                return known[0]
            incumbent = self._incumbent
        loss = resolve_tinker_loss(payload["loss"])
        batches = [[row_from_payload(row) for row in batch] for batch in payload["batches"]]
        checkpoint, metrics = self._client.train(incumbent, batches, loss)
        self._validated(checkpoint)
        directory = self._candidate_directory(identity)
        checkpoint.write(directory)
        candidate = ModelCandidate(
            candidate_id=identity,
            training_job_id=identity,
            checkpoint_path=str(directory),
            current_runtime_load_id=payload.get("source_runtime_load_id"),
            training_metrics=dict(metrics),
            metadata={"scenario_step": payload["scenario_step"]},
        )
        with self._lock:
            if self._incumbent != incumbent:
                raise StaleCandidate({"tinker_stale_samples": 1})
            self._candidates[identity] = (candidate, checkpoint)
        return candidate

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        with self._lock:
            if candidate.candidate_id not in self._candidates:
                raise ValueError("unknown Tinker candidate")
            self._candidates.pop(candidate.candidate_id)
            # The incumbent was never trained or swapped. Remote snapshots stay
            # available for the selection record and an external retention policy.

    def commit_candidate(self, training_job_id: str) -> None:
        """Make the committed candidate the incumbent, also after a restart from its manifest."""
        with self._lock:
            known = self._candidates.pop(training_job_id, None)
            if known is not None:
                self._set_incumbent(known[1])
                return
            directory = self._candidate_directory(training_job_id)
            if (directory / MANIFEST).exists():
                self._set_incumbent(TinkerCheckpoint.read(directory))

    @property
    def supports_checkpoint_restore(self) -> bool:
        return True

    def restore_checkpoint(self, artifact: Artifact) -> None:
        """A rollback's target becomes the incumbent, optimizer state included."""
        path = artifact.materialize().local_path
        if path is None:
            raise ValueError("Tinker requires a materialized checkpoint manifest")
        with self._lock:
            if (path / MANIFEST).exists():
                self._set_incumbent(TinkerCheckpoint.read(path))
                return
            contents = {entry.name for entry in path.iterdir()} - {".git", ".gitattributes", "reef-artifact.json"}
            if contents:
                raise ValueError(f"artifact is missing {MANIFEST}")
            self._set_incumbent(self._base)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._client.close()
        finally:
            self._state_lock.close()
