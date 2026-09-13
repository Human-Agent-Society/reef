"""Snapshot-based training, selection, and publication for Tinker."""

from __future__ import annotations

import fcntl
import hashlib
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from reef.artifact.artifact import Artifact, LiveWeightArtifactRef
from reef.core.batches import TrainingBatch
from reef.core.evaluation import SelectionDecision
from reef.runtime.base import PreparedTrainingStep, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate, StaleCandidate
from reef.runtime.inference import InferenceBackend
from reef.surface.base import PublishedWeightRuntime
from reef.train.tinker_backend.checkpoint import MANIFEST, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import prepare_tinker_step


class TinkerRuntime(TrainingRuntime, PublishedWeightRuntime):
    """One scenario, with immutable remote snapshots and a local publication gate.

    Each training attempt branches from the incumbent's weights AND optimizer.
    Repeating a failed attempt can consume API resources, but cannot apply its
    gradient twice to the incumbent. Reef's commit log remains authoritative.
    """

    def __init__(self, base_model: str, config: TinkerConfig, client: TinkerClient) -> None:
        super().__init__(base_url="https://tinker.thinkingmachines.ai", inference_timeout_s=config.inference_timeout_s)
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
        self._incarnation = uuid.uuid4().hex
        self._snapshots: dict[str, tuple[TinkerCheckpoint, str]] = {}
        self._loads: dict[str, TinkerCheckpoint] = {}
        self._releases: dict[str, tuple[TinkerCheckpoint, str]] = {}
        self._active_release: str | None = None
        self._candidates: dict[str, ModelCandidate] = {}
        self._pending: ModelCandidate | None = None
        self._closed = False
        try:
            base = self._root / "base"
            if (base / MANIFEST).exists():
                self._base = TinkerCheckpoint.read(base)
                self._base.validate_model(base_model, config.lora_rank)
            else:
                self._base = client.initialize()
                self._base.validate_model(base_model, config.lora_rank)
                self._base.write(base)
            self._active, self._version = self._remember(self._base)
            from reef.train.tinker_backend.inference import TinkerInferenceBackend

            self._inference = TinkerInferenceBackend(self, client, base_model)
        except BaseException:
            self._state_lock.close()
            raise

    @property
    def inference_backend(self) -> InferenceBackend:
        return self._inference

    def serving_runtime_load_id(self) -> str:
        return self._version

    def _remember(self, checkpoint: TinkerCheckpoint) -> tuple[TinkerCheckpoint, str]:
        checkpoint.validate_model(self._model, self._config.lora_rank)
        existing = self._snapshots.get(checkpoint.sampler_path)
        if existing is not None:
            if existing[0] != checkpoint:
                raise ValueError("Tinker sampler path cannot identify different training checkpoints")
            return existing
        value = self._new_load(checkpoint)
        self._snapshots[checkpoint.sampler_path] = value
        return value

    def _new_load(self, checkpoint: TinkerCheckpoint) -> tuple[TinkerCheckpoint, str]:
        version = f"{self._incarnation}:{len(self._loads)}"
        self._loads[version] = checkpoint
        return checkpoint, version

    def snapshot(self, artifact: Artifact) -> tuple[TinkerCheckpoint, str]:
        """Resolve exactly the frozen artifact; old in-flight calls keep their snapshot."""
        with self._lock:
            if isinstance(artifact.ref, LiveWeightArtifactRef):
                version = artifact.ref.runtime_load_id
                if version in self._loads:
                    return self._loads[version], version
                raise ValueError("Tinker live snapshot belongs to an unavailable serving incarnation")
            if artifact.ref.release_id in self._releases:
                return self._releases[artifact.ref.release_id]
            materialized = artifact.materialize()
            path = materialized.local_path
            if path is None:
                raise ValueError("Tinker requires a materialized checkpoint manifest")
            if (path / MANIFEST).exists():
                selected = self._remember(TinkerCheckpoint.read(path))
                self._releases[artifact.ref.release_id] = selected
                return selected
            # Reef's empty base tree represents the initial seeded adapter.
            # An arbitrary uploaded weight directory must not silently become base.
            contents = {entry.name for entry in path.iterdir()} - {".git", ".gitattributes", "reef-artifact.json"}
            if contents:
                raise ValueError("artifact is missing tinker-checkpoint.json")
            selected = self._remember(self._base)
            self._releases[artifact.ref.release_id] = selected
            return selected

    def restore_checkpoint(self, artifact: Artifact) -> str:
        # Rollback validates the target first; activate_checkpoint binds the
        # republished artifact. No mutable remote serving slot needs restoring.
        return self.snapshot(artifact)[1]

    def activate_checkpoint(self, artifact: Artifact) -> str:
        with self._lock:
            checkpoint, version = self.snapshot(artifact)
            if self._pending is not None:
                pending = TinkerCheckpoint.read(Path(self._pending.checkpoint_path))
                if pending != checkpoint:
                    # Reload after a failed publication restores the durable head.
                    self._pending = None
            if self._pending is None and self._active_release != artifact.ref.release_id:
                # A rollback is a new serving update even if its immutable
                # sampler was served earlier in this incarnation.
                checkpoint, version = self._new_load(checkpoint)
            self._active, self._version = checkpoint, version
            self._active_release = artifact.ref.release_id
            self._releases[artifact.ref.release_id] = (checkpoint, version)
            if self._pending is None:
                self._inference_admission.open()
            return version

    def prepare_training_step(
        self, batch: TrainingBatch, step_preparer: str, algorithm_state: Mapping[str, Any], scenario_step: int
    ) -> PreparedTrainingStep:
        return prepare_tinker_step(
            batch,
            step_preparer,
            algorithm_state,
            scenario_step,
            runtime_load_id=self._version,
            batch_size=self._config.batch_size,
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("Tinker is waiting for the previous candidate's Reef commit")
            if payload["source_runtime_load_id"] != self._version or payload.get("stale"):
                raise StaleCandidate({"tinker_stale_samples": 1})
            incumbent, version = self._active, self._version
            identity = hashlib.sha256(json.dumps(dict(payload), sort_keys=True, allow_nan=False).encode()).hexdigest()
            if identity in self._candidates:
                return self._candidates[identity]
        loss = resolve_tinker_loss(payload["loss"])
        batches = [[row_from_payload(row) for row in batch] for batch in payload["batches"]]
        checkpoint, metrics = self._client.train(incumbent, batches, loss)
        checkpoint.validate_model(self._model, self._config.lora_rank)
        directory = self._root / "candidates" / identity
        checkpoint.write(directory)
        candidate = ModelCandidate(
            candidate_id=identity,
            training_job_id=identity,
            checkpoint_path=str(directory),
            current_runtime_load_id=version,
            training_metrics=dict(metrics),
            metadata={"scenario_step": payload["scenario_step"]},
        )
        with self._lock:
            if self._active != incumbent or self._version != version:
                raise StaleCandidate({"tinker_stale_samples": 1})
            self._remember(checkpoint)
            self._candidates[identity] = candidate
        return candidate

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        with self._lock:
            if self._candidates.get(candidate.candidate_id) != candidate:
                raise ValueError("unknown Tinker candidate")
            if self._pending == candidate:
                return ActivatedModel(candidate.candidate_id, self._version)
            if candidate.current_runtime_load_id != self._version:
                raise StaleCandidate
            checkpoint = TinkerCheckpoint.read(Path(candidate.checkpoint_path))
            self._inference_admission.close()
            self._active, self._version = self._remember(checkpoint)
            self._pending = candidate
            return ActivatedModel(candidate.candidate_id, self._version)

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("cannot reject an already activated Tinker candidate")
            if self._candidates.get(candidate.candidate_id) != candidate:
                raise ValueError("unknown Tinker candidate")
            self._candidates.pop(candidate.candidate_id)
            # The incumbent was never trained or swapped. Remote snapshots stay
            # available for the selection record and an external retention policy.

    def reconcile_training_job(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
        scenario: str | None = None,
    ) -> None:
        with self._lock:
            if self._pending is None:
                return
            expected_step = self._pending.metadata["scenario_step"] + 1
            if committed_training_job_id == self._pending.training_job_id and scenario_step >= expected_step:
                self._pending = None
                self._inference_admission.open()
            elif scenario_step >= expected_step:
                raise RuntimeError("Tinker candidate does not match Reef's committed training job")

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._inference_admission.close()
            try:
                self._client.close()
            finally:
                self._state_lock.close()
