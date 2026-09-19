"""Tinker as a native training backend for Reef's coordinator.

This is the shape a hosted trainer takes when a local engine serves the
model: Reef's coordinator owns the job marker, staleness admission, the
publication barrier and adapter residency; this backend trains one
optimizer step on Tinker per job and materializes the result as a PEFT
adapter directory under its checkpoint. It delivers adapters as files
through ``adapter_files``; Reef's receiver contract loads them, so the
trainer never holds an engine handle and never sends tensors.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reef.core.batches import StepScheduling, TrainingBatch
from reef.runtime.interfaces import (
    PreparedTrainingJob,
    PreparedTrainingStep,
    TrainingBackend,
    TrainingCheckpoint,
    TrainingContext,
    TrainingCoordinationConfig,
    TrainingJobResult,
    TrainingMetrics,
)
from reef.runtime.recovery import ScenarioHistory, history_path, read_json, write_json
from reef.train.algos.registry import register_loss_family_ref
from reef.train.tinker_backend.checkpoint import MANIFEST, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import BACKEND, TinkerLoss, TokenRow, resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import prepare_tinker_step

#: Where a checkpoint directory keeps the PEFT adapter the engines load.
ADAPTER_DIR = "adapter"


class TinkerTrainingBackend(TrainingBackend):
    """One remote optimizer step per job, branched from the scenario's published checkpoint.

    Each scenario's incumbent is the Tinker checkpoint its last publication
    loaded into the engines, remembered on disk so a restart branches from
    the same weights and optimizer state. A rejected job leaves it unchanged.
    """

    def __init__(
        self,
        base_model: str,
        config: TinkerConfig,
        *,
        loss_family: str | None = None,
        loss_reference: str | None = None,
        api_key: str | None = None,
        client: TinkerClient | None = None,
        start_rollout_id: int = 0,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("Tinker needs an API key or an injected client")
        self._model = base_model
        self._config = config
        self._loss_family = loss_family
        self._loss_reference = loss_reference
        self._api_key = api_key
        self._client: TinkerClient | None = client
        self._base: TinkerCheckpoint | None = None
        self._root = Path(config.state_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._template = str(self._root / "checkpoints" / "rollout_{rollout_id}")
        self._coordination = TrainingCoordinationConfig(
            self._template, lora=True, adapter_capacity=config.max_loaded_adapters
        )
        self._context = TrainingContext(
            next_rollout_id=start_rollout_id, history=ScenarioHistory(history_path(self._template))
        )
        self._active_scenario: str | None = None

    # -- Coordinator contract

    @property
    def config(self) -> TrainingCoordinationConfig:
        return self._coordination

    @property
    def context(self) -> TrainingContext:
        return self._context

    def start(self) -> None:
        """Open the SDK session here, in the coordinator's process, and seed the base checkpoint once.

        The recipe registered its loss family in the driver's process; this
        process learns the same dotted reference, and resolving it imports the
        recipe package here, which registers the training objective the coordinator
        will be asked for. An unknown family fails now, before any job.
        """
        if self._loss_family:
            if self._loss_reference:
                register_loss_family_ref(self._loss_family, self._loss_reference, backend=BACKEND)
            resolve_tinker_loss(self._loss_family)
        if self._client is None:
            from reef.train.tinker_backend.client import TinkerSDKClient

            self._client = TinkerSDKClient(self._model, self._config, self._api_key or "")
        base = self._root / "base"
        if (base / MANIFEST).exists():
            self._base = TinkerCheckpoint.read(base)
        else:
            self._base = self._client.initialize()
            self._base.write(base)
        self._base.validate_model(self._model, self._config.lora_rank)

    def check_health(self) -> None:
        return

    @property
    def client(self) -> TinkerClient:
        if self._client is None:
            raise RuntimeError("the Tinker backend opens its SDK session in start()")
        return self._client

    @property
    def base(self) -> TinkerCheckpoint:
        if self._base is None:
            raise RuntimeError("the Tinker backend seeds its base checkpoint in start()")
        return self._base

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
    ) -> PreparedTrainingStep:
        # Staleness admission is the coordinator's; the payload carries only the rows.
        return prepare_tinker_step(batch, objective, algorithm_state, scheduling, batch_size=self._config.batch_size)

    @contextmanager
    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        rollout_id: int,
        prior_marker: Mapping[str, Any] | None,
    ) -> Iterator[PreparedTrainingJob | TrainingJobResult]:
        scenario = payload.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("Tinker trains one adapter per scenario; the job must name its scenario")
        # Scenario steps are per scenario; the checkpoint index is one sequence across all of them.
        scenario_step = rollout_id
        rollout_id = self._context.next_rollout_id
        directory = Path(self._template.format(rollout_id=rollout_id))
        if directory.exists() or directory.is_symlink():
            raise RuntimeError(f"checkpoint target already exists: {directory}")
        loss = resolve_tinker_loss(payload["loss"])
        batches = [[row_from_payload(row) for row in rows] for rows in payload["batches"]]
        if not batches or any(not rows for rows in batches):
            raise ValueError("a Tinker training job needs at least one non-empty optimizer batch")
        yield _TinkerPreparedJob(
            self,
            TrainingCheckpoint(rollout_id, directory, scenario, scenario_step),
            incumbent=self._incumbent(scenario)[0],
            batches=batches,
            loss=loss,
        )

    def train_job(self, job: _TinkerPreparedJob) -> TrainingMetrics:
        checkpoint, metrics = self.client.train(job.incumbent, job.batches, job.loss)
        checkpoint.validate_model(self._model, self._config.lora_rank)
        job.result = checkpoint
        return TrainingMetrics(training=dict(metrics))

    def save_job_checkpoint(self, job: _TinkerPreparedJob) -> None:
        """Materialize the trained adapter beside its manifest; the job is durable only after this."""
        if job.result is None:
            raise RuntimeError("Tinker job has no trained checkpoint to save")
        directory = job.checkpoint.path
        self.client.download(job.result, directory / ADAPTER_DIR)
        job.result.write(directory)
        self._require_history().record_checkpoint(job.checkpoint.scenario or "", job.checkpoint.rollout_id)

    def adapter_files(self, scenario: str, runtime_load_id: str) -> Path:
        """The adapter to serve as ``runtime_load_id``: the scenario's newest checkpoint, or its incumbent again."""
        _, rollout_id, version = self._incumbent(scenario)
        if version != runtime_load_id:
            entry = self._require_history().entry(scenario)
            latest = None if entry is None else entry.get("rollout_id")
            if not isinstance(latest, int):
                raise RuntimeError(f"scenario {scenario!r} has no Tinker checkpoint to publish")
            rollout_id = latest
        if rollout_id is None:
            raise RuntimeError(f"scenario {scenario!r} has no published Tinker checkpoint to republish")
        directory = Path(self._template.format(rollout_id=rollout_id))
        adapter = directory / ADAPTER_DIR
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"Tinker adapter is missing or incomplete: {adapter}")
        if version != runtime_load_id:
            # Reef publishes what it asked for; the marker retries the same checkpoint on failure.
            self._remember_incumbent(scenario, TinkerCheckpoint.read(directory), rollout_id, runtime_load_id)
        return adapter

    def prepare_weights(self, runtime_load_id: str, *, force_full: bool) -> None:
        raise RuntimeError("Tinker delivers adapter files; Reef's receiver loads them")

    def send_weights(self, runtime_load_id: str, *, force_full: bool) -> str:
        raise RuntimeError("Tinker delivers adapter files; Reef's receiver loads them")

    def initialize_version(self, runtime_load_id: str) -> None:
        return

    def activate_scenario(self, scenario: str) -> None:
        self._active_scenario = scenario

    def send_adapter(self, scenario: str, name: str) -> None:
        raise RuntimeError("Tinker delivers adapter files; Reef's receiver loads them")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    # -- Checkpoint bookkeeping

    def _require_history(self) -> ScenarioHistory:
        history = self._context.history
        if not isinstance(history, ScenarioHistory):
            raise RuntimeError("Tinker's coordinator backend keeps a scenario history")
        return history

    def _incumbent_path(self, scenario: str) -> Path:
        return self._root / "scenarios" / urllib.parse.quote(scenario, safe="") / "incumbent.json"

    def _incumbent(self, scenario: str) -> tuple[TinkerCheckpoint, int | None, str | None]:
        """The checkpoint a scenario's next job branches from: its last publication, else the seeded base."""
        value = read_json(self._incumbent_path(scenario))
        if value is None:
            return self.base, None, None
        checkpoint = TinkerCheckpoint(**value["checkpoint"])
        checkpoint.validate_model(self._model, self._config.lora_rank)
        return checkpoint, int(value["rollout_id"]), str(value["runtime_load_id"])

    def _remember_incumbent(
        self, scenario: str, checkpoint: TinkerCheckpoint, rollout_id: int, runtime_load_id: str
    ) -> None:
        write_json(
            self._incumbent_path(scenario),
            {"checkpoint": asdict(checkpoint), "rollout_id": rollout_id, "runtime_load_id": runtime_load_id},
        )


class _TinkerPreparedJob(PreparedTrainingJob):
    """One admitted job: the incumbent it branches from and the rows it trains on."""

    def __init__(
        self,
        backend: TinkerTrainingBackend,
        checkpoint: TrainingCheckpoint,
        *,
        incumbent: TinkerCheckpoint,
        batches: Sequence[Sequence[TokenRow]],
        loss: TinkerLoss,
    ) -> None:
        self._backend = backend
        self._checkpoint = checkpoint
        self.incumbent = incumbent
        self.batches = batches
        self.loss = loss
        self.result: TinkerCheckpoint | None = None

    @property
    def checkpoint(self) -> TrainingCheckpoint:
        return self._checkpoint

    def train(self) -> TrainingMetrics:
        return self._backend.train_job(self)

    def save_checkpoint(self) -> None:
        self._backend.save_job_checkpoint(self)


def read_incumbent(path: Path) -> dict[str, Any] | None:
    """The persisted incumbent record, for tests and diagnostics."""
    value = read_json(path)
    return None if value is None else json.loads(json.dumps(value))
