"""Tinker as a training component: each job trains its data remotely, from the job's start checkpoint.

A job starts from ``job.start``, a directory that holds a Tinker checkpoint
manifest, or from the base checkpoint when ``start`` is ``None``. The manifest
carries the remote training state, the learning-rate schedule's progress and
the objective's state, so a job that starts from it continues all three. Each
attempt branches from those weights and their optimizer state, so repeating an
attempt can consume API resources but cannot apply its gradient twice.

Serving reads each saved checkpoint from its artifact: Tinker's sampler
(``reef.inference.tinker``) reads the manifest, and Reef's serving engines
(``reef.runtime.serving``) load the PEFT adapter the component exports beside
it. Nothing is shared in process.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from reef.runtime.interfaces import learning_rate_metrics, resolve_learning_rate_schedule_state
from reef.runtime.jobs import ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.runtime.serving import ADAPTER_DIR, holds_weights
from reef.train.algos.registry import job_training
from reef.train.local_jobs import LocalJobComponent
from reef.train.tinker_backend.checkpoint import MANIFEST, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import prepare_tinker_job


class TinkerTrainingComponent(LocalJobComponent):
    """Train each job as remote Tinker sessions from the job's start checkpoint, in a background thread.

    A job's parameters are ``method`` (a ``TrainingMethod`` as a dict) and
    optionally ``scheduling`` (a ``StepScheduling`` as a dict, see
    ``job_training``),
    ``batch_size``, ``kl_coef`` and ``learning_rate``, which override the
    component's ``TinkerConfig``, and ``save_interval``, a positive int: the
    optimizer steps between two saved checkpoints. Without it, the job saves
    one checkpoint when it ends. ``submit`` refuses any other parameter.
    Jobs of several scenarios can share the component, since each job trains
    from its own start.

    ``config.state_dir`` holds ``base/`` (the base checkpoint),
    ``imported/<digest>/`` (starts taken from releases) and ``jobs/`` (see
    ``LocalJobComponent``); a saved checkpoint is ``jobs/<job_id>/<step>/``,
    with its manifest and, with ``config.export_adapter``, its PEFT adapter in
    ``adapter/``. A checkpoint directory is both the start of a later job and
    the release Reef publishes.
    """

    def __init__(self, base_model: str, config: TinkerConfig, client: TinkerClient) -> None:
        self._model = base_model
        self._config = config
        self._client = client
        self._root = Path(config.state_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._state_lock = (self._root / ".lock").open("a")
        try:
            fcntl.flock(self._state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._state_lock.close()
            raise ValueError("Tinker state_dir is already owned by another runtime") from None
        try:
            super().__init__(self._root / "jobs")
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
        self.start_jobs()

    @property
    def name(self) -> str:
        return "tinker"

    @property
    def release_format(self) -> ReleaseFormat | None:
        """The exported PEFT adapter of the configured rank, or ``None`` when releases hold only the manifest."""
        if not self._config.export_adapter:
            return None
        return ReleaseFormat("adapter", self._config.lora_rank, ("all",))

    @property
    def state_dir(self) -> Path:
        return self._root

    def _validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.validate_model(self._model, self._config.lora_rank)
        return checkpoint

    @property
    def shares_scenarios(self) -> bool:
        return True

    def import_start(self, release: Path) -> str | None:
        if not holds_weights(release):
            return None
        if not (release / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        checkpoint = self._validated(TinkerCheckpoint.read(release))
        digest = hashlib.sha256(json.dumps(checkpoint.to_dict(), sort_keys=True).encode()).hexdigest()
        imported = self._root / "imported" / digest
        if not imported.exists():
            # The manifest names the remote state; the exported adapter is not needed to train from it.
            checkpoint.write(imported)
        return str(imported)

    def job_parameters(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """``parameters`` over the component's defaults; raise ``ValueError`` for a parameter it does not accept."""
        defaults: dict[str, Any] = {
            "batch_size": self._config.batch_size,
            "kl_coef": self._config.kl_coef,
            "learning_rate": self._config.learning_rate,
            "save_interval": None,
        }
        unknown = set(parameters) - {"method", "scheduling", *defaults}
        if unknown:
            raise ValueError(f"Tinker jobs accept no parameters {sorted(unknown)}")
        merged = {**defaults, **parameters}
        interval = merged["save_interval"]
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0):
            raise ValueError("save_interval must be a positive int")
        return merged

    def check_job(self, job: TrainingJob) -> None:
        job_training(self.job_parameters(job.parameters))
        if job.start is not None:
            self._validated(TinkerCheckpoint.read(Path(job.start)))

    def close_resources(self) -> None:
        try:
            self._client.close()
        finally:
            self._state_lock.close()

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        parameters = self.job_parameters(job.parameters)
        start = self._base if job.start is None else self._validated(TinkerCheckpoint.read(Path(job.start)))
        method, scheduling = job_training(parameters)
        prepared = prepare_tinker_job(
            job.data,
            method,
            start.state_for(method.objective),
            scheduling,
            batch_size=parameters["batch_size"],
            kl_coef=parameters["kl_coef"],
        )
        if prepared.payload is None:
            # The objective skipped the job's data: the job trains nothing and saves no checkpoint.
            return prepared.metrics
        batches = [[row_from_payload(row) for row in batch] for batch in prepared.payload["batches"]]
        schedule = resolve_learning_rate_schedule_state(
            start.learning_rate_schedule_state, method.learning_rate_schedule
        )
        learning_rates = (
            (parameters["learning_rate"],) * len(batches)
            if schedule is None
            else schedule.learning_rates(len(batches))
        )
        interval = parameters["save_interval"] or len(batches)
        loss = resolve_tinker_loss(prepared.payload["loss"])
        saved = self.saved(directory)
        # A resumed job continues after its last saved checkpoint, from that checkpoint.
        done = saved[-1].step if saved else 0
        current = TinkerCheckpoint.read(Path(saved[-1].path)) if saved else start
        for first in range(done, len(batches), interval):
            if self.stopping(directory):
                break
            last = min(first + interval, len(batches))
            checkpoint, metrics = self._client.train(current, batches[first:last], loss, learning_rates[first:last])
            schedule_state = None if schedule is None else schedule.advanced(last)
            current = replace(
                self._validated(checkpoint),
                learning_rate_schedule_state=schedule_state,
                algorithm_state=dict(prepared.next_algorithm_state),
                objective=method.objective if prepared.next_algorithm_state else None,
            )
            target = directory / str(last)
            if self._config.export_adapter:
                # An earlier attempt may have left part of the adapter.
                shutil.rmtree(target / ADAPTER_DIR, ignore_errors=True)
                self._client.download(current, target / ADAPTER_DIR)
            current.write(target)
            self.save_checkpoint(
                directory,
                SavedCheckpoint(
                    last,
                    str(target),
                    str(target),
                    {
                        **prepared.metrics,
                        **metrics,
                        **learning_rate_metrics(learning_rates[first:last], schedule_state),
                    },
                ),
            )
        return {**prepared.metrics, "optimizer_steps": len(batches)}
