"""Fakes of the Slime training component's worker groups, for CPU tests of ``SlimeTrainingComponent``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from reef.runtime.jobs import JobStatus, TrainingComponent
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import TrainingBatchProcessor


class FakeTrainGroup:
    """An actor or critic group: records its calls; an actor save writes the checkpoint Slime's writers write.

    ``hf_template`` is the actor's ``--save-hf`` template, or ``None`` for a
    critic, which saves no Hugging Face export. With ``lora`` the export is a
    PEFT adapter; with ``megatron_root`` each save also writes the paired
    Megatron iteration and its tracker. ``rank_metrics`` is what each rank
    reports after a step.
    """

    def __init__(
        self,
        hf_template: str | None,
        *,
        lora: bool = False,
        megatron_root: Path | None = None,
        rank_metrics: list[dict[str, Any]] | None = None,
        release_error: Exception | None = None,
    ) -> None:
        self.hf_template = hf_template
        self.lora = lora
        self.megatron_root = megatron_root
        self.rank_metrics = [{}] if rank_metrics is None else rank_metrics
        self.release_error = release_error
        self.train_calls: list[tuple[int, Any]] = []
        self.saved: list[tuple[int, bool]] = []
        self.train_result: list[Any] = [{"train/loss": 0.5}]
        self.events: list[str] = []
        self.rollout_managers: list[Any] = []
        self.train_parallel_config = {"dp_size": 1}

    def set_rollout_manager(self, manager) -> None:
        self.rollout_managers.append(manager)

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.train_calls.append((rollout_id, rollout_data_ref))
        return list(self.train_result)

    def reset_training_state(self) -> None:
        self.events.append("reset")

    def async_pop_metrics(self):
        return list(self.rank_metrics)

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        self.saved.append((rollout_id, force_sync))
        if self.hf_template is None:
            return
        checkpoint = Path(self.hf_template.format(rollout_id=rollout_id))
        checkpoint.mkdir(parents=True)
        if self.lora:
            (checkpoint / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}))
            (checkpoint / "adapter_model.safetensors").write_text(f"adapter-{rollout_id}")
        else:
            (checkpoint / "model.safetensors").write_text(f"weights-{rollout_id}")
        if self.megatron_root is not None:
            iteration = self.megatron_root / f"iter_{rollout_id:07d}"
            iteration.mkdir(parents=True)
            (iteration / "state").write_text("megatron")
            (self.megatron_root / "latest_checkpointed_iteration.txt").write_text(str(rollout_id))

    def release(self) -> None:
        self.events.append("release")
        if self.release_error is not None:
            raise self.release_error


class RecordingBatchProcessor(TrainingBatchProcessor):
    """Keeps each job's rollout data instead of splitting it across data-parallel ranks."""

    def __init__(self) -> None:
        super().__init__(SlimeArguments(), {"dp_size": 1})
        self.calls: list[dict[str, Any]] = []

    def prepare_external_train_data(self, data):
        self.calls.append(data)
        return ["packed"]


def finished(component: TrainingComponent, job_id: str) -> JobStatus:
    """The job's status once it ended; jobs run on the component's own thread."""
    deadline = time.monotonic() + 10
    while (status := component.status(job_id)).state == "running":
        if time.monotonic() > deadline:
            pytest.fail(f"training job {job_id} did not end")
        time.sleep(0.01)
    return status
