"""Fakes of the Slime training component's worker groups and storage, for CPU tests of ``SlimeTrainingComponent``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reef.runtime.jobs import JobStatus, TrainingComponent
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import TrainingBatchProcessor
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig


class FakeTrainGroup:
    """An actor or critic group: records its calls; a save writes the checkpoint Slime's writers write.

    ``hf_template`` is the actor's ``--save-hf`` template, or ``None`` for a
    critic, which saves no Hugging Face export. With ``lora`` the export is a
    PEFT adapter; with ``megatron_root`` each save also writes the group's
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
        self.train_parallel_config = {"dp_size": 1}

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.train_calls.append((rollout_id, rollout_data_ref))
        return list(self.train_result)

    def reset_training_state(self) -> None:
        self.events.append("reset")

    def async_pop_metrics(self):
        return list(self.rank_metrics)

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        self.saved.append((rollout_id, force_sync))
        if self.megatron_root is not None:
            iteration = self.megatron_root / f"iter_{rollout_id:07d}"
            iteration.mkdir(parents=True)
            (iteration / "state").write_text("megatron")
            (self.megatron_root / "latest_checkpointed_iteration.txt").write_text(str(rollout_id))
        if self.hf_template is None:
            return
        checkpoint = Path(self.hf_template.format(rollout_id=rollout_id))
        checkpoint.mkdir(parents=True)
        if self.lora:
            (checkpoint / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}))
            (checkpoint / "adapter_model.safetensors").write_text(f"adapter-{rollout_id}")
        else:
            (checkpoint / "model.safetensors").write_text(f"weights-{rollout_id}")

    def release(self) -> None:
        self.events.append("release")
        if self.release_error is not None:
            raise self.release_error


class RecordingBatchProcessor(TrainingBatchProcessor):
    """Keeps each job's rollout data instead of splitting it across data-parallel ranks."""

    def __init__(self) -> None:
        super().__init__(SlimeArguments(custom_rollout_data_keys=(), reef_rollout_tensor_dtypes={}), {"dp_size": 1})
        self.calls: list[dict[str, Any]] = []

    def prepare_external_train_data(self, data):
        self.calls.append(data)
        return ["packed"]


def checkpoint_storage(
    root: Path, *, critic_root: Path | None = None, critic_save_interval: int = 1
) -> CheckpointStorage:
    """Checkpoint storage over ``root/hf/{rollout_id}`` and ``root/megatron`` that every test checkpoint fits.

    Each asset measures one byte and the filesystem reports a terabyte free,
    so admission never blocks and retention never deletes a checkpoint.
    """
    source = root / "source-hf"
    source.mkdir(parents=True, exist_ok=True)
    return CheckpointStorage(
        RetentionConfig(max_storage_bytes=1 << 30, min_free_space_bytes=0),
        hf_template=str(root / "hf" / "{rollout_id}"),
        megatron_root=root / "megatron",
        critic_root=critic_root,
        source_hf=source,
        measure=lambda path: 1,
        disk_usage=lambda path: SimpleNamespace(total=1 << 40, free=1 << 40),
        critic_save_interval=critic_save_interval,
    )


def finished(component: TrainingComponent, job_id: str) -> JobStatus:
    """The job's status once it ended; jobs run on the component's own thread."""
    deadline = time.monotonic() + 10
    while (status := component.status(job_id)).state == "running":
        if time.monotonic() > deadline:
            pytest.fail(f"training job {job_id} did not end")
        time.sleep(0.01)
    return status
