"""Turn a ``TrainingJob`` into Tinker optimizer batches, and give the learning rates of a job."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from reef.core.batches import trajectories
from reef.runtime.interfaces import (
    LearningRateSchedule,
    LearningRateScheduleState,
    TrainingJob,
    TrainingPayload,
    resolve_learning_rate_schedule_state,
)
from reef.train.algos.schedule import batch_schedule_seed, materialize_schedule
from reef.train.tinker_backend.losses import TokenRow, resolve_tinker_loss


def build_tinker_payload(
    job: TrainingJob, *, batch_size: int, kl_coef: float, runtime_load_id: str | None = None
) -> TrainingPayload:
    """Shape ``job`` into Tinker optimizer batches under its scheduling.

    The job trains with the Tinker loss of ``job.loss_family``, which must be
    registered in this process and accept ``kl_coef``; the payload records
    ``job.method``. With ``runtime_load_id`` the payload also records the
    version that the job trains from. The record-to-record stage already
    dropped stale samples.
    """
    resolve_tinker_loss(job.loss_family).check_kl_coef(kl_coef)
    if job.advantages is None:
        raise ValueError("Tinker policy training requires one advantage per trajectory")
    items = trajectories(job.batch)
    scheduling = job.scheduling
    rows = []
    groups: dict[str, int] = {}
    rollout_ids = []
    for index, (item, advantage) in enumerate(zip(items, job.advantages, strict=True)):
        rows.append(asdict(TokenRow.from_item(item, advantage)))
        key = f"group:{item.group_id}" if item.group_id is not None else f"row:{index}"
        rollout_ids.append(index if scheduling.unit == "sample" else groups.setdefault(key, len(groups)))
    schedule = materialize_schedule(rollout_ids, scheduling, step_size=batch_size, seed=batch_schedule_seed(job.batch))
    batches: list[list[dict[str, Any]]] = []
    cursor = 0
    next_rollout = 0
    for size in schedule.step_sizes:
        next_rollout += size
        indices = []
        while cursor < len(schedule.row_indices) and schedule.rollout_ids[cursor] < next_rollout:
            indices.append(schedule.row_indices[cursor])
            cursor += 1
        batches.append([rows[index] for index in indices])
    if not batches or any(not rows for rows in batches):
        raise ValueError("Tinker scheduling produced an empty optimizer batch")
    return TrainingPayload(
        {
            "batch_id": job.batch.batch_id,
            "loss": job.loss_family,
            "method": job.method.to_dict(),
            "batches": batches,
            **({"source_runtime_load_id": runtime_load_id} if runtime_load_id is not None else {}),
        },
        {"optimizer_steps": len(batches), "dropped_rollouts": schedule.dropped_rollouts},
    )


def job_learning_rates(
    incumbent_schedule_state: LearningRateScheduleState | None,
    requested: LearningRateSchedule | None,
    optimizer_steps: int,
    configured_learning_rate: float,
) -> tuple[tuple[float, ...], LearningRateScheduleState | None]:
    """The rate of each optimizer step of a job, and the schedule state the job leaves.

    ``incumbent_schedule_state`` is the schedule state of the incumbent that
    the job branches from.

    Every attempt of a job branches from the same incumbent, so a retry trains
    with the same rates; the state the job leaves is recorded in its checkpoint.
    """
    schedule_state = resolve_learning_rate_schedule_state(incumbent_schedule_state, requested)
    if schedule_state is None:
        return (configured_learning_rate,) * optimizer_steps, None
    return schedule_state.learning_rates(optimizer_steps), schedule_state.advanced(optimizer_steps)
