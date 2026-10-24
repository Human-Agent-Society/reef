"""Turn backend-neutral algorithm signals into Tinker optimizer batches and their learning rates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from reef.core.batches import StepScheduling, TrainingBatch, trajectories
from reef.runtime.interfaces import (
    LearningRateSchedule,
    LearningRateScheduleState,
    PreparedTrainingStep,
    TrainingMethod,
    resolve_learning_rate_schedule_state,
)
from reef.train.algos.registry import resolve_objective
from reef.train.algos.schedule import batch_schedule_seed, materialize_schedule
from reef.train.tinker_backend.losses import TokenRow, resolve_tinker_loss


def prepare_tinker_step(
    batch: TrainingBatch,
    method: TrainingMethod,
    state: Mapping[str, Any],
    scheduling: StepScheduling,
    *,
    batch_size: int,
    kl_coef: float,
    runtime_load_id: str | None = None,
) -> PreparedTrainingStep:
    """Shape one batch into Tinker optimizer batches under the recipe's ``scheduling``.

    The job trains ``method``'s objective with that objective's Tinker loss,
    which must be registered in this process and accept ``kl_coef``; the
    payload records the method.
    With ``runtime_load_id`` the payload also records the version that the
    job trains from. The record-to-record stage already dropped stale samples.
    """
    objective = resolve_objective(method.objective)
    objective.validate_scheduling(scheduling)
    signal = objective.prepare(batch, state)
    if signal.action == "skip":
        return PreparedTrainingStep("skip", signal.next_algorithm_state, signal.metrics)
    resolve_tinker_loss(objective.loss_family).check_kl_coef(kl_coef)
    items = trajectories(batch)
    if signal.advantages is None or len(signal.advantages) != len(items):
        raise ValueError("Tinker policy training requires one advantage per trajectory")
    rows = []
    groups: dict[str, int] = {}
    rollout_ids = []
    for index, (item, advantage) in enumerate(zip(items, signal.advantages, strict=True)):
        rows.append(asdict(TokenRow.from_item(item, advantage)))
        key = f"group:{item.group_id}" if item.group_id is not None else f"row:{index}"
        rollout_ids.append(index if scheduling.unit == "sample" else groups.setdefault(key, len(groups)))
    schedule = materialize_schedule(rollout_ids, scheduling, step_size=batch_size, seed=batch_schedule_seed(batch))
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
    return PreparedTrainingStep(
        "train",
        signal.next_algorithm_state,
        {**signal.metrics, "optimizer_steps": len(batches), "dropped_rollouts": schedule.dropped_rollouts},
        {
            "batch_id": batch.batch_id,
            "loss": objective.loss_family,
            "method": method.to_dict(),
            "batches": batches,
            **({"source_runtime_load_id": runtime_load_id} if runtime_load_id is not None else {}),
        },
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
