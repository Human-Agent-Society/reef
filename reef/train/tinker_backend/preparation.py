"""Turn backend-neutral algorithm signals into Tinker optimizer batches and their learning rates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from reef.core.job_data import JobData, trajectories
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.registry import resolve_objective
from reef.train.algos.schedule import StepScheduling, schedule_job_data
from reef.train.algos.signals import PreparedJob
from reef.train.tinker_backend.losses import TokenRow, resolve_tinker_loss


def prepare_tinker_job(
    data: JobData,
    method: TrainingMethod,
    state: Mapping[str, Any],
    scheduling: StepScheduling,
    *,
    batch_size: int,
    kl_coef: float,
) -> PreparedJob:
    """Shape one job's data into Tinker optimizer batches under the job's ``scheduling``.

    The job trains ``method``'s objective with that objective's Tinker loss,
    which must be registered in this process and accept ``kl_coef``; the
    payload records the method. The record-to-record stage already dropped
    stale samples.
    """
    objective = resolve_objective(method.objective)
    signal = objective.prepare(data, state)
    if signal.action == "skip":
        return PreparedJob("skip", signal.next_algorithm_state, signal.metrics)
    resolve_tinker_loss(objective.loss_family).check_kl_coef(kl_coef)
    items = trajectories(data)
    if signal.advantages is None or len(signal.advantages) != len(items):
        raise ValueError("Tinker policy training requires one advantage per trajectory")
    rows = [
        asdict(TokenRow.from_item(item, advantage)) for item, advantage in zip(items, signal.advantages, strict=True)
    ]
    schedule = schedule_job_data(data, objective.rollout_unit, scheduling, step_size=batch_size)
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
        raise ValueError("Tinker scheduling produced an empty optimizer data")
    return PreparedJob(
        "train",
        signal.next_algorithm_state,
        {**signal.metrics, "optimizer_steps": len(batches), "dropped_rollouts": schedule.dropped_rollouts},
        {
            "data_id": data.data_id,
            "loss": objective.loss_family,
            "method": method.to_dict(),
            "batches": batches,
        },
    )
