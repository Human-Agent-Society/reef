"""Turn one job's data into its Tinker optimizer steps: the objective's signal as rows, cut by the job's scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.job_data import JobData, trajectories
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.registry import resolve_objective
from reef.train.algos.schedule import StepScheduling, schedule_job_data
from reef.train.tinker_backend.losses import TinkerLoss, TokenRow, resolve_tinker_loss


@dataclass(frozen=True)
class TinkerSteps:
    """The optimizer steps of one Tinker job.

    Values: either the objective trains the job's data, so ``loss`` is its Tinker loss and ``steps`` holds one
    or more non-empty tuples of rows, one per optimizer step in order; or it skipped the data, so ``loss`` is
    None and ``steps`` is empty. ``next_algorithm_state`` is the objective's state after the whole job data,
    and ``metrics`` its telemetry, with ``optimizer_steps`` (the number of steps) and ``dropped_rollouts``
    when it trains.
    Purpose: ``prepare_tinker_job`` makes it; the Tinker component trains it.
    """

    loss: TinkerLoss | None
    steps: tuple[tuple[TokenRow, ...], ...]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (self.loss is None) != (not self.steps):
            raise ValueError("Tinker steps hold a loss exactly when they hold one or more optimizer steps")
        if any(not step for step in self.steps):
            raise ValueError("every Tinker optimizer step trains one or more rows")


def prepare_tinker_job(
    data: JobData,
    method: TrainingMethod,
    state: Mapping[str, Any],
    scheduling: StepScheduling,
    *,
    batch_size: int,
    kl_coef: float,
) -> TinkerSteps:
    """The optimizer steps of ``data`` under ``scheduling``, at most ``batch_size`` rollouts each.

    The job trains ``method``'s objective with that objective's Tinker loss, which must be registered in this
    process and accept ``kl_coef``. The record-to-record stage already dropped stale samples. ``ValueError``:
    the loss refuses ``kl_coef``, the objective gives no advantage for each trajectory of a policy loss, or a
    row is not trainable.
    """
    objective = resolve_objective(method.objective)
    signal = objective.prepare(data, state)
    if signal.action == "skip":
        return TinkerSteps(None, (), signal.next_algorithm_state, signal.metrics)
    loss = resolve_tinker_loss(objective.loss_family)
    loss.check_kl_coef(kl_coef)
    items = trajectories(data)
    if not loss.uses_advantages:
        advantages: tuple[float, ...] = (0.0,) * len(items)
    elif signal.advantages is None or len(signal.advantages) != len(items):
        raise ValueError("Tinker policy training requires one advantage per trajectory")
    else:
        advantages = signal.advantages
    rows = [
        TokenRow.from_item(item, advantage, rollout_log_probs=loss.uses_rollout_log_probs)
        for item, advantage in zip(items, advantages, strict=True)
    ]
    schedule = schedule_job_data(data, objective.rollout_unit, scheduling, step_size=batch_size)
    steps: list[tuple[TokenRow, ...]] = []
    cursor = 0
    next_rollout = 0
    for size in schedule.step_sizes:
        next_rollout += size
        indices = []
        while cursor < len(schedule.row_indices) and schedule.rollout_ids[cursor] < next_rollout:
            indices.append(schedule.row_indices[cursor])
            cursor += 1
        steps.append(tuple(rows[index] for index in indices))
    return TinkerSteps(
        loss,
        tuple(steps),
        signal.next_algorithm_state,
        {**signal.metrics, "optimizer_steps": len(steps), "dropped_rollouts": schedule.dropped_rollouts},
    )
