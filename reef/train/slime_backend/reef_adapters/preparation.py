"""Slime-owned step preparation: resolve an objective signal and build the payload.

Backend-agnostic step signals (which loss family, what advantages) live in
``reef.train.algos`` and are reusable by any training backend.
The only thing left here is ``_build_payload``, which materializes Slime's
own wire tuples from a resolved signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.interfaces import PreparedTrainingStep
from reef.train.algos import StepScheduling
from reef.train.algos.registry import resolve_objective
from reef.train.algos.schedule import MaterializedSchedule, batch_schedule_seed, materialize_schedule
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.types import TrainingBatch, TrajectoryItem, trajectories


def prepare_slime_step(
    batch: TrainingBatch,
    objective_id: str,
    algorithm_state: Mapping[str, Any],
    scheduling: StepScheduling,
) -> PreparedTrainingStep:
    """Resolve a training objective and produce its complete Slime training payload.

    ``scheduling`` is the recipe's step schedule; the objective rejects one its
    loss cannot train before any payload is built.
    """
    objective = resolve_objective(objective_id)
    objective.validate_scheduling(scheduling)
    signal = objective.prepare(batch, algorithm_state)
    if signal.action == "skip":
        return PreparedTrainingStep(
            action="skip",
            next_algorithm_state=signal.next_algorithm_state,
            metrics=signal.metrics,
        )
    schedule = _materialize(batch, scheduling)
    payload = _build_payload(batch, objective.loss_family, signal.advantages, scheduling)
    metrics = dict(signal.metrics)
    if schedule.epochs > 1:
        metrics.setdefault("epochs", schedule.epochs)
    if schedule.optimizer_steps is not None:
        metrics.setdefault("optimizer_steps", schedule.optimizer_steps)
    if schedule.dropped_rollouts:
        metrics.setdefault("dropped_rollouts", schedule.dropped_rollouts)
    if schedule.tail_rollouts:
        metrics.setdefault("partial_step_rollouts", schedule.tail_rollouts)
    return PreparedTrainingStep(
        action="train",
        payload=payload,
        next_algorithm_state=signal.next_algorithm_state,
        metrics=metrics,
    )


def _materialize(batch: TrainingBatch, scheduling: StepScheduling) -> MaterializedSchedule:
    """Rollout grouping for ``batch`` under ``scheduling``, expanded into a row order."""
    samples = trajectories(batch)
    if scheduling.unit == "sample":
        source_rollout_ids = list(range(len(samples)))
    else:
        group_ids: dict[tuple[str, str | int], int] = {}
        source_rollout_ids = []
        for index, item in enumerate(batch.items):
            key = (
                ("group", item.group_id)
                if isinstance(item, TrajectoryItem) and item.group_id is not None
                else ("sample", index)
            )
            source_rollout_ids.append(group_ids.setdefault(key, len(group_ids)))
    return materialize_schedule(source_rollout_ids, scheduling, seed=batch_schedule_seed(batch))


def _build_payload(
    batch: TrainingBatch,
    loss_family: str,
    advantages: tuple[float, ...] | None,
    scheduling: StepScheduling,
) -> dict[str, Any]:
    """Materialize Slime's wire rows in the order ``scheduling`` trains them.

    Rollout ids group the rows one optimizer step must keep together; the
    schedule may repeat rows (epochs), reorder rollouts (shuffle) and fix the
    step layout (``external_step_sizes``) — otherwise the runtime cuts steps
    of its configured size and ``external_remainder`` says what it does with
    a tail. See :class:`StepScheduling`.
    """
    samples = trajectories(batch)
    shape_row = resolve_loss_family(loss_family).shape_sample_row
    if advantages is not None and len(advantages) != len(samples):
        raise ValueError(f"advantages length {len(advantages)} does not match sample count {len(samples)}")
    schedule = _materialize(batch, scheduling)
    payload: dict[str, Any] = {
        "samples": [shape_row(samples[row]) for row in schedule.row_indices],
        "rollout_ids": list(schedule.rollout_ids),
        "loss": loss_family,
        # The batch row behind every wire row, so the runtime layer can attach
        # each row's producing runtime load IDs in the same order —
        # a schedule may repeat (epochs) and reorder (shuffle) rows. Consumed
        # and removed before the payload leaves the runtime.
        "source_rows": list(schedule.row_indices),
    }
    if advantages is not None:
        payload["advantages"] = [advantages[row] for row in schedule.row_indices]
    if schedule.step_sizes is not None:
        payload["external_step_sizes"] = list(schedule.step_sizes)
    else:
        payload["external_remainder"] = scheduling.remainder
    return payload
