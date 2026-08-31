"""Slime-owned step preparation: resolve a preparer signal and build the payload.

Backend-agnostic step signals (which loss family, what advantages) live in
``reef.train.algos`` and are reusable by any training backend.
The only thing left here is ``_build_payload``, which materializes Slime's
own wire tuples from a resolved signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.base import PreparedTrainingStep
from reef.train.algos import StepScheduling
from reef.train.algos.registry import resolve_preparer
from reef.train.algos.schedule import MaterializedSchedule, materialize_schedule, schedule_seed
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.types import GroupedPolicyBatch, TrainingBatch, policy_samples


def prepare_slime_step(
    batch: TrainingBatch,
    preparer_id: str,
    algorithm_state: Mapping[str, Any],
) -> PreparedTrainingStep:
    """Resolve a step preparer and produce its complete Slime training payload."""
    signal = resolve_preparer(preparer_id)(batch, algorithm_state)
    if signal.action == "skip":
        return PreparedTrainingStep(
            action="skip",
            next_algorithm_state=signal.next_algorithm_state,
            metrics=signal.metrics,
        )
    schedule = _materialize(batch, signal.scheduling)
    payload = _build_payload(batch, signal.loss_family, signal.advantages, signal.scheduling)
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
    samples = policy_samples(batch)
    if isinstance(batch, GroupedPolicyBatch) and scheduling.unit != "sample":
        source_rollout_ids = [
            group_index for group_index, comparison_set in enumerate(batch.comparison_sets) for _ in comparison_set
        ]
    else:
        # PolicyBatch, or a grouped batch whose objective schedules per sample;
        # any other batch type already failed policy_samples() above.
        source_rollout_ids = list(range(len(samples)))
    return materialize_schedule(source_rollout_ids, scheduling, seed=schedule_seed(batch.batch_id))


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
    samples = policy_samples(batch)
    shape_row = resolve_loss_family(loss_family).shape_sample_row
    if advantages is not None and len(advantages) != len(samples):
        raise ValueError(f"advantages length {len(advantages)} does not match sample count {len(samples)}")
    schedule = _materialize(batch, scheduling)
    payload: dict[str, Any] = {
        "samples": [shape_row(samples[row]) for row in schedule.row_indices],
        "rollout_ids": list(schedule.rollout_ids),
        "loss": loss_family,
        # The batch row behind every wire row, so the runtime layer can attach
        # per-row provenance (producing weight versions) in the same order —
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
