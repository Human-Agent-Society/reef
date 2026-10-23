"""Slime-owned payload building: materialize Slime's wire rows from a ``TrainingJob``.

Reef's service process resolves the objective and runs its backend-neutral
``prepare`` (``reef.train.algos``); this module only lays out the job's rows
in Slime's optimizer steps.
"""

from __future__ import annotations

from typing import Any

from reef.runtime.interfaces import TrainingJob, TrainingPayload
from reef.train.algos import StepScheduling
from reef.train.algos.schedule import MaterializedSchedule, batch_schedule_seed, materialize_schedule
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.score_centering import sampler_topk_columns
from reef.train.types import TrainingBatch, TrajectoryItem, trajectories


def build_slime_payload(job: TrainingJob, *, step_size: int, sampler_topk: bool = False) -> TrainingPayload:
    """The complete Slime training payload of ``job``, with the metrics of its step layout.

    The payload records ``job.method``, so the method is part of the job's
    identity. ``step_size`` is Slime's ``--global-batch-size``, the rollouts in
    one optimizer step. ``sampler_topk`` adds each row's recorded sampler
    top-K, which score centering reads.
    """
    schedule = _materialize(job.batch, job.scheduling, step_size)
    payload = _build_payload(
        job.batch, job.loss_family, job.advantages, job.scheduling, step_size=step_size, sampler_topk=sampler_topk
    )
    payload["method"] = job.method.to_dict()
    metrics: dict[str, Any] = {"optimizer_steps": schedule.optimizer_steps}
    if schedule.epochs > 1:
        metrics["epochs"] = schedule.epochs
    if schedule.dropped_rollouts:
        metrics["dropped_rollouts"] = schedule.dropped_rollouts
    if schedule.tail_rollouts:
        metrics["partial_step_rollouts"] = schedule.tail_rollouts
    return TrainingPayload(payload, metrics)


def _materialize(batch: TrainingBatch, scheduling: StepScheduling, step_size: int) -> MaterializedSchedule:
    """Rollout grouping for ``batch`` under ``scheduling``, expanded into a row order and steps of ``step_size``."""
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
    return materialize_schedule(source_rollout_ids, scheduling, step_size=step_size, seed=batch_schedule_seed(batch))


def _build_payload(
    batch: TrainingBatch,
    loss_family: str,
    advantages: tuple[float, ...] | None,
    scheduling: StepScheduling,
    *,
    step_size: int,
    sampler_topk: bool = False,
) -> dict[str, Any]:
    """Materialize Slime's wire rows in the order ``scheduling`` trains them, in steps of ``step_size`` rollouts.

    Rollout ids group the rows one optimizer step must keep together; the
    schedule may repeat rows (epochs), reorder rollouts (shuffle), and fixes
    the step layout (``external_step_sizes``). See :class:`StepScheduling`.
    """
    samples = trajectories(batch)
    shape_row = resolve_loss_family(loss_family).shape_sample_row
    schedule = _materialize(batch, scheduling, step_size)
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
    if sampler_topk:
        payload.update(sampler_topk_columns(samples, schedule.row_indices))
    payload["external_step_sizes"] = list(schedule.step_sizes)
    return payload
