"""Slime's job preparation: run the job's objective on its data and build Slime's training payload.

The objective's signal (loss family, advantages, next state) and the step
schedule are backend-neutral (``reef.train.algos``). This module turns them
into Slime's own wire rows (``build_slime_payload``), and cuts the payload
into the steps that one Slime rollout trains (``slime_payload_steps``).
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from typing import Any

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.registry import resolve_objective
from reef.train.algos.schedule import MaterializedSchedule, StepScheduling, schedule_job_data
from reef.train.algos.signals import PreparedJob
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.score_centering import sampler_topk_columns
from reef.train.types import JobData, trajectories


def prepare_slime_job(
    data: JobData,
    method: TrainingMethod,
    algorithm_state: Mapping[str, Any],
    scheduling: StepScheduling,
    *,
    step_size: int,
    sampler_topk: bool = False,
) -> PreparedJob:
    """Resolve the job's training objective and produce its complete Slime training payload.

    ``method`` is the objective and learning-rate schedule the job's plan
    selected; the payload records it, so it is part of the job's identity.
    ``scheduling`` is the job's step schedule, which ``job_training`` already
    checked against the objective. ``step_size`` is Slime's
    ``--global-batch-size``, the rollouts in one optimizer step.
    ``sampler_topk`` adds each row's recorded sampler top-K, which score
    centering reads.
    """
    objective = resolve_objective(method.objective)
    signal = objective.prepare(data, algorithm_state)
    if signal.action == "skip":
        return PreparedJob(
            action="skip",
            next_algorithm_state=signal.next_algorithm_state,
            metrics=signal.metrics,
        )
    schedule = schedule_job_data(data, objective.rollout_unit, scheduling, step_size=step_size)
    payload = build_slime_payload(data, objective.loss_family, signal.advantages, schedule, sampler_topk=sampler_topk)
    payload["method"] = method.to_dict()
    metrics = dict(signal.metrics)
    if schedule.epochs > 1:
        metrics.setdefault("epochs", schedule.epochs)
    metrics.setdefault("optimizer_steps", schedule.optimizer_steps)
    if schedule.dropped_rollouts:
        metrics.setdefault("dropped_rollouts", schedule.dropped_rollouts)
    if schedule.tail_rollouts:
        metrics.setdefault("partial_step_rollouts", schedule.tail_rollouts)
    return PreparedJob(
        action="train",
        payload=payload,
        next_algorithm_state=signal.next_algorithm_state,
        metrics=metrics,
    )


def build_slime_payload(
    data: JobData,
    loss_family: str,
    advantages: tuple[float, ...] | None,
    schedule: MaterializedSchedule,
    *,
    sampler_topk: bool = False,
) -> dict[str, Any]:
    """Slime's wire rows of ``data`` in the order ``schedule`` trains them, with its step layout.

    Rollout ids group the rows one optimizer step must keep together; the
    schedule may repeat rows (epochs), reorder rollouts (shuffle), and fixes
    the step layout (``external_step_sizes``). ``advantages`` has one value
    per trajectory, or is ``None``; raise ``ValueError`` for another length.
    Every list-valued field except ``external_step_sizes`` holds one entry per
    wire row, and the rows carry the schedule's rollout ids in emission order.
    """
    samples = trajectories(data)
    shape_row = resolve_loss_family(loss_family).shape_sample_row
    if advantages is not None and len(advantages) != len(samples):
        raise ValueError(f"advantages length {len(advantages)} does not match sample count {len(samples)}")
    payload: dict[str, Any] = {
        "samples": [shape_row(samples[row]) for row in schedule.row_indices],
        "rollout_ids": list(schedule.rollout_ids),
        "loss": loss_family,
        # Each wire row's producing versions, in the schedule's row order: a schedule may
        # repeat (epochs) and reorder (shuffle) rows. They are training metadata, not a check;
        # the record-to-record stage dropped stale samples before the data formed.
        "producing_runtime_load_ids": [samples[row].training.get("runtime_load_id") for row in schedule.row_indices],
    }
    spans = [
        [
            {"start": span.start, "end": span.end, "runtime_load_id": span.runtime_load_id}
            for span in parse_runtime_load_spans(samples[row].training.get("runtime_load_spans", []))
        ]
        for row in schedule.row_indices
    ]
    if any(spans):
        payload["producing_runtime_load_spans"] = spans
    if advantages is not None:
        payload["advantages"] = [advantages[row] for row in schedule.row_indices]
    if sampler_topk:
        payload.update(sampler_topk_columns(samples, schedule.row_indices))
    payload["external_step_sizes"] = list(schedule.step_sizes)
    return payload


def slime_payload_steps(payload: Mapping[str, Any], first: int, last: int) -> dict[str, Any]:
    """The payload of optimizer steps ``first`` up to ``last`` (exclusive) of ``payload``, as one rollout trains it.

    ``payload`` is a payload that ``build_slime_payload`` made. The result
    holds the rows of those steps in order, with rollout ids that start at 0,
    the sizes of those steps, and every field that is not per row. Raise
    ``ValueError`` unless ``0 <= first < last`` and ``last`` is at most the
    payload's step count.
    """
    step_sizes = payload["external_step_sizes"]
    if isinstance(first, bool) or isinstance(last, bool) or not 0 <= first < last <= len(step_sizes):
        raise ValueError(f"steps {first} to {last} are not a range of the payload's {len(step_sizes)} steps")
    # Step k holds the rollouts numbered from sum(step_sizes[:k]), and the rows carry the ids in order.
    lowest, highest = sum(step_sizes[:first]), sum(step_sizes[:last])
    rollout_ids = payload["rollout_ids"]
    begin, end = bisect.bisect_left(rollout_ids, lowest), bisect.bisect_left(rollout_ids, highest)
    steps = {
        key: value[begin:end] if isinstance(value, list) and key != "external_step_sizes" else value
        for key, value in payload.items()
    }
    steps["rollout_ids"] = [rollout_id - lowest for rollout_id in rollout_ids[begin:end]]
    steps["external_step_sizes"] = list(step_sizes[first:last])
    return steps
