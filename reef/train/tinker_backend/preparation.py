"""Turn backend-neutral algorithm signals into Tinker optimizer batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.core.batches import TrainingBatch, trajectories
from reef.runtime.base import PreparedTrainingStep
from reef.train.algos.registry import resolve_preparer
from reef.train.algos.schedule import materialize_schedule, schedule_seed
from reef.train.tinker_backend.losses import TokenRow, resolve_tinker_loss


def prepare_tinker_step(
    batch: TrainingBatch,
    preparer: str,
    state: Mapping[str, Any],
    scenario_step: int,
    *,
    runtime_load_id: str,
    batch_size: int,
) -> PreparedTrainingStep:
    signal = resolve_preparer(preparer)(batch, state)
    if signal.action == "skip":
        return PreparedTrainingStep("skip", signal.next_algorithm_state, signal.metrics)
    resolve_tinker_loss(signal.loss_family)
    items = trajectories(batch)
    if signal.advantages is None or len(signal.advantages) != len(items):
        raise ValueError("Tinker policy training requires one advantage per trajectory")
    rows = []
    stale = False
    groups: dict[str, int] = {}
    rollout_ids = []
    for index, (item, advantage) in enumerate(zip(items, signal.advantages, strict=True)):
        training = item.training
        if training.get("runtime_load_id") != runtime_load_id:
            stale = True
        spans = training.get("runtime_load_spans")
        if spans:
            parsed_spans = parse_runtime_load_spans(spans, response_length=len(training.get("loss_mask", ())))
            if any(span.runtime_load_id != runtime_load_id for span in parsed_spans):
                stale = True
        rows.append(asdict(TokenRow.from_item(item, advantage)))
        key = f"group:{item.group_id}" if item.group_id is not None else f"row:{index}"
        rollout_ids.append(index if signal.scheduling.unit == "sample" else groups.setdefault(key, len(groups)))
    scheduling = signal.scheduling
    if scheduling.batch_size == "configured":
        if scheduling.remainder == "error" and batch_size > len(set(rollout_ids)):
            raise ValueError("Tinker configured batch_size exceeds the available comparison sets")
        scheduling = replace(scheduling, batch_size=min(batch_size, len(set(rollout_ids))))
    schedule = materialize_schedule(rollout_ids, scheduling, seed=schedule_seed(batch.batch_id))
    batches: list[list[dict[str, Any]]] = []
    cursor = 0
    next_rollout = 0
    for size in schedule.step_sizes or ():
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
            "scenario_step": scenario_step,
            "source_runtime_load_id": runtime_load_id,
            "loss": signal.loss_family,
            "batches": batches,
            "stale": stale,
        },
    )
