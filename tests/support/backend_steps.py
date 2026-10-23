"""Prepare one training step with the Slime or Tinker payload builder, for tests of objectives and wire rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.support.training_steps import PreparedStep

from reef.core.batches import StepScheduling, TrainingBatch
from reef.runtime.interfaces import TrainingMethod
from reef.train.runtime_backend import prepare_training_job
from reef.train.slime_backend.reef_adapters.preparation import build_slime_payload
from reef.train.tinker_backend.preparation import build_tinker_payload


def prepare_slime_step(
    batch: TrainingBatch,
    method: TrainingMethod,
    algorithm_state: Mapping[str, Any],
    scheduling: StepScheduling,
    *,
    step_size: int,
    sampler_topk: bool = False,
) -> PreparedStep:
    signal, job = prepare_training_job(batch, method, algorithm_state, scheduling)
    if job is None:
        return PreparedStep("skip", signal.next_algorithm_state, signal.metrics, None)
    built = build_slime_payload(job, step_size=step_size, sampler_topk=sampler_topk)
    return PreparedStep("train", signal.next_algorithm_state, {**built.metrics, **signal.metrics}, built.payload)


def prepare_tinker_step(
    batch: TrainingBatch,
    method: TrainingMethod,
    algorithm_state: Mapping[str, Any],
    scheduling: StepScheduling,
    *,
    batch_size: int,
    kl_coef: float,
    runtime_load_id: str | None = None,
) -> PreparedStep:
    signal, job = prepare_training_job(batch, method, algorithm_state, scheduling)
    if job is None:
        return PreparedStep("skip", signal.next_algorithm_state, signal.metrics, None)
    built = build_tinker_payload(job, batch_size=batch_size, kl_coef=kl_coef, runtime_load_id=runtime_load_id)
    return PreparedStep("train", signal.next_algorithm_state, {**built.metrics, **signal.metrics}, built.payload)
