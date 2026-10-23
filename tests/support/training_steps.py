"""Prepare one training step the way Reef does, for tests of objectives and training runtimes.

Reef's service process runs the objective (``prepare_training_job``), and the
training runtime builds its payload from the resulting ``TrainingJob``. Tests
of an objective and its payload need both, so ``prepare_runtime_step`` chains
them. This module imports no training backend; ``tests.support.backend_steps``
chains the objective with the Slime and Tinker payload builders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from reef.core.batches import StepScheduling, TrainingBatch
from reef.runtime.interfaces import TrainingMethod, TrainingRuntime
from reef.train.runtime_backend import prepare_training_job


@dataclass(frozen=True)
class PreparedStep:
    """One prepared step: the objective's action, state and metrics, and the payload when it trains.

    ``payload`` is ``None`` exactly when ``action`` is ``"skip"``. ``metrics``
    merges the payload's layout metrics under the objective's metrics, as
    ``RuntimeCandidateBackend`` does.
    """

    action: Literal["train", "skip"]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any]
    payload: Mapping[str, Any] | None


def prepare_runtime_step(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    method: TrainingMethod,
    algorithm_state: Mapping[str, Any],
    scheduling: StepScheduling,
    scenario_step: int,
    *,
    parent_runtime_load_id: str | None = None,
) -> PreparedStep:
    signal, job = prepare_training_job(batch, method, algorithm_state, scheduling)
    if job is None:
        return PreparedStep("skip", signal.next_algorithm_state, signal.metrics, None)
    built = runtime.build_training_payload(job, scenario_step, parent_runtime_load_id=parent_runtime_load_id)
    return PreparedStep("train", signal.next_algorithm_state, {**built.metrics, **signal.metrics}, built.payload)
