"""The per-job training method: learning-rate schedules, their progress, and the recipe's selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from reef.runtime.interfaces import (
    LearningRateSchedule,
    LearningRateScheduleState,
    PreparedTrainingStep,
    TrainingMethod,
    resolve_learning_rate_schedule_state,
)
from reef.runtime.scheduler import training_job_id
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.algos import StepScheduling
from reef.train.runtime_backend import FixedTrainingMethodSelector, RuntimeCandidateBackend, TrainingMethodSelector
from reef.train.types import TrainingBatch

from .test_recipe_config_fields import ConfiguredRecipe

SFT = LearningRateSchedule("sft", 2e-5, warmup_steps=4, decay_style="cosine", decay_steps=8, min_learning_rate=2e-6)
RL = LearningRateSchedule("rl", 1e-6, warmup_steps=2)


@pytest.mark.unit
def test_schedule_warms_up_then_decays_to_its_minimum_counted_in_optimizer_steps() -> None:
    # Warmup from the initial rate, inclusive of the peak at the last warmup step.
    assert [SFT.learning_rate(step) for step in range(5)] == pytest.approx([0.0, 5e-6, 1e-5, 1.5e-5, 2e-5])
    # Cosine over the next eight steps: halfway is the midpoint, the end is the minimum.
    assert SFT.learning_rate(8) == pytest.approx(2e-6 + 0.5 * (2e-5 - 2e-6))
    assert SFT.learning_rate(12) == pytest.approx(2e-6)
    assert SFT.learning_rate(500) == pytest.approx(2e-6)
    linear = LearningRateSchedule("linear", 1.0, decay_style="linear", decay_steps=4, min_learning_rate=0.2)
    assert [linear.learning_rate(step) for step in range(6)] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.2])
    constant = LearningRateSchedule("flat", 3e-6, warmup_steps=3, initial_learning_rate=1e-6)
    assert constant.learning_rate(0) == pytest.approx(1e-6)
    assert constant.learning_rate(3) == constant.learning_rate(1000) == pytest.approx(3e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"name": " "}, "non-empty"),
        ({"peak_learning_rate": 0.0}, "positive"),
        ({"peak_learning_rate": math.inf}, "finite"),
        ({"min_learning_rate": 1.0}, "must not exceed the peak"),
        ({"warmup_steps": -1}, "non-negative integer"),
        ({"warmup_steps": True}, "non-negative integer"),
        ({"decay_style": "step"}, "decay_style must be one of"),
        ({"decay_style": "cosine"}, "needs positive decay_steps"),
        ({"decay_steps": 10}, "constant LearningRateSchedule takes no decay_steps"),
    ],
)
def test_schedule_rejects_invalid_values(values: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {"name": "schedule", "peak_learning_rate": 1e-5, **values}
    with pytest.raises(ValueError, match=message):
        LearningRateSchedule(**arguments)


@pytest.mark.unit
def test_a_job_continues_the_active_schedule_and_starts_another_at_step_zero() -> None:
    active = LearningRateScheduleState(SFT, completed_steps=6)
    # No request keeps the active schedule (or the backend's configured rate).
    assert resolve_learning_rate_schedule_state(active, None) is active
    assert resolve_learning_rate_schedule_state(None, None) is None
    # Selecting the active schedule again, as every retry and every restart does, continues it.
    assert resolve_learning_rate_schedule_state(active, SFT) is active
    # Another schedule starts fresh; the same curve under another name restarts it on purpose.
    assert resolve_learning_rate_schedule_state(active, RL) == LearningRateScheduleState(RL, 0)
    renamed = LearningRateSchedule.from_dict({**SFT.to_dict(), "name": "sft-again"})
    assert resolve_learning_rate_schedule_state(active, renamed) == LearningRateScheduleState(renamed, 0)

    assert active.learning_rates(3) == pytest.approx(tuple(SFT.learning_rate(step) for step in (6, 7, 8)))
    assert active.advanced(3) == LearningRateScheduleState(SFT, 9)
    assert LearningRateScheduleState.from_dict(active.advanced(3).to_dict()) == active.advanced(3)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        LearningRateScheduleState(SFT, -1)


@pytest.mark.unit
def test_the_method_takes_part_in_the_job_identity() -> None:
    method = TrainingMethod("sft", SFT)
    assert TrainingMethod.from_dict(method.to_dict()) == method
    payload = {"samples": [["a", [1, 2], [1], [-0.1], 1.0]], "rollout_ids": [0], "loss": "sft"}

    def identity(value: TrainingMethod) -> str:
        return training_job_id({**payload, "method": value.to_dict(), "scenario_step": 3})

    # A retry selects the same method: the same job.
    assert identity(TrainingMethod("sft", SFT)) == identity(method)
    # Another objective or schedule over the same rows is another job.
    assert identity(TrainingMethod("sft")) != identity(method)
    assert identity(TrainingMethod("sft", RL)) != identity(method)
    assert identity(TrainingMethod("other", SFT)) != identity(method)
    with pytest.raises(ValueError, match="non-empty objective"):
        TrainingMethod("")
    with pytest.raises(TypeError, match="LearningRateSchedule or None"):
        TrainingMethod("sft", SFT.to_dict())  # type: ignore[arg-type]


class RecordingRuntime(StubTrainingRuntime):
    """Record the method of every prepared job and skip it, advancing the objective's step counter."""

    def __init__(self) -> None:
        super().__init__()
        self.prepared: list[tuple[TrainingMethod, dict[str, Any]]] = []

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        self.prepared.append((method, dict(algorithm_state)))
        return PreparedTrainingStep("skip", {"steps": algorithm_state.get("steps", 0) + 1}, {"steps": 1})


class SupervisedThenPolicy(TrainingMethodSelector):
    """Supervised warmup for the first jobs, then policy training with a fresh warmup and lower rate."""

    def __init__(self, supervised_steps: int) -> None:
        self.supervised_steps = supervised_steps

    def select(self, batch: TrainingBatch, algorithm_state: Mapping[str, Any]) -> TrainingMethod:
        if algorithm_state.get("steps", 0) < self.supervised_steps:
            return TrainingMethod("sft", SFT)
        return TrainingMethod("rl-objective", RL)


@pytest.mark.unit
def test_the_backend_selects_every_jobs_method_from_the_committed_state() -> None:
    runtime = RecordingRuntime()
    backend = RuntimeCandidateBackend(
        runtime, SupervisedThenPolicy(supervised_steps=2), StepScheduling(), inference_runtime=runtime.inference
    )
    state: Mapping[str, Any] = {}
    for step in range(4):
        prepared = backend.prepare_step(TrainingBatch(f"batch-{step}"), state, step)
        assert prepared.outcome == "skip"
        state = prepared.state
    assert [method for method, _ in runtime.prepared] == [
        TrainingMethod("sft", SFT),
        TrainingMethod("sft", SFT),
        TrainingMethod("rl-objective", RL),
        TrainingMethod("rl-objective", RL),
    ]
    # The selected method is logged with the job's metrics.
    assert prepared.metrics["training_method"] == TrainingMethod("rl-objective", RL).to_dict()
    assert backend.experiment_config()["training_method_selector"].endswith("SupervisedThenPolicy")

    # A retry prepares the kept batch against the same committed state and selects the same method.
    retry = backend.prepare_step(TrainingBatch("batch-1"), {"steps": 1}, 1)
    assert runtime.prepared[-1][0] == TrainingMethod("sft", SFT)
    assert retry.metrics["training_method"] == TrainingMethod("sft", SFT).to_dict()


@pytest.mark.unit
def test_the_backend_refuses_a_selector_that_returns_no_method() -> None:
    class Broken(TrainingMethodSelector):
        def select(self, batch: TrainingBatch, algorithm_state: Mapping[str, Any]) -> TrainingMethod:
            return "sft"  # type: ignore[return-value]

    runtime = RecordingRuntime()
    with pytest.raises(TypeError, match="TrainingMethodSelector"):
        RuntimeCandidateBackend(runtime, "sft", StepScheduling(), inference_runtime=runtime.inference)  # type: ignore[arg-type]
    backend = RuntimeCandidateBackend(runtime, Broken(), StepScheduling(), inference_runtime=runtime.inference)
    with pytest.raises(TypeError, match=r"Broken\.select must return a TrainingMethod"):
        backend.prepare_step(TrainingBatch("batch"), {}, 0)
    assert runtime.prepared == []


@pytest.mark.unit
def test_a_single_method_recipe_trains_every_job_with_its_spec_objective() -> None:
    runtime = RecordingRuntime()
    recipe = ConfiguredRecipe(training_runtime=runtime, runtime=runtime.inference)
    selector = recipe.training_method_selector()
    assert isinstance(selector, FixedTrainingMethodSelector)
    assert selector.select(TrainingBatch("batch"), {"steps": 40}) == TrainingMethod("sft")
    trainer = recipe.build("scenario", SQLiteRecordStore())
    backend = trainer.candidate_backend
    assert isinstance(backend, RuntimeCandidateBackend)
    assert backend.experiment_config()["objective"] == "sft"
    assert "learning_rate_schedule" not in backend.experiment_config()


@pytest.mark.unit
def test_a_recipe_selector_reaches_the_runtime_through_the_default_build() -> None:
    class SwitchingRecipe(ConfiguredRecipe):
        def training_method_selector(self) -> TrainingMethodSelector:
            return SupervisedThenPolicy(supervised_steps=self.batch_size)

    runtime = RecordingRuntime()
    trainer = SwitchingRecipe(training_runtime=runtime, runtime=runtime.inference, batch_size=1).build(
        "scenario", SQLiteRecordStore()
    )
    backend = trainer.candidate_backend
    assert isinstance(backend, RuntimeCandidateBackend)
    backend.prepare_step(TrainingBatch("first"), {}, 0)
    backend.prepare_step(TrainingBatch("second"), {"steps": 1}, 1)
    assert [method.objective for method, _ in runtime.prepared] == ["sft", "rl-objective"]
