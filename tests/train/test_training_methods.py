"""The per-job training method: learning-rate schedules, their progress, and the recipe's method of each job."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.reef_recipe.test_recipe_config_fields import StagedRecipe
from tests.support._trajectories import policy_trajectory
from tests.support.runtime_stubs import StubTrainingComponent, runtime_bindings

from reef.core.job_data import JobOutcome, TaskSample
from reef.runtime.interfaces import (
    LearningRateSchedule,
    LearningRateScheduleState,
    TrainingMethod,
    resolve_learning_rate_schedule_state,
)
from reef.runtime.jobs import job_id_for
from reef.storage.samples import SampleKind
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import PipelineConfig
from reef.train.types import JobData

SFT = LearningRateSchedule("sft", 2e-5, warmup_steps=4, decay_style="cosine", decay_steps=8, min_learning_rate=2e-6)
RL = LearningRateSchedule("rl", 1e-6, warmup_steps=2)
#: The staged recipe's stages with jobs of one sample.
ONE_SAMPLE_JOBS = PipelineConfig.from_setting(
    {"sample2job": {"config": {"job_samples": 1}}}, spec=StagedRecipe.pipeline_spec(), environ={}
)


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
    data = {"scenario": "math", "items": [["a", [1, 2], [1], [-0.1], 1.0]]}

    def identity(value: TrainingMethod) -> str:
        return job_id_for("stub", None, data, {"method": value.to_dict()})

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


def batch(name: str) -> JobData:
    return JobData(name, (policy_trajectory(f"s-{name}", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, "v1"),))


def task(name: str) -> JobData:
    return JobData(name, (TaskSample(Path(name)),))


@pytest.mark.unit
def test_a_single_method_recipe_trains_every_job_with_its_spec_objective() -> None:
    recipe = StagedRecipe(**runtime_bindings(), pipeline_stages=ONE_SAMPLE_JOBS)
    sft = {"method": TrainingMethod("sft").to_dict()}
    assert recipe.parameters_by_kind() == {"trajectory": sft, "task": sft}
    trainer = recipe.build("scenario", SQLiteRecordStore())
    stage = trainer.pipeline.sample2job
    assert stage.job_parameters(batch("b")) == stage.job_parameters(task("t")) == sft
    assert stage.experiment_config() == {"job_parameters": {"trajectory": sft, "task": sft}}
    assert sft["method"]["learning_rate_schedule"] is None


@pytest.mark.unit
def test_a_recipe_method_for_each_kind_reaches_the_component_through_the_default_build() -> None:
    class SupervisedThenPolicy(StagedRecipe):
        """Supervised trajectories with a warmup, then policy training on tasks with a lower rate."""

        def parameters_by_kind(self) -> Mapping[SampleKind, Mapping[str, Any]]:
            return {
                "trajectory": {"method": TrainingMethod("sft", SFT).to_dict()},
                "task": {"method": TrainingMethod("rl-objective", RL).to_dict()},
            }

    component = StubTrainingComponent()
    trainer = SupervisedThenPolicy(**runtime_bindings(component), pipeline_stages=ONE_SAMPLE_JOBS).build(
        "scenario", SQLiteRecordStore()
    )
    loop, stage = trainer.job_loop, trainer.pipeline.sample2job
    assert loop is not None
    loop.submit(JobLoopState(component.base_start), batch("first"), stage.job_parameters(batch("first")))
    finished = JobLoopState(component.base_start, finished_jobs=1, last_job=JobOutcome("j", "stub", "finished"))
    loop.submit(finished, task("second"), stage.job_parameters(task("second")))
    assert [TrainingMethod.from_dict(job.parameters["method"]) for job in component.submitted] == [
        TrainingMethod("sft", SFT),
        TrainingMethod("rl-objective", RL),
    ]


@pytest.mark.unit
def test_the_job_loop_refuses_a_value_that_is_no_training_component() -> None:
    with pytest.raises(TypeError, match="component must be a TrainingComponent"):
        JobLoop("sft", scenario="s")  # type: ignore[arg-type]
