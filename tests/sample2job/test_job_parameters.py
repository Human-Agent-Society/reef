"""A sample-to-job stage gives each job the parameters of its samples' kind."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.core.job_data import JobData, TaskSample, TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.pipeline_context import PipelineContext
from reef.sample2job import DistillJobData

TRAJECTORY = TrajectorySample({"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1}]})
TASK = TaskSample(Path("task-a"))


def stage(job_parameters) -> DistillJobData:
    return DistillJobData(PipelineContext("s", job_parameters=job_parameters), None, 1)


def test_a_job_gets_the_parameters_of_its_samples_kind() -> None:
    parameters = {"trajectory": {"method": {"objective": "sft"}}, "task": {"method": {"objective": "task_group"}}}
    jobs = stage(parameters)
    assert jobs.job_parameters(JobData("s:1", (TRAJECTORY,))) == {"method": {"objective": "sft"}}
    assert jobs.job_parameters(JobData("s:2", (TASK,))) == {"method": {"objective": "task_group"}}
    # A harness step's data can hold no sample; it takes no parameters.
    assert jobs.job_parameters(JobData("s:3")) == {}
    assert jobs.experiment_config() == {"job_parameters": parameters}


def test_mixed_kinds_and_a_kind_without_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="both trajectories and tasks"):
        stage({"trajectory": {}, "task": {}}).job_parameters(JobData("s:1", (TRAJECTORY, TASK)))
    with pytest.raises(ValueError, match="no job parameters for a job of task samples"):
        stage({"trajectory": {}}).job_parameters(JobData("s:1", (TASK,)))


def test_the_context_keeps_its_own_copy_of_json_parameters_by_sample_kind() -> None:
    parameters = {"method": {"objective": "sft"}}
    context = PipelineContext("s", job_parameters={"trajectory": parameters})
    parameters["method"] = "changed"
    assert context.job_parameters == {"trajectory": {"method": {"objective": "sft"}}}
    with pytest.raises(ValueError, match="maps sample kinds"):
        PipelineContext("s", job_parameters={"record": {}})
    with pytest.raises(ValueError):
        PipelineContext("s", job_parameters={"trajectory": {"learning_rate": float("nan")}})


def test_a_stage_with_a_job_size_limit_refuses_a_larger_buffer() -> None:
    class OneGridJobData(DistillJobData):
        def __init__(self, context: PipelineContext, buffer_size: int | None) -> None:
            super(DistillJobData, self).__init__(context, buffer_size, 4, max_job_samples=4)

    assert OneGridJobData(PipelineContext("s"), None).buffer_size == 4
    with pytest.raises(PluginConfigError, match="at most 4 samples in one job's data, not buffer_size 8"):
        OneGridJobData(PipelineContext("s"), 8)
