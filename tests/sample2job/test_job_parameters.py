"""A sample-to-job stage gives each job the parameters of its samples' kind, with its request's merged in."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.core.job_data import JobData, TaskSample, TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.core.training_request import TrainingRequest
from reef.pipeline_context import PipelineContext
from reef.sample2job import DistillJobData

TRAJECTORY = TrajectorySample({"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1}]})
TASK = TaskSample(Path("task-a"))


def stage(job_parameters) -> DistillJobData:
    return DistillJobData(PipelineContext("s", job_parameters=job_parameters), None, 1)


def test_a_job_gets_the_parameters_of_its_samples_kind() -> None:
    parameters = {"trajectory": {"objective": "sft"}, "task": {"objective": "provided:Objective"}}
    jobs = stage(parameters)
    assert jobs.job_parameters(JobData("s:1", (TRAJECTORY,))) == {"objective": "sft"}
    assert jobs.job_parameters(JobData("s:2", (TASK,))) == {"objective": "provided:Objective"}
    # A harness step's data can hold no sample; it takes no parameters.
    assert jobs.job_parameters(JobData("s:3")) == {}
    assert jobs.experiment_config() == {"job_parameters": parameters}


def training_request(**parameters: object) -> TrainingRequest:
    payload = {"text": "train", "session": "s", "artifact_id": "r", "parameters": parameters}
    return TrainingRequest.from_dict(payload, request_id="train-1")


def test_a_requests_parameters_merge_into_the_parameters_of_its_kind() -> None:
    recipe = {
        "objective": "sft",
        "learning_rate": 1e-5,
        "slime": {"seq-length": "32768", "tensor-model-parallel-size": "4"},
        "env": {"TAGS": ["a", "b"]},
    }
    jobs = stage({"trajectory": recipe})
    overrides = {"learning_rate": 2e-5, "batch_size": 8, "slime": {"seq-length": "131072"}, "env": {"TAGS": ["c"]}}
    data = JobData("s:request:train-1", (TRAJECTORY,), request=training_request(**overrides))
    # Nested objects merge key by key; every other value of the request wins, a list included.
    assert jobs.job_parameters(data) == {
        "objective": "sft",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "slime": {"seq-length": "131072", "tensor-model-parallel-size": "4"},
        "env": {"TAGS": ["c"]},
    }
    # A kind that the recipe gives no parameters trains with the request's alone.
    rl = {"objective": "provided:Objective", "rollout": {"rollouts_per_task": 4}}
    assert jobs.job_parameters(JobData("s:request:train-1", (TASK,), request=training_request(**rl))) == rl


def test_mixed_kinds_and_a_kind_without_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="both trajectories and tasks"):
        stage({"trajectory": {}, "task": {}}).job_parameters(JobData("s:1", (TRAJECTORY, TASK)))
    with pytest.raises(ValueError, match="no job parameters for a job of task samples"):
        stage({"trajectory": {}}).job_parameters(JobData("s:1", (TASK,)))


def test_the_context_keeps_its_own_copy_of_json_parameters_by_sample_kind() -> None:
    parameters = {"objective": "sft"}
    context = PipelineContext("s", job_parameters={"trajectory": parameters})
    parameters["objective"] = "changed"
    assert context.job_parameters == {"trajectory": {"objective": "sft"}}
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
