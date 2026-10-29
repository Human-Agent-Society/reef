"""The built-in SFT-then-RL recipe: held samples train first, then imported train tasks as RL task jobs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.support.runtime_stubs import StubTrainingComponent
from tests.support.task_service_stubs import task_for

from reef.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.job_data import JobData, JobOutcome, TaskSample, TrajectorySample
from reef.core.reports import TeacherContextReport
from reef.core.tasks import write_harbor_task
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.errors import RecipeConfigError
from reef.recipe.sft import SFTRecord2Record
from reef.recipe.sft_then_rl import SFTThenRLJobData, SFTThenRLJobSettings, SFTThenRLRecipe
from reef.record2sample import NoSamples
from reef.runtime.jobs import TrainingComponent
from reef.storage.commits import CommitRecord
from reef.storage.samples import SampleStore
from reef.storage.scenario import ScenarioStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import PipelineContext
from reef.train.pipeline import TrainingPipeline

ROLLOUT = {"rollouts_per_task": 2}
RUNTIME = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000")


def atif(session: str) -> TrajectorySample:
    return TrajectorySample(
        {
            "schema_version": "ATIF-v1.4",
            "session_id": session,
            "agent": {"name": "fixture", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": session}],
        }
    )


class RenderingSamples(NoSamples):
    """Renders an uploaded trajectory as it is, so the test needs no tokenizer."""

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        return trajectory


def import_tasks(samples: SampleStore, root: Path, split: str, *names: str) -> None:
    directories = tuple(
        write_harbor_task(task_for(name, "record-1", 9000 + port + 10 * len(name)), root / split)
        for port, name in enumerate(names)
    )
    samples.add_tasks(directories, "train" if split == "train" else "eval")


def commit(session: ScenarioStore, consumed: frozenset[str]) -> None:
    """Commit the next step, listing ``consumed``, as the trainer does when it submits a job."""
    history = session.history()
    step = history[-1].step + 1 if history else 1
    session.commit_step(
        expected_step=step - 1,
        commit=CommitRecord(
            scenario="support",
            step=step,
            artifact_ref=ArtifactRef(f"content:{step}", f"release:{step}", "base"),
            checkpoint=True,
            algorithm_state={},
            high_water_sequence=0,
            high_water_offset=0,
            consumed_ids=consumed,
        ),
    )


def pipeline_for(
    samples: SampleStore, *, tasks_per_job: int = 2, job_samples: int = 2, training_mode: str = "auto"
) -> TrainingPipeline:
    context = PipelineContext("support", TeacherContextReport, samples=samples, training_mode=training_mode)
    settings = SFTThenRLJobSettings(job_samples=job_samples, tasks_per_job=tasks_per_job)
    return TrainingPipeline(
        context,
        record2record=SFTRecord2Record(context),
        record2sample=RenderingSamples(),
        sample2job=SFTThenRLJobData(context, settings),
    )


def task_names(data: JobData) -> list[str]:
    return [item.metadata["task"]["name"] for item in data.samples if isinstance(item, TaskSample)]


def test_held_trajectories_train_before_the_imported_train_tasks_and_each_task_trains_once(tmp_path: Path) -> None:
    session = SQLiteScenarioStorage(tmp_path / "records").open("support")
    samples = session.samples
    import_tasks(samples, tmp_path / "tasks", "train", "task-a", "task-b", "task-c")
    import_tasks(samples, tmp_path / "tasks", "eval", "held-out")
    pipeline = pipeline_for(samples)
    pipeline.push_trajectory("t1", atif("t1"))
    pipeline.push_trajectory("t2", atif("t2"))

    # Two uploads make SFT data ready, so they go first although tasks wait.
    sft = pipeline.build_job_data(2)
    assert [type(item) for item in sft.samples] == [TrajectorySample, TrajectorySample]
    assert pipeline.acknowledge(sft.data_id) == {"sample:t1", "sample:t2"}

    first = pipeline.build_job_data(2)
    assert task_names(first) == ["task-a", "task-b"]
    assert all(item.task_path.is_dir() for item in first.samples if isinstance(item, TaskSample))
    consumed = pipeline.acknowledge(first.data_id)
    assert consumed == {"sample:task-a", "sample:task-b"}
    commit(session, consumed)
    pipeline.release_records(frozenset(consumed))
    assert {task.sample_id: task.train_counts for task in samples.tasks(split="train")} == {
        "task-a": {"": 1},
        "task-b": {"": 1},
        "task-c": {},
    }
    pipeline.job_ended(JobOutcome("job-1", "stub", "finished"))

    # A task uploaded later joins the pool; the eval task never trains.
    import_tasks(samples, tmp_path / "tasks", "train", "task-d")
    second = pipeline.build_job_data(2)
    assert task_names(second) == ["task-c", "task-d"]
    consumed = pipeline.acknowledge(second.data_id)
    commit(session, consumed)
    pipeline.release_records(frozenset(consumed))
    assert not pipeline_for(samples).ready(2), "a restart pools only the tasks that did not train"
    pipeline.close()
    session.close()


def test_a_task_whose_submission_did_not_commit_trains_again_after_a_restart(tmp_path: Path) -> None:
    session = SQLiteScenarioStorage(tmp_path / "records").open("support")
    import_tasks(session.samples, tmp_path / "tasks", "train", "task-a")
    pipeline = pipeline_for(session.samples)
    data = pipeline.build_job_data(2)
    pipeline.acknowledge(data.data_id)
    restarted = pipeline_for(session.samples)
    assert task_names(restarted.build_job_data(2)) == ["task-a"]
    session.close()


def instruction(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="support",
        request_type=RequestType.TRAIN,
        agent_record_id=record_id,
        payload={"text": record_id, "session": "session-1", "release_id": "release-1"},
    )


def test_in_manual_mode_an_imported_task_job_waits_for_an_instruction_and_consumes_it(tmp_path: Path) -> None:
    samples = SQLiteScenarioStorage(tmp_path / "records").open("support").samples
    import_tasks(samples, tmp_path / "tasks", "train", "task-a")
    pipeline = pipeline_for(samples, training_mode="manual")
    assert not pipeline.ready(2), "manual mode starts no task job without an instruction"
    pipeline.push(instruction("train-1"))
    data = pipeline.build_job_data(2)
    assert task_names(data) == ["task-a"]
    assert data.request is not None and data.request.id == "train-1"
    assert "train-1" in pipeline.acknowledge(data.data_id)
    assert pipeline.buffered_requests() == 0
    pipeline.close()


def test_a_task_job_gets_the_rl_parameters_and_any_other_job_the_sft_parameters(tmp_path: Path) -> None:
    built = recipe(task_job_parameters={"rollout": ROLLOUT})
    samples = SQLiteScenarioStorage(None).open("support").samples
    context = PipelineContext(
        "support", TeacherContextReport, samples=samples, job_parameters=built.parameters_by_kind()
    )
    stage = SFTThenRLJobData(context, SFTThenRLJobSettings(job_samples=2))
    sft, rl = built.parameters_by_kind()["trajectory"], built.parameters_by_kind()["task"]
    task = TaskSample(tmp_path)
    assert stage.job_parameters(JobData("support:data:1", (atif("t1"),))) == sft
    assert stage.job_parameters(JobData("support:data:2", (task, replace(task, task_path=tmp_path / "b")))) == rl
    with pytest.raises(ValueError, match="holds both trajectories and tasks"):
        stage.job_parameters(JobData("support:data:3", (atif("t1"), task)))


def recipe(**config: Any) -> SFTThenRLRecipe:
    return SFTThenRLRecipe(StubTrainingComponent(), runtime=RUNTIME, **config)


def test_the_recipe_trains_sft_jobs_with_sft_and_task_jobs_with_task_group() -> None:
    built = recipe(job_parameters={"save_interval": 4}, task_job_parameters={"rollout": ROLLOUT})
    parameters = built.parameters_by_kind()
    assert parameters["trajectory"] == {
        "save_interval": 4,
        "method": {"objective": "reef.recipe.sft:SftObjective", "learning_rate_schedule": None},
    }
    assert parameters["task"] == {
        "rollout": ROLLOUT,
        "method": {"objective": "task_group", "learning_rate_schedule": None},
    }
    assert built.pipeline_spec().sample2job is SFTThenRLJobData and SFTThenRLJobData.accepts_imported_tasks


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "needs recipe.config.task_job_parameters"),
        ({"task_job_parameters": {"rollout": ROLLOUT, "method": "x"}}, "cannot set method"),
        ({"task_job_parameters": {}}, "needs the rollout parameter as an object"),
    ],
)
def test_the_recipe_refuses_task_job_parameters_that_cannot_run(config: dict[str, Any], message: str) -> None:
    with pytest.raises(RecipeConfigError, match=message):
        recipe(**config)


def test_the_recipe_refuses_a_training_component_that_runs_no_task_jobs() -> None:
    class TrajectoryOnly(StubTrainingComponent):
        check_task_jobs = TrainingComponent.check_task_jobs

    with pytest.raises(RecipeConfigError, match="the stub training component trains no task jobs"):
        SFTThenRLRecipe(TrajectoryOnly(), runtime=RUNTIME, task_job_parameters={"rollout": ROLLOUT})


def test_tasks_per_job_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tasks_per_job must be a positive int"):
        SFTThenRLJobSettings(tasks_per_job=0)
