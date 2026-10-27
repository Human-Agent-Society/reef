"""The built-in SFT-then-RL recipe: held samples train first, then imported train tasks as RL task jobs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.support.runtime_stubs import StubTrainingComponent
from tests.support.task_service_stubs import task_for

from reef.core.evaluation import CheckpointCandidate
from reef.core.job_data import JobData, JobOutcome, TaskSample, TrajectorySample
from reef.core.reports import TeacherContextReport
from reef.core.tasks import write_harbor_task
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.errors import RecipeConfigError
from reef.recipe.sft import SFTRecord2Record
from reef.recipe.sft_then_rl import (
    ImportedTaskEvaluation,
    SFTThenRLJobData,
    SFTThenRLJobPlan,
    SFTThenRLJobSettings,
    SFTThenRLRecipe,
)
from reef.record2sample import NoSamples
from reef.runtime.jobs import TrainingComponent
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import IMPORTED_TASK_STORE, TaskStorage
from reef.train import PipelineContext
from reef.train.jobs import JobPlanContext, JobRequest
from reef.train.pipeline import TrainingPipeline

ROLLOUT = {"episodes_per_task": 2}
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


def import_tasks(storage: TaskStorage, root: Path, split: str, *names: str) -> None:
    store = storage.open(IMPORTED_TASK_STORE)
    for port, name in enumerate(names):
        directory = write_harbor_task(task_for(name, "record-1", 9000 + port + 10 * len(name)), root / split)
        store.add_task(directory=directory, split=split)


def pipeline_for(storage: TaskStorage, *, tasks_per_job: int = 2, job_samples: int = 2) -> TrainingPipeline:
    context = PipelineContext("support", TeacherContextReport, task_storage=storage)
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
    storage = SQLiteScenarioStorage(tmp_path / "records").open("support").tasks
    import_tasks(storage, tmp_path / "tasks", "train", "task-a", "task-b", "task-c")
    import_tasks(storage, tmp_path / "tasks", "eval", "held-out")
    pipeline = pipeline_for(storage)
    pipeline.push_trajectory("t1", atif("t1"))
    pipeline.push_trajectory("t2", atif("t2"))

    # Two uploads make SFT data ready, so they go first although tasks wait.
    sft = pipeline.build_job_data(2)
    assert [type(item) for item in sft.samples] == [TrajectorySample, TrajectorySample]
    assert pipeline.acknowledge(sft.data_id) == {"trajectory:t1", "trajectory:t2"}

    first = pipeline.build_job_data(2)
    assert task_names(first) == ["task-a", "task-b"]
    assert all(item.task_path.is_dir() for item in first.samples if isinstance(item, TaskSample))
    consumed = pipeline.acknowledge(first.data_id)
    assert all(identifier.startswith("rollout:support:rollout:") for identifier in consumed) and len(consumed) == 2
    pipeline.release_records(frozenset(consumed))
    store = storage.open(IMPORTED_TASK_STORE)
    assert store.trained_task_names() == {"task-a", "task-b"}
    pipeline.job_ended(JobOutcome("job-1", "stub", "finished"))

    # A task uploaded later joins the pool; the eval task never trains.
    import_tasks(storage, tmp_path / "tasks", "train", "task-d")
    second = pipeline.build_job_data(2)
    assert task_names(second) == ["task-c", "task-d"]
    pipeline.release_records(frozenset(pipeline.acknowledge(second.data_id)))
    assert not pipeline_for(storage).ready(2), "a restart pools only the tasks that did not train"
    pipeline.close()


def test_a_task_whose_submission_did_not_commit_trains_again_after_a_restart(tmp_path: Path) -> None:
    storage = SQLiteScenarioStorage(tmp_path / "records").open("support").tasks
    import_tasks(storage, tmp_path / "tasks", "train", "task-a")
    pipeline = pipeline_for(storage)
    data = pipeline.build_job_data(2)
    pipeline.acknowledge(data.data_id)
    restarted = pipeline_for(storage)
    assert task_names(restarted.build_job_data(2)) == ["task-a"]


def test_the_plan_gives_a_task_job_the_rl_request_and_any_other_job_the_sft_request(tmp_path: Path) -> None:
    sft, rl = JobRequest({"method": "sft"}), JobRequest({"method": "rl", "rollout": ROLLOUT})
    plan = SFTThenRLJobPlan(sft, rl)

    def context(*items: Any) -> JobPlanContext:
        return JobPlanContext(JobData("support:data:1", items), 0, None)

    task = TaskSample(tmp_path)
    assert plan.next_job(context(atif("t1"))) is sft
    assert plan.next_job(context()) is sft, "an instruction without samples is not a task job"
    assert plan.next_job(context(task, replace(task, task_path=tmp_path / "b"))) is rl
    with pytest.raises(ValueError, match="holds both trajectories and tasks"):
        plan.next_job(context(atif("t1"), task))


def recipe(**config: Any) -> SFTThenRLRecipe:
    return SFTThenRLRecipe(StubTrainingComponent(), runtime=RUNTIME, **config)


def test_the_recipe_trains_sft_jobs_with_sft_and_task_jobs_with_task_group() -> None:
    built = recipe(job_parameters={"save_interval": 4}, task_job_parameters={"rollout": ROLLOUT})
    plan = built.job_plan()
    assert isinstance(plan, SFTThenRLJobPlan)
    assert plan.sft.parameters == {
        "save_interval": 4,
        "method": {"objective": "reef.recipe.sft:SftObjective", "learning_rate_schedule": None},
    }
    assert plan.rl.parameters == {
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


def test_every_artifact_is_evaluated_on_the_imported_eval_tasks_with_the_rl_parameters(tmp_path: Path) -> None:
    storage = SQLiteScenarioStorage(tmp_path / "records").open("support").tasks
    built = recipe(task_job_parameters={"rollout": ROLLOUT})
    evaluation = built.artifact_evaluation(storage)
    assert isinstance(evaluation, ImportedTaskEvaluation) and built.artifact_evaluation(None) is None
    candidate = CheckpointCandidate(candidate_id="job-1:4", training_job_id="job-1", step=4, release_path="/r")
    assert evaluation.evaluation_job(candidate) is None, "no eval task, nothing to evaluate on"
    import_tasks(storage, tmp_path / "tasks", "train", "task-a")
    import_tasks(storage, tmp_path / "tasks", "eval", "held-out", "held-out-2")
    job = evaluation.evaluation_job(candidate)
    assert job is not None and task_names(job.data) == ["held-out", "held-out-2"]
    assert job.parameters == {
        "rollout": ROLLOUT,
        "method": {"objective": "task_group", "learning_rate_schedule": None},
        "evaluation_of": "job-1:4",
    }
