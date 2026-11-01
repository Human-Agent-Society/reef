"""The harness training component: one harness step per training job, its starts, and its state across jobs.

Each job runs on the component's own job thread over the pi recipe of ``tests.support.harness_steps``, whose fake
binary scores rules that hold the marker higher.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_steps import (
    COMPONENT,
    PI_SEED,
    UNCHANGED,
    TreeReleases,
    changed_rules,
    harness_backend,
    harness_recipe,
)

from reef.artifact.files import DirectoryFiles
from reef.core.job_data import JobData
from reef.harness.adapters import get_adapter
from reef.harness.tree import HarnessTree, TreePatch
from reef.recipe.harness import HarnessJobLoop
from reef.runtime.jobs import JobStatus, TrainingJob
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import StepProgress
from reef.train.harness_backend.component import (
    PROGRESS_FILE,
    HarnessTrainingComponent,
    ProgressFile,
    read_step_progress,
)
from reef.train.jobs import JobLoopState, RunningJob

MARKER = TreePatch(changed_rules("marker rules\n"))


def component_of(tmp_path: Path, method) -> HarnessTrainingComponent:
    component = harness_recipe(tmp_path, method).harness_component
    assert isinstance(component, HarnessTrainingComponent)
    return component


def job(component: HarnessTrainingComponent, name: str, *, start: str | None = None, previous: str | None = None):
    """The job ``name`` of scenario ``demo``: one trace from ``start`` (the base by default), after ``previous``."""
    return TrainingJob(
        name,
        component.name,
        component.base_start if start is None else start,
        JobData(name, (recorded_trajectory(f"{name}-trace", {"messages": []}, 0.0),)),
        {
            "component": COMPONENT,
            "head_release_id": "base",
            "evaluation_tasks": None,
            "previous_job": previous,
        },
        scenario="demo",
    )


def ended(component: HarnessTrainingComponent, job_id: str, seconds: float = 30.0) -> JobStatus:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = component.status(job_id)
        if status.state != "running":
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not end in {seconds}s")


@pytest.mark.unit
def test_a_kept_tree_is_the_jobs_checkpoint_and_names_the_parts_it_changed(tmp_path: Path) -> None:
    component = component_of(tmp_path, lambda workspace, services: MARKER)
    component.start_jobs()
    try:
        component.submit(job(component, "one"))
        status = ended(component, "one")
        assert status.state == "finished" and status.metrics == {}
        (checkpoint,) = status.checkpoints
        # The step's metrics are the checkpoint's: the job's end carries none of its own.
        assert checkpoint.metrics["selected"] is True
        assert checkpoint.path == checkpoint.release_path
        assert HarnessTree.read(Path(checkpoint.path)) == PI_SEED.apply(MARKER)
        # The next job starts from the checkpoint and continues the state after this job.
        component.submit(job(component, "two", start=checkpoint.path, previous="one"))
        second = ended(component, "two")
        assert second.state == "finished" and second.checkpoints == ()
        assert second.metrics["skipped"] == UNCHANGED and second.metrics["head_release_id"] == "base"
        assert component.state_after("two")[0] == 2 and component.state_after(None)[0] == 0
    finally:
        component.close()


@pytest.mark.unit
def test_a_checkpoint_commit_names_the_parts_that_its_tree_changed_from_the_jobs_start(tmp_path: Path) -> None:
    component = component_of(tmp_path, lambda workspace, services: MARKER)
    component.start_jobs()
    try:
        component.submit(job(component, "one"))
        (checkpoint,) = ended(component, "one").checkpoints
    finally:
        component.close()
    loop = HarnessJobLoop(
        component,
        scenario="demo",
        releases=ComponentReleases(TreeReleases(PI_SEED), COMPONENT),
        eval_samples=None,
        seed=PI_SEED,
        layout=get_adapter("pi").layout,
    )
    # The activation rule reads the parts: the step rewrote the rules of the base tree.
    published = loop.publish(JobLoopState(component.base_start, RunningJob("one")), checkpoint)
    assert published.result.artifact is not None and published.result.changed_kinds == {"rules"}
    # From a start that holds the same tree, nothing changed.
    again = loop.publish(JobLoopState(checkpoint.path, RunningJob("two")), checkpoint)
    assert again.result.changed_kinds == frozenset()
    # The harness parameters are the loop's own: the stage cannot name them.
    with pytest.raises(ValueError, match="already name"):
        loop.submit(JobLoopState(component.base_start), JobData("three"), {"component": "other"})


@pytest.mark.unit
def test_a_resumed_job_commits_its_settled_step_and_runs_no_second_step(tmp_path: Path) -> None:
    calls = []

    def method(workspace, services):
        calls.append(workspace.step)
        return MARKER

    component = component_of(tmp_path, method)
    component.start_jobs()
    component.submit(job(component, "one"))
    first = ended(component, "one")
    component.close()
    # A restart before the job's end was durable: the job has its outcome, but no result yet.
    (component.job_directory("one") / "result.json").unlink()
    (component.job_directory("one") / "checkpoints.json").unlink()
    restarted = component_of(tmp_path, method)
    restarted.start_jobs()
    try:
        assert ended(restarted, "one").checkpoints == first.checkpoints
        assert calls == [1]
    finally:
        restarted.close()


@pytest.mark.unit
def test_a_failed_step_ends_its_job_with_its_error_and_keeps_the_state(tmp_path: Path) -> None:
    def method(workspace, services):
        raise RuntimeError("the method failed")

    component = component_of(tmp_path, method)
    component.start_jobs()
    try:
        component.submit(job(component, "one"))
        status = ended(component, "one")
        assert status.state == "failed" and status.error == "RuntimeError: the method failed"
        assert status.metrics["skipped"] == "step failed" and status.metrics["error"] == status.error
        # The failed step counts, so the next one claims the next step record; the harness state stays.
        steps, state = component.state_after("one")
        assert steps == 1 and state == component.state_after(None)[1]
        # A job that never settled changed nothing: the state after it is the state after the job it names.
        (component.job_directory("lost")).mkdir()
        assert component.state_after("lost") == component.state_after(None)
    finally:
        component.close()


@pytest.mark.unit
def test_starts_and_jobs_are_checked_before_a_job_runs(tmp_path: Path) -> None:
    component = component_of(tmp_path, lambda workspace, services: None)
    assert component.base_start == f"base:{PI_SEED.checksum()}"
    assert component.start_tree(component.base_start) == PI_SEED
    assert component.release_format is None and component.shares_scenarios
    assert component.base_artifact_source().files == dict(PI_SEED.files)
    # An imported start is a copy of the release's tree, under its checksum, without the repository's own files.
    release = tmp_path / "release"
    PI_SEED.apply(MARKER).write(release)
    # An import's directory holds the tree alone.
    component.check_release(DirectoryFiles(release))
    (release / ".gitattributes").write_text("* filter=lfs\n")
    (release / "reef-artifact.json").write_text("{}\n")
    start = component.import_start(release)
    assert Path(start).name == PI_SEED.apply(MARKER).checksum()
    assert component.start_tree(start) == PI_SEED.apply(MARKER) and component.import_start(release) == start
    broken = tmp_path / "broken"
    HarnessTree({"rules.md": ""}).write(broken)
    with pytest.raises(ValueError, match="fails validation"):
        component.import_start(broken)
    with pytest.raises(ValueError, match="base of another seed tree"):
        component.check_job(job(component, "other", start="base:other"))
    valid = job(component, "valid")
    for name, value, message in (
        ("component", "", "names the scenario component"),
        ("head_release_id", None, "names the release of its start"),
        ("evaluation_tasks", "t", "evaluation_tasks"),
        ("previous_job", "", "previous_job"),
    ):
        broken_job = TrainingJob(
            valid.job_id,
            valid.backend,
            valid.start,
            valid.data,
            {**valid.parameters, name: value},
            scenario="demo",
        )
        with pytest.raises(ValueError, match=message):
            component.check_job(broken_job)
    component.close()


@pytest.mark.unit
def test_the_progress_file_holds_the_running_steps_progress_until_it_ends(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda workspace, services: None)
    progress = StepProgress(request_id="ask-1", phase="proposing", started_at=1.0, step_record=str(tmp_path / "1"))
    path = tmp_path / PROGRESS_FILE
    with ProgressFile(backend, path, interval_seconds=0.01):
        assert read_step_progress(path) is None
        backend.progress = progress
        deadline = time.monotonic() + 5
        while read_step_progress(path) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert read_step_progress(path) == progress
    assert read_step_progress(path) is None
