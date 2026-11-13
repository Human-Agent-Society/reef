"""The harness training component: each training job as one run of the method, its starts, and its state across jobs.

Each job runs on the component's own job thread over the pi recipe of ``tests.support.harness_jobs``, whose fake
binary scores rules that hold the marker higher.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.support.harness_jobs import COMPONENT, PI_SEED, UNCHANGED, changed_rules, harness_recipe
from tests.support.job_data import job_data_ref
from tests.support.trajectories import recorded_trajectory

from reef.artifact.files import DirectoryFiles
from reef.core.job_data import JobData
from reef.harness.tree import HarnessTree, TreePatch
from reef.train.component import JobStatus, TrainingArtifactLayout, TrainingJob
from reef.train.harness_backend.backend import JobProgress
from reef.train.harness_backend.component import HarnessTrainingComponent

MARKER = TreePatch(changed_rules("marker rules\n"))


def component_of(tmp_path: Path, method) -> HarnessTrainingComponent:
    _, live = harness_recipe(tmp_path, method)
    component = live.harness_component
    assert isinstance(component, HarnessTrainingComponent)
    return component


def seed_start(component: HarnessTrainingComponent, tmp_path: Path) -> str:
    """The start of a job from the seed tree: the component's import of a snapshot that holds the seed."""
    artifact_dir = tmp_path / "seed-snapshot"
    if not artifact_dir.is_dir():
        PI_SEED.write(artifact_dir)
    return component.import_start(artifact_dir)


def job(component: HarnessTrainingComponent, name: str, *, start: str, previous: str | None = None):
    """The job ``name`` of scenario ``demo``: one trace from ``start``, after ``previous``."""
    return TrainingJob(
        name,
        component.name,
        start,
        job_data_ref(JobData(name, (recorded_trajectory(f"{name}-trace", {"messages": []}, 0.0),))),
        {"evaluation_tasks": None},
        scenario="demo",
        component=COMPONENT,
        previous_job_id=previous,
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
def test_a_kept_tree_is_the_jobs_checkpoint_and_names_the_parts_it_changed(
    tmp_path: Path,
) -> None:
    component = component_of(tmp_path, lambda workspace, services: MARKER)
    component.start_jobs()
    try:
        first = job(component, "one", start=seed_start(component, tmp_path))
        component.submit(first)
        status = ended(component, "one")
        assert status.state == "finished" and status.metrics == {}
        (checkpoint,) = status.checkpoints
        # The job's metrics are the checkpoint's: the job's end carries none of its own.
        assert checkpoint.metrics["selected"] is True
        # The tree's checksum names the head that the job ran on.
        assert checkpoint.metrics["head_checksum"] == PI_SEED.checksum()
        assert "head_release_id" not in checkpoint.metrics
        assert checkpoint.path == checkpoint.artifact_path
        assert HarnessTree.read(Path(checkpoint.path)) == PI_SEED.apply(MARKER)
        # The next job starts from the checkpoint and continues the state after this job.
        component.submit(job(component, "two", start=checkpoint.path, previous="one"))
        second = ended(component, "two")
        assert second.state == "finished" and second.checkpoints == ()
        assert second.metrics["skipped"] == UNCHANGED
        assert second.metrics["head_checksum"] == PI_SEED.apply(MARKER).checksum()
        assert component.state_after("two")[0] == 2 and component.state_after(None)[0] == 0
    finally:
        component.close()


@pytest.mark.unit
def test_a_checkpoint_commit_names_the_parts_that_its_tree_changed_from_the_jobs_start(tmp_path: Path) -> None:
    component = component_of(tmp_path, lambda workspace, services: MARKER)
    component.start_jobs()
    try:
        component.submit(job(component, "one", start=seed_start(component, tmp_path)))
        (checkpoint,) = ended(component, "one").checkpoints
    finally:
        component.close()
    # The training worker asks the component for the parts and records them with the checkpoint; the activation
    # rule reads them. The parts are those that changed from the job's own start: the job rewrote the rules of the
    # seed tree.
    assert component.changed_parts(seed_start(component, tmp_path), checkpoint) == {"rules"}
    # From a start that holds the same tree, nothing changed.
    assert component.changed_parts(checkpoint.path, checkpoint) == frozenset()


@pytest.mark.unit
def test_a_resumed_job_commits_its_settled_step_and_runs_no_second_step(tmp_path: Path) -> None:
    calls = []

    def method(workspace, services):
        calls.append(workspace.step)
        return MARKER

    component = component_of(tmp_path, method)
    component.start_jobs()
    component.submit(job(component, "one", start=seed_start(component, tmp_path)))
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
        component.submit(job(component, "one", start=seed_start(component, tmp_path)))
        status = ended(component, "one")
        assert status.state == "failed" and status.error == "RuntimeError: the method failed"
        assert status.metrics["skipped"] == "step failed" and status.metrics["error"] == status.error
        # The failed job counts, so the next one claims the next job record; the harness state stays.
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
    assert component.artifact_format is None and component.shares_scenarios
    assert component.facts.default_start().files == dict(PI_SEED.files)
    # An imported start is a copy of the snapshot's tree, under its checksum, without the repository's own files.
    artifact_dir = tmp_path / "snapshot"
    PI_SEED.apply(MARKER).write(artifact_dir)
    # An import's directory holds the tree alone.
    TrainingArtifactLayout(component.facts, component).check(DirectoryFiles(artifact_dir))
    (artifact_dir / ".gitattributes").write_text("* filter=lfs\n")
    (artifact_dir / "reef-artifact.json").write_text("{}\n")
    start = component.import_start(artifact_dir)
    assert Path(start).name == PI_SEED.apply(MARKER).checksum()
    assert HarnessTree.read(Path(start)) == PI_SEED.apply(MARKER) and component.import_start(artifact_dir) == start
    broken = tmp_path / "broken"
    HarnessTree({"rules.md": ""}).write(broken)
    with pytest.raises(ValueError, match="fails validation"):
        component.import_start(broken)
    valid = job(component, "valid", start=seed_start(component, tmp_path))
    for broken_job, message in (
        (replace(valid, component=None), "names the scenario component"),
        (replace(valid, parameters={**valid.parameters, "evaluation_tasks": "t"}), "evaluation_tasks"),
        (replace(valid, parameters={**valid.parameters, "request": "r"}), "training request"),
    ):
        with pytest.raises(ValueError, match=message):
            component.check_job(broken_job)
    component.close()


@pytest.mark.unit
def test_the_component_reports_the_running_jobs_progress_and_the_ended_jobs_records(tmp_path: Path) -> None:
    """No other process reads the component's directories: the training worker copies ``job_progress`` into the
    job's item and stores ``job_records`` in the artifact store."""
    seen: list[Mapping[str, Any] | None] = []

    def method(workspace, services):
        seen.append(component.job_progress("one"))
        return MARKER

    component = component_of(tmp_path, method)
    assert component.job_progress("one") is None and component.job_records("one") is None
    component.start_jobs()
    try:
        component.submit(job(component, "one", start=seed_start(component, tmp_path)))
        ended(component, "one")
    finally:
        component.close()
    (progress,) = seen
    assert progress is not None and JobProgress.from_json(progress).phase == "proposing"
    assert component.job_progress("one") is None
    records = component.job_records("one")
    assert records is not None and records.is_dir()
    assert records.parent.name == "steps"
    # A deleted scenario's directory moves aside, under its ID, which no later scenario uses.
    scenario_dir = records.parent.parent.parent
    component.archive_scenario(scenario_dir.name)
    assert not scenario_dir.exists() and component.job_records("one") is None
    (archived,) = (scenario_dir.parent / "archived").iterdir()
    assert archived.name == scenario_dir.name
