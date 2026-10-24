"""JobTrainingRuntime: each training step is one job on a training component, started from the committed weights."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory

from reef.artifact.artifact import Artifact
from reef.core.batches import StepScheduling
from reef.runtime.interfaces import StaleCandidate, TrainingMethod, TrainingRuntimeError
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.train.job_runtime import JobTrainingRuntime
from reef.train.types import TrainingBatch


class FakeComponent(TrainingComponent):
    """Run each job in ``submit``: save one checkpoint directory, or none when the batch is empty."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: list[TrainingJob] = []
        self.statuses: dict[str, list[JobStatus]] = {}
        self.refused_starts: set[str] = set()

    @property
    def name(self) -> str:
        return "fake"

    @property
    def batch_size(self) -> int:
        return 2

    def check_start(self, path: Path) -> None:
        if path.name in self.refused_starts:
            raise ValueError("cannot start from these weights")

    def submit(self, job: TrainingJob) -> None:
        if job.job_id in self.statuses:
            return
        self.jobs.append(job)
        state = {"steps": job.parameters["algorithm_state"].get("steps", 0) + 1}
        if not job.data.items:
            self.statuses[job.job_id] = [
                JobStatus(job.job_id, "finished", metrics={"skipped": 1}, algorithm_state=state)
            ]
            return
        directory = self.root / job.job_id
        directory.mkdir(parents=True)
        (directory / "weights").write_text(str(job.start))
        checkpoint = SavedCheckpoint(1, str(directory))
        self.statuses[job.job_id] = [
            JobStatus(job.job_id, "running"),
            JobStatus(job.job_id, "finished", (checkpoint,), metrics={"loss": 0.5}, algorithm_state=state),
        ]

    def status(self, job_id: str) -> JobStatus:
        statuses = self.statuses[job_id]
        return statuses.pop(0) if len(statuses) > 1 else statuses[0]

    def cancel(self, job_id: str) -> None:
        return


def batch(name: str = "b") -> TrainingBatch:
    return TrainingBatch(name, (policy_trajectory(f"s-{name}", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, "v1"),))


def prepare(runtime: JobTrainingRuntime, data: TrainingBatch, state=None, step: int = 0, scenario=None):
    return runtime.prepare_training_step(
        data, TrainingMethod("sft"), state or {}, StepScheduling(), step, scenario=scenario
    )


@pytest.fixture
def component(tmp_path):
    return FakeComponent(tmp_path / "component")


@pytest.fixture
def runtime(component, tmp_path):
    return JobTrainingRuntime(component, tmp_path / "runtime", poll_interval_seconds=0.001)


@pytest.mark.unit
def test_a_step_is_one_job_and_its_checkpoint_is_the_candidate(runtime, component):
    prepared = prepare(runtime, batch(), {"steps": 3}, step=7)
    assert prepared.action == "train" and prepared.next_algorithm_state == {"steps": 4}
    [job] = component.jobs
    assert job.backend == "fake" and job.start is None
    assert job.parameters == {
        "method": {"objective": "sft", "learning_rate_schedule": None},
        "scheduling": {"unit": "comparison_set", "epochs": 1, "shuffle": False, "remainder": "partial"},
        "algorithm_state": {"steps": 3},
    }
    candidate = runtime.train_candidate(prepared.payload)
    assert candidate.training_job_id == job.job_id
    assert candidate.checkpoint_path == str(component.root / job.job_id)
    assert candidate.training_metrics == {"loss": 0.5}
    assert candidate.metadata == {"scenario_step": 7}


@pytest.mark.unit
def test_a_job_that_saved_nothing_is_a_skip(runtime):
    prepared = prepare(runtime, TrainingBatch("empty"), {"steps": 3})
    assert prepared.action == "skip"
    assert prepared.next_algorithm_state == {"steps": 4}
    assert prepared.metrics == {"skipped": 1}


@pytest.mark.unit
def test_the_next_job_starts_from_the_committed_checkpoint_and_a_rejection_keeps_the_start(runtime, component):
    first = runtime.train_candidate(prepare(runtime, batch("one")).payload)
    runtime.reject_candidate(first)
    second = runtime.train_candidate(prepare(runtime, batch("two")).payload)
    assert [job.start for job in component.jobs] == [None, None]
    runtime.commit_candidate(second.training_job_id)
    assert runtime.start() == second.checkpoint_path
    prepare(runtime, batch("three"))
    assert component.jobs[-1].start == second.checkpoint_path
    with pytest.raises(ValueError, match="unknown training candidate"):
        runtime.reject_candidate(first)


@pytest.mark.unit
def test_the_same_request_is_the_same_job_whatever_the_batch_number(runtime, component):
    first = prepare(runtime, batch("one"))
    again = prepare(runtime, TrainingBatch("renumbered", batch("one").items))
    assert first.payload == again.payload
    assert len(component.jobs) == 1


@pytest.mark.unit
def test_a_commit_or_a_rollback_while_a_job_ran_makes_its_candidate_stale(runtime, tmp_path):
    payload = prepare(runtime, batch()).payload
    release = tmp_path / "release"
    release.mkdir()
    (release / "weights").write_text("restored")
    runtime.restore_checkpoint(Artifact.local(release))
    with pytest.raises(StaleCandidate):
        runtime.train_candidate(payload)


@pytest.mark.unit
def test_the_start_survives_a_restart_and_a_restart_commits_by_the_components_status(component, tmp_path):
    runtime = JobTrainingRuntime(component, tmp_path / "runtime", poll_interval_seconds=0.001)
    candidate = runtime.train_candidate(prepare(runtime, batch()).payload)
    restarted = JobTrainingRuntime(component, tmp_path / "runtime", poll_interval_seconds=0.001)
    restarted.commit_candidate(candidate.training_job_id)
    assert restarted.start() == candidate.checkpoint_path
    # A job the component never saw, such as one of another trainer, changes nothing.
    restarted.commit_candidate("unknown")
    assert JobTrainingRuntime(component, tmp_path / "runtime").start() == candidate.checkpoint_path


@pytest.mark.unit
def test_a_rollback_copies_the_release_into_the_start_and_an_empty_release_is_the_base_model(runtime, tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "weights").write_text("restored")
    (release / "reef-artifact.json").write_text("{}")
    artifact = Artifact.local(release)
    runtime.restore_checkpoint(artifact)
    start = Path(runtime.start())
    assert start.parent == runtime.state_dir / "restored"
    assert (start / "weights").read_text() == "restored"
    assert not (start / "reef-artifact.json").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "reef-artifact.json").write_text("{}")
    runtime.restore_checkpoint(Artifact.local(empty))
    assert runtime.start() is None


@pytest.mark.unit
def test_a_rollback_the_component_refuses_keeps_the_start(runtime, component, tmp_path):
    release = tmp_path / "foreign"
    release.mkdir()
    (release / "weights").write_text("other model")
    component.refused_starts.add("foreign")
    with pytest.raises(ValueError, match="cannot start from these weights"):
        runtime.restore_checkpoint(Artifact.local(release))
    assert runtime.start() is None


@pytest.mark.unit
def test_each_scenario_trains_its_own_weights_from_its_own_start(component, tmp_path):
    runtime = JobTrainingRuntime(component, tmp_path / "runtime", per_scenario=True, poll_interval_seconds=0.001)
    assert runtime.concurrent_training_scenarios
    math = runtime.train_candidate(prepare(runtime, batch(), scenario="math").payload)
    # The same rows are another job for another scenario.
    code = runtime.train_candidate(prepare(runtime, batch(), scenario="code").payload)
    assert math.training_job_id != code.training_job_id
    # The component learns no scenario: a job names only its start, data and parameters.
    assert all("scenario" not in job.parameters for job in component.jobs)
    assert (math.metadata, code.metadata) == (
        {"scenario_step": 0, "scenario": "math"},
        {"scenario_step": 0, "scenario": "code"},
    )
    runtime.commit_candidate(math.training_job_id, scenario="math")
    assert (runtime.start("math"), runtime.start("code")) == (math.checkpoint_path, None)
    restarted = JobTrainingRuntime(component, tmp_path / "runtime", per_scenario=True)
    assert restarted.start("math") == math.checkpoint_path
    with pytest.raises(TrainingRuntimeError, match="needs the scenario"):
        prepare(runtime, batch("other"))
    with pytest.raises(TrainingRuntimeError, match="takes no scenario"):
        JobTrainingRuntime(component, tmp_path / "single").start("math")


@pytest.mark.unit
def test_a_failed_job_fails_the_step(runtime, component):
    class Failing(FakeComponent):
        def submit(self, job):
            self.statuses[job.job_id] = [JobStatus(job.job_id, "failed", error="out of memory")]

    failing = JobTrainingRuntime(Failing(component.root), runtime.state_dir / "other", poll_interval_seconds=0.001)
    with pytest.raises(TrainingRuntimeError, match="out of memory"):
        prepare(failing, batch())
