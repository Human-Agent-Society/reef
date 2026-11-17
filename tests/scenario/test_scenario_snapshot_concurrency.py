"""Snapshot readers stay live during evolution without observing partial publication."""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.polling import wait_until
from tests.support.runtime_stubs import StubTrainingComponent, live_objects
from tests.support.samples_service import make_samples
from tests.support.threshold_processor import ThresholdJobData, ThresholdSamples

from reef.artifact import Artifact, ArtifactPublicationError, DefaultStart, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import CommitPendingError
from reef.dispatcher import Dispatcher
from reef.execution.workers import SERVICE_COMPONENTS
from reef.scenario import Scenario
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import TrainingCommit
from reef.train.component import SavedCheckpoint, TrainingJob
from reef.train.job_queue import CHECKS_QUEUE, JobQueueComponent
from reef.train.pipeline import PipelineSpec
from reef.train.types import JobData


class _LocalJobs(StubTrainingComponent):
    """A training component whose ``n``-th job saves one checkpoint holding ``model.txt`` = ``step <n>``, with the
    metric ``score`` = ``n``. Its scenarios start from ``model.txt`` = ``initial``."""

    def __init__(self, artifact_dir: Path) -> None:
        super().__init__(artifact_dir, name="local", shares_scenarios=True)

    def stub_default_start(self) -> DefaultStart:
        return DefaultStart(files={"model.txt": "initial"})

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        step = len(self.submitted)
        path = self.checkpoint_root / str(step)
        path.mkdir(parents=True)
        (path / "model.txt").write_text(f"step {step}", encoding="utf-8")
        return (SavedCheckpoint(0, str(path), str(path), {"score": float(step)}),)


class _BlockingJobData(ThresholdJobData):
    """Batches like ``ThresholdJobData``. While ``block`` is true, ``job_parameters`` sets ``started`` and waits for
    ``resume``, so the trainer's cycle that submits the job waits with it. The ``local_scenario`` fixture resets the
    three class attributes."""

    block = False
    started = Event()
    resume = Event()

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        if type(self).block:
            type(self).started.set()
            assert type(self).resume.wait(10), "test did not release the cycle"
        return super().job_parameters(data)


@dataclass(frozen=True)
class _LocalRecipe(TestPolicyRecipe):
    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(ThresholdSamples, _BlockingJobData)


@pytest.fixture
def local_scenario(tmp_path):
    backend = _LocalJobs(tmp_path / "candidates")
    _BlockingJobData.block = False
    _BlockingJobData.started = Event()
    _BlockingJobData.resume = Event()
    dispatcher = Dispatcher(
        _LocalRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        live=live_objects(backend),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
        # The test makes the samples and commits each step itself. No samples worker makes them, and control's local
        # training workers run no cycle: a woken worker would prepare and commit the same result as the test.
        components=frozenset(SERVICE_COMPONENTS) - {"samples"},
        hold_local_cycles=True,
    )
    scenario = dispatcher.get_or_create_scenario("math")
    assert scenario is not None
    try:
        yield scenario, backend
    finally:
        _BlockingJobData.resume.set()
        dispatcher.close()


def _queue_batch(scenario: Scenario, step: int) -> None:
    # Append directly and make the samples here so the test, rather than the dispatcher's workers,
    # owns exactly when the real local training cycle starts and commits.
    scenario.records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            # The artifact that serves the scenario answered the call, so the sample is not stale.
            payload={
                "tokens": [1, 2],
                "loss_mask": [0, 1],
                "rollout_log_probs": [-0.2],
                "artifact_id": scenario.serving_artifact_id(),
            },
            agent_record_id=f"i{step}",
        )
    )
    scenario.records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0, "references": [f"i{step}"]},
            agent_record_id=f"r{step}",
            references=(f"i{step}",),
        )
    )
    make_samples(scenario)


def _next_commit(scenario: Scenario) -> TrainingCommit:
    """The trainer's next commit, once it has one: a job's submission after the training service checked the job,
    or the news of the running job after the training worker reported it."""
    deadline = time.monotonic() + 10
    while (result := scenario.next_commit()) is None:
        assert time.monotonic() < deadline, "the trainer had no commit"
        time.sleep(0.01)
    return result


def _finish(scenario: Scenario) -> None:
    """Commit what is left of the running job, when one runs, up to its end."""
    while scenario.trainer.job_running:
        scenario.commit(_next_commit(scenario))


def _prepare(scenario: Scenario) -> TrainingCommit:
    """Commit the end of the last job and the submission of the next one; the next job's checkpoint, which publishes
    its artifact, not committed yet."""
    _finish(scenario)
    _queue_batch(scenario, scenario.commit_number + 1)
    scenario.commit(_next_commit(scenario))
    result = _next_commit(scenario)
    assert result.artifact is not None
    return result


def _wait_for_job_check(scenario: Scenario) -> None:
    """Ask the training service to check the trainer's next job, and wait for its answer: the trainer's next cycle
    then submits the job."""
    job_loop = scenario.trainer.job_loop_required()
    component = job_loop.component
    assert isinstance(component, JobQueueComponent)
    wait_until(lambda: scenario.next_commit() is None and job_loop.waiting_job is not None)
    job = job_loop.waiting_job
    assert job is not None

    def answered() -> bool:
        item = component.queue.read(CHECKS_QUEUE, f"job:{job.job_id}")
        return item is not None and item.final

    wait_until(answered, message="the training service did not check the job")


def _model_text(artifact: Artifact) -> str:
    local_path = artifact.materialize().local_path
    assert local_path is not None
    return local_path.joinpath("model.txt").read_text(encoding="utf-8")


@pytest.mark.unit
def test_snapshot_reads_remain_live_during_local_training(local_scenario) -> None:
    scenario, _ = local_scenario
    initial_ref = scenario.current_snapshot()
    scenario.commit(_prepare(scenario))
    _finish(scenario)
    current_ref = scenario.current_snapshot()
    expected_rows = scenario.artifact_rows()
    expected_selection = scenario.artifact_selection()
    _queue_batch(scenario, 2)
    # The cycle that waits is the one that submits the job, so the job's check has its answer first.
    _wait_for_job_check(scenario)
    _BlockingJobData.block = True
    writer_started = Event()

    def write():
        writer_started.set()
        return scenario.activate(scenario.repository.resolve(initial_ref).artifact_id)

    with ThreadPoolExecutor(max_workers=7) as executor:
        training = executor.submit(scenario.next_commit)
        try:
            assert _BlockingJobData.started.wait(2), "the cycle did not start"
            writer = executor.submit(write)
            assert writer_started.wait(2), "writer did not start"
            # A queued writer must wait on the operation lock without first
            # taking the publication lock and starving the read-side callers.
            catalog = executor.submit(scenario.artifact_rows)
            selection = executor.submit(scenario.artifact_selection)
            version = executor.submit(scenario.snapshot_queries.artifact_for_version, current_ref.revision)
            old_version = executor.submit(scenario.snapshot_queries.artifact_for_version, initial_ref.revision)

            assert catalog.result(timeout=2) == expected_rows
            assert selection.result(timeout=2) == expected_selection
            assert next(row for row in expected_rows if row["active"])["metrics"]["score"] == 1.0
            artifact = version.result(timeout=2)
            assert artifact.ref == current_ref and _model_text(artifact) == "step 1"
            artifact = old_version.result(timeout=2)
            assert artifact.ref == initial_ref and _model_text(artifact) == "initial"
            with pytest.raises(TimeoutError):
                writer.result(timeout=0.1)
            assert not training.done()
        finally:
            _BlockingJobData.resume.set()

        result = training.result(timeout=2)
        assert result.metrics["training_job"]["event"] == "submitted"
        # The submission waits for its commit, so the activation is refused.
        with pytest.raises(CommitPendingError, match="waits for its commit"):
            writer.result(timeout=2)
        assert scenario.current_snapshot() == current_ref
        assert scenario.commit_number == 3


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["commit", "activate"])
def test_snapshot_reads_wait_for_complete_publication(local_scenario, monkeypatch, operation) -> None:
    scenario, _ = local_scenario
    initial_ref = scenario.current_snapshot()
    scenario.commit(_prepare(scenario))
    scenario.commit(_prepare(scenario))
    previous_ref = scenario.current_snapshot()
    result = _prepare(scenario) if operation == "commit" else None
    publication_started = Event()
    resume_publication = Event()

    assert scenario.store.durable
    assert isinstance(scenario.store, CommitHistoryScenarioStore)
    journal = scenario.store.commit_history
    assert journal is not None
    original = journal.append

    def block_publication(value):
        publication_started.set()
        assert resume_publication.wait(10), "test did not release publication"
        return original(value)

    monkeypatch.setattr(journal, "append", block_publication)
    readers_started = [Event() for _ in range(3)]

    def read_catalog():
        readers_started[0].set()
        return scenario.artifact_rows()

    def read_selection():
        readers_started[1].set()
        return scenario.artifact_selection()

    def read_version():
        readers_started[2].set()
        return scenario.snapshot_queries.artifact_for_version(previous_ref.revision)

    with ThreadPoolExecutor(max_workers=5) as executor:
        writer = (
            executor.submit(scenario.commit, result)
            if operation == "commit"
            else executor.submit(scenario.activate, scenario.repository.resolve(initial_ref).artifact_id)
        )
        try:
            assert publication_started.wait(2), "publication did not start"
            # The head stays put until the journal commits, and publication reads wait for it.
            assert scenario.current_snapshot() == previous_ref
            # Each job commits its submission, its checkpoint and its end; the next job's submission is committed.
            assert scenario.commit_number == (7 if operation == "commit" else 5)
            readers = [executor.submit(read) for read in (read_catalog, read_selection, read_version)]
            for started in readers_started:
                assert started.wait(2), "reader did not start"
            for reader in readers:
                with pytest.raises(TimeoutError):
                    reader.result(timeout=0.1)
        finally:
            resume_publication.set()

        writer.result(timeout=2)
        catalog, selection, version = [reader.result(timeout=2) for reader in readers]

    current_ref = scenario.current_snapshot()
    assert scenario.commit_number == (8 if operation == "commit" else 6)
    # The catalog marks exactly one active artifact; an activation adds no row.
    [active] = [row for row in catalog if row["active"]]
    assert active["snapshot"].revision == current_ref.revision
    assert active["operation"] == ("training" if operation == "commit" else "creation")
    assert (active.get("metrics") or {}).get("score") == (3.0 if operation == "commit" else None)
    assert selection == scenario.artifact_selection(current_ref)
    assert _model_text(scenario.repository.resolve(current_ref)) == ("step 3" if operation == "commit" else "initial")
    assert next(row for row in catalog if row["snapshot"] == previous_ref)["metrics"]["score"] == 2.0
    assert version.ref == previous_ref


def test_pointer_failure_is_reported_and_repaired_before_next_commit(local_scenario, monkeypatch):
    scenario, _ = local_scenario
    backend = scenario.repository.backend
    first = _prepare(scenario)

    def unavailable(*args, **kwargs):
        raise ArtifactPublicationError("storage unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "commit_snapshot", unavailable)
        scenario.commit(first)
        committed = scenario.current_snapshot()
        # The job's submission, then its checkpoint.
        assert scenario.commit_number == 2
        assert scenario.repository.checkpoint_artifact == committed
        assert scenario.commit_status["artifact_head_sync"] == {
            "state": "pending",
            "error": "storage unavailable",
        }
        # The job's end is the next commit: it repairs the committed parent first, which fails.
        ended = _next_commit(scenario)
        assert ended.artifact is None

        def unexpected_publish(*args, **kwargs):
            pytest.fail("a commit must not publish while its committed parent is unsynchronized")

        patch.setattr(backend, "publish", unexpected_publish)
        with pytest.raises(ArtifactPublicationError, match="storage unavailable"):
            scenario.commit(ended)
        assert scenario.commit_number == 2
        assert len(scenario.store.history()) == 2
        assert scenario.current_snapshot() == committed

    # Retry the same prepared commit after storage recovers; the committed
    # parent is repaired first, and the commit advances exactly once.
    scenario.commit(ended)
    assert scenario.commit_number == 3
    assert len(scenario.store.history()) == 3
    assert scenario.repository.checkpoint_artifact == backend.current()
    assert scenario.commit_status["artifact_head_sync"] == {
        "state": "synchronized",
        "error": None,
    }


def test_pointer_conflict_preserves_the_commit_and_activation_leaves_the_backend_head(
    local_scenario, monkeypatch, tmp_path
):
    scenario, _ = local_scenario
    backend = scenario.repository.backend
    initial = scenario.current_snapshot()
    first = _prepare(scenario)
    foreign_path = tmp_path / "foreign"
    foreign_path.mkdir()
    (foreign_path / "model.txt").write_text("another writer")
    commit_snapshot = backend.commit_snapshot
    unrelated = None

    def competing_writer(ref, *, expected_parent):
        nonlocal unrelated
        unrelated = backend.publish(Artifact.local(foreign_path), expected_parent=expected_parent)
        commit_snapshot(ref, expected_parent=expected_parent)

    with monkeypatch.context() as patch:
        patch.setattr(backend, "commit_snapshot", competing_writer)
        scenario.commit(first)
    assert scenario.commit_number == 2
    assert scenario.commit_status["artifact_head_sync"]["state"] == "conflict"
    assert backend.current() == unrelated
    # An activation writes to the commit history only: the conflicted backend head stays as it is.
    assert (
        scenario.activate(scenario.repository.resolve(initial).artifact_id).artifact_id
        == scenario.repository.resolve(initial).artifact_id
    )
    assert scenario.current_snapshot() == initial
    assert scenario.commit_number == 3
    assert backend.current() == unrelated
    assert scenario.commit_status["artifact_head_sync"]["state"] == "conflict"


def test_activation_does_no_storage_io(local_scenario, monkeypatch):
    scenario, _ = local_scenario
    initial = scenario.current_snapshot()
    scenario.commit(_prepare(scenario))
    backend = scenario.repository.backend
    tip = backend.current()

    def unavailable(*args, **kwargs):
        raise ArtifactPublicationError("storage unavailable")

    with monkeypatch.context() as patch:
        for method in ("commit_snapshot", "publish", "register"):
            patch.setattr(backend, method, unavailable)
        activated = scenario.activate(scenario.repository.resolve(initial).artifact_id)
    assert activated.artifact_id == scenario.repository.resolve(initial).artifact_id
    assert scenario.commit_number == 3
    assert scenario.store.history()[-1].operation == "activate"
    assert scenario.current_snapshot() == initial
    # The chain tip stays the parent of the next publication, and the backend head never moves backward.
    assert scenario.repository.checkpoint_artifact == tip
    assert backend.current() == tip
    assert scenario.commit_status["artifact_head_sync"] == {
        "state": "synchronized",
        "error": None,
    }
    scenario.commit(_prepare(scenario))
    assert scenario.current_snapshot().parent_revision == tip.revision


def test_committed_record_retry_reports_pointer_failure_without_repeating_step(local_scenario, monkeypatch):
    scenario, _ = local_scenario
    result = _prepare(scenario)
    append_commit = scenario.store.append_commit

    def lose_ack(**keywords):
        append_commit(**keywords)
        raise OSError("commit acknowledgment lost")

    with monkeypatch.context() as patch:
        patch.setattr(scenario.store, "append_commit", lose_ack)
        with pytest.raises(OSError, match="acknowledgment lost"):
            scenario.commit(result)
    committed = scenario.store.history()[-1]

    def unavailable(*args, **kwargs):
        raise ArtifactPublicationError("storage unavailable")

    monkeypatch.setattr(scenario.repository.backend, "commit_snapshot", unavailable)
    scenario.commit(result)
    assert scenario.commit_number == 2
    assert len(scenario.store.history()) == 2
    assert scenario.trainer.state == result.state
    assert scenario.current_snapshot() == committed.snapshot
    assert scenario.repository.checkpoint_artifact == committed.snapshot
    assert scenario.commit_status["artifact_head_sync"] == {
        "state": "pending",
        "error": "storage unavailable",
    }
