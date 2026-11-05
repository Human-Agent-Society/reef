"""Release readers stay live during evolution without observing partial publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Event

import pytest
from tests.support._threshold_processor import threshold_pipeline
from tests.support.runtime_stubs import StubTrainingComponent
from tests.support.samples_service import make_samples

from reef.artifact import Artifact, ArtifactPublicationError, DefaultStart, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import CommitPending
from reef.dispatcher import Dispatcher
from reef.execution.workers import SERVICE_COMPONENTS
from reef.recipe import Recipe
from reef.scenario import Scenario
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import Trainer, TrainingCommit
from reef.train.component import SavedCheckpoint, TrainingJob
from reef.train.jobs import JobLoop


class _LocalJobs(StubTrainingComponent):
    """A training component whose ``n``-th job saves one checkpoint holding ``model.txt`` = ``step <n>``, with the
    metric ``score`` = ``n``. With ``block``, a job waits in ``submit``, so the trainer's preparation waits with it."""

    def __init__(self, artifact_dir: Path) -> None:
        super().__init__(artifact_dir, name="local", shares_scenarios=True)
        self.block = False
        self.started = Event()
        self.resume = Event()

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        if self.block:
            self.started.set()
            assert self.resume.wait(10), "test did not release the job"
        step = len(self.submitted)
        path = self.checkpoint_root / str(step)
        path.mkdir(parents=True)
        (path / "model.txt").write_text(f"step {step}", encoding="utf-8")
        return (SavedCheckpoint(0, str(path), str(path), {"score": float(step)}),)


@dataclass(frozen=True)
class _LocalRecipe(Recipe):
    backend: _LocalJobs

    def default_start(self) -> DefaultStart:
        return DefaultStart(files={"model.txt": "initial"})

    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        artifacts=None,
        samples=None,
        root=None,
    ):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=partial(threshold_pipeline, job_samples=1),
            job_loop=JobLoop(self.backend, scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            root=root,
        )


@pytest.fixture
def local_scenario(tmp_path):
    backend = _LocalJobs(tmp_path / "candidates")
    dispatcher = Dispatcher(
        _LocalRecipe(backend=backend),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
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
        backend.resume.set()
        dispatcher.close()


def _queue_batch(scenario: Scenario, step: int) -> None:
    # Append directly and make the samples here so the test, rather than the dispatcher's workers,
    # owns exactly when the real local training cycle starts and commits.
    scenario.records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
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


def _finish(scenario: Scenario) -> None:
    """Commit the end of the running job, when one runs: the job ended in ``submit``."""
    while scenario.trainer.job_running:
        result = scenario.next_commit()
        assert result is not None
        scenario.commit(result)


def _prepare(scenario: Scenario) -> TrainingCommit:
    """Commit the end of the last job and the submission of the next one; the next job's checkpoint, which publishes
    its artifact, not committed yet."""
    _finish(scenario)
    _queue_batch(scenario, scenario.scenario_step + 1)
    submitted = scenario.next_commit()
    assert submitted is not None
    scenario.commit(submitted)
    result = scenario.next_commit()
    assert result is not None and result.artifact is not None
    return result


def _model_text(artifact: Artifact) -> str:
    local_path = artifact.materialize().local_path
    assert local_path is not None
    return local_path.joinpath("model.txt").read_text(encoding="utf-8")


@pytest.mark.unit
def test_release_reads_remain_live_during_local_training(local_scenario) -> None:
    scenario, backend = local_scenario
    initial_ref = scenario.current_snapshot()
    scenario.commit(_prepare(scenario))
    _finish(scenario)
    current_ref = scenario.current_snapshot()
    expected_rows = scenario.artifact_rows()
    expected_selection = scenario.artifact_selection()
    backend.block = True
    _queue_batch(scenario, 2)
    writer_started = Event()

    def write():
        writer_started.set()
        return scenario.activate(scenario.repository.resolve(initial_ref).artifact_id)

    with ThreadPoolExecutor(max_workers=7) as executor:
        training = executor.submit(scenario.next_commit)
        try:
            assert backend.started.wait(2), "the job did not start"
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
            backend.resume.set()

        result = training.result(timeout=2)
        assert result.metrics["training_job"]["event"] == "submitted"
        # The submission waits for its commit, so the activation is refused.
        with pytest.raises(CommitPending, match="waits for its commit"):
            writer.result(timeout=2)
        assert scenario.current_snapshot() == current_ref
        assert scenario.scenario_step == 3


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["commit", "activate"])
def test_release_reads_wait_for_complete_publication(local_scenario, monkeypatch, operation) -> None:
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
            assert scenario.scenario_step == (7 if operation == "commit" else 5)
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
    assert scenario.scenario_step == (8 if operation == "commit" else 6)
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
        assert scenario.scenario_step == 2
        assert scenario.repository.checkpoint_artifact == committed
        assert scenario.commit_status["artifact_head_sync"] == {
            "state": "pending",
            "error": "storage unavailable",
        }
        # The job's end is the next commit: it repairs the committed parent first, which fails.
        ended = scenario.next_commit()
        assert ended is not None and ended.artifact is None

        def unexpected_publish(*args, **kwargs):
            pytest.fail("a commit must not publish while its committed parent is unsynchronized")

        patch.setattr(backend, "publish", unexpected_publish)
        with pytest.raises(ArtifactPublicationError, match="storage unavailable"):
            scenario.commit(ended)
        assert scenario.scenario_step == 2
        assert len(scenario.store.history()) == 2
        assert scenario.current_snapshot() == committed

    # Retry the same prepared commit after storage recovers; the committed
    # parent is repaired first, and the commit advances exactly once.
    scenario.commit(ended)
    assert scenario.scenario_step == 3
    assert len(scenario.store.history()) == 3
    assert scenario.repository.checkpoint_artifact == backend.current()
    assert scenario.commit_status["artifact_head_sync"] == {
        "state": "synchronized",
        "error": None,
    }


def test_pointer_conflict_preserves_committed_step_and_activation_leaves_the_backend_head(
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
    assert scenario.scenario_step == 2
    assert scenario.commit_status["artifact_head_sync"]["state"] == "conflict"
    assert backend.current() == unrelated
    # An activation writes to the commit history only: the conflicted backend head stays as it is.
    assert (
        scenario.activate(scenario.repository.resolve(initial).artifact_id).artifact_id
        == scenario.repository.resolve(initial).artifact_id
    )
    assert scenario.current_snapshot() == initial
    assert scenario.scenario_step == 3
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
    assert scenario.scenario_step == 3
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
    commit_step = scenario.store.commit_step

    def lose_ack(**keywords):
        commit_step(**keywords)
        raise OSError("commit acknowledgment lost")

    with monkeypatch.context() as patch:
        patch.setattr(scenario.store, "commit_step", lose_ack)
        with pytest.raises(OSError, match="acknowledgment lost"):
            scenario.commit(result)
    committed = scenario.store.history()[-1]

    def unavailable(*args, **kwargs):
        raise ArtifactPublicationError("storage unavailable")

    monkeypatch.setattr(scenario.repository.backend, "commit_snapshot", unavailable)
    scenario.commit(result)
    assert scenario.scenario_step == 2
    assert len(scenario.store.history()) == 2
    assert scenario.trainer.state == result.state
    assert scenario.current_snapshot() == committed.snapshot
    assert scenario.repository.checkpoint_artifact == committed.snapshot
    assert scenario.commit_status["artifact_head_sync"] == {
        "state": "pending",
        "error": "storage unavailable",
    }
