"""Durability protocol tests for the per-scenario commit history (issue #78, phase 3).

One append-only log per scenario is the single commit point: record-store
compaction, the checkpoint head, and the trainer's algorithm
state + read progress are all derived from it at recovery. These tests walk
the crash windows around the commit point and assert each is recoverable.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support._threshold_processor import ThresholdJobData, ThresholdSamples, threshold_pipeline
from tests.support.harness_service import harness_repository
from tests.support.harness_steps import PI_SEED, harness_recipe
from tests.support.runtime_stubs import StubTrainingComponent, runtime_bindings

from reef.artifact import Artifact, ArtifactRef, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher, local_workers
from reef.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.recipe.base import Recipe
from reef.record2record import ReportGrouping, ReportRecord2Record
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.storage.commit_history import CommitHistoryError, CommitHistoryScenarioStore
from reef.storage.commits import CommitRecord, active_artifact_ref
from reef.storage.samples import SAMPLE_KINDS, consumed_sample_id
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train import CandidateBackend, PreparedStep, Trainer, TrainStepResult
from reef.train.harness_backend.state import HarnessState
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import TrainingPipeline


def sample_record(
    *,
    step: int = 1,
    runtime_load_id: str = "w1",
    checkpoint: bool = False,
    training_job_id: str | None = None,
    metrics: dict | None = None,
) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        artifact_ref=ArtifactRef(
            content_id=f"content:{step}", release_id=f"release:{runtime_load_id}:{step}", parent_release_id="base"
        ),
        checkpoint=checkpoint,
        algorithm_state={"steps": step},
        high_water_sequence=step * 2,
        high_water_offset=step * 2,
        consumed_ids=frozenset({f"i{step}"}),
        recorded_at=1000.0 + step,
        training_job_id=training_job_id,
        metrics=metrics,
    )


@pytest.fixture
def open_history(tmp_path):
    """Open commit histories over one SQLite database file; the fixture closes them."""
    stores: list[SQLiteRecordStore] = []

    def open_one():
        stores.append(SQLiteRecordStore(tmp_path / "records.sqlite3"))
        return stores[-1].commit_history()

    yield open_one
    for store in stores:
        store.close()


@pytest.mark.unit
def test_commit_record_round_trips_through_the_history(open_history) -> None:
    log = open_history()
    log.append(sample_record(step=1))
    metrics = {"changed": "config/format.json", "wins": 3, "losses": 1, "ties": 0, "published": True}
    log.append(sample_record(step=2, runtime_load_id="w2", metrics=metrics, training_job_id="job-1"))

    records = log.records()
    assert [record.step for record in records] == [1, 2]
    first, second = records
    assert first.scenario == "math"
    assert first.artifact_ref == ArtifactRef("content:1", "release:w1:1", "base")
    assert first.checkpoint is False
    assert first.algorithm_state == {"steps": 1}
    assert first.high_water_sequence == 2
    assert first.high_water_offset == 2
    assert first.consumed_ids == frozenset({"i1"})
    assert first.recorded_at == 1001.0
    assert first.metrics is None  # pre-metrics records stay readable
    assert first.training_job_id is None
    assert second.metrics == metrics
    assert second.training_job_id == "job-1"
    assert second.to_dict()["training_job_id"] == "job-1"
    # A fresh reader over the same database sees the same history.
    assert open_history().records() == records


@pytest.mark.unit
def test_commit_record_detaches_nested_metrics() -> None:
    metrics = {"selection": {"selected": True}}
    record = sample_record(metrics=metrics)

    metrics["selection"]["selected"] = False
    encoded = record.to_dict()
    encoded["metrics"]["selection"]["selected"] = False

    assert record.metrics == {"selection": {"selected": True}}


@pytest.mark.unit
def test_commit_history_tracks_the_current_training_run_position(open_history) -> None:
    log = open_history()
    log.append(sample_record(step=1))
    log.append(sample_record(step=2, runtime_load_id="w2"))
    assert log.training_run_position() == (0, 2)

    start = CommitRecord(
        scenario="math",
        step=3,
        artifact_ref=ArtifactRef("artifact:1", "checkpoint:1", None),
        checkpoint=False,
        algorithm_state=None,
        high_water_sequence=0,
        high_water_offset=0,
        operation="train_from",
    )
    log.append(start)
    assert log.training_run_position() == (3, 0)

    log.append(sample_record(step=4, runtime_load_id="w4"))
    assert log.training_run_position() == (3, 1)
    # An activation changes what serves, not the training run.
    log.append(
        CommitRecord(
            scenario="math",
            step=5,
            artifact_ref=ArtifactRef("artifact:1", "checkpoint:1", None),
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
            operation="activate",
        )
    )
    assert log.training_run_position() == (3, 1)


def operation_record(operation: str, step: int, release_id: str, **changes) -> CommitRecord:
    """An activation or a new training start at ``step`` naming ``release_id``, with ``changes`` applied."""
    values = {
        "scenario": "math",
        "step": step,
        "artifact_ref": ArtifactRef(f"content:{release_id}", release_id, None),
        "checkpoint": False,
        "algorithm_state": None,
        "high_water_sequence": 0,
        "high_water_offset": 0,
        "operation": operation,
    }
    return CommitRecord(**{**values, **changes})


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["activate", "train_from", "serve", "stop_serving"])
@pytest.mark.parametrize(
    "changes",
    [
        {"activated": True},
        {"algorithm_state": {"steps": 1}},
        {"metrics": {"loss": 1.0}},
        {"training_job_id": "job-1"},
        {"high_water_sequence": 3},
        {"high_water_offset": 3},
        {"consumed_ids": frozenset({"i1"})},
    ],
)
def test_activations_and_training_starts_carry_no_trainer_state(operation, changes) -> None:
    assert CommitRecord.from_dict(operation_record(operation, 1, "r1").to_dict()).operation == operation
    with pytest.raises(CommitHistoryError, match=r"must not carry|made an artifact is activated"):
        operation_record(operation, 1, "r1", **changes)


@pytest.mark.unit
def test_each_operation_names_its_artifact_exactly() -> None:
    # A one-component activation names the release that made the artifact; a composed one names the release it
    # published, with the component and the manifest.
    # A serve and its stop name the release that serves the artifact in the same way.
    for operation in ("activate", "serve", "stop_serving"):
        operation_record(operation, 1, "r1")
        operation_record(operation, 1, "r1", component="harness", components={"harness": "c", "weights": "w"})
        for changes in (
            {"component": "harness"},
            {"components": {"harness": "c"}},
            {"component": "config", "components": {"harness": "c"}},
        ):
            with pytest.raises(CommitHistoryError, match=f"operation '{operation}' names a component"):
                operation_record(operation, 1, "r1", **changes)
    for operation in ("activate", "train_from", "serve", "stop_serving"):
        with pytest.raises(CommitHistoryError, match="makes no artifact"):
            operation_record(operation, 1, "r1", checkpoint=True)
    # A new start names a trainer's component and publishes nothing.
    operation_record("train_from", 1, "r1", component="weights")
    with pytest.raises(CommitHistoryError, match="publishes no release"):
        operation_record("train_from", 1, "r1", components={"weights": "c"})
    for operation in ("rollback", "promote", "creation"):
        with pytest.raises(CommitHistoryError, match="operation must be"):
            operation_record(operation, 1, "r1")
    with pytest.raises(CommitHistoryError, match="scenario must be a non-empty string"):
        operation_record("activate", 1, "r1", scenario="")
    with pytest.raises(CommitHistoryError, match="only a training commit that made an artifact is activated"):
        replace(sample_record(), activated=True)


@pytest.mark.unit
def test_the_active_artifact_is_the_last_activation_in_the_log() -> None:
    creation = ArtifactRef("content:base", "base", None)
    made = replace(sample_record(step=1, checkpoint=True), activated=True)
    held = sample_record(step=2, checkpoint=True, runtime_load_id="w2")
    nothing = replace(sample_record(step=3), artifact_ref=made.artifact_ref)
    assert active_artifact_ref((), creation) == creation
    assert active_artifact_ref((made,), creation) == made.artifact_ref
    assert made.to_dict()["activated"] is True and "activated" not in held.to_dict()
    assert CommitRecord.from_dict(made.to_dict()) == made
    # An artifact the rule did not activate, a step that made nothing and a new training start activate nothing.
    start = operation_record("train_from", 4, "base")
    assert active_artifact_ref((made, held, nothing, start), creation) == made.artifact_ref
    activation = operation_record("activate", 5, held.artifact_ref.release_id, artifact_ref=held.artifact_ref)
    assert active_artifact_ref((made, held, nothing, start, activation), creation) == held.artifact_ref
    back = operation_record("activate", 6, "base", artifact_ref=creation)
    assert active_artifact_ref((made, held, nothing, start, activation, back), creation) == creation
    # A training step that makes and activates an artifact afterwards is the newest activation.
    newest = replace(sample_record(step=7, checkpoint=True, runtime_load_id="w7"), activated=True)
    assert active_artifact_ref((made, held, activation, back, newest), creation) == newest.artifact_ref


@pytest.mark.unit
def test_commit_history_retry_is_idempotent_but_conflicting_step_is_rejected(open_history) -> None:
    log = open_history()
    original = sample_record(step=1)
    retry = CommitRecord.from_dict({**original.to_dict(), "recorded_at": original.recorded_at + 1})

    log.append(original)
    log.append(retry)
    second = sample_record(step=2, runtime_load_id="w2")
    log.append(second)
    log.append(retry)

    assert log.records() == (original, second)
    with pytest.raises(CommitHistoryError, match="conflicts"):
        log.append(sample_record(step=1, runtime_load_id="different"))


@pytest.mark.unit
def test_commit_history_reads_the_appends_of_another_writer(open_history) -> None:
    first = open_history()
    second = open_history()
    first.append(sample_record(step=1))
    assert len(first.records()) == 1

    second.append(sample_record(step=2, runtime_load_id="w2"))
    # The first writer's cache stops at step 1, but the append still sees step 2.
    with pytest.raises(CommitHistoryError, match="conflicts"):
        first.append(sample_record(step=2, runtime_load_id="other"))
    first.append(sample_record(step=3, runtime_load_id="w3"))

    assert [record.step for record in first.records()] == [1, 2, 3]
    assert second.records() == first.records()


@pytest.mark.unit
def test_commit_record_serializes_its_artifact_ref() -> None:
    durable = CommitRecord(
        scenario="math",
        step=1,
        artifact_ref=ArtifactRef("artifact:1", "checkpoint:1", None),
        checkpoint=True,
        algorithm_state=None,
        high_water_sequence=0,
        high_water_offset=0,
    )
    assert durable.to_dict()["artifact_ref"] == {
        "kind": "artifact",
        "content_id": "artifact:1",
        "release_id": "checkpoint:1",
        "parent_release_id": None,
    }


@pytest.mark.unit
def test_commit_record_holds_json_values() -> None:
    record = replace(sample_record(), algorithm_state={"path": ("a",)}, metrics={1: {"set": ()}})
    assert record.algorithm_state == {"path": ["a"]}
    assert record.metrics == {"1": {"set": []}}
    assert CommitRecord.from_dict(record.to_dict()) == record
    with pytest.raises(CommitHistoryError, match="JSON values"):
        replace(sample_record(), metrics={"ids": {"a"}})


@pytest.mark.unit
def test_commit_record_validates_its_schema() -> None:
    with pytest.raises(CommitHistoryError, match="step"):
        replace(sample_record(), step=0)
    with pytest.raises(CommitHistoryError, match="high_water_sequence"):
        replace(sample_record(), high_water_sequence=-1)
    with pytest.raises(CommitHistoryError, match="record kind"):
        CommitRecord.from_dict({"record": "something-else"})
    encoded = sample_record().to_dict()
    encoded["record_progress"] = {"high_water_sequence": 0, "high_water_offset": 0, "compacted_ids": [1]}
    with pytest.raises(CommitHistoryError, match="compacted_ids"):
        CommitRecord.from_dict(encoded)


class RecordingComponent(StubTrainingComponent):
    """A training component that records the source records of each batch it trained."""

    def __init__(self, checkpoint_root: Path | None = None) -> None:
        super().__init__(checkpoint_root)
        self.trained_data: list[list[str]] = []

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        self.trained_data.append([source_record_id(sample) for sample in job.data.samples])
        return super().train(job)


def sft_inference(agent_record_id: str, runtime_load_id: str | None = None) -> AgentRecord:
    """An inference record; ``runtime_load_id`` is the version that served it, when the runtime reports one."""
    payload: dict[str, object] = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    if runtime_load_id is not None:
        payload["runtime_load_id"] = runtime_load_id
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
    )


def sft_report(agent_record_id: str, reference: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [reference]},
        agent_record_id=agent_record_id,
        references=(reference,),
    )


def build_training_dispatcher(
    component,
    tmp_path,
    backend_factory,
    *,
    agent_record_dir=None,
    hold_local_cycles=False,
):
    return Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component)),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        hold_local_cycles=hold_local_cycles,
    )


def served_version(dispatcher: Dispatcher, scenario: str = "math") -> str | None:
    """The version that serves the scenario now: fresh inference records claim it, so they pass the staleness check."""
    return dispatcher.get_or_create_scenario(scenario).serving_runtime_load_id()


def scenario_database(agent_record_dir, scenario: str = "math"):
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    return agent_record_dir / f"{key}.sqlite3"


def committed_records(agent_record_dir, scenario: str = "math") -> tuple[CommitRecord, ...]:
    with SQLiteRecordStore(scenario_database(agent_record_dir, scenario)) as records:
        return records.commit_history().records()


def delete_commits(agent_record_dir, *steps: int) -> None:
    """Remove committed steps behind the store's back, as a lost write or a damaged file would."""
    with closing(sqlite3.connect(scenario_database(agent_record_dir))) as connection, connection:
        deleted = connection.executemany("DELETE FROM scenario_commit WHERE step = ?", [(step,) for step in steps])
        assert deleted.rowcount == len(steps)


def repository_factory(tmp_path):
    """The in-memory repository factory over an empty test base."""
    initial = tmp_path / "initial"
    initial.mkdir()
    return InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")


def wait_until(condition, message: str) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(0.001)


def wait_for_step(dispatcher: Dispatcher, step: int, *, scenario: str = "math") -> None:
    wait_until(
        lambda: dispatcher.get_or_create_scenario(scenario).scenario_step >= step, "async training did not commit"
    )


def wait_for_jobs(dispatcher: Dispatcher, count: int, *, scenario: str = "math") -> None:
    """Wait until the scenario's trainer committed the end of ``count`` training jobs."""
    wait_until(
        lambda: dispatcher.get_or_create_scenario(scenario).trainer.state["finished_jobs"] >= count,
        "the training job did not finish",
    )


def train_pair(dispatcher: Dispatcher, index: int, runtime_load_id: str | None = None) -> None:
    """Send inference ``i<index>`` (served by ``runtime_load_id``, else by the served version) and its report
    ``r<index>``, then wait until ``index`` training jobs ended."""
    version = served_version(dispatcher) if runtime_load_id is None else runtime_load_id
    dispatcher.accept_record(sft_inference(f"i{index}", version))
    dispatcher.accept_record(sft_report(f"r{index}", f"i{index}"))
    wait_for_jobs(dispatcher, index)


class _SavedArtifactBackend(CandidateBackend):
    def __init__(self, artifact_path: Path) -> None:
        self._artifact_path = artifact_path
        self.data_ids: list[str] = []

    def initial_state(self):
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        del scenario_step
        self.data_ids.append(batch.data_id)
        return PreparedStep.with_candidate(
            UpdateCandidate(batch.data_id),
            state={"steps": int(state["steps"]) + 1},
        )

    def evaluate(self, candidate):
        del candidate
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision: SelectionDecision):
        del decision
        return TrainStepResult(prepared.state, artifact=Artifact.local(self._artifact_path))

    def abort_step(self, prepared):
        del prepared


@dataclass(frozen=True)
class _SavedArtifactRecipe(Recipe):
    backend: CandidateBackend

    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
    ):
        del experiment_logger
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=threshold_pipeline,
            candidate_backend=self.backend,
            algorithm_state=algorithm_state,
            default_job_samples=1,
        )


def _build_saved_artifact_dispatcher(tmp_path, *, agent_record_dir=None):
    artifact_path = tmp_path / "candidate"
    artifact_path.mkdir()
    (artifact_path / "model.txt").write_text("trained")
    backend = _SavedArtifactBackend(artifact_path)
    dispatcher = Dispatcher(
        _SavedArtifactRecipe(backend=backend),
        repository_factory(tmp_path),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        # The tests drive the steps themselves; a local cycle would race them for the batch.
        hold_local_cycles=True,
    )
    return dispatcher, backend


@pytest.mark.unit
def test_each_committed_step_appends_one_atomic_record_and_a_restart_recovers_from_the_log(tmp_path) -> None:
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = repository_factory(tmp_path)
    component = RecordingComponent(tmp_path / "exported")
    dispatcher = build_training_dispatcher(component, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    train_pair(dispatcher, 1)
    # The staleness check trains only data of the served version, so the second record comes from w1.
    train_pair(dispatcher, 2, "w1")

    # Each job commits its submission, its checkpoint decision and its end, one record each.
    records = committed_records(agent_record_dir)
    assert [record.step for record in records] == [1, 2, 3, 4, 5, 6]
    assert [record.checkpoint for record in records] == [False, True, False, False, True, False]
    first_job, second_job = (job.job_id for job in component.submitted)
    assert [record.training_job_id for record in records] == [first_job] * 3 + [second_job] * 3
    submitted, _, finished = records[:3]
    # The submission consumes the batch's records; the later commits of the job consume none.
    assert submitted.algorithm_state["job"] == {"job_id": first_job, "checkpoints_handled": 0}
    assert submitted.consumed_ids == frozenset({"i1", "r1", consumed_sample_id("r1:0")})
    assert submitted.high_water_sequence == 2
    assert finished.algorithm_state["finished_jobs"] == 1
    assert finished.consumed_ids == frozenset()
    assert records[3].consumed_ids == frozenset({"i2", "r2", consumed_sample_id("r2:0")})
    assert records[3].high_water_sequence == 4
    # The record's ref is exactly the serving head the scenario advanced to.
    scenario = dispatcher.get_or_create_scenario("math")
    committed_head = scenario.repository.require_current_artifact()
    assert committed_head == records[4].artifact_ref == records[5].artifact_ref
    committed_state = scenario.trainer.state
    assert committed_state["last_job"]["job_id"] == second_job
    # The checkpoint's metadata carries its commit record; the checkpoint commit consumes nothing.
    metadata = scenario.repository.backend.metadata()["scenario_commit_record"]
    assert metadata["scenario_step"] == 5
    assert metadata["algorithm_state"] == records[4].algorithm_state
    assert metadata["training_job_id"] == second_job
    assert metadata["record_progress"] == {"high_water_sequence": 4, "high_water_offset": 4, "consumed_ids": []}

    # A fresh dispatcher over the same dirs is a process restart: nothing is
    # durable except the release chain, the record store, and the commit history.
    second = build_training_dispatcher(
        RecordingComponent(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    recovered = second.get_or_create_scenario("math")
    assert recovered.scenario_step == 6
    assert recovered.repository.require_current_artifact() == committed_head
    assert recovered.trainer.state == committed_state
    # The next job starts from the second job's checkpoint.
    assert recovered.trainer.state["finished_jobs"] == 2
    assert recovered.trainer.state["start"] == str(tmp_path / "exported" / "2")


class ProtectAllPipeline(TrainingPipeline):
    """Pairs like SFT but never releases a row: every pushed id is protected."""

    def __init__(self, context) -> None:
        super().__init__(
            context,
            record2record=ReportRecord2Record(context, ReportGrouping()),
            record2sample=ThresholdSamples(),
            sample2job=ThresholdJobData(context, None, None),
        )
        self._seen: set[str] = set()

    def push(self, item) -> None:
        self._seen.add(item.agent_record_id)
        super().push(item)

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class ProtectAllPolicyRecipe(TestPolicyRecipe):
    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
    ):
        component = self.training_component
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=ProtectAllPipeline,
            job_loop=JobLoop(component, scenario=scenario),
            job_parameters={kind: {"method": {"objective": "sft"}} for kind in SAMPLE_KINDS},
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            default_job_samples=1,
        )


@pytest.mark.unit
def test_recovery_resumes_record_progress_without_retraining(tmp_path) -> None:
    """Consumed rows must never be trained twice, even when they survive compaction.

    The protect-all pipeline keeps every row physically present, so recovery
    without the history's high-water mark would page from sequence 0, push again
    the consumed pair, and train it again.
    """
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = repository_factory(tmp_path)

    def build(component):
        return Dispatcher(
            ProtectAllPolicyRecipe(**runtime_bindings(component)),
            backend_factory,
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        )

    first_component = RecordingComponent(tmp_path / "first")
    first = build(first_component)
    train_pair(first, 1)
    assert first_component.trained_data == [["i1"]]
    # Nothing was compacted: the consumed pair is still visible to training reads.
    assert first.get_or_create_scenario("math").records.get("math", "i1") is not None

    second_component = RecordingComponent(tmp_path / "second")
    second = build(second_component)
    recovered = second.get_or_create_scenario("math")
    assert recovered.trainer.data_offset == 2  # resumed at the committed watermark

    train_pair(second, 2)
    # The only newly trained batch is the genuinely new pair; i1 never retrains.
    assert second_component.trained_data == [["i2"]]
    assert recovered.scenario_step == 6


@pytest.mark.unit
def test_recovery_replays_retained_rows_behind_the_watermark(tmp_path) -> None:
    """Issue #344: the consumption cursor passes rows whose batches have not trained.

    The drain that readies the first batch also consumes the next batch's
    inference, so the committed high-water mark covers it while retention
    keeps it stored. Recovery must feed that row back to the rebuilt
    processor; otherwise the second report waits forever on a reference that
    can never be pushed again and training stalls silently. The consumed
    first pair, compacted and so absent from replay, still never trains twice.
    """
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = repository_factory(tmp_path)
    # The components save no checkpoint, so the base release serves both inferences and neither is stale.
    first_component = RecordingComponent()
    first = build_training_dispatcher(first_component, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    base_version = served_version(first)
    first.accept_record(sft_inference("i1", base_version))
    first.accept_record(sft_inference("i2", base_version))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_jobs(first, 1)
    assert first_component.trained_data == [["i1"]]
    # The second inference sits below the committed watermark yet survives
    # compaction: it is protected, and only processor memory knew about it.
    assert committed_records(agent_record_dir)[-1].high_water_sequence == 3
    assert first.get_or_create_scenario("math").records.get("math", "i2") is not None

    second_component = RecordingComponent()
    second = build_training_dispatcher(second_component, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("math")
    assert recovered.trainer.data_offset == 3  # resumed at the committed watermark
    assert served_version(second) == base_version

    second.accept_record(sft_report("r2", "i2"))
    wait_for_jobs(second, 2)
    # The replayed inference pairs with the late report; i1 never retrains.
    assert second_component.trained_data == [["i2"]]


@pytest.mark.unit
def test_recovery_skips_consumption_after_processor_release_failure(tmp_path, monkeypatch) -> None:
    """A committed batch cannot train twice even if processor cleanup fails."""
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = repository_factory(tmp_path)
    first_component = RecordingComponent(tmp_path / "first")
    first = build_training_dispatcher(first_component, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    scenario = first.get_or_create_scenario("math")

    original_release = scenario.trainer.pipeline.release_records
    crash = {"armed": True}

    def exploding_release(*args, **kwargs) -> None:
        if crash["armed"]:
            crash["armed"] = False
            raise RuntimeError("simulated crash between record append and buffer release")
        original_release(*args, **kwargs)

    monkeypatch.setattr(scenario.trainer.pipeline, "release_records", exploding_release)
    # The failed cycle does not wake its worker again; the worker polls the recovered running job.
    monkeypatch.setattr(local_workers, "POLL_SECONDS", 0.01)
    train_pair(first, 1)
    assert first_component.trained_data == [["i1"]]
    assert crash["armed"] is False

    second_component = RecordingComponent(tmp_path / "second")
    second = build_training_dispatcher(second_component, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("math")
    # Recovery uses committed consumption while leaving bodies readable.
    assert recovered.records.get("math", "i1") is not None
    assert recovered.scenario_step == 3

    train_pair(second, 2)
    assert second_component.trained_data == [["i2"]]
    assert recovered.scenario_step == 6


@pytest.mark.unit
def test_recovery_adopts_a_checkpoint_whose_record_was_lost(tmp_path) -> None:
    """Crash window: checkpoint published to the release chain, record never appended.

    The checkpoint head is then ahead of the commit history. Its metadata carries
    the same record fields, so recovery adopts it and heals the history.
    """
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = repository_factory(tmp_path)
    component = RecordingComponent(tmp_path / "exported")
    first = build_training_dispatcher(component, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    train_pair(first, 1)
    submitted, checkpoint, _ = committed_records(agent_record_dir)
    assert checkpoint.checkpoint is True
    assert checkpoint.activated is True

    # Simulate the crash after the checkpoint's publication: the history loses its checkpoint record and the
    # job's end, which a crashed process would never have committed.
    delete_commits(agent_record_dir, 2, 3)

    # The training component keeps its jobs across the restart; cycles stay held until the history is checked.
    second = build_training_dispatcher(
        component, tmp_path, backend_factory, agent_record_dir=agent_record_dir, hold_local_cycles=True
    )
    recovered = second.get_or_create_scenario("math")
    assert recovered.scenario_step == 2
    assert recovered.trainer.state == checkpoint.algorithm_state
    # The healed history carries the adopted checkpoint record, with the record
    # high-water mark taken from the checkpoint's metadata.
    records = committed_records(agent_record_dir)
    assert records[0] == submitted
    adopted = records[1]
    assert len(records) == 2
    assert adopted.step == 2
    assert adopted.checkpoint is True
    assert adopted.artifact_ref == recovered.repository.require_current_artifact()
    assert adopted.high_water_sequence == 2
    assert adopted.training_job_id == component.submitted[0].job_id
    assert adopted.operation == "training"
    assert adopted.operation_verified is True
    assert adopted.metrics == checkpoint.metrics
    recovered_status = second.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert recovered_status["step"] == 2
    assert recovered_status["metrics"] == checkpoint.metrics

    # The recovered trainer polls the job on, commits its end, and trains nothing again.
    second.open_local_cycles()
    wait_for_jobs(second, 1)
    assert [record.step for record in committed_records(agent_record_dir)] == [1, 2, 3]
    assert component.trained_data == [["i1"]]


@pytest.mark.unit
def test_recovery_rejects_a_gapped_log(tmp_path) -> None:
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = repository_factory(tmp_path)
    # The component saves no checkpoint, so every commit lies after the checkpointed step 0.
    first = build_training_dispatcher(
        RecordingComponent(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    base_version = served_version(first)
    for index in (1, 2, 3):
        train_pair(first, index, base_version)
    assert [record.step for record in committed_records(agent_record_dir)] == [1, 2, 3, 4, 5, 6]

    delete_commits(agent_record_dir, 3)

    second = build_training_dispatcher(
        RecordingComponent(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    with pytest.raises(ReefError, match="corrupt"):
        second.get_or_create_scenario("math")


@pytest.mark.unit
def test_no_commit_log_without_an_agent_record_dir(tmp_path) -> None:
    """In-memory deployments keep pre-log semantics: version-chain-only recovery."""
    backend_factory = repository_factory(tmp_path)
    component = RecordingComponent()
    first = build_training_dispatcher(component, tmp_path, backend_factory)
    train_pair(first, 1)

    store = first.get_or_create_scenario("math").store
    assert not store.durable
    assert [record.step for record in store.history()] == [1, 2]
    committed = first.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert committed["step"] == 2
    # The component saved no checkpoint, so the job failed and published no release.
    assert component.submitted[0].parameters == {"method": {"objective": "sft"}}
    assert committed["metrics"]["training_job"]["event"] == "failed"
    assert not list(tmp_path.rglob("*.sqlite3"))

    # Recovery still works from the release chain, as before the commit history
    # existed: the step reverts to the checkpointed 0.
    second = build_training_dispatcher(RecordingComponent(), tmp_path, backend_factory)
    recovered = second.get_or_create_scenario("math")
    assert recovered.scenario_step == 0
    assert recovered.store.history() == ()
    assert second.build_training_status()["scenarios"]["math"]["last_committed_step"] is None


@pytest.mark.unit
def test_checkpoint_status_metrics_survive_without_a_commit_log(tmp_path) -> None:
    backend_factory = repository_factory(tmp_path)
    first = build_training_dispatcher(RecordingComponent(tmp_path / "exported"), tmp_path, backend_factory)
    train_pair(first, 1)
    checkpoint = first.get_or_create_scenario("math").store.history()[1]
    assert checkpoint.checkpoint is True

    first.close()
    # Inspect checkpoint recovery before the worker can commit the missing job's failure.
    second = build_training_dispatcher(RecordingComponent(), tmp_path, backend_factory, hold_local_cycles=True)
    recovered = second.get_or_create_scenario("math")

    # The release chain ends at the checkpoint, so recovery returns to its step and metrics.
    assert recovered.scenario_step == 2
    assert not recovered.store.durable
    recovered_status = second.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert recovered_status["step"] == 2
    assert recovered_status["metrics"] == checkpoint.metrics
    assert not list(tmp_path.rglob("*.sqlite3"))

    second.close()


def build_harness_evolve_dispatcher(
    tmp_path, backend_factory, *, method=lambda workspace, services: None, agent_record_dir=None
):
    """A dispatcher over a pi harness recipe; these tests need an artifact-producing recipe, not its semantics."""
    return Dispatcher(
        harness_recipe(tmp_path, method),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
    )


def trace_inference(
    agent_record_id: str,
    system_text: str = "skill v0",
    *,
    scenario: str = "skills",
) -> AgentRecord:
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "system", "content": system_text}, {"role": "user", "content": "q"}]},
        agent_record_id=agent_record_id,
    )


def trace_report(agent_record_id: str, reference: str, *, scenario: str = "skills") -> AgentRecord:
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 0.0, "references": [reference]},
        agent_record_id=agent_record_id,
        references=(reference,),
    )


@pytest.mark.unit
def test_harness_growth_does_not_block_acceptance_or_other_scenarios(tmp_path) -> None:
    started = Event()
    release = Event()

    def blocking_method(workspace, services):
        if workspace.traces()[0].record != "a-i1":
            return
        started.set()
        assert release.wait(10)

    dispatcher = build_harness_evolve_dispatcher(
        tmp_path, harness_repository(tmp_path, PI_SEED.files), method=blocking_method
    )
    dispatcher.accept_record(trace_inference("a-i1", scenario="skills-a"))

    returned = Event()
    errors: list[Exception] = []

    def accept_report() -> None:
        try:
            dispatcher.accept_record(
                trace_report("a-r1", "a-i1", scenario="skills-a"),
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            returned.set()

    request = Thread(target=accept_report)
    request.start()
    assert started.wait(5), "harness growth did not start in time"
    status_values = []
    status_returned = Event()

    def read_status() -> None:
        try:
            status_values.append(dispatcher.build_training_status())
        except Exception as exc:
            errors.append(exc)
        finally:
            status_returned.set()

    status_request = Thread(target=read_status)
    status_request.start()
    try:
        # The proposer holds until ``release``, which only this block's end sets.
        assert returned.wait(5), "report acceptance waited for harness growth"
        assert status_returned.wait(5), "status waited for harness growth"
        status = status_values[0]["scenarios"]["skills-a"]
        assert status["scenario_step"] == 0
        assert status["last_committed_step"] is None
        dispatcher.accept_record(trace_inference("b-i1", scenario="skills-b"))
        dispatcher.accept_record(
            trace_report("b-r1", "b-i1", scenario="skills-b"),
        )
        wait_for_step(dispatcher, 1, scenario="skills-b")
    finally:
        release.set()
        request.join()
        status_request.join()

    assert errors == []
    scenario = dispatcher.get_or_create_scenario("skills-a")
    wait_for_step(dispatcher, 1, scenario="skills-a")
    assert scenario.scenario_step == 1
    dispatcher.close()


@pytest.mark.unit
def test_local_backend_failure_reaches_status_and_retries_pending_job_data(tmp_path) -> None:
    calls = 0
    should_fail = True

    def method(workspace, services):
        nonlocal calls
        calls += 1
        if should_fail:
            raise RuntimeError("the method failed")

    dispatcher = build_harness_evolve_dispatcher(tmp_path, harness_repository(tmp_path, PI_SEED.files), method=method)
    dispatcher.accept_record(trace_inference("i1"))
    dispatcher.accept_record(trace_report("r1", "i1"))

    wait_until(lambda: dispatcher.build_training_status()["error"] is not None, "the failure did not reach the status")
    assert dispatcher.build_training_status()["error"] == "skills: RuntimeError: the method failed"

    failed_attempts = calls
    should_fail = False
    dispatcher.accept_record(trace_inference("i2"))
    wait_for_step(dispatcher, 1, scenario="skills")
    assert calls == failed_attempts + 1
    assert dispatcher.build_training_status()["error"] is None
    dispatcher.close()


@pytest.mark.unit
@pytest.mark.parametrize("failure_point", ["stage", "publish", "activation", "commit_history"])
def test_artifact_commit_failure_keeps_the_pending_job_data_retryable(tmp_path, monkeypatch, failure_point) -> None:
    agent_record_dir = tmp_path / "records" if failure_point == "commit_history" else None
    dispatcher, backend = _build_saved_artifact_dispatcher(
        tmp_path,
        agent_record_dir=agent_record_dir,
    )
    scenario = dispatcher.get_or_create_scenario("math")
    scenario.records.append(sft_inference("i1"))
    scenario.records.append(sft_report("r1", "i1"))
    result = scenario.prepare_training_step()
    assert result is not None
    pending = scenario.trainer.pending_job_data
    assert pending is not None
    committer = scenario._committer

    if failure_point == "commit_history":
        assert isinstance(scenario.store, CommitHistoryScenarioStore)
        target = scenario.store.commit_history
        method_name = "append"
        assert target is not None
    elif failure_point == "activation":
        target = committer
        method_name = "_activate"
    else:
        target = committer._artifacts
        method_name = failure_point
    original = getattr(target, method_name)
    calls_before_failure = 2 if failure_point == "activation" else 1

    def fail_once(*args, **kwargs):
        nonlocal calls_before_failure
        calls_before_failure -= 1
        if calls_before_failure == 0:
            raise RuntimeError(f"injected {failure_point} failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method_name, fail_once)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        scenario.commit(result)

    assert scenario.scenario_step == 0
    assert scenario.trainer.state == {"steps": 0}
    assert scenario.trainer.pending_job_data is pending
    assert [record.agent_record_id for record in scenario.records.replay("math")] == ["i1", "r1"]

    scenario.records.append(sft_inference("i2"))
    scenario.commit(result)

    assert backend.data_ids == [pending.data_id]
    assert scenario.trainer.state == {"steps": 1}
    assert scenario.trainer.pending_job_data is None
    assert [record.agent_record_id for record in scenario.records.replay("math")] == ["i1", "r1", "i2"]
    assert len(scenario.store.history()) == 1
    dispatcher.close()


@pytest.mark.unit
@pytest.mark.parametrize("failure_point", ["commit_applied", "release_records"])
def test_post_commit_failure_resumes_the_recorded_step_without_republication(
    tmp_path,
    monkeypatch,
    failure_point,
) -> None:
    dispatcher, backend = _build_saved_artifact_dispatcher(
        tmp_path,
        agent_record_dir=tmp_path / "records",
    )
    scenario = dispatcher.get_or_create_scenario("math")
    scenario.records.append(sft_inference("i1"))
    scenario.records.append(sft_report("r1", "i1"))
    result = scenario.prepare_training_step()
    assert result is not None
    pending = scenario.trainer.pending_job_data
    assert pending is not None
    target = scenario.trainer.pipeline if failure_point == "release_records" else scenario.trainer
    original = getattr(target, failure_point)
    should_fail = True

    def fail_after_effect(*args, **kwargs):
        nonlocal should_fail
        outcome = original(*args, **kwargs)
        if should_fail:
            should_fail = False
            raise RuntimeError(f"injected {failure_point} failure")
        return outcome

    monkeypatch.setattr(target, failure_point, fail_after_effect)

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        scenario.commit(result)

    assert scenario.scenario_step == 0
    assert scenario.trainer.state == {"steps": 0}
    assert scenario.trainer.pending_job_data is pending
    assert scenario.store.durable
    [record] = scenario.store.history()
    published_release = record.artifact_ref.release_id
    with pytest.raises(RuntimeError, match="does not match the pending step"):
        scenario.commit(replace(result, state={"steps": 999}))

    scenario.commit(result)

    assert scenario.scenario_step == 1
    assert scenario.trainer.state == {"steps": 1}
    assert scenario.trainer.pending_job_data is None
    assert len(scenario.store.history()) == 1
    assert scenario.repository.require_current_artifact().release_id == published_release
    assert backend.data_ids == [pending.data_id]
    dispatcher.close()


@pytest.mark.unit
def test_a_commit_failure_keeps_one_retryable_batch_and_one_record(tmp_path, monkeypatch) -> None:
    agent_record_dir = tmp_path / "records"
    component = RecordingComponent(tmp_path / "exported")
    dispatcher = build_training_dispatcher(
        component,
        tmp_path,
        repository_factory(tmp_path),
        agent_record_dir=agent_record_dir,
        hold_local_cycles=True,
    )
    scenario = dispatcher.get_or_create_scenario("math")
    scenario.records.append(sft_inference("i1", scenario.serving_runtime_load_id()))
    scenario.records.append(sft_report("r1", "i1"))
    # The job's submission is the step that consumes the batch.
    result = scenario.prepare_training_step()
    assert result is not None
    pending = scenario.trainer.pending_job_data
    assert pending is not None

    # The step is stored, but the caller sees an error.
    original = scenario.store.commit_step
    should_fail = True

    def fail_once(*args, **kwargs):
        nonlocal should_fail
        if should_fail:
            should_fail = False
            original(*args, **kwargs)
            raise RuntimeError("injected commit_step failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(scenario.store, "commit_step", fail_once)

    with pytest.raises(RuntimeError, match="injected commit_step failure"):
        scenario.commit(result)

    assert scenario.scenario_step == 0
    assert scenario.trainer.state == JobLoopState(scenario.trainer.job_loop_required().component.base_start).to_dict()
    assert scenario.trainer.pending_job_data is pending
    assert scenario.store.durable
    assert len(scenario.store.history()) == 1

    scenario.commit(result)

    assert scenario.scenario_step == 1
    assert scenario.trainer.state == result.state
    assert scenario.trainer.state["job"]["job_id"] == component.submitted[0].job_id
    assert scenario.trainer.pending_job_data is None
    assert scenario.store.training_run_position() == (0, 1)
    assert len(scenario.store.history()) == 1
    assert component.trained_data == [["i1"]]
    dispatcher.close()


@pytest.mark.unit
def test_durable_local_backend_recovers_after_post_commit_notification_failure(tmp_path) -> None:
    agent_record_dir = tmp_path / "agent-record"
    dispatcher = build_harness_evolve_dispatcher(
        tmp_path, harness_repository(tmp_path, PI_SEED.files), agent_record_dir=agent_record_dir
    )
    original = dispatcher.get_or_create_scenario("skills")
    assert original is not None
    # The creation carries the base artifact, whose artifact_id is the repository base release's.
    activation_target = original.repository.backend.resolve_release("base").release_id
    release_failed = Event()
    reload_started = Event()
    allow_reload = Event()
    reload_scenario = dispatcher.registry.reload

    def fail_release(self, record_ids) -> None:
        del self, record_ids
        release_failed.set()
        raise RuntimeError("simulated failure after commit record append")

    def blocking_reload(scenario: str):
        reload_started.set()
        assert allow_reload.wait(1)
        return reload_scenario(scenario)

    original.trainer.pipeline.release_records = fail_release.__get__(
        original.trainer.pipeline, type(original.trainer.pipeline)
    )
    dispatcher.registry.reload = blocking_reload
    dispatcher.accept_record(trace_inference("i1"))
    dispatcher.accept_record(trace_report("r1", "i1"))

    assert release_failed.wait(1)
    assert reload_started.wait(1)
    activation_started = Event()
    activation_returned = Event()
    activation_errors: list[Exception] = []

    def activate() -> None:
        activation_started.set()
        try:
            dispatcher.activate("skills", activation_target)
        except Exception as exc:
            activation_errors.append(exc)
        finally:
            activation_returned.set()

    activation_thread = Thread(target=activate)
    activation_thread.start()
    assert activation_started.wait(1)
    try:
        assert not activation_returned.wait(0.05), "activation entered before durable recovery"
    finally:
        allow_reload.set()
    activation_thread.join(1)
    assert not activation_thread.is_alive()
    assert activation_errors == []

    wait_for_step(dispatcher, 1, scenario="skills")
    recovered = dispatcher.get_or_create_scenario("skills")
    assert recovered is not original
    assert recovered.records.get("skills", "i1") is not None
    assert recovered.records.get("skills", "r1") is not None
    status = dispatcher.build_training_status()["scenarios"]["skills"]
    assert status["scenario_step"] == 1
    assert status["last_committed_step"]["step"] == 1
    assert status["last_committed_step"]["metrics"]["skipped"] == "the method left the head tree unchanged"
    records = committed_records(agent_record_dir, scenario="skills")
    assert [record.step for record in records] == [1]
    assert records[0].metrics is not None
    assert records[0].metrics["skipped"] == "the method left the head tree unchanged"

    dispatcher.accept_record(trace_inference("i2"))
    dispatcher.accept_record(trace_report("r2", "i2"))
    wait_for_step(dispatcher, 2, scenario="skills")

    assert [record.step for record in committed_records(agent_record_dir, scenario="skills")] == [1, 2]
    status = dispatcher.build_training_status()["scenarios"]["skills"]
    assert status["scenario_step"] == 2
    assert status["last_committed_step"]["step"] == 2
    dispatcher.close()


@pytest.mark.unit
def test_no_artifact_commit_appends_a_record_and_survives_a_restart(tmp_path) -> None:
    """A step that publishes no artifact must still enter the commit history.

    Harness evolution frequently proposes nothing (or a losing candidate);
    the trainer still consumed records and advanced its algorithm state.
    Recording that progress keeps the commit history the single durable commit point: a
    crash loses neither the batch nor the record watermark, and recovery
    restores the step, the state, and the record progress without re-training compacted rows.
    """
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = harness_repository(tmp_path, PI_SEED.files)
    first = build_harness_evolve_dispatcher(tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(trace_inference("i1"))
    first.accept_record(trace_report("r1", "i1"))
    wait_for_step(first, 1, scenario="skills")
    scenario = first.get_or_create_scenario("skills")
    assert scenario.scenario_step == 1
    committed = first.build_training_status()["scenarios"]["skills"]["last_committed_step"]
    assert committed["step"] == 1
    assert committed["metrics"]["skipped"] == "the method left the head tree unchanged"
    [record] = committed_records(agent_record_dir, scenario="skills")
    assert record.step == 1
    assert record.checkpoint is False
    assert record.artifact_ref == scenario.repository.require_current_artifact()  # no new artifact published
    assert record.algorithm_state == HarnessState().to_json()
    assert record.high_water_sequence == 2
    assert record.consumed_ids == frozenset({"i1", "r1", consumed_sample_id("r1:0")})

    second = build_harness_evolve_dispatcher(tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("skills")
    assert recovered.scenario_step == 1
    assert recovered.trainer.state == HarnessState().to_json()
    assert second.build_training_status()["scenarios"]["skills"]["last_committed_step"] == committed
    # Record progress resumes at the committed watermark, not sequence 0.
    assert recovered.trainer.data_offset == 2
    first.close()
    second.close()
