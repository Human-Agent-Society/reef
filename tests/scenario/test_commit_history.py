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
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from threading import Event, Thread

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support._strategies import resolve_episode_scorer, resolve_proposer
from tests.support._threshold_processor import ThresholdBatches, ThresholdTasks, threshold_pipeline
from tests.support.runtime_stubs import StubTrainingRuntime, candidate_backend, runtime_bindings

from reef.artifact import Artifact, ArtifactRef, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.evaluation.harness import ScoreComparisonPlugin
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.recipe.base import Recipe
from reef.record2record import ReportGrouping, ReportRecord2Record
from reef.runtime.interfaces import ActivatedModel, ModelCandidate, PreparedTrainingStep
from reef.storage.commit_history import RECORD_KIND, CommitHistoryError, CommitHistoryScenarioStore
from reef.storage.commits import CommitRecord
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.surface import Surface
from reef.surface.harnesses import create_harness_surface
from reef.train import CandidateBackend, PreparedStep, Trainer, TrainStepResult
from reef.train.algos import StepScheduling
from reef.train.cordis_backend import CordisBackend
from reef.train.cordis_backend.processor import TraceBatches, TraceTasks
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
    log.append(sample_record(step=2, runtime_load_id="w2"))

    records = log.records()
    assert [record.step for record in records] == [1, 2]
    first = records[0]
    assert first.scenario == "math"
    assert first.artifact_ref == ArtifactRef("content:1", "release:w1:1", "base")
    assert first.checkpoint is False
    assert first.algorithm_state == {"steps": 1}
    assert first.high_water_sequence == 2
    assert first.high_water_offset == 2
    assert first.consumed_ids == frozenset({"i1"})
    assert first.recorded_at == 1001.0
    # A fresh reader over the same database sees the same history.
    assert open_history().records() == records


@pytest.mark.unit
def test_commit_record_round_trips_gate_metrics(open_history) -> None:
    log = open_history()
    log.append(sample_record(step=1))
    metrics = {"changed": "config/format.json", "wins": 3, "losses": 1, "ties": 0, "published": True}
    log.append(sample_record(step=2, runtime_load_id="w2", metrics=metrics))

    first, second = open_history().records()
    assert first.metrics is None  # pre-metrics records stay readable
    assert second.metrics == metrics


@pytest.mark.unit
def test_commit_record_detaches_nested_metrics() -> None:
    metrics = {"selection": {"selected": True}}
    record = sample_record(metrics=metrics)

    metrics["selection"]["selected"] = False
    encoded = record.to_dict()
    encoded["metrics"]["selection"]["selected"] = False

    assert record.metrics == {"selection": {"selected": True}}


@pytest.mark.unit
def test_commit_record_round_trips_training_job_identity(open_history) -> None:
    log = open_history()
    log.append(sample_record(training_job_id="job-1"))

    [record] = log.records()
    assert record.training_job_id == "job-1"
    assert record.to_dict()["training_job_id"] == "job-1"


@pytest.mark.unit
def test_commit_history_tracks_the_current_training_run_position(open_history) -> None:
    log = open_history()
    log.append(sample_record(step=1))
    log.append(sample_record(step=2, runtime_load_id="w2"))
    assert log.training_run_position() == (0, 2)

    rollback = CommitRecord(
        scenario="math",
        step=3,
        artifact_ref=ArtifactRef("artifact:1", "checkpoint:3", "checkpoint:1"),
        checkpoint=True,
        algorithm_state={"steps": 2},
        high_water_sequence=4,
        high_water_offset=4,
        operation="rollback",
        rollback_target_release_id="checkpoint:1",
    )
    log.append(rollback)
    assert log.training_run_position() == (3, 0)

    log.append(sample_record(step=4, runtime_load_id="w4"))
    assert log.training_run_position() == (3, 1)


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
        CommitRecord(
            scenario="math",
            step=0,
            artifact_ref=ArtifactRef(content_id="a", release_id="r", parent_release_id=None),
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
        )
    with pytest.raises(CommitHistoryError, match="high_water_sequence"):
        CommitRecord(
            scenario="math",
            step=1,
            artifact_ref=ArtifactRef(content_id="a", release_id="r", parent_release_id=None),
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=-1,
            high_water_offset=0,
        )
    with pytest.raises(CommitHistoryError, match="record kind"):
        CommitRecord.from_dict({"record": "something-else"})
    with pytest.raises(CommitHistoryError, match="compacted_ids"):
        CommitRecord.from_dict(
            {
                "record": RECORD_KIND,
                "scenario": "math",
                "step": 1,
                "artifact_ref": {
                    "kind": "artifact",
                    "content_id": "a",
                    "release_id": "r",
                    "parent_release_id": None,
                },
                "checkpoint": False,
                "algorithm_state": None,
                "record_progress": {"high_water_sequence": 0, "high_water_offset": 0, "compacted_ids": [1]},
                "recorded_at": 1.0,
            }
        )


class RecordingRuntime(StubTrainingRuntime):
    """Training runtime that records the batches it trained."""

    def __init__(
        self,
        *,
        served_version: str | None = None,
    ) -> None:
        super().__init__(base_url="http://trainer")
        self.trained_batches: list[list[str]] = []
        self._served_version = served_version
        self._checkpoint_dir = None
        self._candidate_versions: dict[str, str] = {}

    @property
    def inference_handler(self):
        return None

    def prepare_training_step(
        self, batch, objective, algorithm_state, scheduling, scenario_step, *, parent_runtime_load_id=None
    ):
        del objective
        payload = {
            "scenario_step": scenario_step,
            "sources": [source_record_id(sample) for sample in batch.items],
        }
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state={"steps": int(algorithm_state.get("steps", 0)) + 1},
            metrics={},
        )

    def train_candidate(self, payload):
        self.trained_batches.append(payload["sources"])
        runtime_load_id = f"w{len(self.trained_batches)}"
        if self._checkpoint_dir is None:
            # Without a checkpoint directory, the runtime saves nothing, so its jobs publish nothing.
            checkpoint_path = None
        else:
            checkpoint = self._checkpoint_dir / runtime_load_id
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.txt").write_text(runtime_load_id)
            checkpoint_path = str(checkpoint)
        job_id = f"job-{payload['scenario_step']}"
        self._candidate_versions[job_id] = runtime_load_id
        return ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=checkpoint_path,
            current_runtime_load_id=self._served_version,
        )

    def activate_candidate(self, candidate):
        runtime_load_id = self._candidate_versions[candidate.candidate_id]
        self._served_version = runtime_load_id
        return ActivatedModel(candidate.candidate_id, runtime_load_id)

    def reject_candidate(self, candidate):
        self._candidate_versions.pop(candidate.candidate_id, None)

    def serving_runtime_load_id(self):
        return self._served_version


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
    runtime,
    tmp_path,
    backend_factory,
    *,
    agent_record_dir=None,
):
    return Dispatcher(
        TestPolicyRecipe(**runtime_bindings(runtime)),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
    )


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


def wait_for_step(dispatcher: Dispatcher, step: int, *, scenario: str = "math") -> None:
    for _ in range(1000):
        if dispatcher.get_or_create_scenario(scenario).scenario_step >= step:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not commit")


class _SavedArtifactBackend(CandidateBackend):
    def __init__(self, artifact_path: Path) -> None:
        self._artifact_path = artifact_path
        self.batch_ids: list[str] = []

    def initial_state(self):
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        del scenario_step
        self.batch_ids.append(batch.batch_id)
        return PreparedStep.with_candidate(
            UpdateCandidate(batch.batch_id),
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

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        del experiment_logger
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=threshold_pipeline,
            candidate_backend=self.backend,
            algorithm_state=algorithm_state,
        )


def _build_saved_artifact_dispatcher(tmp_path, *, agent_record_dir=None):
    initial = tmp_path / "initial"
    initial.mkdir()
    artifact_path = tmp_path / "candidate"
    artifact_path.mkdir()
    (artifact_path / "model.txt").write_text("trained")
    backend = _SavedArtifactBackend(artifact_path)
    dispatcher = Dispatcher(
        _SavedArtifactRecipe(backend=backend),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
    )
    return dispatcher, backend


@pytest.mark.unit
def test_each_committed_step_appends_one_atomic_record(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=agent_record_dir,
    )
    dispatcher.accept_record(sft_inference("i1"))
    dispatcher.accept_record(sft_report("r1", "i1"))
    wait_for_step(dispatcher, 1)
    # The staleness check trains only data of the served version, so the second record comes from w1.
    dispatcher.accept_record(sft_inference("i2", "w1"))
    dispatcher.accept_record(sft_report("r2", "i2"))
    wait_for_step(dispatcher, 2)

    records = committed_records(agent_record_dir)
    assert [record.step for record in records] == [1, 2]
    first, second = records
    assert first.checkpoint is True and second.checkpoint is True
    assert first.algorithm_state == {"steps": 1}
    assert first.consumed_ids == frozenset({"i1", "r1"})
    assert first.high_water_sequence == 2
    assert second.high_water_sequence == 4
    # The record's ref is exactly the serving head the scenario advanced to.
    scenario = dispatcher.get_or_create_scenario("math")
    assert scenario.repository.require_current_artifact() == second.artifact_ref
    assert scenario.committed_training_job_id == "job-1"


@pytest.mark.unit
def test_training_commit_proves_its_training_job_identity(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=agent_record_dir,
    )
    dispatcher.accept_record(sft_inference("i1"))
    dispatcher.accept_record(sft_report("r1", "i1"))
    wait_for_step(dispatcher, 1)

    scenario = dispatcher.get_or_create_scenario("math")
    [record] = committed_records(agent_record_dir)
    assert record.training_job_id == "job-0"
    assert scenario.committed_training_job_id == "job-0"


@pytest.mark.unit
def test_recovery_restores_the_head_step_and_state_from_the_log(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    first = build_training_dispatcher(runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(sft_inference("i1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)
    first.accept_record(sft_inference("i2", "w1"))
    first.accept_record(sft_report("r2", "i2"))
    wait_for_step(first, 2)
    committed_head = first.get_or_create_scenario("math").repository.require_current_artifact()

    # A fresh dispatcher over the same dirs is a process restart: nothing is
    # durable except the release chain, the record store, and the commit history.
    second = build_training_dispatcher(
        RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    recovered = second.get_or_create_scenario("math")
    assert recovered.scenario_step == 2
    assert recovered.repository.require_current_artifact() == committed_head
    assert recovered.trainer.state == {"steps": 2}


class ProtectAllPipeline(TrainingPipeline):
    """Pairs like SFT but never releases a row: every pushed id is protected."""

    def __init__(self, context) -> None:
        super().__init__(
            context,
            record2record=ReportRecord2Record(context, ReportGrouping()),
            record2task=ThresholdTasks(),
            task2batch=ThresholdBatches(context, None),
        )
        self._seen: set[str] = set()

    def push(self, item) -> None:
        self._seen.add(item.agent_record_id)
        super().push(item)

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class ProtectAllPolicyRecipe(TestPolicyRecipe):
    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=ProtectAllPipeline,
            candidate_backend=candidate_backend(self.training_runtime, "sft", StepScheduling()),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )


@pytest.mark.unit
def test_recovery_resumes_record_progress_without_retraining(tmp_path) -> None:
    """Consumed rows must never be trained twice, even when they survive compaction.

    The protect-all pipeline keeps every row physically present, so recovery
    without the history's high-water mark would page from sequence 0, push again
    the consumed pair, and train it again.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    recipe = lambda runtime: ProtectAllPolicyRecipe(**runtime_bindings(runtime))  # noqa: E731

    def build(runtime):
        return Dispatcher(
            recipe(runtime),
            backend_factory,
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        )

    first_runtime = RecordingRuntime()
    first = build(first_runtime)
    first.accept_record(sft_inference("i1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)
    assert first_runtime.trained_batches == [["i1"]]
    # Nothing was compacted: the consumed pair is still visible to training reads.
    assert first.get_or_create_scenario("math").records.get("math", "i1") is not None

    second_runtime = RecordingRuntime()
    second = build(second_runtime)
    recovered = second.get_or_create_scenario("math")
    assert recovered.trainer.data_offset == 2  # resumed at the committed watermark

    # The recovered head serves w1, so the new pair comes from w1.
    second.accept_record(sft_inference("i2", "w1"))
    second.accept_record(sft_report("r2", "i2"))
    wait_for_step(second, 2)
    # The only newly trained batch is the genuinely new pair; i1 never retrains.
    assert second_runtime.trained_batches == [["i2"]]
    assert recovered.scenario_step == 2


@pytest.mark.unit
def test_recovery_replays_retained_rows_behind_the_watermark(tmp_path) -> None:
    """Issue #344: the consumption cursor passes rows whose batches have not trained.

    The drain that readies step 1 also consumes the next step's inference, so
    the committed high-water mark covers it while retention keeps it stored.
    Recovery must feed that row back to the rebuilt processor; otherwise the
    step-2 report waits forever on a reference that can never be pushed
    again and training stalls silently. The consumed step-1 pair, compacted
    and so absent from replay, still never trains twice.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first_runtime = RecordingRuntime()
    first = build_training_dispatcher(first_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(sft_inference("i1"))
    # The staleness check trains only data of the served version; the step-2 inference claims w1 for this test.
    first.accept_record(sft_inference("i2", "w1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)
    assert first_runtime.trained_batches == [["i1"]]
    # The step-2 inference sits below the committed watermark yet survives
    # compaction: it is protected, and only processor memory knew about it.
    assert committed_records(agent_record_dir)[-1].high_water_sequence == 3
    assert first.get_or_create_scenario("math").records.get("math", "i2") is not None

    second_runtime = RecordingRuntime()
    second = build_training_dispatcher(second_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("math")
    assert recovered.trainer.data_offset == 3  # resumed at the committed watermark

    second.accept_record(sft_report("r2", "i2"))
    wait_for_step(second, 2)
    # The replayed inference pairs with the late report; i1 never retrains.
    assert second_runtime.trained_batches == [["i2"]]
    assert recovered.scenario_step == 2


@pytest.mark.unit
def test_recovery_skips_consumption_after_processor_release_failure(tmp_path, monkeypatch) -> None:
    """A committed batch cannot train twice even if processor cleanup fails."""
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first_runtime = RecordingRuntime()
    first = build_training_dispatcher(first_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    scenario = first.get_or_create_scenario("math")

    original_release = scenario.trainer.pipeline.release_records
    crash = {"armed": True}

    def exploding_release(*args, **kwargs) -> None:
        if crash["armed"]:
            crash["armed"] = False
            raise RuntimeError("simulated crash between record append and buffer release")
        original_release(*args, **kwargs)

    monkeypatch.setattr(scenario.trainer.pipeline, "release_records", exploding_release)
    first.accept_record(sft_inference("i1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)
    assert first_runtime.trained_batches == [["i1"]]
    assert crash["armed"] is False

    second_runtime = RecordingRuntime()
    second = build_training_dispatcher(second_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("math")
    # Recovery uses committed consumption while leaving bodies readable.
    assert recovered.records.get("math", "i1") is not None
    assert recovered.scenario_step == 1

    second.accept_record(sft_inference("i2", "w1"))
    second.accept_record(sft_report("r2", "i2"))
    wait_for_step(second, 2)
    assert second_runtime.trained_batches == [["i2"]]
    assert recovered.scenario_step == 2


@pytest.mark.unit
def test_recovery_adopts_a_checkpoint_whose_record_was_lost(tmp_path) -> None:
    """Crash window: checkpoint published to the release chain, record never appended.

    The checkpoint head is then ahead of the commit history. Its metadata carries
    the same record fields, so recovery adopts it and heals the history.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    first = build_training_dispatcher(
        runtime,
        tmp_path,
        backend_factory,
        agent_record_dir=agent_record_dir,
    )
    first.accept_record(sft_inference("i1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)
    assert first.get_or_create_scenario("math").scenario_step == 1
    committed = first.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert committed["step"] == 1
    assert committed["metrics"]["selected"] is True

    # Simulate the crash: the history loses its last (only) record.
    delete_commits(agent_record_dir, 1)

    second = build_training_dispatcher(
        RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    recovered = second.get_or_create_scenario("math")
    assert recovered.scenario_step == 1
    assert recovered.trainer.state == {"steps": 1}
    # The healed history carries the adopted checkpoint record, with the record
    # high-water mark taken from the checkpoint's metadata.
    records = committed_records(agent_record_dir)
    assert len(records) == 1
    adopted = records[0]
    assert adopted.step == 1
    assert adopted.checkpoint is True
    assert adopted.artifact_ref == recovered.repository.require_current_artifact()
    assert adopted.high_water_sequence == 2
    assert adopted.training_job_id == "job-0"
    assert adopted.operation == "training"
    assert adopted.operation_verified is True
    assert adopted.metrics == committed["metrics"]
    recovered_status = second.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert recovered_status["step"] == 1
    assert recovered_status["metrics"] == committed["metrics"]


@pytest.mark.unit
def test_checkpoint_metadata_restores_a_commit_record(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=tmp_path / "agent-record",
    )
    dispatcher.accept_record(sft_inference("i1"))
    dispatcher.accept_record(sft_report("r1", "i1"))
    wait_for_step(dispatcher, 1)

    backend = dispatcher.get_or_create_scenario("math").repository.backend
    metadata = backend.metadata()["scenario_commit_record"]
    assert metadata["scenario_step"] == 1
    assert metadata["algorithm_state"] == {"steps": 1}
    assert metadata["record_progress"] == {
        "high_water_sequence": 2,
        "high_water_offset": 2,
        "consumed_ids": ["i1", "r1"],
    }


@pytest.mark.unit
def test_recovery_rejects_a_gapped_log(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    for index in (1, 2, 3):
        first.accept_record(sft_inference(f"i{index}", None if index == 1 else f"w{index - 1}"))
        first.accept_record(sft_report(f"r{index}", f"i{index}"))
        wait_for_step(first, index)

    delete_commits(agent_record_dir, 2)

    second = build_training_dispatcher(
        RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    with pytest.raises(ReefError, match="corrupt"):
        second.get_or_create_scenario("math")


@pytest.mark.unit
def test_no_commit_log_without_an_agent_record_dir(tmp_path) -> None:
    """In-memory deployments keep pre-log semantics: version-chain-only recovery."""
    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory)
    first.accept_record(sft_inference("i1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)

    store = first.get_or_create_scenario("math").store
    assert not store.durable
    assert [record.step for record in store.history()] == [1]
    committed = first.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert committed["step"] == 1
    # The runtime saved nothing, so the step trained but published no release.
    assert committed["metrics"]["training_method"]["objective"] == "sft"
    assert not list(tmp_path.rglob("*.sqlite3"))

    # Recovery still works from the release chain, as before the commit history
    # existed: the step reverts to the checkpointed 0.
    second = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory)
    recovered = second.get_or_create_scenario("math")
    assert recovered.scenario_step == 0
    assert recovered.store.history() == ()
    assert second.build_training_status()["scenarios"]["math"]["last_committed_step"] is None


@pytest.mark.unit
def test_checkpoint_status_metrics_survive_without_a_commit_log(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    first = build_training_dispatcher(
        runtime,
        tmp_path,
        backend_factory,
    )
    first.accept_record(sft_inference("i1"))
    first.accept_record(sft_report("r1", "i1"))
    wait_for_step(first, 1)
    committed = first.build_training_status()["scenarios"]["math"]["last_committed_step"]

    second = build_training_dispatcher(
        RecordingRuntime(),
        tmp_path,
        backend_factory,
    )
    recovered = second.get_or_create_scenario("math")

    assert recovered.scenario_step == 1
    assert not recovered.store.durable
    recovered_status = second.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert recovered_status["step"] == 1
    assert recovered_status["metrics"] == committed["metrics"]
    assert not list(tmp_path.rglob("*.sqlite3"))


@dataclass(frozen=True)
class _HarnessEvolveTestRecipe(Recipe):
    """Private harness-evolution stand-in for these mechanics tests.

    These tests exercise commit-history/version-chain mechanics, not
    harness-evolution semantics specifically; they just need a concrete,
    artifact-producing Recipe. Constructs CordisBackend and
    the reported trace pipeline directly, the same shape
    CordisRecipe.build() has, minus the config-boot layer these
    tests never touch (see tests/reef_service/test_harness_recipe.py for
    the backend's own guarantees).
    """

    propose: object
    evaluate: object
    tasks: tuple[str, ...]
    batch_size: int = 1
    _: KW_ONLY
    name: str = "harness_evolve"

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface()

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None) -> Trainer:
        candidate_backend = CordisBackend(
            descriptor=get_adapter("pi"),
            propose=resolve_proposer(self.propose),
            score_episode=resolve_episode_scorer(self.evaluate),
            tasks=self.tasks,
            binary="fake-pi",
            models=ModelBinding(base_url="http://localhost:8000", model="qwen3-8b"),
        )
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(
                context,
                record2record=ReportRecord2Record(context, ReportGrouping()),
                record2task=TraceTasks(),
                task2batch=TraceBatches(context, None),
            ),
            candidate_backend=candidate_backend,
            candidate_evaluator=ScoreComparisonPlugin(candidate_backend),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            batch_size=self.batch_size,
        )


def build_harness_evolve_dispatcher(
    runtime,
    tmp_path,
    backend_factory,
    *,
    propose=lambda nodes, samples, model: None,
    evaluate=lambda task, result: 0.0,
    tasks=("task one",),
    agent_record_dir=None,
):
    recipe = _HarnessEvolveTestRecipe(propose=propose, evaluate=evaluate, tasks=tasks, **runtime_bindings(runtime))
    return Dispatcher(
        recipe,
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

    def blocking_proposer(nodes, samples, model):
        del nodes
        if source_record_id(samples[0]) != "a-i1":
            return
        started.set()
        assert release.wait(1)

    initial = tmp_path / "initial"
    (initial / "skills").mkdir(parents=True)
    (initial / "skills" / "SKILL.md").write_text("skill v0", encoding="utf-8")
    dispatcher = build_harness_evolve_dispatcher(
        None,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        propose=blocking_proposer,
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
        assert returned.wait(0.1), "report acceptance waited for harness growth"
        assert status_returned.wait(0.5), "status waited for harness growth"
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
def test_local_backend_failure_reaches_status_and_retries_pending_batch(tmp_path) -> None:
    calls = 0
    should_fail = True

    def proposer(nodes, samples, model):
        nonlocal calls, should_fail
        del nodes, samples
        calls += 1
        if should_fail:
            raise RuntimeError("proposal failed")

    initial = tmp_path / "initial"
    (initial / "skills").mkdir(parents=True)
    (initial / "skills" / "SKILL.md").write_text("skill v0", encoding="utf-8")
    dispatcher = build_harness_evolve_dispatcher(
        None,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        propose=proposer,
    )
    dispatcher.accept_record(trace_inference("i1"))
    dispatcher.accept_record(trace_report("r1", "i1"))

    for _ in range(1000):
        if dispatcher.build_training_status()["error"] is not None:
            break
        time.sleep(0.001)
    assert dispatcher.build_training_status()["error"] == "skills: RuntimeError: proposal failed"

    failed_attempts = calls
    should_fail = False
    dispatcher.accept_record(trace_inference("i2"))
    wait_for_step(dispatcher, 1, scenario="skills")
    assert calls == failed_attempts + 1
    assert dispatcher.build_training_status()["error"] is None
    dispatcher.close()


@pytest.mark.unit
@pytest.mark.parametrize("failure_point", ["stage", "publish", "activation", "commit_history"])
def test_artifact_commit_failure_keeps_the_pending_batch_retryable(tmp_path, monkeypatch, failure_point) -> None:
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
    pending = scenario.trainer.pending_batch
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
    assert scenario.trainer.pending_batch is pending
    assert [record.agent_record_id for record in scenario.records.replay("math")] == ["i1", "r1"]

    scenario.records.append(sft_inference("i2"))
    scenario.commit(result)

    assert backend.batch_ids == [pending.batch_id]
    assert scenario.trainer.state == {"steps": 1}
    assert scenario.trainer.pending_batch is None
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
    pending = scenario.trainer.pending_batch
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
    assert scenario.trainer.pending_batch is pending
    assert scenario.store.durable
    [record] = scenario.store.history()
    published_release = record.artifact_ref.release_id
    with pytest.raises(RuntimeError, match="does not match the pending step"):
        scenario.commit(replace(result, state={"steps": 999}))

    scenario.commit(result)

    assert scenario.scenario_step == 1
    assert scenario.trainer.state == {"steps": 1}
    assert scenario.trainer.pending_batch is None
    assert len(scenario.store.history()) == 1
    assert scenario.repository.require_current_artifact().release_id == published_release
    assert backend.batch_ids == [pending.batch_id]
    dispatcher.close()


@pytest.mark.unit
def test_a_commit_failure_keeps_one_retryable_batch_and_one_record(tmp_path, monkeypatch) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "records"
    runtime = RecordingRuntime()
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=agent_record_dir,
    )
    monkeypatch.setattr(dispatcher.training_loop, "start", lambda: None)
    scenario = dispatcher.get_or_create_scenario("math")
    scenario.records.append(sft_inference("i1"))
    scenario.records.append(sft_report("r1", "i1"))
    assert scenario.reserve_training_batch() is not None
    execution = scenario.execute_reserved_training_step()
    result = execution.result
    assert result is not None
    pending = scenario.trainer.pending_batch
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
    assert scenario.trainer.state == {}
    assert scenario.trainer.pending_batch is pending
    assert scenario.store.durable
    assert len(scenario.store.history()) == 1

    scenario.commit(result)

    assert scenario.scenario_step == 1
    assert scenario.trainer.state == {"steps": 1}
    assert scenario.trainer.pending_batch is None
    assert scenario.store.training_run_position() == (0, 1)
    assert len(scenario.store.history()) == 1
    assert runtime.trained_batches == [["i1"]]
    dispatcher.close()


@pytest.mark.unit
def test_durable_local_backend_recovers_after_post_commit_notification_failure(tmp_path) -> None:
    initial = tmp_path / "initial"
    (initial / "skills").mkdir(parents=True)
    (initial / "skills" / "SKILL.md").write_text("skill v0", encoding="utf-8")
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = build_harness_evolve_dispatcher(
        None,
        tmp_path,
        backend_factory,
        agent_record_dir=agent_record_dir,
    )
    original = dispatcher.get_or_create_scenario("skills")
    assert original is not None
    rollback_target = original.current_artifact_ref().release_id
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
    rollback_started = Event()
    rollback_returned = Event()
    rollback_errors: list[Exception] = []

    def rollback() -> None:
        rollback_started.set()
        try:
            dispatcher.rollback("skills", rollback_target)
        except Exception as exc:
            rollback_errors.append(exc)
        finally:
            rollback_returned.set()

    rollback_thread = Thread(target=rollback)
    rollback_thread.start()
    assert rollback_started.wait(1)
    try:
        assert not rollback_returned.wait(0.05), "rollback entered before durable recovery"
    finally:
        allow_reload.set()
    rollback_thread.join(1)
    assert not rollback_thread.is_alive()
    assert rollback_errors == []

    wait_for_step(dispatcher, 1, scenario="skills")
    recovered = dispatcher.get_or_create_scenario("skills")
    assert recovered is not original
    assert recovered.records.get("skills", "i1") is not None
    assert recovered.records.get("skills", "r1") is not None
    status = dispatcher.build_training_status()["scenarios"]["skills"]
    assert status["scenario_step"] == 1
    assert status["last_committed_step"]["step"] == 1
    assert status["last_committed_step"]["metrics"]["skipped"] == "no proposal"
    records = committed_records(agent_record_dir, scenario="skills")
    assert [record.step for record in records] == [1]
    assert records[0].metrics is not None
    assert records[0].metrics["skipped"] == "no proposal"

    dispatcher.accept_record(trace_inference("i2"))
    dispatcher.accept_record(trace_report("r2", "i2"))
    wait_for_step(dispatcher, 2, scenario="skills")

    assert [record.step for record in committed_records(agent_record_dir, scenario="skills")] == [1, 2]
    status = dispatcher.build_training_status()["scenarios"]["skills"]
    assert status["scenario_step"] == 2
    assert status["last_committed_step"]["step"] == 2
    dispatcher.close()


@pytest.mark.unit
def test_no_artifact_commit_appends_a_record_and_advances_the_step(tmp_path) -> None:
    """A step that publishes no artifact must still enter the commit history.

    Harness evolution frequently proposes nothing (or a losing candidate);
    the trainer still consumed records and advanced its algorithm state.
    Recording that progress keeps the commit history the single durable commit point: a
    crash loses neither the batch nor the record watermark, and recovery
    resumes without re-training compacted rows.
    """
    initial = tmp_path / "initial"
    (initial / "skills").mkdir(parents=True)
    (initial / "skills" / "SKILL.md").write_text("skill v0", encoding="utf-8")
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = build_harness_evolve_dispatcher(None, tmp_path, backend_factory, agent_record_dir=agent_record_dir)

    dispatcher.accept_record(trace_inference("i1"))
    dispatcher.accept_record(trace_report("r1", "i1"))

    scenario = dispatcher.get_or_create_scenario("skills")
    wait_for_step(dispatcher, 1, scenario="skills")
    assert scenario.scenario_step == 1
    head = scenario.repository.require_current_artifact()

    records = committed_records(agent_record_dir, scenario="skills")
    assert len(records) == 1
    record = records[0]
    assert record.step == 1
    assert record.checkpoint is False
    assert record.artifact_ref == head  # head unchanged: no new artifact published
    assert record.algorithm_state == {"steps": 1, "entries": []}
    assert record.high_water_sequence == 2
    assert record.consumed_ids == frozenset({"i1", "r1"})
    dispatcher.close()


@pytest.mark.unit
def test_no_artifact_commit_survives_a_restart(tmp_path) -> None:
    """A no-artifact step must restore step, state, and record progress."""
    initial = tmp_path / "initial"
    (initial / "skills").mkdir(parents=True)
    (initial / "skills" / "SKILL.md").write_text("skill v0", encoding="utf-8")
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first = build_harness_evolve_dispatcher(None, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(trace_inference("i1"))
    first.accept_record(trace_report("r1", "i1"))
    wait_for_step(first, 1, scenario="skills")
    assert first.get_or_create_scenario("skills").scenario_step == 1
    committed = first.build_training_status()["scenarios"]["skills"]["last_committed_step"]
    assert committed["step"] == 1
    assert committed["metrics"]["skipped"] == "no proposal"

    second = build_harness_evolve_dispatcher(None, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("skills")
    assert recovered.scenario_step == 1
    assert recovered.trainer.state["steps"] == 1
    assert second.build_training_status()["scenarios"]["skills"]["last_committed_step"] == committed
    # Record progress resumes at the committed watermark, not sequence 0.
    assert recovered.trainer.data_offset == 2
    first.close()
    second.close()
