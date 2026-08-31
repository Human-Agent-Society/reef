"""Durability protocol tests for the per-scenario commit log (issue #78, phase 3).

One append-only log per scenario is the single commit point: record-store
compaction, the checkpoint head, and the trainer's algorithm
state + read progress are all derived from it at recovery. These tests walk
the crash windows around the commit point and assert each is recoverable.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import KW_ONLY, dataclass
from threading import Event, Thread

import pytest

from reef.artifact import ArtifactRef, InMemoryRepositoryBackend, LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding
from reef.recipe import RecipeRegistry
from reef.recipe.base import Recipe
from reef.runtime import ActivatedModel, ModelCandidate, PreparedTrainingStep, TrainingRuntime
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario.commit_log import RECORD_KIND, CommitLog, CommitLogError, CommitRecord
from reef.scenario.scenario import SCENARIO_SNAPSHOT_METADATA_KEY
from reef.surface import Surface
from reef.surface.harnesses import create_harness_surface
from reef.train import RetentionDecision, Trainer
from reef.train.cordis_backend import CordisBackend, ScoreComparisonSelector
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.evaluation import DefaultCandidateEvaluationPlugin
from reef.train.slime_backend.backend import SlimeTrainingBackend

from ._policy_recipe import TestPolicyRecipe
from ._threshold_processor import ThresholdProcessor


def sample_record(
    *,
    step: int = 1,
    weight_version: str = "w1",
    checkpoint: bool = False,
    training_job_id: str | None = None,
) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        artifact_ref=LiveWeightArtifactRef(
            artifact_id=f"live:{step}",
            version=f"live:proc:{weight_version}:{step}",
            parent_version="base",
            weight_version=weight_version,
        ),
        checkpoint=checkpoint,
        algorithm_state={"steps": step},
        high_water_sequence=step * 2,
        high_water_offset=step * 2,
        compacted_ids=frozenset({f"i{step}"}),
        recorded_at=1000.0 + step,
        training_job_id=training_job_id,
    )


@pytest.mark.unit
def test_commit_record_round_trips_through_the_log(tmp_path) -> None:
    log = CommitLog(tmp_path / "commits.jsonl")
    log.append(sample_record(step=1))
    log.append(sample_record(step=2, weight_version="w2"))

    records = log.records()
    assert [record.step for record in records] == [1, 2]
    first = records[0]
    assert first.scenario == "math"
    assert isinstance(first.artifact_ref, LiveWeightArtifactRef)
    assert first.artifact_ref.weight_version == "w1"
    assert first.checkpoint is False
    assert first.algorithm_state == {"steps": 1}
    assert first.high_water_sequence == 2
    assert first.high_water_offset == 2
    assert first.compacted_ids == frozenset({"i1"})
    assert first.recorded_at == 1001.0
    # A fresh reader over the same file sees the same history.
    assert CommitLog(tmp_path / "commits.jsonl").records() == records


@pytest.mark.unit
def test_commit_record_round_trips_gate_metrics(tmp_path) -> None:
    log = CommitLog(tmp_path / "commits.jsonl")
    log.append(sample_record(step=1))
    metrics = {"changed": "config/format.json", "wins": 3, "losses": 1, "ties": 0, "published": True}
    record = sample_record(step=2, weight_version="w2")
    record.metrics = dict(metrics)
    log.append(record)

    first, second = CommitLog(tmp_path / "commits.jsonl").records()
    assert first.metrics is None  # pre-metrics records stay readable
    assert second.metrics == metrics


@pytest.mark.unit
def test_commit_record_round_trips_training_job_identity(tmp_path) -> None:
    log = CommitLog(tmp_path / "commits.jsonl")
    log.append(sample_record(training_job_id="job-1"))

    [record] = log.records()
    assert record.training_job_id == "job-1"
    assert record.to_dict()["training_job_id"] == "job-1"


@pytest.mark.unit
def test_commit_record_serializes_the_concrete_artifact_ref_kind() -> None:
    weight_value = sample_record().to_dict()["artifact_ref"]
    assert weight_value == {
        "kind": "weight",
        "artifact_id": "live:1",
        "version": "live:proc:w1:1",
        "parent_version": "base",
        "weight_version": "w1",
    }

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
        "artifact_id": "artifact:1",
        "version": "checkpoint:1",
        "parent_version": None,
    }


@pytest.mark.unit
def test_commit_record_validates_its_schema() -> None:
    with pytest.raises(CommitLogError, match="step"):
        CommitRecord(
            scenario="math",
            step=0,
            artifact_ref=ArtifactRef(artifact_id="a", version="r", parent_version=None),
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
        )
    with pytest.raises(CommitLogError, match="high_water_sequence"):
        CommitRecord(
            scenario="math",
            step=1,
            artifact_ref=ArtifactRef(artifact_id="a", version="r", parent_version=None),
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=-1,
            high_water_offset=0,
        )
    with pytest.raises(CommitLogError, match="record kind"):
        CommitRecord.from_dict({"record": "something-else"})
    with pytest.raises(CommitLogError, match="compacted_ids"):
        CommitRecord.from_dict(
            {
                "record": RECORD_KIND,
                "scenario": "math",
                "step": 1,
                "artifact_ref": {
                    "kind": "artifact",
                    "artifact_id": "a",
                    "version": "r",
                    "parent_version": None,
                },
                "checkpoint": False,
                "algorithm_state": None,
                "record_progress": {"high_water_sequence": 0, "high_water_offset": 0, "compacted_ids": [1]},
                "recorded_at": 1.0,
            }
        )


@pytest.mark.unit
def test_records_tolerate_a_torn_tail(tmp_path) -> None:
    path = tmp_path / "commits.jsonl"
    log = CommitLog(path)
    log.append(sample_record())
    # A crash mid-append leaves a partial final line; the commit never happened.
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"record":"reef-commit/4","scenario":"ma')

    records = log.records()
    assert [record.step for record in records] == [1]


@pytest.mark.unit
def test_records_reject_corruption_before_the_tail(tmp_path) -> None:
    path = tmp_path / "commits.jsonl"
    log = CommitLog(path)
    log.append(sample_record(step=1))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("not json\n")
    log.append(sample_record(step=2))

    with pytest.raises(CommitLogError, match="line 2"):
        log.records()


class RecordingRuntime(TrainingRuntime):
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
    def inference_backend(self):
        return None

    def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
        del step_preparer
        payload = {
            "rollout_id": scenario_step,
            "sources": [sample.source_agent_record_id for sample in batch.samples],
        }
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state={"steps": int(algorithm_state.get("steps", 0)) + 1},
            metrics={},
        )

    def train_candidate(self, payload):
        self.trained_batches.append(payload["sources"])
        weight_version = f"w{len(self.trained_batches)}"
        checkpoint = self._checkpoint_dir / str(payload["rollout_id"]) if self._checkpoint_dir else "/unused"
        if self._checkpoint_dir:
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.txt").write_text(weight_version)
        job_id = f"job-{payload['rollout_id']}"
        self._candidate_versions[job_id] = weight_version
        return ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=str(checkpoint),
            current_weight_version=self._served_version,
        )

    def activate_candidate(self, candidate):
        weight_version = self._candidate_versions[candidate.candidate_id]
        self._served_version = weight_version
        return ActivatedModel(candidate.candidate_id, weight_version)

    def reject_candidate(self, candidate, decision):
        del decision
        self._candidate_versions.pop(candidate.candidate_id, None)

    def serving_weight_version(self):
        return self._served_version


def sft_inference(agent_record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
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
    checkpoint_strategy: CheckpointStrategy | None = None,
    agent_record_dir=None,
):
    return Dispatcher(
        RecipeRegistry(
            recipes={
                "test_policy": TestPolicyRecipe(
                    runtime,
                    batch_size=1,
                    checkpoint_strategy=(
                        checkpoint_strategy if checkpoint_strategy is not None else EveryNVersions(1000)
                    ),
                )
            }
        ),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=agent_record_dir,
    )


def commit_log_path(agent_record_dir, scenario: str = "math"):
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    return agent_record_dir / f"{key}.commits.jsonl"


def wait_for_step(dispatcher: Dispatcher, step: int, *, scenario: str = "math") -> None:
    for _ in range(1000):
        if dispatcher.get_or_create_scenario(scenario).scenario_step >= step:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not commit")


@pytest.mark.unit
def test_each_committed_step_appends_one_atomic_record(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    runtime = RecordingRuntime()
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=agent_record_dir,
    )
    dispatcher.accept_record(sft_inference("i1"), recipe="test_policy")
    dispatcher.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    dispatcher.accept_record(sft_inference("i2"), recipe="test_policy")
    dispatcher.accept_record(sft_report("r2", "i2"), recipe="test_policy")
    wait_for_step(dispatcher, 2)

    records = CommitLog(commit_log_path(agent_record_dir)).records()
    assert [record.step for record in records] == [1, 2]
    first, second = records
    assert isinstance(first.artifact_ref, LiveWeightArtifactRef)
    assert isinstance(second.artifact_ref, LiveWeightArtifactRef)
    assert first.artifact_ref.weight_version == "w1"
    assert first.checkpoint is False
    assert first.algorithm_state == {"steps": 1}
    assert first.compacted_ids == frozenset({"i1", "r1"})
    assert first.high_water_sequence == 2
    assert second.artifact_ref.weight_version == "w2"
    assert second.high_water_sequence == 4
    # The record's ref is exactly the serving head the scenario advanced to.
    scenario = dispatcher.get_or_create_scenario("math")
    assert scenario.repository.require_current_artifact() == second.artifact_ref
    assert scenario.committed_training_job_id == "job-1"
    assert scenario.committed_training_without_job_id is False


@pytest.mark.unit
def test_training_commit_proves_its_training_job_identity(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    runtime = RecordingRuntime()
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=agent_record_dir,
    )
    dispatcher.accept_record(sft_inference("i1"), recipe="test_policy")
    dispatcher.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(dispatcher, 1)

    scenario = dispatcher.get_or_create_scenario("math")
    [record] = CommitLog(commit_log_path(agent_record_dir)).records()
    assert record.training_job_id == "job-0"
    assert scenario.committed_training_job_id == "job-0"
    assert scenario.committed_training_without_job_id is False


@pytest.mark.unit
def test_recovery_restores_the_live_head_step_and_state_from_the_log(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    first.accept_record(sft_inference("i2"), recipe="test_policy")
    first.accept_record(sft_report("r2", "i2"), recipe="test_policy")
    wait_for_step(first, 2)
    committed_head = first.get_or_create_scenario("math").repository.require_current_artifact()
    assert isinstance(committed_head, LiveWeightArtifactRef)
    assert committed_head.weight_version == "w2"

    # A fresh dispatcher over the same dirs is a process restart: nothing is
    # durable except the version chain, the record store, and the commit log.
    second = build_training_dispatcher(
        RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    recovered = second.get_or_create_scenario("math", "test_policy")
    assert recovered.scenario_step == 2
    # The live head survives the restart: it is the log's head record, even
    # though the version chain only knows the bootstrap checkpoint.
    assert recovered.repository.require_current_artifact() == committed_head
    assert recovered.trainer.state == {"steps": 2}


class ProtectAllProcessor(ThresholdProcessor):
    """Pairs like SFT but never releases a row: every ingested id is protected."""

    def __init__(self, context) -> None:
        super().__init__(context)
        self._seen: set[str] = set()

    def ingest(self, item) -> None:
        self._seen.add(item.agent_record_id)
        super().ingest(item)

    def retention_decision(self) -> RetentionDecision:
        return RetentionDecision(protected_agent_record_ids=frozenset(self._seen))


@dataclass(frozen=True)
class ProtectAllPolicyRecipe(TestPolicyRecipe):
    def build(self, scenario, records, *, algorithm_state=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: ProtectAllProcessor(
                context.with_config({"batch_size": self.batch_size, "min_score": self.min_score})
            ),
            training_backend=SlimeTrainingBackend(self.runtime, "sft"),
            algorithm_state=algorithm_state,
        )


@pytest.mark.unit
def test_recovery_resumes_record_progress_without_retraining(tmp_path) -> None:
    """Consumed rows must never be trained twice, even when they survive compaction.

    The protect-all processor keeps every row physically present, so recovery
    without the log's high-water mark would page from sequence 0, re-ingest
    the consumed pair, and train it again.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    registry = lambda runtime: RecipeRegistry(  # noqa: E731
        recipes={
            "test_policy": ProtectAllPolicyRecipe(
                runtime,
                batch_size=1,
                checkpoint_strategy=EveryNVersions(1000),
            )
        }
    )

    def build(runtime):
        return Dispatcher(
            registry(runtime),
            backend_factory,
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=agent_record_dir,
        )

    first_runtime = RecordingRuntime()
    first = build(first_runtime)
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(first, 1)
    assert first_runtime.trained_batches == [["i1"]]
    # Nothing was compacted: the consumed pair is still physically present.
    assert first.get_or_create_scenario("math").records.get("math", "i1") is not None

    second_runtime = RecordingRuntime()
    second = build(second_runtime)
    recovered = second.get_or_create_scenario("math", "test_policy")
    assert recovered.trainer.data_offset == 2  # resumed at the committed watermark

    second.accept_record(sft_inference("i2"), recipe="test_policy")
    second.accept_record(sft_report("r2", "i2"), recipe="test_policy")
    wait_for_step(second, 2)
    # The only newly trained batch is the genuinely new pair; i1 never retrains.
    assert second_runtime.trained_batches == [["i2"]]
    assert recovered.scenario_step == 2


@pytest.mark.unit
def test_recovery_reingests_retained_rows_behind_the_watermark(tmp_path) -> None:
    """Issue #344: the consumption cursor passes rows whose batches have not trained.

    The drain that readies step 1 also consumes the next step's inference, so
    the committed high-water mark covers it while retention keeps it stored.
    Recovery must feed that row back to the rebuilt processor; otherwise the
    step-2 report waits forever on a reference that can never be ingested
    again and training stalls silently. The consumed step-1 pair, compacted
    and so absent from replay, still never trains twice.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first_runtime = RecordingRuntime()
    first = build_training_dispatcher(first_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_inference("i2"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(first, 1)
    assert first_runtime.trained_batches == [["i1"]]
    # The step-2 inference sits below the committed watermark yet survives
    # compaction: it is protected, and only processor memory knew about it.
    assert CommitLog(commit_log_path(agent_record_dir)).records()[-1].high_water_sequence == 3
    assert first.get_or_create_scenario("math").records.get("math", "i2") is not None

    second_runtime = RecordingRuntime()
    second = build_training_dispatcher(second_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("math", "test_policy")
    assert recovered.trainer.data_offset == 3  # resumed at the committed watermark

    second.accept_record(sft_report("r2", "i2"), recipe="test_policy")
    wait_for_step(second, 2)
    # The reingested inference pairs with the late report; i1 never retrains.
    assert second_runtime.trained_batches == [["i2"]]
    assert recovered.scenario_step == 2


@pytest.mark.unit
def test_recovery_replays_a_compaction_interrupted_by_a_crash(tmp_path) -> None:
    """Crash window: commit record appended, compaction never applied.

    Recovery must re-apply the recorded deletions; otherwise the rows would be
    re-ingested (the state came from the record) and trained twice.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first_runtime = RecordingRuntime()
    first = build_training_dispatcher(first_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    scenario = first.get_or_create_scenario("math", "test_policy")

    from reef.train.trainer import Trainer

    original_apply_compaction = Trainer.apply_compaction
    crash = {"armed": True}

    def exploding_compaction(self, compacted_ids) -> None:
        if crash["armed"]:
            crash["armed"] = False
            raise RuntimeError("simulated crash between record append and compaction")
        original_apply_compaction(self, compacted_ids)

    scenario.trainer.apply_compaction = exploding_compaction.__get__(scenario.trainer, Trainer)
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(first, 1)
    assert first_runtime.trained_batches == [["i1"]]
    assert crash["armed"] is False

    second_runtime = RecordingRuntime()
    second = build_training_dispatcher(second_runtime, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("math", "test_policy")
    # Recovery replayed the interrupted compaction and resumed at step 1.
    assert recovered.records.get("math", "i1") is None
    assert recovered.scenario_step == 1

    second.accept_record(sft_inference("i2"), recipe="test_policy")
    second.accept_record(sft_report("r2", "i2"), recipe="test_policy")
    wait_for_step(second, 2)
    assert second_runtime.trained_batches == [["i2"]]
    assert recovered.scenario_step == 2


@pytest.mark.unit
def test_recovery_adopts_a_checkpoint_whose_record_was_lost(tmp_path) -> None:
    """Crash window: checkpoint published to the version chain, record never appended.

    The checkpoint head is then ahead of the log. Its snapshot metadata carries
    the same record fields, so recovery adopts it and heals the log.
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
        checkpoint_strategy=EveryNVersions(1),
        agent_record_dir=agent_record_dir,
    )
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(first, 1)
    assert first.get_or_create_scenario("math").scenario_step == 1

    # Simulate the crash: the log loses its last (only) record.
    path = commit_log_path(agent_record_dir)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    path.write_text("", encoding="utf-8")

    second = build_training_dispatcher(
        RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    recovered = second.get_or_create_scenario("math", "test_policy")
    assert recovered.scenario_step == 1
    assert recovered.trainer.state == {"steps": 1}
    # The healed log carries the adopted checkpoint record, with the record
    # high-water mark taken from the checkpoint's snapshot metadata.
    records = CommitLog(path).records()
    assert len(records) == 1
    adopted = records[0]
    assert adopted.step == 1
    assert adopted.checkpoint is True
    assert adopted.artifact_ref == recovered.repository.require_current_artifact()
    assert adopted.high_water_sequence == 2
    assert adopted.compacted_ids == frozenset({"i1", "r1"})
    assert adopted.training_job_id == "job-0"
    assert adopted.operation == "training"
    assert adopted.operation_verified is True


@pytest.mark.unit
def test_checkpoint_snapshot_metadata_doubles_as_a_commit_record(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = RecordingRuntime()
    runtime._checkpoint_dir = tmp_path / "exported"
    dispatcher = build_training_dispatcher(
        runtime,
        tmp_path,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        checkpoint_strategy=EveryNVersions(1),
        agent_record_dir=tmp_path / "agent-record",
    )
    dispatcher.accept_record(sft_inference("i1"), recipe="test_policy")
    dispatcher.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(dispatcher, 1)

    backend = dispatcher.get_or_create_scenario("math").repository.backend
    snapshot = backend.metadata()[SCENARIO_SNAPSHOT_METADATA_KEY]
    assert snapshot["scenario_step"] == 1
    assert snapshot["algorithm_state"] == {"steps": 1}
    assert snapshot["record_progress"] == {
        "high_water_sequence": 2,
        "high_water_offset": 2,
        "compacted_ids": ["i1", "r1"],
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
        first.accept_record(sft_inference(f"i{index}"), recipe="test_policy")
        first.accept_record(sft_report(f"r{index}", f"i{index}"), recipe="test_policy")
    wait_for_step(first, 3)

    path = commit_log_path(agent_record_dir)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    second = build_training_dispatcher(
        RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir
    )
    with pytest.raises(ReefError, match="corrupt"):
        second.get_or_create_scenario("math", "test_policy")


@pytest.mark.unit
def test_recovery_warns_when_the_engine_disagrees_with_the_live_head(tmp_path, caplog) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(first, 1)

    disagreeing = RecordingRuntime(served_version="w999")
    second = build_training_dispatcher(disagreeing, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    with caplog.at_level(logging.WARNING, logger="reef.surface.weights"):
        recovered = second.get_or_create_scenario("math", "test_policy")
    assert recovered.scenario_step == 1
    assert any("w1" in record.message and "w999" in record.message for record in caplog.records)

    agreeing = RecordingRuntime(served_version="w1")
    third = build_training_dispatcher(agreeing, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="reef.surface.weights"):
        third.get_or_create_scenario("math", "test_policy")
    assert not caplog.records


@pytest.mark.unit
def test_no_commit_log_without_an_agent_record_dir(tmp_path) -> None:
    """In-memory deployments keep pre-log semantics: version-chain-only recovery."""
    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory)
    first.accept_record(sft_inference("i1"), recipe="test_policy")
    first.accept_record(sft_report("r1", "i1"), recipe="test_policy")
    wait_for_step(first, 1)

    assert first.get_or_create_scenario("math").commit_log is None
    assert not list(tmp_path.rglob("*.commits.jsonl"))

    # Recovery still works from the version chain: the live head is forgotten, as
    # before the log existed, and the step reverts to the checkpointed 0.
    second = build_training_dispatcher(RecordingRuntime(), tmp_path, backend_factory)
    recovered = second.get_or_create_scenario("math", "test_policy")
    assert recovered.scenario_step == 0
    assert not isinstance(recovered.repository.require_current_artifact(), LiveWeightArtifactRef)


@dataclass(frozen=True)
class _HarnessEvolveTestRecipe(Recipe):
    """Private harness-evolution stand-in for these mechanics tests.

    These tests exercise commit-log/version-chain mechanics, not
    harness-evolution semantics specifically; they just need a concrete,
    artifact-producing Recipe. Constructs CordisBackend and
    CordisProcessor directly, the same shape
    CordisRecipe.build() has, minus the config-boot layer these
    tests never touch (see tests/reef_service/test_harness_recipe.py for
    the backend's own guarantees).
    """

    propose: object
    evaluate: object
    tasks: tuple[str, ...]
    batch_size: int = 1
    max_score: float = 0.0
    _: KW_ONLY
    name: str = "harness_evolve"

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface()

    def build(self, scenario, records, *, algorithm_state=None) -> Trainer:
        from reef.train.cordis_backend.processor import CordisProcessor

        training_backend = CordisBackend(
            descriptor=get_adapter("pi"),
            propose=resolve_proposer(self.propose),
            score_episode=resolve_episode_scorer(self.evaluate),
            tasks=self.tasks,
            models=ModelBinding(base_url="http://localhost:8000", model="qwen3-8b"),
        )
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: CordisProcessor(
                context.with_config({"batch_size": self.batch_size, "max_score": self.max_score})
            ),
            training_backend=training_backend,
            candidate_evaluator=DefaultCandidateEvaluationPlugin(
                training_backend,
                ScoreComparisonSelector(),
            ),
            algorithm_state=algorithm_state,
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
    recipe = _HarnessEvolveTestRecipe(propose=propose, evaluate=evaluate, tasks=tasks, runtime=runtime)
    return Dispatcher(
        RecipeRegistry(recipes={"harness_evolve": recipe}),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=agent_record_dir,
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
        if samples[0].source_agent_record_id != "a-i1":
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
    dispatcher.accept_record(trace_inference("a-i1", scenario="skills-a"), recipe="harness_evolve")

    returned = Event()
    errors: list[Exception] = []

    def accept_report() -> None:
        try:
            dispatcher.accept_record(
                trace_report("a-r1", "a-i1", scenario="skills-a"),
                recipe="harness_evolve",
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            returned.set()

    request = Thread(target=accept_report)
    request.start()
    assert started.wait(1)
    try:
        assert returned.wait(0.1), "report acceptance waited for harness growth"
        dispatcher.accept_record(trace_inference("b-i1", scenario="skills-b"), recipe="harness_evolve")
        dispatcher.accept_record(
            trace_report("b-r1", "b-i1", scenario="skills-b"),
            recipe="harness_evolve",
        )
        wait_for_step(dispatcher, 1, scenario="skills-b")
    finally:
        release.set()
        request.join()

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
    dispatcher.accept_record(trace_inference("i1"), recipe="harness_evolve")
    dispatcher.accept_record(trace_report("r1", "i1"), recipe="harness_evolve")

    for _ in range(1000):
        if dispatcher.training_status["error"] is not None:
            break
        time.sleep(0.001)
    assert dispatcher.training_status["error"] == "skills: RuntimeError: proposal failed"

    failed_attempts = calls
    should_fail = False
    dispatcher.accept_record(trace_inference("i2"), recipe="harness_evolve")
    wait_for_step(dispatcher, 1, scenario="skills")
    assert calls == failed_attempts + 1
    assert dispatcher.training_status["error"] is None
    dispatcher.close()


@pytest.mark.unit
def test_no_artifact_commit_appends_a_record_and_advances_the_step(tmp_path) -> None:
    """A step that publishes no artifact must still enter the WAL.

    Harness evolution frequently proposes nothing (or a losing candidate);
    the trainer still consumed records and advanced its algorithm state.
    Recording that progress keeps the WAL the single durable commit point: a
    crash loses neither the batch nor the record watermark, and recovery
    resumes without re-training compacted rows.
    """
    initial = tmp_path / "initial"
    (initial / "skills").mkdir(parents=True)
    (initial / "skills" / "SKILL.md").write_text("skill v0", encoding="utf-8")
    agent_record_dir = tmp_path / "agent-record"
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = build_harness_evolve_dispatcher(None, tmp_path, backend_factory, agent_record_dir=agent_record_dir)

    dispatcher.accept_record(trace_inference("i1"), recipe="harness_evolve")
    dispatcher.accept_record(trace_report("r1", "i1"), recipe="harness_evolve")

    scenario = dispatcher.get_or_create_scenario("skills")
    wait_for_step(dispatcher, 1, scenario="skills")
    assert scenario.scenario_step == 1
    head = scenario.repository.require_current_artifact()
    assert not isinstance(head, LiveWeightArtifactRef)  # harness surface is pull-based

    records = CommitLog(commit_log_path(agent_record_dir, scenario="skills")).records()
    assert len(records) == 1
    record = records[0]
    assert record.step == 1
    assert record.checkpoint is False
    assert record.artifact_ref == head  # head unchanged: no new artifact published
    assert record.algorithm_state == {"steps": 1, "entries": []}
    assert record.high_water_sequence == 2
    assert record.compacted_ids == frozenset({"i1", "r1"})
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
    first.accept_record(trace_inference("i1"), recipe="harness_evolve")
    first.accept_record(trace_report("r1", "i1"), recipe="harness_evolve")
    wait_for_step(first, 1, scenario="skills")
    assert first.get_or_create_scenario("skills").scenario_step == 1

    second = build_harness_evolve_dispatcher(None, tmp_path, backend_factory, agent_record_dir=agent_record_dir)
    recovered = second.get_or_create_scenario("skills", "harness_evolve")
    assert recovered.scenario_step == 1
    assert recovered.trainer.state["steps"] == 1
    # Record progress resumes at the committed watermark, not sequence 0.
    assert recovered.trainer.data_offset == 2
    first.close()
    second.close()
