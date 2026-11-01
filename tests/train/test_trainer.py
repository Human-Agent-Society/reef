from __future__ import annotations

import time
from contextlib import closing
from dataclasses import replace
from functools import partial

import pytest
from tests.support._grouped_pg import GROUPED_PG_OBJECTIVE as _GROUPED_PG_OBJECTIVE
from tests.support._grouped_pg import grouped_policy_pipeline
from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support.chat_templates import CharTemplate
from tests.support.conversations import conversation
from tests.support.runtime_stubs import (
    STUB_ROOT,
    RecordingComponent,
    StubInferenceRuntime,
    StubTrainingComponent,
    runtime_bindings,
)
from tests.support.samples_service import make_samples

from recipes.sao import SAOJobData, SAOSamples
from recipes.tttd import TTTDGroupedRolloutReport, TTTDJobData, TTTDSamples, TTTDSettings
from reef.artifact import ArtifactRef, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import rendered_job_data
from reef.core.errors import TrainingStartRefused
from reef.core.plugins import PluginConfigError
from reef.core.reports import ReportValidationError
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.dispatcher import Dispatcher
from reef.recipe import WeightTrainingRecipe, WeightTrainingSpec
from reef.record2sample.assembly import SampleAssemblySettings
from reef.record2sample.service import ProducedSamples
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import ScenarioRoot
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.commits import CommitRecord
from reef.storage.samples import TrainerSamples
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train import PipelineContext, Trainer, TrainingPipeline
from reef.train.algos.schedule import StepScheduling
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import PipelineConfig, PipelineSpec
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData

NUM_GPUS = 0


def sao_pipeline(context: PipelineContext, *, accept_multi_turn: bool = False) -> TrainingPipeline:
    return TrainingPipeline(
        context,
        record2sample=SAOSamples(context, SampleAssemblySettings(accept_multi_turn_policy_samples=accept_multi_turn)),
        sample2job=SAOJobData(context, None, None),
    )


def preparing_trainer(records: SQLiteRecordStore, **options) -> Trainer:
    """A trainer over ``records`` with the threshold pipeline, one sample a job, and a component whose jobs end in
    ``submit`` without a checkpoint."""
    values = {
        "pipeline_factory": partial(threshold_pipeline, job_samples=1),
        "job_loop": JobLoop(RecordingComponent(), scenario="math"),
        "root": STUB_ROOT,
    }
    return Trainer.build("math", records, **{**values, **options})


def finish(trainer: Trainer) -> None:
    """Commit the end of ``trainer``'s running job, when one runs: its job ended in ``submit``."""
    while trainer.job_running:
        ended = trainer.run_once()
        assert ended is not None
        trainer.commit(trainer.prepare_commit(ended))


def record_commit(session: CommitHistoryScenarioStore, step: int, prepared) -> None:
    """Append ``prepared`` to the commit history as training step ``step``, as the scenario committer does, so the
    sample store counts the samples that the step trained."""
    session.commit_step(
        expected_step=step - 1,
        commit=CommitRecord(
            scenario="math",
            step=step,
            artifact_ref=ArtifactRef(f"content:{step}", f"release:{step}", "base"),
            checkpoint=True,
            algorithm_state=prepared.algorithm_state,
            high_water_sequence=prepared.high_water_sequence,
            high_water_offset=prepared.high_water_offset,
            consumed_ids=prepared.consumed_ids,
        ),
    )


def inference(agent_record_id: str, *, candidate: str | None = None) -> AgentRecord:
    payload = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    if candidate is not None:
        payload["candidate"] = candidate
    return AgentRecord.create(
        scenario="math", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=agent_record_id
    )


def served_call(agent_record_id: str, turns: int, *, filler: str = "") -> AgentRecord:
    """A call of the conversation ``filler`` names (``conversation``), which the inference service recorded as served
    by ``artifact-1``; a call with more turns continues a call of the same conversation with fewer."""
    record = conversation(agent_record_id, turns, filler=filler)
    return replace(record, payload={**record.payload, "artifact_id": "artifact-1"})


def report(agent_record_id: str, references: str | tuple[str, ...], score: float, **metadata) -> AgentRecord:
    reference_ids = (references,) if isinstance(references, str) else references
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": list(reference_ids), "metadata": metadata},
        agent_record_id=agent_record_id,
        references=reference_ids,
    )


def one_pair_records() -> SQLiteRecordStore:
    """An in-memory store holding inference i1 and its report r1, scored 1."""
    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    return records


def positioned_inference(sequence: int) -> AgentRecord:
    payload = {"tokens": [sequence], "loss_mask": [1], "rollout_log_probs": [-0.2]}
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=f"i{sequence}",
    )


def positioned_report(sequence: int, reference: str, score: float) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": [reference]},
        agent_record_id=f"r{sequence}",
        references=(reference,),
    )


@pytest.mark.unit
def test_recipe_processor_never_becomes_ready() -> None:
    processor = TrainingPipeline(PipelineContext("math"))
    processor.push(inference("i1"))

    assert not processor.ready(1)
    # A processor that never becomes ready says so: it holds no ready samples, whatever it has been pushed.
    assert processor.status()["ready_samples"] == 0
    assert processor.releasable_record_ids().isdisjoint(frozenset({"i1"}))


@pytest.mark.unit
def test_pairing_processor_uses_feedback_without_score_filtering() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("bad"))
    processor.push(inference("good"))
    processor.push(report("bad-r", "bad", 0.1))
    processor.push(report("good-r", "good", 1.0))
    assert [trajectory_reward(sample) for sample in processor.build_job_data(2).samples] == [0.1, 1.0]


@pytest.mark.unit
def test_pairing_processor_emits_policy_samples_for_spo() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("i1"))
    processor.push(report("r1", "i1", 0.8))

    batch = processor.build_job_data(1)
    assert isinstance(batch, JobData)
    assert batch.data_id == "math:data:1"
    (item,) = batch.samples
    assert source_record_id(item) == "i1"
    assert trajectory_reward(item) == 0.8
    # The sample is the ATIF trajectory of the call, which the training component renders: it carries no tensors.
    assert not {"tokens", "loss_mask", "rollout_log_probs"} & set(item.training)
    assert item.artifact_id is None
    assert item.metadata["report_agent_record_id"] == "r1"


@pytest.mark.unit
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_pairing_processor_rejects_a_non_finite_score(bad_score: float) -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("i1"))
    with pytest.raises(ReportValidationError, match="finite"):
        processor.push(report("r1", "i1", bad_score))
    assert not processor.ready(1)


@pytest.mark.unit
def test_pairing_processor_assembles_ordered_multi_reference_report() -> None:
    processor = threshold_pipeline(PipelineContext("math"), accept_multi_turn=True)
    # Two conversations: the report names two trajectories, and the sample holds the calls of both in order.
    first = served_call("i1", 1)
    second = served_call("i2", 1, filler=" again")
    final_report = report("r1", ("i1", "i2"), 0.75)

    processor.push(first)
    processor.push(second)
    assert not processor.ready(1)
    processor.push(final_report)

    batch = processor.build_job_data(1)
    (item,) = batch.samples
    assert source_record_id(item) == "i2"
    assert trajectory_reward(item) == 0.75
    # Both replies train: no agent step is copied context.
    agent_steps = [step for step in item.trajectory["steps"] if step["source"] == "agent"]
    assert [step.get("is_copied_context", False) for step in agent_steps] == [False, False]
    assert item.artifact_id == "artifact-1"
    assert [record["payload"] for record in item.metadata["records"]] == [first.payload, second.payload]
    processor.acknowledge(batch.data_id)
    assert processor.releasable_record_ids() == frozenset({"i1", "i2", "r1"})


@pytest.mark.unit
@pytest.mark.parametrize(
    "processor_type",
    [
        pytest.param(SAOJobData, id="sao"),
        # openclawrl is absent: it consumes no reports, so multi-reference
        # assembly never happens there by construction.
        pytest.param(TTTDJobData, id="tttd"),
    ],
)
def test_cookbook_processors_wait_for_references_before_rejecting_a_report_of_several_trajectories(
    processor_type,
) -> None:
    metadata = {}
    if processor_type is TTTDJobData:
        metadata = {
            "algorithm": "tttd",
            "step": 0,
            "group": 0,
            "rollout": 0,
            "groups_per_step": 1,
            "rollouts_per_group": 2,
            "comparison_set": "tttd-step-0-group-0",
        }
        context = PipelineContext("math", TTTDGroupedRolloutReport)
        tttd_samples = TTTDSamples(context, TTTDSettings(groups_per_step=1, rollouts_per_group=2))
        processor = TrainingPipeline(
            context,
            record2sample=tttd_samples,
            sample2job=TTTDJobData(context, tttd_samples, None, None),
        )
    else:
        context = PipelineContext("math")
        processor = TrainingPipeline(
            context,
            record2sample=SAOSamples(context, SampleAssemblySettings()),
            sample2job=SAOJobData(context, None, None),
        )
    processor.push(served_call("i1", 1))
    # i2 does not continue i1, so the report names two trajectories.
    multi_trajectory_report = report("r1", ("i1", "i2"), 1.0, **metadata)
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.push(multi_trajectory_report)
    processor.push(served_call("i2", 1))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        processor.push(multi_trajectory_report)
    assert processor.releasable_record_ids().isdisjoint({"i1", "i2", "r1"})


@pytest.mark.unit
def test_failed_multi_turn_assembly_preserves_inputs_and_other_live_reports() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    for record_id in ("i1", "i2", "i3"):
        processor.push(served_call(record_id, 1))
    processor.push(report("single", "i1", 1.0))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        processor.push(report("multi", ("i2", "i3"), 1.0))
    # The single report's sample is stored, which consumes its records; the failed report keeps its inputs held.
    assert {"i1", "single"} <= processor.releasable_record_ids()
    assert processor.releasable_record_ids().isdisjoint({"i2", "i3", "multi"})
    assert processor.acknowledge(processor.build_job_data(1).data_id) == {"sample:single:0"}
    assert processor.releasable_record_ids().isdisjoint({"i2", "i3", "multi"})


@pytest.mark.unit
def test_pairing_processor_rejects_report_eligibility_flags() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("i1"))
    with pytest.raises(ReportValidationError, match="eligible"):
        processor.push(report("r1", "i1", 0.0, training={"eligible": False}))
    assert not processor.ready(1)
    assert processor.releasable_record_ids().isdisjoint({"i1"})


@pytest.mark.unit
def test_pairing_retention_trains_one_report_per_inference_and_releases_it() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("i1"))
    processor.push(report("r1", "i1", 1.0))
    # r1's sample is stored, which consumes r1 and i1; r2 ends at once without training.
    processor.push(report("r2", "i1", 0.5))
    assert processor.releasable_record_ids() == frozenset({"i1", "r1", "r2"})

    batch = processor.build_job_data(1)
    assert len(batch.samples) == 1 and trajectory_reward(batch.samples[0]) == 1.0
    assert processor.acknowledge(batch.data_id) == {"sample:r1:0"}
    assert not processor.ready(1)
    assert processor.releasable_record_ids() == frozenset({"i1", "r1", "r2"})

    processor.push(report("late", "i1", 0.25))
    assert not processor.ready(1)
    assert processor.releasable_record_ids() == frozenset({"i1", "r1", "r2", "late"})


@pytest.mark.unit
def test_pairing_retention_keeps_low_score_feedback_until_its_sample_is_stored() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("i1"))
    assert processor.releasable_record_ids().isdisjoint({"i1"})
    processor.push(report("low", "i1", 0.1))
    # The low-score sample is stored, so its records are consumed; the sample trains in the next job data.
    assert processor.releasable_record_ids() == {"i1", "low"}
    batch = processor.build_job_data(1)
    assert trajectory_reward(batch.samples[0]) == 0.1
    assert processor.acknowledge(batch.data_id) == {"sample:low:0"}


@pytest.mark.unit
def test_grpo_retention_releases_complete_comparison_set_after_acknowledgement() -> None:
    processor = grouped_policy_pipeline(PipelineContext("math"))
    for agent_record_id, score in (("i1", 0.2), ("i2", 0.8)):
        processor.push(inference(agent_record_id))
        processor.push(report("r" + agent_record_id, agent_record_id, score, comparison_set="set-a"))

    batch = processor.build_job_data(1)
    processor.acknowledge(batch.data_id)
    decision = processor.releasable_record_ids()
    assert decision == frozenset({"i1", "i2", "ri1", "ri2"})


@pytest.mark.unit
def test_algorithms_consume_formatted_batches_and_keep_algorithm_state() -> None:
    sft_processor = threshold_pipeline(PipelineContext("math"))
    sft_processor.push(inference("i1"))
    sft_processor.push(report("r1", "i1", 1.0))
    sft_result = prepare_slime_job(
        sft_processor.build_job_data(1), TrainingMethod("sft"), {}, StepScheduling(), step_size=1
    )
    assert sft_result.next_algorithm_state == {"steps": 1}

    grpo_processor = grouped_policy_pipeline(PipelineContext("math"))
    for rid, score in (("i3", 0.2), ("i4", 0.8)):
        grpo_processor.push(inference(rid))
        grpo_processor.push(report("r" + rid, rid, score, comparison_set="x"))
    result = prepare_slime_job(
        grpo_processor.build_job_data(1), TrainingMethod(_GROUPED_PG_OBJECTIVE), {}, StepScheduling(), step_size=1
    )
    assert result.metrics["advantages"] == pytest.approx((-1.0, 1.0))


@pytest.mark.unit
def test_trainer_reserves_batch_and_commits_backend_preparation() -> None:
    trainer = preparing_trainer(one_pair_records())
    make_samples(trainer)

    result = trainer.run_once()

    assert result is not None
    assert trainer.pending_job_data is not None
    # The reserved batch's sample leaves the buffer, and the buffer takes one job's data.
    assert trainer.pipeline_status() == {
        "ready_samples": 0,
        "buffer_size": 1,
        "buffered_requests": 0,
        "job_samples": 1,
    }
    assert len(trainer.pending_job_data.samples) == 1
    assert result.metrics["training_job"]["event"] == "submitted"
    assert result.artifact is None
    assert trainer.state == JobLoopState("recording:base").to_dict()
    prepared = trainer.prepare_commit(result)
    trainer.commit(prepared)
    assert trainer.job_running
    finish(trainer)
    assert JobLoopState.from_dict(trainer.state).finished_jobs == 1


@pytest.mark.unit
def test_the_trainer_builds_jobs_only_from_samples_that_the_samples_service_made() -> None:
    trainer = preparing_trainer(one_pair_records())

    # The trainer's own cycle reads the sample store only: no record became a sample yet.
    assert trainer.run_once() is None
    assert trainer.pipeline_status()["ready_samples"] == 0

    # The samples service makes the samples; the next cycle builds the job from them.
    assert trainer.produce_samples() == ProducedSamples(changed=True, background=False)
    result = trainer.run_once()
    assert result is not None and len(trainer.pending_job_data.samples) == 1


@pytest.mark.unit
def test_trainer_takes_job_samples_and_buffers_one_job_only_with_a_backend() -> None:
    records = SQLiteRecordStore()
    for sequence in range(3):
        records.append(inference(f"i{sequence}"))
        records.append(report(f"r{sequence}", f"i{sequence}", 1.0))
    trainer = preparing_trainer(records, pipeline_factory=partial(threshold_pipeline, job_samples=2))
    assert (trainer.job_samples, trainer.pipeline.buffer_size) == (2, 2)
    make_samples(trainer)
    assert trainer.run_once() is not None
    data = trainer.pending_job_data
    assert data is not None and len(data.samples) == 2
    # The samples service read every record; the third sample waits in the buffer for the next job.
    assert trainer.pipeline_status() == {
        "ready_samples": 1,
        "buffer_size": 2,
        "buffered_requests": 0,
        "job_samples": 2,
    }

    # A trainer that trains nothing has no job size, so its pipeline keeps every sample.
    observer = Trainer.build("math", records, pipeline_factory=threshold_pipeline)
    assert observer.job_samples is None
    assert observer.pipeline_status()["buffer_size"] is None


@pytest.mark.unit
def test_the_stage_config_sets_the_trainer_job_samples() -> None:
    def pipeline(context: PipelineContext) -> TrainingPipeline:
        return PipelineConfig.from_setting(
            {"sample2job": {"config": {"job_samples": 3}}}, spec=PipelineSpec(sample2job=ThresholdJobData), environ={}
        ).build(context)

    trainer = preparing_trainer(SQLiteRecordStore(), pipeline_factory=pipeline)
    assert (trainer.job_samples, trainer.pipeline.buffer_size) == (3, 3)


@pytest.mark.unit
def test_a_training_trainer_without_job_samples_is_refused() -> None:
    with pytest.raises(PluginConfigError, match=r"set pipeline\.sample2job\.config\.job_samples"):
        Trainer.build(
            "math",
            SQLiteRecordStore(),
            pipeline_factory=threshold_pipeline,
            job_loop=JobLoop(RecordingComponent(), scenario="math"),
            root=STUB_ROOT,
        )


class _RefusingRoots(RecordingComponent):
    """A component that cannot start from any scenario root."""

    def root_start(self, root: ScenarioRoot) -> str:
        raise ValueError("the root holds other weights")


@pytest.mark.unit
def test_a_new_trainer_starts_from_its_root_and_a_refused_root_refuses_the_trainer() -> None:
    """A trainer without committed state starts at the component's start of its scenario's root; it needs the root,
    and a root that the component cannot start from is a ``TrainingStartRefused`` that names the root."""
    trainer = preparing_trainer(SQLiteRecordStore())
    assert JobLoopState.from_dict(trainer.state).start == "recording:base"
    with pytest.raises(ValueError, match="starts from the scenario's root: give root"):
        preparing_trainer(SQLiteRecordStore(), root=None)
    with pytest.raises(TrainingStartRefused, match=r"from its root release local:.*: the root holds other weights"):
        preparing_trainer(SQLiteRecordStore(), job_loop=JobLoop(_RefusingRoots(), scenario="math"))
    # Committed state wins: the root is not read.
    committed = JobLoopState("committed").to_dict()
    restored = preparing_trainer(
        SQLiteRecordStore(), job_loop=JobLoop(_RefusingRoots(), scenario="math"), algorithm_state=committed
    )
    assert restored.state == committed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("buffer_size", "max_job_samples", "message"),
    [
        (2, None, "ThresholdJobData buffer_size 2 is smaller than job_samples 3"),
        (None, 2, "ThresholdJobData holds at most 2 samples in one job's data, not job_samples 3"),
    ],
)
def test_trainer_refuses_job_samples_that_its_task_to_job_stage_cannot_hold(
    buffer_size, max_job_samples, message
) -> None:
    def pipeline(context: PipelineContext) -> TrainingPipeline:
        stage = ThresholdJobData(context, buffer_size, 3, max_job_samples=max_job_samples)
        return threshold_pipeline(context, sample2job=stage)

    with pytest.raises(PluginConfigError, match=message):
        preparing_trainer(SQLiteRecordStore(), pipeline_factory=pipeline)


@pytest.mark.unit
def test_trainer_restores_algorithm_state_from_metadata() -> None:
    session = SQLiteScenarioStorage(None).open("math")
    records = session.records
    samples = TrainerSamples(session.samples)
    for item in (
        inference("i1"),
        report("r1", "i1", 1.0),
        inference("i2"),
        report("r2", "i2", 0.5),
    ):
        records.append(item)

    def pipeline(context: PipelineContext) -> TrainingPipeline:
        return threshold_pipeline(
            PipelineContext(context.scenario, samples=context.samples, job_parameters=context.job_parameters),
            job_samples=1,
        )

    first = preparing_trainer(records, pipeline_factory=pipeline, samples=samples)
    make_samples(first)

    first_result = first.run_once()
    assert first_result is not None
    assert source_record_id(first.pending_job_data.samples[0]) == "i1"
    prepared = first.prepare_commit(first_result)
    first.commit(prepared)
    record_commit(session, 1, prepared)
    finish(first)
    assert JobLoopState.from_dict(first.state).finished_jobs == 1
    # The samples service read every record: both pairs are stored samples, and only i1's trained.
    assert first.data_offset == prepared.high_water_offset == 4

    # Recovery restores consumption independently of the retained record bodies; i2's stored sample trains next.
    recovered_state = first.algorithm_state_dict()
    second = preparing_trainer(records, pipeline_factory=pipeline, algorithm_state=recovered_state, samples=samples)

    second.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    assert second.state == recovered_state
    make_samples(second)
    second_result = second.run_once()
    assert second_result is not None
    assert source_record_id(second.pending_job_data.samples[0]) == "i2"
    prepared = second.prepare_commit(second_result)
    second.commit(prepared)
    finish(second)
    assert second.run_once() is None
    session.close()


@pytest.mark.unit
def test_commit_retains_readable_records_and_recovery_skips_committed_consumption(tmp_path) -> None:
    storage = SQLiteScenarioStorage(tmp_path)
    first_inference = positioned_inference(1)
    first_report = positioned_report(2, first_inference.agent_record_id, 1.0)
    second_inference = positioned_inference(3)
    second_report = positioned_report(4, second_inference.agent_record_id, 0.5)

    with closing(storage.open("math")) as first_session:
        first_store = first_session.records
        for item in (first_inference, first_report, second_inference, second_report):
            first_store.append(item)
        first = preparing_trainer(first_store, samples=TrainerSamples(first_session.samples))
        make_samples(first)
        first_result = first.run_once()
        assert first_result is not None
        prepared = first.prepare_commit(first_result)
        first.commit(prepared)
        record_commit(first_session, 1, prepared)
        finish(first)
        first_state = first.algorithm_state_dict()

        assert first_store.get("math", first_inference.agent_record_id) == first_inference
        assert first_store.get("math", first_report.agent_record_id) == first_report
        assert first_store.get_for_audit("math", first_inference.agent_record_id).item == first_inference
        assert first_store.get_for_audit("math", first_report.agent_record_id).item == first_report
        assert [item.agent_record_id for item in first_store.replay("math")] == [
            first_inference.agent_record_id,
            first_report.agent_record_id,
            second_inference.agent_record_id,
            second_report.agent_record_id,
        ]

    with closing(storage.open("math")) as second_session:
        second_store = second_session.records
        second = preparing_trainer(
            second_store, algorithm_state=first_state, samples=TrainerSamples(second_session.samples)
        )
        second.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
        second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        assert second.state == first_state
        make_samples(second)
        second_result = second.run_once()
        assert second_result is not None
        assert source_record_id(second.pending_job_data.samples[0]) == second_inference.agent_record_id
        prepared = second.prepare_commit(second_result)
        second.commit(prepared)
        finish(second)
        assert second_store.count("math") == 4
        archived = second_store.audit_page("math")
        assert [entry.item for entry in archived] == [first_inference, first_report, second_inference, second_report]
        assert second.run_once() is None
    storage.close()


@pytest.mark.integration
def test_a_grouped_recipe_trains_one_comparison_set_as_one_job(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    component = StubTrainingComponent(tmp_path / "checkpoints")
    inference_runtime = StubInferenceRuntime()

    # The minimal grouped pg recipe: the cookbook grouped-method shape the
    # define-a-recipe guide describes, kept local to this suite.
    class GroupedPgRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(objective=_GROUPED_PG_OBJECTIVE)

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
            root=None,
        ):
            return Trainer.build(
                scenario,
                records,
                pipeline_factory=partial(grouped_policy_pipeline, job_samples=1),
                job_loop=JobLoop(component, scenario=scenario),
                job_parameters=self.parameters_by_kind(),
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
                root=root,
            )

    dispatcher = Dispatcher(
        GroupedPgRecipe(**runtime_bindings(component, inference_runtime), name="grouped_pg"),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    scenario = dispatcher.get_or_create_scenario("math")
    serving = scenario.serving_artifact_id()
    for rid, score in (("i1", 0.2), ("i2", 0.8)):
        record = inference(rid)
        dispatcher.accept_record(replace(record, payload={**record.payload, "artifact_id": serving}))
        dispatcher.accept_record(report("r" + rid, rid, score, comparison_set="set-a"))

    deadline = time.monotonic() + 10
    while scenario.trainer.state["finished_jobs"] != 1:
        assert time.monotonic() < deadline, "the training job did not finish"
        time.sleep(0.001)
    [job] = component.submitted
    assert TrainingMethod.from_dict(job.parameters["method"]) == TrainingMethod(_GROUPED_PG_OBJECTIVE)
    # The comparison set is one batch: the component computes its group-relative advantages from the job's data.
    assert [trajectory_reward(item) for item in job.data.samples] == [0.2, 0.8]
    prepared = prepare_slime_job(job.data, TrainingMethod(_GROUPED_PG_OBJECTIVE), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None and prepared.payload["advantages"] == pytest.approx([-1.0, 1.0])
    assert scenario.repository.current_artifact == scenario.repository.checkpoint_artifact
    # After the root release, the job's checkpoint is what the scenario serves.
    assert inference_runtime.activated[1:] == ["w1"]


@pytest.mark.unit
def test_the_slime_sao_row_defaults_the_action_mask_to_the_loss_mask_of_a_rendered_rollout() -> None:
    """A record-made SAO sample carries no tensors and no action mask; once the training component renders it, the
    SAO row's action mask is the loss mask, so every trained token is an action."""
    processor = sao_pipeline(PipelineContext("math"), accept_multi_turn=True)
    processor.push(served_call("i1", 1))
    processor.push(served_call("i2", 3))
    processor.push(report("r1", "i2", 1.0))

    assert processor.ready(1)
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert "action_mask" not in sample.training
    rendered, left_out = rendered_job_data(batch, CharTemplate())
    assert left_out == ()
    prepared = prepare_slime_job(rendered, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    (row,) = prepared.payload["samples"]
    loss_mask, action_mask, producing_artifact_id = row[2], row[5], row[6]
    assert action_mask == loss_mask and any(action_mask)
    assert producing_artifact_id == "artifact-1"


def test_reported_samples_leave_rendering_and_tensor_validation_to_the_training_component() -> None:
    from reef.train.slime_backend.data_builder import to_slime_rollout_data

    processor = sao_pipeline(PipelineContext("math"))
    processor.push(served_call("i1", 1))
    processor.push(report("r1", "i1", 1.0))
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    # The stage stores the trajectory without tensors; a job that is prepared without rendering it fails explicitly.
    assert "tokens" not in sample.training
    prepared = prepare_slime_job(batch, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    with pytest.raises(ValueError):
        to_slime_rollout_data(prepared.payload)
    assert processor.build_job_data(1) is batch
    # The job data trains nothing until it is acknowledged: released, the next job data holds the same sample.
    processor.release_job_data(batch.data_id)
    assert processor.build_job_data(1).samples == batch.samples


def test_capacity_loss_skips_orphan_report_and_remains_visible_after_restart(tmp_path, caplog):
    database = tmp_path / "records.sqlite3"
    with SQLiteRecordStore(database) as records:
        records.append(positioned_inference(1))
        records.append(positioned_report(2, "i1", 1.0))
        # Keep the report but evict its input before the processor sees either.
        with records._transaction(write=False) as connection:
            keep_bytes = connection.exec_driver_sql(
                "SELECT body_bytes FROM agent_record WHERE agent_record_id='r2'"
            ).scalar_one()
        with closing(SQLiteScenarioStorage(tmp_path)) as storage:
            assert storage.prune(max_bytes=keep_bytes) == 1
        records.append(positioned_inference(3))
        records.append(positioned_report(4, "i3", 1.0))
        trainer = preparing_trainer(records)
        assert trainer.pipeline_status()["record_data_incomplete"] is True
        make_samples(trainer)
        result = trainer.run_once()
        assert result is not None
        assert source_record_id(trainer.pending_job_data.samples[0]) == "i3"
        prepared = trainer.prepare_commit(result)
        assert "r2" in prepared.consumed_ids
        assert prepared.metrics["records/data_incomplete"] == 1
        trainer.commit(prepared)
        finish(trainer)
        ended_state = trainer.algorithm_state_dict()
        assert trainer.run_once() is None
        assert "Skipping report r2" in caplog.text
    with SQLiteRecordStore(database) as records:
        resumed = preparing_trainer(records, algorithm_state=ended_state)
        resumed.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
        resumed.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        make_samples(resumed)
        assert resumed.run_once() is None
        assert resumed.pipeline_status()["evicted_record_count"] == 1
        assert records.get("math", "i3") is not None
