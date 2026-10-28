from __future__ import annotations

import time
from contextlib import closing
from dataclasses import replace

import pytest
from tests.support._grouped_pg import GROUPED_PG_OBJECTIVE as _GROUPED_PG_OBJECTIVE
from tests.support._grouped_pg import grouped_policy_pipeline
from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings

import reef.record2sample.assembly as reported_module
from recipes.sao import SAOJobData, SAOSamples
from recipes.tttd import TTTDGridSettings, TTTDGroupedRolloutReport, TTTDJobData, TTTDRecord2Record, TTTDSamples
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginConfigError
from reef.core.reports import ReportValidationError
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.dispatcher import Dispatcher
from reef.evaluation import (
    BackendEvaluateMixin,
    CandidateEvaluationPlugin,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.recipe import WeightTrainingRecipe, WeightTrainingSpec
from reef.record2record import ExclusiveReportRecord2Record
from reef.record2sample.assembly import SampleAssemblySettings
from reef.runtime.interfaces import TrainingMethod
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train import PipelineContext, Trainer, TrainingPipeline
from reef.train.algos.schedule import StepScheduling
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.jobs import JobLoop
from reef.train.pipeline import PipelineConfig, PipelineSpec
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData, TrainStepResult, TrajectorySample

NUM_GPUS = 0


def sao_pipeline(context: PipelineContext, *, accept_multi_turn: bool = False) -> TrainingPipeline:
    return TrainingPipeline(
        context,
        record2record=ExclusiveReportRecord2Record(context),
        record2sample=SAOSamples(SampleAssemblySettings(accept_multi_turn_policy_samples=accept_multi_turn)),
        sample2job=SAOJobData(context, None, None),
    )


def policy_plugin(backend: object, policy: type) -> CandidateEvaluationPlugin:
    """A plugin that measures through ``backend`` and decides with ``policy``'s mixin."""

    class _Plugin(policy, BackendEvaluateMixin, CandidateEvaluationPlugin):  # type: ignore[misc, valid-type]
        pass

    plugin = _Plugin()
    plugin._candidate_backend = backend
    return plugin


class _PreparingBackend(CandidateBackend):
    """Test backend that commits only the objective's state transition."""

    def __init__(self, objective: str = "sft") -> None:
        self._objective = objective

    def initial_state(self):
        return {}

    def prepare_step(self, batch, state, scenario_step):
        del scenario_step
        prepared = prepare_slime_job(batch, TrainingMethod(self._objective), state, StepScheduling(), step_size=1)
        return PreparedStep.skipped(state=prepared.next_algorithm_state, metrics=prepared.metrics)

    def evaluate(self, candidate):
        raise AssertionError("state-only test backend does not produce candidates")

    def settle_step(self, prepared, decision):
        raise AssertionError("state-only test backend does not settle candidates")

    def abort_step(self, prepared):
        del prepared


def inference(agent_record_id: str, *, candidate: str | None = None) -> AgentRecord:
    payload = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    if candidate is not None:
        payload["candidate"] = candidate
    return AgentRecord.create(
        scenario="math", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=agent_record_id
    )


def training_inference(
    agent_record_id: str,
    tokens: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float],
) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        payload={
            "response": {
                "training": {
                    "tokens": tokens,
                    "loss_mask": loss_mask,
                    "rollout_log_probs": rollout_log_probs,
                    "runtime_load_id": "wv-1",
                }
            }
        },
    )


def report(agent_record_id: str, references: str | tuple[str, ...], score: float, **metadata) -> AgentRecord:
    reference_ids = (references,) if isinstance(references, str) else references
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": list(reference_ids), "metadata": metadata},
        agent_record_id=agent_record_id,
        references=reference_ids,
    )


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
    assert item.training["tokens"] == [1, 2]
    assert item.training["loss_mask"] == [0, 1]
    assert item.training["rollout_log_probs"] == [-0.2]
    assert item.training["runtime_load_id"] is None
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
    first = training_inference("i1", [10, 20], [1], [-0.1])
    second = training_inference("i2", [10, 20, 11, 21], [1], [-0.2])
    final_report = report("r1", ("i1", "i2"), 0.75)

    processor.push(first)
    processor.push(second)
    assert not processor.ready(1)
    processor.push(final_report)

    batch = processor.build_job_data(1)
    (item,) = batch.samples
    assert source_record_id(item) == "r1"
    assert trajectory_reward(item) == 0.75
    assert item.training["tokens"] == [10, 20, 11, 21]
    assert item.training["loss_mask"] == [1, 0, 1]
    assert item.training["rollout_log_probs"] == [-0.1, 0.0, -0.2]
    assert item.training["runtime_load_id"] == "wv-1"
    assert item.training["turn_count"] == 2
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
def test_cookbook_processors_assemble_before_rejecting_multi_turn_reports(
    processor_type,
    monkeypatch,
) -> None:
    assembled_samples: list[TrajectorySample] = []
    assembler = reported_module.make_multi_turn_policy_trajectory

    def record_assembly(*args, **kwargs):
        sample = assembler(*args, **kwargs)
        assert sample is not None
        assembled_samples.append(sample)
        return sample

    monkeypatch.setattr(reported_module, "make_multi_turn_policy_trajectory", record_assembly)

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

    if processor_type is TTTDJobData:
        context = PipelineContext("math", TTTDGroupedRolloutReport)
        tttd_record2record = TTTDRecord2Record(context, TTTDGridSettings(groups_per_step=1, rollouts_per_group=2))
        processor = TrainingPipeline(
            context,
            record2record=tttd_record2record,
            record2sample=TTTDSamples(SampleAssemblySettings(), tttd_record2record),
            sample2job=TTTDJobData(context, tttd_record2record, None, None),
        )
    else:
        context = PipelineContext("math")
        processor = TrainingPipeline(
            context,
            record2record=ExclusiveReportRecord2Record(context),
            record2sample=SAOSamples(SampleAssemblySettings()),
            sample2job=SAOJobData(context, None, None),
        )
    processor.push(training_inference("i1", [10, 20], [1], [-0.1]))
    multi_turn_report = report("r1", ("i1", "i2"), 1.0, **metadata)
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.push(multi_turn_report)
    assert assembled_samples == []
    processor.push(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        processor.push(multi_turn_report)
    assert assembled_samples
    assert all(
        sample.training.get("turn_count", 1) == 2 and (sample.training.get("turn_count", 1) > 1)
        for sample in assembled_samples
    )
    assert processor.releasable_record_ids().isdisjoint({"i1", "i2", "r1"})


@pytest.mark.unit
def test_failed_multi_turn_assembly_preserves_inputs_and_other_live_reports() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(training_inference("i1", [10, 20], [1], [-0.1]))
    processor.push(training_inference("i2", [10, 20], [1], [-0.1]))
    processor.push(training_inference("i3", [10, 20, 11, 21], [1], [-0.2]))
    processor.push(report("single", "i1", 1.0))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        processor.push(report("multi", ("i2", "i3"), 1.0))
    assert processor.releasable_record_ids().isdisjoint({"i1", "i2", "i3", "single", "multi"})
    processor.acknowledge(processor.build_job_data(1).data_id)
    assert processor.releasable_record_ids().isdisjoint({"i2", "i3", "multi"})


@pytest.mark.unit
def test_pairing_processor_raises_for_forked_multi_turn_rollout() -> None:
    processor = threshold_pipeline(PipelineContext("math"), accept_multi_turn=True)
    processor.push(training_inference("i1", [1, 2, 3], [1], [-0.1]))
    processor.push(training_inference("i2", [1, 9, 3, 4], [1], [-0.2]))
    with pytest.raises(ValueError, match="cannot assemble"):
        processor.push(report("r1", ("i1", "i2"), 1.0))
    assert not processor.ready(1)
    assert processor.releasable_record_ids().isdisjoint({"i1", "i2", "r1"})


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
    # r2 ends at once: r1 holds i1.
    processor.push(report("r2", "i1", 0.5))
    assert processor.releasable_record_ids() == frozenset({"r2"})

    batch = processor.build_job_data(1)
    assert trajectory_reward(batch.samples[0]) == 1.0
    processor.acknowledge(batch.data_id)
    assert not processor.ready(1)
    assert processor.releasable_record_ids() == frozenset({"i1", "r1", "r2"})

    processor.push(report("late", "i1", 0.25))
    assert not processor.ready(1)
    assert processor.releasable_record_ids() == frozenset({"i1", "r1", "r2", "late"})


@pytest.mark.unit
def test_pairing_retention_protects_low_score_feedback_until_consumed() -> None:
    processor = threshold_pipeline(PipelineContext("math"))
    processor.push(inference("i1"))
    processor.push(report("low", "i1", 0.1))
    assert processor.releasable_record_ids().isdisjoint({"i1", "low"})
    processor.acknowledge(processor.build_job_data(1).data_id)
    assert processor.releasable_record_ids() == {"i1", "low"}


@pytest.mark.unit
def test_pairing_retention_does_not_protect_consumed_inferences() -> None:
    first = threshold_pipeline(PipelineContext("math"))
    first.push(inference("i1"))
    first.push(report("r1", "i1", 1.0))
    batch = first.build_job_data(1)
    first.acknowledge(batch.data_id)

    # Consumed inferences are released and may be compacted; the processor does
    # not require them to be retained across restart.
    decision = first.releasable_record_ids()
    assert decision == frozenset({"i1", "r1"})


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
    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=threshold_pipeline,
        candidate_backend=_PreparingBackend(),
        default_job_samples=1,
    )

    result = trainer.run_once(0)

    assert result is not None
    assert trainer.pending_job_data is not None
    # The reserved batch has not been consumed yet, so its sample is still counted as held.
    assert trainer.pipeline_status() == {"ready_samples": 1, "buffer_size": 4, "job_samples": 1}
    assert result.metrics["samples"] == 1
    assert result.artifact is None
    assert trainer.state == {}
    prepared = trainer.prepare_commit(result)
    trainer.commit(prepared)
    assert trainer.state == {"steps": 1}


@pytest.mark.unit
def test_trainer_takes_job_samples_and_buffers_four_jobs_only_with_a_backend() -> None:
    records = SQLiteRecordStore()
    for sequence in range(3):
        records.append(inference(f"i{sequence}"))
        records.append(report(f"r{sequence}", f"i{sequence}", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=threshold_pipeline,
        candidate_backend=_PreparingBackend(),
        default_job_samples=2,
    )
    assert (trainer.job_samples, trainer.pipeline.buffer_size) == (2, 8)
    assert trainer.run_once() is not None
    data = trainer.pending_job_data
    assert data is not None and len(data.samples) == 2
    # The trainer stops reading records once it holds one job's data.
    assert trainer.pipeline_status() == {"ready_samples": 2, "buffer_size": 8, "job_samples": 2}

    observer = Trainer.build("math", records, pipeline_factory=threshold_pipeline, default_job_samples=2)
    assert observer.pipeline.context.default_job_samples is None
    assert observer.job_samples is None
    assert observer.pipeline_status()["buffer_size"] is None


@pytest.mark.unit
def test_the_stage_config_sets_job_samples_over_the_default() -> None:
    def pipeline(context: PipelineContext) -> TrainingPipeline:
        return PipelineConfig.from_setting(
            {"sample2job": {"config": {"job_samples": 3}}}, spec=PipelineSpec(sample2job=ThresholdJobData), environ={}
        ).build(context)

    trainer = Trainer.build(
        "math",
        SQLiteRecordStore(),
        pipeline_factory=pipeline,
        candidate_backend=_PreparingBackend(),
        default_job_samples=2,
    )
    assert (trainer.job_samples, trainer.pipeline.buffer_size) == (3, 12)


@pytest.mark.unit
def test_a_training_trainer_without_job_samples_is_refused() -> None:
    with pytest.raises(PluginConfigError, match=r"set pipeline\.sample2job\.config\.job_samples"):
        Trainer.build(
            "math", SQLiteRecordStore(), pipeline_factory=threshold_pipeline, candidate_backend=_PreparingBackend()
        )


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
        stage = ThresholdJobData(context, buffer_size, None, max_job_samples=max_job_samples)
        return threshold_pipeline(context, sample2job=stage)

    with pytest.raises(PluginConfigError, match=message):
        Trainer.build(
            "math",
            SQLiteRecordStore(),
            pipeline_factory=pipeline,
            candidate_backend=_PreparingBackend(),
            default_job_samples=3,
        )


@pytest.mark.unit
def test_trainer_executes_candidate_policy_between_evaluation_and_settlement() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {"steps": 0}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            calls.append("prepare")
            return PreparedStep.with_candidate(UpdateCandidate(batch.data_id), state=state)

        def evaluate(self, candidate):
            calls.append("evaluate")
            return EvaluationResult("test", "1", {"score": 1.0})

        def settle_step(self, prepared, decision):
            calls.append("settle")
            return TrainStepResult({"steps": 1}, {"selection": decision.to_dict()})

        def abort_step(self, prepared):
            calls.append("abort")

    class Policy:
        def decide(self, candidate, evaluation):
            calls.append("decide")
            return SelectionDecision(True, "test", "selected by test", evaluation)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    backend = Backend()
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=threshold_pipeline,
        candidate_backend=backend,
        candidate_evaluator=policy_plugin(backend, Policy),
        default_job_samples=1,
    )

    result = trainer.run_once()

    assert result is not None
    assert result.metrics["selection"]["selected"] is True
    assert calls == ["prepare", "evaluate", "decide", "settle"]


@pytest.mark.unit
def test_trainer_rejects_structural_plugin_before_constructing_processor() -> None:
    from tests.support._activation_rule_plugin import DuckPlugin

    def pipeline_factory(context):
        raise AssertionError("an invalid plugin must fail before processor construction")

    with pytest.raises(TypeError, match="must inherit CandidateEvaluationPlugin"):
        Trainer.build(
            "math",
            SQLiteRecordStore(),
            pipeline_factory=pipeline_factory,
            candidate_backend=_PreparingBackend(),
            candidate_evaluator=DuckPlugin(),
            default_job_samples=1,
        )


def test_trainer_uses_explicit_candidate_evaluator_instead_of_backend_fallback() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            return PreparedStep.with_candidate(UpdateCandidate(batch.data_id), state=state)

        def evaluate(self, candidate):
            raise AssertionError("the backend evaluator must be replaced by the plugin")

        def settle_step(self, prepared, decision):
            calls.append(("settle", decision.evaluation.evaluator))
            return TrainStepResult({}, {"selection": decision.to_dict()})

        def abort_step(self, prepared):
            raise AssertionError("the successful plugin evaluation must not abort")

    class ExternalEvaluator(CandidateEvaluationPlugin):
        def evaluate(self, candidate):
            calls.append(("evaluate", candidate.candidate_id))
            return EvaluationResult("external", "1", {"score": 0.9})

        def decide(self, candidate, evaluation):
            del candidate
            calls.append(("decide", evaluation.evaluator))
            return SelectionDecision(True, "external", "selected by module", evaluation)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    candidate_evaluator = ExternalEvaluator()
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=threshold_pipeline,
        candidate_backend=Backend(),
        candidate_evaluator=candidate_evaluator,
        default_job_samples=1,
    )

    result = trainer.run_once()

    assert result is not None
    assert trainer.candidate_evaluator is candidate_evaluator
    assert calls == [
        ("evaluate", "math:data:1"),
        ("decide", "external"),
        ("settle", "external"),
    ]


@pytest.mark.unit
def test_trainer_aborts_candidate_when_policy_execution_fails() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            return PreparedStep.with_candidate(UpdateCandidate(batch.data_id), state=state)

        def evaluate(self, candidate):
            return EvaluationResult("test", "1", {})

        def settle_step(self, prepared, decision):
            raise AssertionError("settlement must not run")

        def abort_step(self, prepared):
            assert prepared.candidate is not None
            calls.append(("abort", prepared.candidate.candidate_id))

    class BrokenPolicy:
        def decide(self, candidate, evaluation):
            raise RuntimeError("policy failed")

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    backend = Backend()
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=threshold_pipeline,
        candidate_backend=backend,
        candidate_evaluator=policy_plugin(backend, BrokenPolicy),
        default_job_samples=1,
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        trainer.run_once()
    assert calls == [("abort", "math:data:1")]


@pytest.mark.unit
def test_trainer_rejects_an_evaluator_that_replaces_its_evaluation_result() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            return PreparedStep.with_candidate(UpdateCandidate(batch.data_id), state=state)

        def evaluate(self, candidate):
            raise AssertionError("the explicit evaluator must replace the backend evaluator")

        def settle_step(self, prepared, decision):
            raise AssertionError("invalid evaluator decision must not settle")

        def abort_step(self, prepared):
            assert prepared.candidate is not None
            calls.append(("abort", prepared.candidate.candidate_id))

    class ReplacingEvaluator(CandidateEvaluationPlugin):
        def evaluate(self, candidate):
            del candidate
            return EvaluationResult("external", "1", {"score": 1.0})

        def decide(self, candidate, evaluation):
            del candidate, evaluation
            replacement = EvaluationResult("external", "1", {"score": 0.0})
            return SelectionDecision(False, "broken", "replaced result", replacement)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=threshold_pipeline,
        candidate_backend=Backend(),
        candidate_evaluator=ReplacingEvaluator(),
        default_job_samples=1,
    )

    with pytest.raises(ValueError, match="retain the evaluation result"):
        trainer.run_once()
    assert calls == [("abort", "math:data:1")]


@pytest.mark.unit
def test_trainer_restores_algorithm_state_from_metadata() -> None:
    records = SQLiteRecordStore()
    for item in (
        inference("i1"),
        report("r1", "i1", 1.0),
        inference("i2"),
        report("r2", "i2", 0.5),
    ):
        records.append(item)
    first = Trainer.build(
        "math",
        records,
        pipeline_factory=lambda context: threshold_pipeline(
            PipelineContext(context.scenario, default_job_samples=context.default_job_samples)
        ),
        candidate_backend=_PreparingBackend(),
        default_job_samples=1,
    )

    first_result = first.run_once(0)
    assert first_result is not None
    assert source_record_id(first.pending_job_data.samples[0]) == "i1"
    prepared = first.prepare_commit(first_result)
    first.commit(prepared)
    assert first.state == {"steps": 1}
    assert first.data_offset == prepared.high_water_offset == 2

    # Recovery restores consumption independently of the retained record bodies.
    recovered_state = first.algorithm_state_dict()
    second = Trainer.build(
        "math",
        records,
        pipeline_factory=lambda context: threshold_pipeline(
            PipelineContext(context.scenario, default_job_samples=context.default_job_samples)
        ),
        candidate_backend=_PreparingBackend(),
        algorithm_state=recovered_state,
        default_job_samples=1,
    )

    second.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    assert second.state == {"steps": 1}
    second_result = second.run_once(1)
    assert second_result is not None
    assert source_record_id(second.pending_job_data.samples[0]) == "i2"
    prepared = second.prepare_commit(second_result)
    second.commit(prepared)
    assert second.run_once(2) is None


@pytest.mark.unit
def test_commit_retains_readable_records_and_recovery_skips_committed_consumption(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    first_inference = positioned_inference(1)
    first_report = positioned_report(2, first_inference.agent_record_id, 1.0)
    second_inference = positioned_inference(3)
    second_report = positioned_report(4, second_inference.agent_record_id, 0.5)

    with SQLiteRecordStore(database) as first_store:
        for item in (first_inference, first_report, second_inference, second_report):
            first_store.append(item)
        first = Trainer.build(
            "math",
            first_store,
            pipeline_factory=threshold_pipeline,
            candidate_backend=_PreparingBackend(),
            default_job_samples=1,
        )
        first_result = first.run_once(0)
        assert first_result is not None
        prepared = first.prepare_commit(first_result)
        first.commit(prepared)

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

    with SQLiteRecordStore(database) as second_store:
        second = Trainer.build(
            "math",
            second_store,
            pipeline_factory=threshold_pipeline,
            candidate_backend=_PreparingBackend(),
            algorithm_state={"steps": 1},
            default_job_samples=1,
        )
        second.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
        second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        assert second.state == {"steps": 1}
        second_result = second.run_once(1)
        assert second_result is not None
        assert source_record_id(second.pending_job_data.samples[0]) == second_inference.agent_record_id
        prepared = second.prepare_commit(second_result)
        second.commit(prepared)
        assert second_store.count("math") == 4
        archived = second_store.audit_page("math")
        assert [entry.item for entry in archived] == [first_inference, first_report, second_inference, second_report]
        assert second.run_once(2) is None


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

        def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
            return Trainer.build(
                scenario,
                records,
                pipeline_factory=grouped_policy_pipeline,
                job_loop=JobLoop(component, self.job_plan(), scenario=scenario),
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
                default_job_samples=1,
            )

    dispatcher = Dispatcher(
        GroupedPgRecipe(**runtime_bindings(component, inference_runtime), name="grouped_pg"),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    scenario = dispatcher.get_or_create_scenario("math")
    serving = scenario.serving_runtime_load_id()
    for rid, score in (("i1", 0.2), ("i2", 0.8)):
        record = inference(rid)
        dispatcher.accept_record(replace(record, payload={**record.payload, "runtime_load_id": serving}))
        dispatcher.accept_record(report("r" + rid, rid, score, comparison_set="set-a"))

    for _ in range(1000):
        if scenario.trainer.state["finished_jobs"] == 1:
            break
        time.sleep(0.001)
    else:
        raise AssertionError("the training job did not finish")
    [job] = component.submitted
    assert TrainingMethod.from_dict(job.parameters["method"]) == TrainingMethod(_GROUPED_PG_OBJECTIVE)
    # The comparison set is one batch: the component computes its group-relative advantages from the job's data.
    assert [trajectory_reward(item) for item in job.data.samples] == [0.2, 0.8]
    prepared = prepare_slime_job(job.data, TrainingMethod(_GROUPED_PG_OBJECTIVE), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None and prepared.payload["advantages"] == pytest.approx([-1.0, 1.0])
    assert scenario.repository.current_artifact == scenario.repository.checkpoint_artifact
    # After the base release, the job's checkpoint is what the scenario serves.
    assert inference_runtime.activated[1:] == ["w1"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.unit
def test_sao_processor_defaults_action_mask_for_assembled_rollouts() -> None:
    """A multi-turn rollout reaches SAO through the shared assembly path,
    which fills no SAO-only fields; the processor must apply its documented
    single-turn default (action mask = loss mask) instead of rejecting the
    assembled sample on an empty action mask."""
    processor = sao_pipeline(PipelineContext("math"), accept_multi_turn=True)
    processor.push(training_inference("i1", [10, 20], [1], [-0.1]))
    processor.push(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    processor.push(report("r1", ("i1", "i2"), 1.0))

    assert processor.ready(1)
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert sample.training.get("turn_count", 1) > 1
    assert tuple(sample.training.get("action_mask", [])) == tuple(sample.training.get("loss_mask", []))
    assert any(tuple(sample.training.get("action_mask", [])))


@pytest.mark.parametrize("missing", ["tokens", "rollout_log_probs"])
def test_reported_samples_leave_required_tensor_validation_to_training_backend(missing: str) -> None:
    from reef.train.slime_backend.data_builder import to_slime_rollout_data

    processor = sao_pipeline(PipelineContext("math"))
    payload = {"tokens": [10, 20], "loss_mask": [1], "rollout_log_probs": [-0.1]}
    payload.pop(missing)
    processor.push(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            agent_record_id="i1",
            payload=payload,
        )
    )
    processor.push(report("r1", "i1", 1.0))
    batch = processor.build_job_data(1)
    assert len(batch.samples) == 1
    prepared = prepare_slime_job(batch, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    with pytest.raises(ValueError):
        to_slime_rollout_data(prepared.payload)
    assert processor.build_job_data(1) is batch
    assert processor.releasable_record_ids().isdisjoint({"i1", "r1"})


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
        trainer = Trainer.build(
            "math",
            records,
            pipeline_factory=threshold_pipeline,
            candidate_backend=_PreparingBackend(),
            default_job_samples=1,
        )
        assert trainer.pipeline_status()["record_data_incomplete"] is True
        result = trainer.run_once(0)
        assert result is not None
        assert source_record_id(trainer.pending_job_data.samples[0]) == "i3"
        prepared = trainer.prepare_commit(result)
        assert "r2" in prepared.consumed_ids
        assert prepared.metrics["records/data_incomplete"] == 1
        trainer.commit(prepared)
        assert trainer.run_once(1) is None
        assert "Skipping report r2" in caplog.text
    with SQLiteRecordStore(database) as records:
        resumed = Trainer.build(
            "math",
            records,
            pipeline_factory=threshold_pipeline,
            candidate_backend=_PreparingBackend(),
            algorithm_state=prepared.algorithm_state,
            default_job_samples=1,
        )
        resumed.replay_retained(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
        resumed.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        assert resumed.run_once(1) is None
        assert resumed.pipeline_status()["evicted_record_count"] == 1
        assert records.get("math", "i3") is not None
