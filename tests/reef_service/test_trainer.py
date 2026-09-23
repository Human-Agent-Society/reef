from __future__ import annotations

import time
from contextlib import closing

import pytest
from reef_service.runtime_stubs import StubInferenceRuntime, StubTrainingRuntime, candidate_backend, runtime_bindings

import reef.train.processors.reported as reported_module
from recipes.sao import SAOProcessor
from recipes.tttd import TTTDGroupedRolloutReport, TTTDProcessor
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.dispatcher import Dispatcher
from reef.recipe import WeightTrainingRecipe
from reef.runtime.interfaces import ActivatedModel, ModelCandidate, PreparedTrainingStep
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train import ProcessorContext, Trainer
from reef.train.algos import StepScheduling
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.evaluation import (
    BackendEvaluateMixin,
    CandidateEvaluationPlugin,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.train.processors import DataProcessor
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import TrainingBatch, TrainStepResult, TrajectoryItem, trajectory_groups

from ._grouped_pg import GROUPED_PG_OBJECTIVE as _GROUPED_PG_OBJECTIVE
from ._grouped_pg import GroupedPolicyProcessor
from ._threshold_processor import ThresholdProcessor

NUM_GPUS = 0


def policy_plugin(backend: object, policy: type) -> CandidateEvaluationPlugin:
    """A plugin that measures through ``backend`` and decides with ``policy``'s mixin."""

    class _Plugin(policy, BackendEvaluateMixin, CandidateEvaluationPlugin):  # type: ignore[misc, valid-type]
        pass

    plugin = _Plugin()
    plugin._candidate_backend = backend
    return plugin


class _PreparingBackend(CandidateBackend):
    """Dispatched test backend that commits only the objective's state transition."""

    def __init__(self, objective: str = "sft") -> None:
        self._objective = objective

    @property
    def dispatched(self) -> bool:
        return True

    def initial_state(self):
        return {}

    def prepare_step(self, batch, state, scenario_step):
        del scenario_step
        prepared = prepare_slime_step(batch, self._objective, state, StepScheduling())
        return PreparedStep.skipped(state=prepared.next_algorithm_state, metrics=prepared.metrics)

    def evaluate(self, candidate):
        raise AssertionError("state-only test backend does not produce candidates")

    def settle_step(self, prepared, decision):
        raise AssertionError("state-only test backend does not settle candidates")

    def abort_step(self, prepared):
        del prepared


@pytest.mark.unit
def test_training_backend_names_both_sides_of_the_durable_commit_handshake() -> None:
    calls = []

    class Receiver(StubInferenceRuntime):
        def acknowledge_publication(self, training_job_id):
            calls.append(training_job_id)

    runtime = StubTrainingRuntime()
    runtime.inference = Receiver(runtime, base_url="http://inference")
    backend = candidate_backend(runtime, "sft", StepScheduling())

    assert not hasattr(CandidateBackend, "reconcile")
    assert hasattr(CandidateBackend, "recover_pending_step")
    assert hasattr(CandidateBackend, "acknowledge_commit")
    backend.recover_pending_step(4)
    assert calls == []
    backend.acknowledge_commit(5, "job-4")
    assert calls == ["job-4"]
    assert runtime.inference.inference_admission_status["open"]


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
    processor = DataProcessor(ProcessorContext("math"))
    processor.ingest(inference("i1"))

    assert not processor.ready()
    assert processor.status() == {}
    assert processor.retention_decision().protected_agent_record_ids == frozenset({"i1"})


@pytest.mark.unit
def test_pairing_processor_uses_feedback_without_score_filtering() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 2}))
    processor.ingest(inference("bad"))
    processor.ingest(inference("good"))
    processor.ingest(report("bad-r", "bad", 0.1))
    processor.ingest(report("good-r", "good", 1.0))
    assert [trajectory_reward(sample) for sample in processor.build_batch().items] == [0.1, 1.0]


@pytest.mark.unit
def test_pairing_processor_emits_policy_samples_for_spo() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1", 0.8))

    batch = processor.build_batch()
    assert isinstance(batch, TrainingBatch)
    assert batch.batch_id == "math:threshold:1"
    (item,) = batch.items
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
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(inference("i1"))
    with pytest.raises(ReportValidationError, match="finite"):
        processor.ingest(report("r1", "i1", bad_score))
    assert not processor.ready()


@pytest.mark.unit
def test_pairing_processor_assembles_ordered_multi_reference_report() -> None:
    processor = ThresholdProcessor(
        ProcessorContext("math", {"batch_size": 1, "accept_multi_turn_policy_samples": True})
    )
    first = training_inference("i1", [10, 20], [1], [-0.1])
    second = training_inference("i2", [10, 20, 11, 21], [1], [-0.2])
    final_report = report("r1", ("i1", "i2"), 0.75)

    processor.ingest(first)
    processor.ingest(second)
    assert not processor.ready()
    processor.ingest(final_report)

    batch = processor.build_batch()
    (item,) = batch.items
    assert source_record_id(item) == "r1"
    assert trajectory_reward(item) == 0.75
    assert item.training["tokens"] == [10, 20, 11, 21]
    assert item.training["loss_mask"] == [1, 0, 1]
    assert item.training["rollout_log_probs"] == [-0.1, 0.0, -0.2]
    assert item.training["runtime_load_id"] == "wv-1"
    assert item.training["turn_count"] == 2
    assert [record["payload"] for record in item.metadata["records"]] == [first.payload, second.payload]
    processor.acknowledge(batch.batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"i1", "i2", "r1"})


@pytest.mark.unit
@pytest.mark.parametrize(
    "processor_type",
    [
        pytest.param(SAOProcessor, id="sao"),
        # openclawrl is absent: it consumes no reports, so multi-reference
        # assembly never happens there by construction.
        pytest.param(TTTDProcessor, id="tttd"),
    ],
)
def test_cookbook_processors_assemble_before_rejecting_multi_turn_reports(
    processor_type,
    monkeypatch,
) -> None:
    assembled_samples: list[TrajectoryItem] = []
    assembler = reported_module.make_multi_turn_policy_trajectory

    def record_assembly(*args, **kwargs):
        sample = assembler(*args, **kwargs)
        assert sample is not None
        assembled_samples.append(sample)
        return sample

    monkeypatch.setattr(reported_module, "make_multi_turn_policy_trajectory", record_assembly)

    config = {"batch_size": 1}
    metadata = {}
    if processor_type is TTTDProcessor:
        config.update(groups_per_step=1, rollouts_per_group=2)
        metadata = {
            "algorithm": "tttd",
            "step": 0,
            "group": 0,
            "rollout": 0,
            "groups_per_step": 1,
            "rollouts_per_group": 2,
            "comparison_set": "tttd-step-0-group-0",
        }

    report_type = TTTDGroupedRolloutReport if processor_type is TTTDProcessor else None
    processor = processor_type(ProcessorContext("math", config, report_type=report_type))
    processor.ingest(training_inference("i1", [10, 20], [1], [-0.1]))
    multi_turn_report = report("r1", ("i1", "i2"), 1.0, **metadata)
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.ingest(multi_turn_report)
    assert assembled_samples == []
    processor.ingest(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        processor.ingest(multi_turn_report)
    assert assembled_samples
    assert all(
        sample.training.get("turn_count", 1) == 2 and (sample.training.get("turn_count", 1) > 1)
        for sample in assembled_samples
    )
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "i2", "r1"}


@pytest.mark.unit
def test_failed_multi_turn_assembly_preserves_inputs_and_other_live_reports() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(training_inference("i1", [10, 20], [1], [-0.1]))
    processor.ingest(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    processor.ingest(report("single", "i1", 1.0))
    with pytest.raises(ValueError, match="accept_multi_turn"):
        processor.ingest(report("multi", ("i1", "i2"), 1.0))
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "i2", "single", "multi"}
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "i2", "multi"}


@pytest.mark.unit
def test_pairing_processor_raises_for_forked_multi_turn_episode() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"accept_multi_turn_policy_samples": True}))
    processor.ingest(training_inference("i1", [1, 2, 3], [1], [-0.1]))
    processor.ingest(training_inference("i2", [1, 9, 3, 4], [1], [-0.2]))
    with pytest.raises(ValueError, match="cannot assemble"):
        processor.ingest(report("r1", ("i1", "i2"), 1.0))
    assert not processor.ready()
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "i2", "r1"}


@pytest.mark.unit
def test_pairing_processor_rejects_report_eligibility_flags() -> None:
    processor = ThresholdProcessor(ProcessorContext("math"))
    processor.ingest(inference("i1"))
    with pytest.raises(ReportValidationError, match="eligible"):
        processor.ingest(report("r1", "i1", 0.0, training={"eligible": False}))
    assert not processor.ready()
    assert processor.retention_decision().protected_agent_record_ids == {"i1"}


@pytest.mark.unit
def test_pairing_retention_consumes_reports_exactly_and_releases_trained_inference() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1", 1.0))
    processor.ingest(report("r2", "i1", 0.5))

    first = processor.build_batch()
    processor.acknowledge(first.batch_id)
    decision = processor.retention_decision()
    assert decision.releasable_agent_record_ids == frozenset({"r1"})
    assert decision.protected_agent_record_ids == frozenset({"i1", "r2"})

    second = processor.build_batch()
    assert trajectory_reward(second.items[0]) == 0.5
    processor.acknowledge(second.batch_id)
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "r1", "r2"})

    processor.ingest(report("late", "i1", 0.25))
    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"i1", "r1", "r2", "late"})


@pytest.mark.unit
def test_pairing_retention_protects_low_score_feedback_until_consumed() -> None:
    processor = ThresholdProcessor(ProcessorContext("math"))
    processor.ingest(inference("i1"))
    processor.ingest(report("low", "i1", 0.1))
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "low"}
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == {"i1", "low"}


@pytest.mark.unit
def test_pairing_retention_does_not_protect_consumed_inferences() -> None:
    first = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    first.ingest(inference("i1"))
    first.ingest(report("r1", "i1", 1.0))
    batch = first.build_batch()
    first.acknowledge(batch.batch_id)

    # Consumed inferences are released and may be compacted; the processor does
    # not require them to be retained across restart.
    decision = first.retention_decision()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "r1"})


@pytest.mark.unit
def test_grpo_retention_releases_complete_comparison_set_after_acknowledgement() -> None:
    processor = GroupedPolicyProcessor(ProcessorContext("math", {"batch_size": 1}))
    for agent_record_id, score in (("i1", 0.2), ("i2", 0.8)):
        processor.ingest(inference(agent_record_id))
        processor.ingest(report("r" + agent_record_id, agent_record_id, score, comparison_set="set-a"))

    batch = processor.build_batch()
    processor.acknowledge(batch.batch_id)
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "i2", "ri1", "ri2"})


@pytest.mark.unit
def test_algorithms_consume_formatted_batches_and_keep_algorithm_state() -> None:
    sft_processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    sft_processor.ingest(inference("i1"))
    sft_processor.ingest(report("r1", "i1", 1.0))
    sft_result = prepare_slime_step(sft_processor.build_batch(), "sft", {}, StepScheduling())
    assert sft_result.next_algorithm_state == {"steps": 1}

    grpo_processor = GroupedPolicyProcessor(ProcessorContext("math", {"batch_size": 1}))
    for rid, score in (("i3", 0.2), ("i4", 0.8)):
        grpo_processor.ingest(inference(rid))
        grpo_processor.ingest(report("r" + rid, rid, score, comparison_set="x"))
    result = prepare_slime_step(grpo_processor.build_batch(), _GROUPED_PG_OBJECTIVE, {}, StepScheduling())
    assert result.metrics["advantages"] == pytest.approx((-1.0, 1.0))


@pytest.mark.unit
def test_trainer_reserves_batch_and_commits_backend_preparation() -> None:
    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(ProcessorContext(context.scenario, {"batch_size": 1})),
        candidate_backend=_PreparingBackend(),
    )

    batch = trainer.reserve_training_batch()
    assert batch is not None
    assert trainer.processor_status() == {}
    execution = trainer.execute_reserved_step(0)
    result = execution.result

    assert result is not None
    assert result.metrics["samples"] == 1
    assert result.artifact is None
    assert result.runtime_load_id is None
    assert trainer.state == {}
    prepared = trainer.prepare_commit(result)
    trainer.commit(prepared)
    trainer.apply_compaction(prepared.compacted_ids)
    assert trainer.state == {"steps": 1}


@pytest.mark.unit
def test_trainer_tells_the_processor_a_dropped_batch_before_acknowledging_it() -> None:
    events: list[str] = []

    class NotingProcessor(ThresholdProcessor):
        def dropped(self, batch_id: str) -> None:
            events.append(f"dropped {batch_id}")

        def acknowledge(self, batch_id: str) -> frozenset[str]:
            events.append(f"acknowledged {batch_id}")
            return super().acknowledge(batch_id)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: NotingProcessor(context.with_config({"batch_size": 1})),
        candidate_backend=_PreparingBackend(),
    )
    batch = trainer.reserve_training_batch()
    assert batch is not None
    trainer.reject_pending({"reason": "stale"})
    assert events == [f"dropped {batch.batch_id}", f"acknowledged {batch.batch_id}"]
    assert trainer.reserve_training_batch() is None, "the dropped batch is consumed"


@pytest.mark.unit
def test_trainer_executes_candidate_policy_between_evaluation_and_settlement() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {"steps": 0}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            calls.append("prepare")
            return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state=state)

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
            return SelectionDecision("select", "test", "1", "selected by test", evaluation)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    backend = Backend()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        candidate_backend=backend,
        candidate_evaluator=policy_plugin(backend, Policy),
    )

    result = trainer.run_once()

    assert result is not None
    assert result.metrics["selection"]["outcome"] == "select"
    assert calls == ["prepare", "evaluate", "decide", "settle"]


@pytest.mark.unit
def test_trainer_rejects_structural_plugin_before_constructing_processor() -> None:
    from ._candidate_evaluation_plugin import DuckPlugin

    def processor_factory(context):
        raise AssertionError("an invalid plugin must fail before processor construction")

    with pytest.raises(TypeError, match="must inherit CandidateEvaluationPlugin"):
        Trainer.build(
            "math",
            SQLiteRecordStore(),
            processor_factory=processor_factory,
            candidate_backend=_PreparingBackend(),
            candidate_evaluator=DuckPlugin(),
        )


def test_trainer_uses_explicit_candidate_evaluator_instead_of_backend_fallback() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state=state)

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
            return SelectionDecision("select", "external", "1", "selected by module", evaluation)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    candidate_evaluator = ExternalEvaluator()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        candidate_backend=Backend(),
        candidate_evaluator=candidate_evaluator,
    )

    result = trainer.run_once()

    assert result is not None
    assert trainer.candidate_evaluator is candidate_evaluator
    assert calls == [
        ("evaluate", "math:threshold:1"),
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
            return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state=state)

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
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        candidate_backend=backend,
        candidate_evaluator=policy_plugin(backend, BrokenPolicy),
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        trainer.run_once()
    assert calls == [("abort", "math:threshold:1")]


@pytest.mark.unit
def test_trainer_rejects_an_evaluator_that_replaces_its_evaluation_result() -> None:
    calls = []

    class Backend(CandidateBackend):
        def initial_state(self):
            return {}

        def prepare_step(self, batch, state, scenario_step):
            del scenario_step
            return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state=state)

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
            return SelectionDecision("reject", "broken", "1", "replaced result", replacement)

    records = SQLiteRecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        candidate_backend=Backend(),
        candidate_evaluator=ReplacingEvaluator(),
    )

    with pytest.raises(ValueError, match="retain the evaluation result"):
        trainer.run_once()
    assert calls == [("abort", "math:threshold:1")]


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
        processor_factory=lambda context: ThresholdProcessor(ProcessorContext(context.scenario, {"batch_size": 1})),
        candidate_backend=_PreparingBackend(),
    )

    first_batch = first.reserve_training_batch()
    assert first_batch is not None
    first_result = first.execute_reserved_step(0).result
    assert first_result is not None
    assert source_record_id(first.pending_batch.items[0]) == "i1"
    prepared = first.prepare_commit(first_result)
    first.commit(prepared)
    first.apply_compaction(prepared.compacted_ids)
    assert first.state == {"steps": 1}
    assert first.data_offset == prepared.high_water_offset == 2

    # Recovery restores consumption independently of the retained record bodies.
    recovered_state = first.algorithm_state_dict()
    second = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(ProcessorContext(context.scenario, {"batch_size": 1})),
        candidate_backend=_PreparingBackend(),
        algorithm_state=recovered_state,
    )

    second.reingest(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
    second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
    assert second.state == {"steps": 1}
    second_batch = second.reserve_training_batch()
    assert second_batch is not None
    second_result = second.execute_reserved_step(1).result
    assert second_result is not None
    assert source_record_id(second.pending_batch.items[0]) == "i2"
    prepared = second.prepare_commit(second_result)
    second.commit(prepared)
    second.apply_compaction(prepared.compacted_ids)
    assert second.reserve_training_batch() is None


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
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            candidate_backend=_PreparingBackend(),
        )
        batch = first.reserve_training_batch()
        assert batch is not None
        first_result = first.execute_reserved_step(0).result
        assert first_result is not None
        prepared = first.prepare_commit(first_result)
        first.commit(prepared)
        first.apply_compaction(prepared.compacted_ids)

        assert first_store.get("math", first_inference.agent_record_id) == first_inference
        assert first_store.get("math", first_report.agent_record_id) == first_report
        assert first_store.get_for_audit("math", first_inference.agent_record_id).item == first_inference
        assert first_store.get_for_audit("math", first_report.agent_record_id).item == first_report
        assert prepared.compacted_ids == frozenset()
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
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            candidate_backend=_PreparingBackend(),
            algorithm_state={"steps": 1},
        )
        second.reingest(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
        second.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        assert second.state == {"steps": 1}
        batch = second.reserve_training_batch()
        assert batch is not None
        second_result = second.execute_reserved_step(1).result
        assert second_result is not None
        assert source_record_id(second.pending_batch.items[0]) == second_inference.agent_record_id
        prepared = second.prepare_commit(second_result)
        second.commit(prepared)
        second.apply_compaction(prepared.compacted_ids)
        assert second_store.count("math") == 4
        archived = second_store.audit_page("math")
        assert [entry.item for entry in archived] == [first_inference, first_report, second_inference, second_report]
        assert all(entry.compacted_at is None for entry in archived)
        assert second.reserve_training_batch() is None


@pytest.mark.integration
def test_scenario_runtime_executes_grpo_as_one_async_transaction(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_text("trained")

    class FakeTrainingRuntime(StubTrainingRuntime):
        def __init__(self):
            super().__init__(base_url="http://trainer")
            self.calls = []

        @property
        def inference_handler(self):
            return None

        def prepare_training_step(
            self, batch, objective, algorithm_state, scheduling, scenario_step, *, serving_runtime_load_id=None
        ):
            prepared = prepare_slime_step(batch, objective, algorithm_state, scheduling)
            assert prepared.payload is not None
            self.calls.append(("prepare", batch, objective))
            return PreparedTrainingStep(
                action="train",
                payload={**prepared.payload, "rollout_id": scenario_step},
                next_algorithm_state=prepared.next_algorithm_state,
                metrics=prepared.metrics,
            )

        def train_candidate(self, payload):
            self.calls.append(("execute", payload))
            job_id = f"job-{payload['rollout_id']}"
            return ModelCandidate(
                candidate_id=job_id,
                training_job_id=job_id,
                checkpoint_path=str(checkpoint),
                current_runtime_load_id=None,
            )

        def activate_candidate(self, candidate):
            return ActivatedModel(candidate.candidate_id, "slime-v1")

        def reject_candidate(self, candidate, decision):
            raise AssertionError("candidate should be selected")

    training_runtime = FakeTrainingRuntime()
    initial = tmp_path / "initial"
    initial.mkdir()

    # The minimal grouped pg recipe — the cookbook grouped-method shape the
    # define-a-recipe guide describes, kept local to this suite.
    class GroupedPgRecipe(WeightTrainingRecipe):
        objective = _GROUPED_PG_OBJECTIVE
        loss_family = "pg"

        def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
            return Trainer.build(
                scenario,
                records,
                processor_factory=lambda context: GroupedPolicyProcessor(context.with_config({"batch_size": 1})),
                candidate_backend=candidate_backend(self.training_runtime, self.objective, StepScheduling()),
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
            )

    dispatcher = Dispatcher(
        GroupedPgRecipe(**runtime_bindings(training_runtime), name="grouped_pg"),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    for rid, score in (("i1", 0.2), ("i2", 0.8)):
        dispatcher.accept_record(inference(rid))
        dispatcher.accept_record(report("r" + rid, rid, score, comparison_set="set-a"))

    runtime = dispatcher.get_or_create_scenario("math")
    for _ in range(1000):
        if runtime.scenario_step == 1:
            break
        time.sleep(0.001)
    else:
        raise AssertionError("async training did not commit")
    assert runtime.trainer.state == {"steps": 1}
    assert runtime.scenario_step == 1
    assert runtime.repository.current_artifact == runtime.repository.checkpoint_artifact
    prepare = training_runtime.calls[0]
    assert prepare[2] == _GROUPED_PG_OBJECTIVE
    assert trajectory_reward(trajectory_groups(prepare[1])[0][1]) == 0.8
    assert training_runtime.calls[1][1]["advantages"] == pytest.approx([-1.0, 1.0])
    assert [call[0] for call in training_runtime.calls] == ["prepare", "execute"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.unit
def test_sao_processor_defaults_action_mask_for_assembled_episodes() -> None:
    """A multi-turn episode reaches SAO through the shared assembly path,
    which fills no SAO-only fields; the processor must apply its documented
    single-turn default (action mask = loss mask) instead of rejecting the
    assembled sample on an empty action mask."""
    processor = SAOProcessor(ProcessorContext("math", {"batch_size": 1, "accept_multi_turn_policy_samples": True}))
    processor.ingest(training_inference("i1", [10, 20], [1], [-0.1]))
    processor.ingest(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    processor.ingest(report("r1", ("i1", "i2"), 1.0))

    assert processor.ready()
    batch = processor.build_batch()
    (sample,) = batch.items
    assert sample.training.get("turn_count", 1) > 1
    assert tuple(sample.training.get("action_mask", [])) == tuple(sample.training.get("loss_mask", []))
    assert any(tuple(sample.training.get("action_mask", [])))


@pytest.mark.parametrize("missing", ["tokens", "rollout_log_probs"])
def test_reported_samples_leave_required_tensor_validation_to_training_backend(missing: str) -> None:
    from reef.train.slime_backend.data_builder import to_slime_rollout_data

    processor = SAOProcessor(ProcessorContext("math"))
    payload = {"tokens": [10, 20], "loss_mask": [1], "rollout_log_probs": [-0.1]}
    payload.pop(missing)
    processor.ingest(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            agent_record_id="i1",
            payload=payload,
        )
    )
    processor.ingest(report("r1", "i1", 1.0))
    batch = processor.build_batch()
    assert len(batch.items) == 1
    prepared = prepare_slime_step(batch, "sao", {}, StepScheduling(unit="sample"))
    assert prepared.payload is not None
    with pytest.raises(ValueError):
        to_slime_rollout_data(prepared.payload)
    assert processor.build_batch() is batch
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r1"}


def test_capacity_loss_skips_orphan_report_and_remains_visible_after_restart(tmp_path, caplog):
    database = tmp_path / "records.sqlite3"
    with SQLiteRecordStore(database) as records:
        records.append(positioned_inference(1))
        records.append(positioned_report(2, "i1", 1.0))
        # Keep the report but evict its input before the processor sees either.
        with records._transaction("math", write=False) as connection:
            keep_bytes = connection.exec_driver_sql(
                "SELECT body_bytes FROM agent_record WHERE agent_record_id='r2'"
            ).scalar_one()
        with closing(SQLiteScenarioStorage(tmp_path)) as storage:
            assert storage.prune(days=7, max_bytes=keep_bytes) == 1
        records.append(positioned_inference(3))
        records.append(positioned_report(4, "i3", 1.0))
        trainer = Trainer.build(
            "math",
            records,
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            candidate_backend=_PreparingBackend(),
        )
        assert trainer.processor_status()["record_data_incomplete"] is True
        batch = trainer.reserve_training_batch()
        assert batch is not None
        assert source_record_id(batch.items[0]) == "i3"
        result = trainer.execute_reserved_step(0).result
        assert result is not None
        prepared = trainer.prepare_commit(result)
        assert "r2" in prepared.consumed_ids
        assert prepared.metrics["records/data_incomplete"] == 1
        trainer.commit(prepared)
        assert trainer.reserve_training_batch() is None
        assert "Skipping report r2" in caplog.text
    with SQLiteRecordStore(database) as records:
        resumed = Trainer.build(
            "math",
            records,
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            candidate_backend=_PreparingBackend(),
            algorithm_state=prepared.algorithm_state,
        )
        resumed.reingest(up_to_sequence=prepared.high_water_sequence, consumed_ids=prepared.consumed_ids)
        resumed.restore_record_progress(after_sequence=prepared.high_water_sequence, offset=prepared.high_water_offset)
        assert resumed.reserve_training_batch() is None
        assert resumed.processor_status()["evicted_record_count"] == 1
        assert records.get("math", "i3") is not None


def test_stale_batches_survive_restart_without_retiring_records(tmp_path):
    from reef.scenario.factory import _consumed_by_committed_steps
    from reef.storage.commit_log import CommitLogScenarioStore

    database = tmp_path / "records.sqlite3"
    for index in (1, 2):
        with SQLiteRecordStore(database) as records:
            records.append(inference(f"i{index}"))
            records.append(report(f"r{index}", f"i{index}", 1.0))
            trainer = Trainer.build(
                "math",
                records,
                processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
                candidate_backend=_PreparingBackend(),
            )
            with closing(CommitLogScenarioStore("math", records)) as session:
                consumed = _consumed_by_committed_steps(session, None, "math")
                trainer.reingest(up_to_sequence=0, consumed_ids=consumed)
                batch = trainer.reserve_training_batch()
                assert batch is not None
                assert source_record_id(batch.items[0]) == f"i{index}"
                trainer.reject_pending({"reason": "stale"})
                assert trainer.reserve_training_batch() is None
                assert records.get("math", f"i{index}") is not None
                assert len(records.compaction_receipts("math")) == index
