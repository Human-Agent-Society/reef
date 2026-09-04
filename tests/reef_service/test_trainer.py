from __future__ import annotations

import time

import pytest

import reef.train.processors.reported as reported_module
from recipes.sao import SAOProcessor
from recipes.tttd import TTTDGroupedRolloutReport, TTTDProcessor
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.recipe import WeightTrainingRecipe
from reef.records import RecordStore
from reef.runtime import ActivatedModel, ModelCandidate, PreparedTrainingStep, TrainingRuntime
from reef.train import ProcessorContext, Trainer
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.evaluation import (
    DefaultCandidateEvaluationPlugin,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.train.processors import DataProcessor
from reef.train.slime_backend.backend import SlimeTrainingBackend
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicyBatch, PolicySample, TrainStepResult

from ._grouped_pg import GROUPED_PG_PREPARER as _GROUPED_PG_PREPARER
from ._grouped_pg import GroupedPolicyProcessor
from ._threshold_processor import ThresholdProcessor

NUM_GPUS = 0


class _PreparingBackend(TrainingBackend):
    """Dispatched test backend that commits only the preparer's state transition."""

    def __init__(self, preparer: str = "sft") -> None:
        self._preparer = preparer

    @property
    def dispatched(self) -> bool:
        return True

    def initial_state(self):
        return {}

    def prepare_step(self, batch, state, scenario_step):
        del scenario_step
        prepared = prepare_slime_step(batch, self._preparer, state)
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

    class Runtime(TrainingRuntime):
        @property
        def inference_backend(self):
            return None

        def reconcile_training_job(
            self,
            scenario_step,
            *,
            committed_training_job_id=None,
            committed_training_without_job_id=False,
        ):
            calls.append((scenario_step, committed_training_job_id, committed_training_without_job_id))

        def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
            raise AssertionError("not used")

        def train_candidate(self, payload):
            raise AssertionError("not used")

        def activate_candidate(self, candidate):
            raise AssertionError("not used")

        def reject_candidate(self, candidate, decision):
            raise AssertionError("not used")

    backend = SlimeTrainingBackend(Runtime(base_url="http://trainer"), "sft")

    assert not hasattr(TrainingBackend, "reconcile")
    assert hasattr(TrainingBackend, "recover_pending_step")
    assert hasattr(TrainingBackend, "acknowledge_commit")
    backend.recover_pending_step(
        4,
        committed_training_job_id=None,
        committed_training_without_job_id=True,
    )
    backend.acknowledge_commit(5, "job-4")

    assert calls == [(4, None, True), (5, "job-4", False)]


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
def test_pairing_processor_correlates_report_and_applies_improved_only_policy() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1, "min_score": 0.5}))
    processor.ingest(report("late", "good", 1.0))
    assert not processor.ready()
    processor.ingest(inference("bad"))
    processor.ingest(report("bad-r", "bad", 0.1))
    processor.ingest(inference("good"))

    batch = processor.build_batch()
    assert isinstance(batch, PolicyBatch)
    assert [sample.source_agent_record_id for sample in batch.samples] == ["good"]


@pytest.mark.unit
def test_pairing_processor_emits_policy_samples_for_spo() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1", 0.8))

    batch = processor.build_batch()
    assert isinstance(batch, PolicyBatch)
    assert batch.batch_id == "math:threshold:1"
    assert batch.samples == (PolicySample("i1", (1, 2), (0, 1), (-0.2,), 0.8),)


@pytest.mark.unit
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_pairing_processor_never_trains_a_non_finite_score(bad_score: float) -> None:
    # NaN slips past any `score < min_score` comparison; the shared
    # eligibility gate rejects every non-finite score as terminal.
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1", bad_score))

    assert not processor.ready()
    assert "r1" in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
def test_pairing_processor_assembles_ordered_multi_reference_report() -> None:
    processor = ThresholdProcessor(
        ProcessorContext("math", {"batch_size": 1, "accept_multi_turn_policy_samples": True})
    )
    first = training_inference("i1", [10, 20], [1], [-0.1])
    second = training_inference("i2", [10, 20, 11, 21], [1], [-0.2])
    final_report = report("r1", ("i1", "i2"), 0.75)

    processor.ingest(first)
    processor.ingest(final_report)
    assert not processor.ready()
    processor.ingest(second)

    batch = processor.build_batch()
    assert batch.samples == (
        PolicySample(
            "r1",
            (10, 20, 11, 21),
            (1, 0, 1),
            (-0.1, 0.0, -0.2),
            0.75,
            runtime_load_id="wv-1",
            turn_count=2,
        ),
    )
    assert batch.samples[0].is_multi_turn
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
    assembled_samples: list[PolicySample] = []
    assembler = reported_module.make_multi_turn_policy_sample

    def record_assembly(*args, **kwargs):
        sample = assembler(*args, **kwargs)
        assert sample is not None
        assembled_samples.append(sample)
        return sample

    monkeypatch.setattr(reported_module, "make_multi_turn_policy_sample", record_assembly)

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
    processor.ingest(multi_turn_report)

    assert not processor.ready()
    assert assembled_samples == []
    assert processor.retention_decision().protected_agent_record_ids == frozenset({"i1", "i2", "r1"})

    processor.ingest(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))

    assert not processor.ready()
    assert assembled_samples
    assert all(sample.turn_count == 2 and sample.is_multi_turn for sample in assembled_samples)
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "i2", "r1"})


@pytest.mark.unit
def test_rejected_multi_turn_report_keeps_source_owned_by_live_report() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1}))
    processor.ingest(training_inference("i1", [10, 20], [1], [-0.1]))
    processor.ingest(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    processor.ingest(report("single", "i1", 1.0))
    processor.ingest(report("multi", ("i1", "i2"), 1.0))

    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset({"i1", "single"})
    assert decision.releasable_agent_record_ids == frozenset({"i2", "multi"})

    batch = processor.build_batch()
    processor.acknowledge(batch.batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"i1", "i2", "single", "multi"})


@pytest.mark.unit
def test_pairing_processor_releases_forked_multi_turn_episode() -> None:
    processor = ThresholdProcessor(
        ProcessorContext("math", {"batch_size": 1, "accept_multi_turn_policy_samples": True})
    )
    processor.ingest(training_inference("i1", [1, 2, 3], [1], [-0.1]))
    processor.ingest(training_inference("i2", [1, 9, 3, 4], [1], [-0.2]))
    processor.ingest(report("r1", ("i1", "i2"), 1.0))

    assert not processor.ready()
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "i2", "r1"})


@pytest.mark.unit
def test_pairing_processor_releases_report_marked_ineligible_for_training() -> None:
    processor = ThresholdProcessor(
        ProcessorContext("math", {"batch_size": 1, "accept_multi_turn_policy_samples": True})
    )
    processor.ingest(inference("i1"))
    processor.ingest(inference("i2"))
    processor.ingest(report("r1", ("i1", "i2"), 0.0, training={"eligible": False}))

    assert not processor.ready()
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "i2", "r1"})


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
    assert second.samples[0].reward == 0.5
    processor.acknowledge(second.batch_id)
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "r1", "r2"})

    processor.ingest(report("late", "i1", 0.25))
    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"i1", "r1", "r2", "late"})


@pytest.mark.unit
def test_pairing_retention_releases_terminal_filtered_reports() -> None:
    processor = ThresholdProcessor(ProcessorContext("math", {"batch_size": 1, "min_score": 0.5}))
    processor.ingest(inference("i1"))
    processor.ingest(report("low", "i1", 0.1))

    decision = processor.retention_decision()

    assert decision.protected_agent_record_ids == frozenset({"i1"})
    assert decision.releasable_agent_record_ids == frozenset({"low"})


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
    sft_result = prepare_slime_step(sft_processor.build_batch(), "sft", {})
    assert sft_result.next_algorithm_state == {"steps": 1}

    grpo_processor = GroupedPolicyProcessor(ProcessorContext("math", {"batch_size": 1}))
    for rid, score in (("i3", 0.2), ("i4", 0.8)):
        grpo_processor.ingest(inference(rid))
        grpo_processor.ingest(report("r" + rid, rid, score, comparison_set="x"))
    result = prepare_slime_step(grpo_processor.build_batch(), _GROUPED_PG_PREPARER, {})
    assert result.metrics["advantages"] == pytest.approx((-1.0, 1.0))


@pytest.mark.unit
def test_trainer_reserves_batch_and_commits_backend_preparation() -> None:
    records = RecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(ProcessorContext(context.scenario, {"batch_size": 1})),
        training_backend=_PreparingBackend(),
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
def test_trainer_executes_candidate_policy_between_evaluation_and_settlement() -> None:
    calls = []

    class Backend(TrainingBackend):
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

    records = RecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    backend = Backend()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        training_backend=backend,
        candidate_evaluator=DefaultCandidateEvaluationPlugin(backend, Policy()),
    )

    result = trainer.run_once()

    assert result is not None
    assert result.metrics["selection"]["outcome"] == "select"
    assert calls == ["prepare", "evaluate", "decide", "settle"]


@pytest.mark.unit
def test_trainer_uses_explicit_candidate_evaluator_instead_of_backend_fallback() -> None:
    calls = []

    class Backend(TrainingBackend):
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

    class ExternalEvaluator:
        def evaluate(self, candidate):
            calls.append(("evaluate", candidate.candidate_id))
            return EvaluationResult("external", "1", {"score": 0.9})

        def decide(self, candidate, evaluation):
            del candidate
            calls.append(("decide", evaluation.evaluator))
            return SelectionDecision("select", "external", "1", "selected by module", evaluation)

    records = RecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    candidate_evaluator = ExternalEvaluator()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        training_backend=Backend(),
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

    class Backend(TrainingBackend):
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

    records = RecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    backend = Backend()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        training_backend=backend,
        candidate_evaluator=DefaultCandidateEvaluationPlugin(backend, BrokenPolicy()),
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        trainer.run_once()
    assert calls == [("abort", "math:threshold:1")]


@pytest.mark.unit
def test_trainer_rejects_an_evaluator_that_replaces_its_evaluation_result() -> None:
    calls = []

    class Backend(TrainingBackend):
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

    class ReplacingEvaluator:
        def evaluate(self, candidate):
            del candidate
            return EvaluationResult("external", "1", {"score": 1.0})

        def decide(self, candidate, evaluation):
            del candidate, evaluation
            replacement = EvaluationResult("external", "1", {"score": 0.0})
            return SelectionDecision("reject", "broken", "1", "replaced result", replacement)

    records = RecordStore()
    records.append(inference("i1"))
    records.append(report("r1", "i1", 1.0))
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        training_backend=Backend(),
        candidate_evaluator=ReplacingEvaluator(),
    )

    with pytest.raises(ValueError, match="retain the evaluation result"):
        trainer.run_once()
    assert calls == [("abort", "math:threshold:1")]


@pytest.mark.unit
def test_trainer_restores_algorithm_state_from_metadata() -> None:
    records = RecordStore()
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
        training_backend=_PreparingBackend(),
    )

    first_batch = first.reserve_training_batch()
    assert first_batch is not None
    first_result = first.execute_reserved_step(0).result
    assert first_result is not None
    assert first.pending_batch.samples[0].source_agent_record_id == "i1"
    prepared = first.prepare_commit(first_result)
    first.commit(prepared)
    first.apply_compaction(prepared.compacted_ids)
    assert first.state == {"steps": 1}
    assert first.data_offset == prepared.high_water_offset == 2

    # Simulate restart: a fresh trainer is built with the algorithm_state
    # recovered from artifact metadata. the record store on disk retains only the
    # untrained prefix (compaction removed i1/r1).
    recovered_state = first.algorithm_state_dict()
    second = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ThresholdProcessor(ProcessorContext(context.scenario, {"batch_size": 1})),
        training_backend=_PreparingBackend(),
        algorithm_state=recovered_state,
    )

    assert second.state == {"steps": 1}
    second_batch = second.reserve_training_batch()
    assert second_batch is not None
    second_result = second.execute_reserved_step(1).result
    assert second_result is not None
    assert second.pending_batch.samples[0].source_agent_record_id == "i2"
    prepared = second.prepare_commit(second_result)
    second.commit(prepared)
    second.apply_compaction(prepared.compacted_ids)
    assert second.reserve_training_batch() is None


@pytest.mark.unit
def test_commit_compacts_consumed_payloads_physically(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    first_inference = positioned_inference(1)
    first_report = positioned_report(2, first_inference.agent_record_id, 1.0)
    second_inference = positioned_inference(3)
    second_report = positioned_report(4, second_inference.agent_record_id, 0.5)

    with RecordStore(database) as first_store:
        for item in (first_inference, first_report, second_inference, second_report):
            first_store.append(item)
        first = Trainer.build(
            "math",
            first_store,
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            training_backend=_PreparingBackend(),
        )
        batch = first.reserve_training_batch()
        assert batch is not None
        first_result = first.execute_reserved_step(0).result
        assert first_result is not None
        prepared = first.prepare_commit(first_result)
        first.commit(prepared)
        first.apply_compaction(prepared.compacted_ids)

        assert first_store.get("math", first_inference.agent_record_id) is None
        assert first_store.get("math", first_report.agent_record_id) is None
        assert [item.agent_record_id for item in first_store.replay("math")] == [
            second_inference.agent_record_id,
            second_report.agent_record_id,
        ]

    with RecordStore(database) as second_store:
        second = Trainer.build(
            "math",
            second_store,
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            training_backend=_PreparingBackend(),
            algorithm_state={"steps": 1},
        )
        assert second.state == {"steps": 1}
        batch = second.reserve_training_batch()
        assert batch is not None
        second_result = second.execute_reserved_step(1).result
        assert second_result is not None
        assert second.pending_batch.samples[0].source_agent_record_id == second_inference.agent_record_id
        prepared = second.prepare_commit(second_result)
        second.commit(prepared)
        second.apply_compaction(prepared.compacted_ids)
        assert second_store.count("math") == 0


@pytest.mark.integration
def test_scenario_runtime_executes_grpo_as_one_async_transaction(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_text("trained")

    class FakeTrainingRuntime(TrainingRuntime):
        def __init__(self):
            super().__init__(base_url="http://trainer")
            self.calls = []

        @property
        def inference_backend(self):
            return None

        def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
            prepared = prepare_slime_step(batch, step_preparer, algorithm_state)
            assert prepared.payload is not None
            self.calls.append(("prepare", batch, step_preparer))
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
        step_preparer = _GROUPED_PG_PREPARER
        loss_family = "pg"

        def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
            return Trainer.build(
                scenario,
                records,
                processor_factory=lambda context: GroupedPolicyProcessor(context.with_config({"batch_size": 1})),
                training_backend=SlimeTrainingBackend(self.runtime, self.step_preparer),
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
            )

    dispatcher = Dispatcher(
        GroupedPgRecipe(training_runtime, name="grouped_pg"),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
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
    assert prepare[2] == _GROUPED_PG_PREPARER
    assert prepare[1].comparison_sets[0][1].reward == 0.8
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
    processor = SAOProcessor(ProcessorContext("swe", {"batch_size": 1, "accept_multi_turn_policy_samples": True}))
    processor.ingest(training_inference("i1", [10, 20], [1], [-0.1]))
    processor.ingest(training_inference("i2", [10, 20, 11, 21], [1], [-0.2]))
    processor.ingest(report("r1", ("i1", "i2"), 1.0))

    assert processor.ready()
    batch = processor.build_batch()
    (sample,) = batch.samples
    assert sample.is_multi_turn
    assert sample.action_mask == sample.loss_mask
    assert any(sample.action_mask)
