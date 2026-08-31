from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from reef.core import AgentRecord, RequestType
from reef.records import RecordStore
from reef.train import (
    DataProcessor,
    PreparedStep,
    ProcessorContext,
    RetentionDecision,
    Trainer,
    TrainingBackend,
    TrainingBatch,
    TrainStepResult,
)
from reef.train.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate


@dataclass(frozen=True)
class ExampleBatch(TrainingBatch):
    values: tuple[str, ...]


class ExampleProcessor(DataProcessor):
    required_request_types = frozenset({RequestType.REPORT})

    @property
    def output_schema(self) -> type[ExampleBatch]:
        return ExampleBatch

    def __init__(self, context: ProcessorContext, events: list[str]) -> None:
        super().__init__(context)
        self.events = events
        self.items: list[AgentRecord] = []
        self.pending: ExampleBatch | None = None

    def ingest(self, item: AgentRecord) -> None:
        self.events.append(f"ingest:{item.agent_record_id}")
        self.items.append(item)

    def ready(self) -> bool:
        return bool(self.items)

    def build_batch(self) -> ExampleBatch:
        self.events.append("build")
        self.pending = ExampleBatch(batch_id="batch-1", values=(self.items[0].agent_record_id,))
        return self.pending

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        self.events.append(f"ack:{batch_id}")
        consumed = frozenset(item.agent_record_id for item in self.items)
        self.items.clear()
        self.pending = None
        return consumed


class ExampleBackend(TrainingBackend):
    def __init__(self, scenario: str, events: list[str]) -> None:
        self.scenario = scenario
        self.events = events

    def initial_state(self) -> Mapping[str, Any]:
        self.events.append(f"initialize:{self.scenario}")
        return {"version": 0}

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        del scenario_step
        assert isinstance(batch, ExampleBatch)
        self.events.append(f"prepare:{batch.batch_id}")
        return PreparedStep.with_candidate(
            UpdateCandidate(batch.batch_id),
            state={"version": int(state["version"]) + 1},
        )

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return EvaluationResult("example", "1", {})

    def settle_step(
        self,
        prepared: PreparedStep,
        decision: SelectionDecision,
    ) -> TrainStepResult:
        return TrainStepResult(
            state=prepared.state,
            metrics={"trained": 1, "selection": decision.to_dict()},
        )

    def abort_step(self, prepared: PreparedStep) -> None:
        del prepared


@pytest.mark.unit
def test_data_processor_is_the_no_update_default() -> None:
    processor = DataProcessor(ProcessorContext(scenario="math"))
    assert processor.ready() is False
    record = AgentRecord.create(scenario="math", request_type=RequestType.REPORT, payload={}, agent_record_id="r1")
    processor.ingest(record)
    assert processor.retention_decision() == RetentionDecision(protected_agent_record_ids=frozenset({"r1"}))
    # The default holds no batch-ready units, so it never becomes ready and
    # build_batch refuses on that ground — the same refusal every engine gives.
    with pytest.raises(RuntimeError, match="not ready"):
        processor.build_batch()
    with pytest.raises(ValueError, match="unknown batch_id"):
        processor.acknowledge("math:whatever:1")


@pytest.mark.unit
def test_training_backend_cannot_be_instantiated_without_contract_methods() -> None:
    with pytest.raises(TypeError):
        TrainingBackend()


@pytest.mark.unit
def test_prepared_step_enforces_its_single_outcome() -> None:
    candidate = UpdateCandidate("candidate-1")

    with pytest.raises(ValueError, match="requires a candidate"):
        PreparedStep("candidate", {})
    with pytest.raises(ValueError, match="cannot carry a candidate"):
        PreparedStep("skip", {}, candidate=candidate)
    with pytest.raises(ValueError, match="outcome"):
        PreparedStep("unknown", {})  # type: ignore[arg-type]


@pytest.mark.unit
def test_custom_processors_default_to_releasing_no_records() -> None:
    processor = ExampleProcessor(ProcessorContext("math"), [])

    assert processor.retention_decision() == RetentionDecision()


@pytest.mark.unit
def test_trainer_dispatches_only_required_data_types() -> None:
    events: list[str] = []
    records = RecordStore()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ExampleProcessor(context, events),
        training_backend=ExampleBackend("math", events),
    )

    assert isinstance(trainer, Trainer)

    records.append(
        AgentRecord.create(
            scenario="math", request_type=RequestType.INFERENCE, payload={}, agent_record_id="inference"
        )
    )
    records.append(
        AgentRecord.create(
            scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
        )
    )
    trainer.run_once()

    assert "ingest:report" in events
    assert "ingest:inference" not in events
    assert trainer.data_offset == 2


@pytest.mark.unit
def test_trainer_waits_for_commit_before_acknowledging_batch() -> None:
    events: list[str] = []
    records = RecordStore()
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ExampleProcessor(context, events),
        training_backend=ExampleBackend("math", events),
    )
    records.append(
        AgentRecord.create(
            scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
        )
    )

    result = trainer.run_once()

    assert result is not None
    assert result.metrics["selection"]["policy"] == "always"
    assert events == [
        "initialize:math",
        "ingest:report",
        "build",
        "prepare:batch-1",
    ]
    assert result.state is not None
    events.append(f"commit:math:{result.state['version']}")
    prepared = trainer.commit()
    trainer.apply_compaction(prepared.compacted_ids)
    assert events == [
        "initialize:math",
        "ingest:report",
        "build",
        "prepare:batch-1",
        "commit:math:1",
        "ack:batch-1",
    ]


@pytest.mark.unit
def test_trainer_finishes_a_skipped_preparation_without_selection() -> None:
    class SkippingBackend(ExampleBackend):
        def prepare_step(
            self,
            batch: TrainingBatch,
            state: Mapping[str, Any],
            scenario_step: int,
        ) -> PreparedStep:
            del batch, scenario_step
            return PreparedStep.skipped(state=state, metrics={"skipped": "no update"})

        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            raise AssertionError("a skipped preparation must not be evaluated")

    records = RecordStore()
    records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1},
            agent_record_id="report",
        )
    )
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ExampleProcessor(context, []),
        training_backend=SkippingBackend("math", []),
    )

    result = trainer.run_once()

    assert result is not None
    assert result.metrics == {"skipped": "no update"}
    assert result.state == {"version": 0}


@pytest.mark.unit
def test_trainer_keeps_pending_result_when_external_commit_fails() -> None:
    events: list[str] = []
    records = RecordStore()

    def fail_commit(result: TrainStepResult) -> None:
        assert result.state is not None
        events.append(f"commit:{result.state['version']}")
        raise RuntimeError("storage failed")

    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ExampleProcessor(context, events),
        training_backend=ExampleBackend("math", events),
    )
    records.append(
        AgentRecord.create(
            scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
        )
    )

    result = trainer.run_once()
    assert result is not None
    with pytest.raises(RuntimeError, match="storage failed"):
        fail_commit(result)

    assert "ack:batch-1" not in events
    assert trainer.state == {"version": 0}
    assert trainer.run_once() is result


@pytest.mark.unit
def test_trainer_reads_only_new_records() -> None:
    events: list[str] = []
    records = RecordStore()
    report = AgentRecord.create(
        scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
    )
    records.append(report)
    trainer = Trainer.build(
        "math",
        records,
        processor_factory=lambda context: ExampleProcessor(context, events),
        training_backend=ExampleBackend("math", events),
    )

    trainer.run_once()
    prepared = trainer.commit()
    trainer.apply_compaction(prepared.compacted_ids)
    records.append(report)
    trainer.run_once()

    assert events.count("ingest:report") == 1
    assert trainer.data_offset == 1
