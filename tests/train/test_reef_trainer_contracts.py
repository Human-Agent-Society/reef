from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from tests.support._threshold_processor import ThresholdJobData
from tests.support.runtime_stubs import STUB_ROOT, StubTrainingComponent
from tests.support.samples_service import make_samples

from reef.core import AgentRecord, RequestType
from reef.evaluation import UpdateCandidate
from reef.runtime.jobs import SavedCheckpoint, TrainingComponent, TrainingJob
from reef.storage.sqlite import SQLiteRecordStore
from reef.train import JobData, PipelineContext, PreparedStep, Trainer, TrainingPipeline, TrainStepResult
from reef.train.jobs import JobLoop, JobLoopState


@dataclass(frozen=True)
class ExampleJobData(JobData):
    values: tuple[str, ...] = field(kw_only=True)


class ExamplePipeline(TrainingPipeline):
    @property
    def required_request_types(self) -> frozenset[RequestType]:
        return frozenset({RequestType.REPORT})

    @property
    def output_schema(self) -> type[ExampleJobData]:
        return ExampleJobData

    def __init__(self, context: PipelineContext, events: list[str]) -> None:
        # The stage gives the job size only: one report is one job.
        super().__init__(context, sample2job=ThresholdJobData(context, None, 1))
        self.events = events
        self.items: list[AgentRecord] = []
        self.pending: ExampleJobData | None = None

    def push(self, item: AgentRecord) -> None:
        self.events.append(f"push:{item.agent_record_id}")
        self.items.append(item)

    def ready(self, sample_count: int) -> bool:
        return len(self.items) >= sample_count

    def build_job_data(self, sample_count: int) -> ExampleJobData:
        self.events.append("build")
        self.pending = ExampleJobData(data_id="batch-1", values=(self.items[0].agent_record_id,))
        return self.pending

    def acknowledge(self, data_id: str) -> frozenset[str]:
        self.events.append(f"ack:{data_id}")
        consumed = frozenset(item.agent_record_id for item in self.items)
        self.items.clear()
        self.pending = None
        return consumed


class ExampleComponent(StubTrainingComponent):
    """A training component that logs each job it takes, as ``submit:<data_id>``, and ends it without a checkpoint."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(name="example")
        self.events = events

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        assert isinstance(job.data, ExampleJobData)
        self.events.append(f"submit:{job.data.data_id}")
        return ()


def example_jobs(events: list[str]) -> JobLoop:
    return JobLoop(ExampleComponent(events), scenario="math")


@pytest.mark.unit
def test_training_pipeline_is_the_no_update_default() -> None:
    processor = TrainingPipeline(PipelineContext(scenario="math"))
    assert processor.ready(1) is False
    record = AgentRecord.create(scenario="math", request_type=RequestType.REPORT, payload={}, agent_record_id="r1")
    processor.push(record)
    assert processor.releasable_record_ids() == frozenset()
    # The default holds no units, so it never becomes ready and
    # build_job_data refuses on that ground — the same refusal every pipeline gives.
    with pytest.raises(RuntimeError, match="no job data ready"):
        processor.build_job_data(1)
    with pytest.raises(ValueError, match="unknown data_id"):
        processor.acknowledge("math:whatever:1")


@pytest.mark.unit
def test_training_component_cannot_be_instantiated_without_contract_methods() -> None:
    with pytest.raises(TypeError):
        TrainingComponent()  # type: ignore[abstract]


@pytest.mark.unit
def test_prepared_step_enforces_its_single_outcome() -> None:
    candidate = UpdateCandidate("candidate-1")

    with pytest.raises(ValueError, match="only a candidate step, carries a candidate"):
        PreparedStep("candidate", {})
    with pytest.raises(ValueError, match="only a candidate step, carries a candidate"):
        PreparedStep("skip", {}, candidate=candidate)
    with pytest.raises(ValueError, match="outcome"):
        PreparedStep("unknown", {})  # type: ignore[arg-type]


@pytest.mark.unit
def test_custom_pipelines_default_to_releasing_no_records() -> None:
    processor = ExamplePipeline(PipelineContext("math"), [])

    assert processor.releasable_record_ids() == frozenset()


@pytest.mark.unit
def test_trainer_dispatches_only_required_data_types() -> None:
    events: list[str] = []
    records = SQLiteRecordStore()
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=lambda context: ExamplePipeline(context, events),
        job_loop=example_jobs(events),
        root=STUB_ROOT,
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
    make_samples(trainer)
    trainer.run_once()

    assert "push:report" in events
    assert "push:inference" not in events
    assert trainer.data_offset == 2


@pytest.mark.unit
def test_trainer_waits_for_commit_before_acknowledging_batch() -> None:
    events: list[str] = []
    records = SQLiteRecordStore()
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=lambda context: ExamplePipeline(context, events),
        job_loop=example_jobs(events),
        root=STUB_ROOT,
    )
    records.append(
        AgentRecord.create(
            scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
        )
    )
    make_samples(trainer)

    result = trainer.run_once()

    assert result is not None
    assert result.metrics["training_job"]["event"] == "submitted"
    assert events == ["push:report", "build", "submit:batch-1"]
    events.append("commit:math")
    prepared = trainer.prepare_commit(result)
    trainer.commit(prepared)
    assert events == ["push:report", "build", "submit:batch-1", "commit:math", "ack:batch-1"]


@pytest.mark.unit
def test_a_job_without_a_checkpoint_ends_without_an_artifact() -> None:
    records = SQLiteRecordStore()
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
        pipeline_factory=lambda context: ExamplePipeline(context, []),
        job_loop=example_jobs([]),
        root=STUB_ROOT,
    )
    make_samples(trainer)
    submitted = trainer.run_once()
    assert submitted is not None
    trainer.commit(trainer.prepare_commit(submitted))

    ended = trainer.run_once()

    assert ended is not None and ended.artifact is None
    assert ended.metrics["training_job"]["event"] == "finished"
    trainer.commit(trainer.prepare_commit(ended))
    assert JobLoopState.from_dict(trainer.state).finished_jobs == 1


@pytest.mark.unit
def test_trainer_keeps_pending_result_when_external_commit_fails() -> None:
    events: list[str] = []
    records = SQLiteRecordStore()

    def fail_commit(result: TrainStepResult) -> None:
        events.append("commit")
        raise RuntimeError("storage failed")

    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=lambda context: ExamplePipeline(context, events),
        job_loop=example_jobs(events),
        root=STUB_ROOT,
    )
    records.append(
        AgentRecord.create(
            scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
        )
    )
    make_samples(trainer)

    result = trainer.run_once()
    assert result is not None
    with pytest.raises(RuntimeError, match="storage failed"):
        fail_commit(result)

    assert "ack:batch-1" not in events
    assert trainer.state == JobLoopState("example:base").to_dict()
    assert trainer.run_once() is result


@pytest.mark.unit
def test_trainer_reads_only_new_records() -> None:
    events: list[str] = []
    records = SQLiteRecordStore()
    report = AgentRecord.create(
        scenario="math", request_type=RequestType.REPORT, payload={"score": 1}, agent_record_id="report"
    )
    records.append(report)
    trainer = Trainer.build(
        "math",
        records,
        pipeline_factory=lambda context: ExamplePipeline(context, events),
        job_loop=example_jobs(events),
        root=STUB_ROOT,
    )

    make_samples(trainer)
    first = trainer.run_once()
    assert first is not None
    prepared = trainer.prepare_commit(first)
    trainer.commit(prepared)
    records.append(report)
    make_samples(trainer)
    trainer.run_once()

    assert events.count("push:report") == 1
    assert trainer.data_offset == 1
