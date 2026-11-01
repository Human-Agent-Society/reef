"""Harness evolution's job data: trace batching, and offline runs of the imported train tasks.

The trace pipeline pairs reports with their inference records and batches
every valid scored trace; the evolution backend (``HarnessBackend``)
consumes the batch. The stages are built from a ``pipeline`` section, as a
deployment configures them, rather than through any Recipe wrapper.
``TrainTaskJobData`` also takes the scenario's imported train tasks from the
sample store into the buffer, for offline steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.task_service_stubs import task_for

from reef.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.job_data import TaskSample
from reef.core.reports import ReportValidationError
from reef.core.tasks import write_harbor_task
from reef.core.trajectories import recorded_payload, recorded_payloads, source_record_id
from reef.pipeline_context import PipelineContext
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2sample.traces import InferenceSamples, TraceSamples
from reef.sample2job.traces import TraceJobData, TrainTaskJobData
from reef.storage.commits import CommitRecord
from reef.storage.samples import SampleStore, TrainerSamples, consumed_sample_id
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.pipeline import PipelineConfig, TrainingPipeline
from reef.train.types import JobData


def _inference(agent_record_id: str, payload: dict) -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
    )


def _report(agent_record_id: str, score: object, references: list[str], feedback: object = None) -> AgentRecord:
    payload: dict = {"score": score, "references": references}
    if feedback is not None:
        payload["feedback"] = feedback
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload=payload,
        agent_record_id=agent_record_id,
    )


#: The ``pipeline`` section that evolves on recorded inference traffic alone, without reports.
INFERENCE_STAGES = {
    "record2sample": {"module": "reef.record2sample.traces:InferenceSamples"},
}


def harness_stages(
    *, inference: bool = False, buffer_size: int | None = None, job_samples: int | None = None
) -> PipelineConfig:
    """The harness evolution stages as a deployment's ``pipeline`` section configures them."""
    section: dict = dict(INFERENCE_STAGES) if inference else {}
    sizes = {"buffer_size": buffer_size, "job_samples": job_samples}
    stage_config = {key: value for key, value in sizes.items() if value is not None}
    if stage_config:
        section["sample2job"] = {"config": stage_config}
    return PipelineConfig.from_setting(section, spec=HarnessEvolutionRecipe.pipeline_spec(), environ={})


def reports_pipeline(
    context: PipelineContext, buffer_size: int | None = None, job_samples: int | None = None
) -> TrainingPipeline:
    """Harness traces from reported scores: the default stages, built directly."""
    return TrainingPipeline(
        context,
        record2sample=TraceSamples(context),
        sample2job=TraceJobData(context, buffer_size, job_samples),
    )


def records_pipeline(
    context: PipelineContext, buffer_size: int | None = None, job_samples: int | None = None
) -> TrainingPipeline:
    """Harness traces from recorded inference traffic, unscored: the inference stages, built directly."""
    return TrainingPipeline(
        context,
        record2sample=InferenceSamples(),
        sample2job=TraceJobData(context, buffer_size, job_samples),
    )


def _processor() -> TrainingPipeline:
    return harness_stages().build(PipelineContext("s"))


def test_the_default_stages_train_reports_and_the_inference_stages_train_records() -> None:
    reported = harness_stages(buffer_size=6, job_samples=2).build(PipelineContext("s"))
    assert (type(reported.record2sample), type(reported.sample2job)) == (TraceSamples, TraceJobData)
    assert reported.buffer_size == 6
    recorded = harness_stages(inference=True, job_samples=2).build(PipelineContext("s"))
    assert (type(recorded.record2sample), type(recorded.sample2job)) == (InferenceSamples, TraceJobData)
    assert recorded.buffer_size == 2, "the buffer takes one job's data"


@pytest.mark.parametrize("stage", [TraceJobData, TrainTaskJobData])
def test_a_trace_job_takes_one_sample_unless_job_samples_is_set(stage: type[TraceJobData]) -> None:
    defaulted = stage(PipelineContext("s"), None, None)
    assert (defaulted.job_samples, defaulted.buffer_size) == (1, 1), "each trace starts a step"
    configured = stage(PipelineContext("s"), 6, 3)
    assert (configured.job_samples, configured.buffer_size) == (3, 6)


def test_trace_processor_batches_a_failed_trace() -> None:
    processor = _processor()
    payload = {"messages": [{"role": "system", "content": "skill"}, {"role": "user", "content": "q"}]}
    processor.push(_inference("inf-1", payload))
    processor.push(_report("rep-1", 0.0, ["inf-1"]))
    assert processor.ready(1)
    batch = processor.build_job_data(1)
    assert isinstance(batch, JobData)
    assert len(batch.samples) == 1
    sample = batch.samples[0]
    assert source_record_id(sample) == "inf-1"
    assert sample.metadata.get("reward") == 0.0
    assert recorded_payload(sample)["messages"][0]["content"] == "skill"
    processor.acknowledge(batch.data_id)
    retention = processor.releasable_record_ids()
    assert "rep-1" in retention


def test_trace_processor_batches_successful_reports() -> None:
    processor = _processor()
    processor.push(_inference("inf-1", {"messages": []}))
    processor.push(_report("rep-1", 1.0, ["inf-1"]))
    assert processor.build_job_data(1).samples[0].metadata.get("reward") == 1.0


def test_trace_processor_carries_the_task_a_report_names() -> None:
    processor = _processor()
    processor.push(_inference("inf-1", {"messages": []}))
    report = _report("rep-1", 1.0, ["inf-1"])
    task = {"name": "hello-file", "path": "/tasks/hello-file", "digest": "ab" * 32}
    named = AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload={**report.payload, "metadata": {"task": task}},
        agent_record_id="rep-1",
    )
    processor.push(named)
    sample = processor.build_job_data(1).samples[0]
    assert sample.metadata["task"] == task and sample.metadata["reward"] == 1.0


def test_trace_processor_batches_a_multi_reference_report_as_one_trajectory() -> None:
    """A report over a whole run becomes one sample: the trajectory holds
    every referenced payload in reference order, the payload is the last
    exchange, and the feedback rides along verbatim."""
    processor = _processor()
    first = {"messages": [{"role": "user", "content": "first"}]}
    second = {"messages": [{"role": "user", "content": "second"}]}
    processor.push(_inference("inf-1", first))
    processor.push(_inference("inf-2", second))
    processor.push(_report("rep-1", 0.0, ["inf-1", "inf-2"], feedback="wrong file"))

    assert processor.ready(1)
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert source_record_id(sample) == "inf-2"
    assert recorded_payload(sample) == second
    assert recorded_payloads(sample) == (first, second)
    assert sample.metadata.get("feedback") == "wrong file"
    assert sample.metadata.get("reward") == 0.0


def test_trace_processor_keeps_single_reference_trajectory_and_carries_feedback() -> None:
    processor = _processor()
    payload = {"messages": [{"role": "user", "content": "q"}]}
    processor.push(_inference("inf-1", payload))
    processor.push(_report("rep-1", 0.0, ["inf-1"], feedback={"reason": "timeout"}))

    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert recorded_payload(sample) == payload
    assert recorded_payloads(sample) == (payload,)
    assert sample.metadata.get("feedback") == {"reason": "timeout"}


def test_trace_processor_rejects_a_report_without_references() -> None:
    processor = _processor()
    with pytest.raises(ReportValidationError, match="references"):
        processor.push(_report("rep-1", 0.0, []))
    assert not processor.ready(1)


def test_trace_processor_has_no_judge_hook() -> None:
    for stage in (TrainingPipeline, TraceSamples, TraceJobData):
        assert not hasattr(stage, "judge")


def test_trace_processor_has_no_training_preparation_hook() -> None:
    for stage in (TrainingPipeline, TraceSamples, TraceJobData):
        assert not hasattr(stage, "prepare_training_step")
        assert not hasattr(stage, "prepare")


def test_trace_processor_refuses_a_non_finite_score() -> None:
    for bad in (float("inf"), float("-inf"), float("nan")):
        processor = _processor()
        processor.push(_inference("inf-1", {"messages": []}))
        with pytest.raises(ReportValidationError, match="finite"):
            processor.push(_report("rep-1", bad, ["inf-1"]))
        assert not processor.ready(1)


def _record_processor() -> TrainingPipeline:
    return harness_stages(inference=True).build(PipelineContext("s"))


def test_record_driven_processor_batches_every_n_inferences_unscored() -> None:
    """The report-free policy: recorded traffic alone forms the batch, in
    arrival order, with score None on every sample."""
    processor = harness_stages(inference=True, job_samples=2).build(PipelineContext("s"))
    first = {"messages": [{"role": "user", "content": "first"}]}
    second = {"messages": [{"role": "user", "content": "second"}]}
    processor.push(_inference("inf-1", first))
    assert not processor.ready(2)
    processor.push(_inference("inf-2", second))
    assert processor.ready(2)

    batch = processor.build_job_data(2)
    assert isinstance(batch, JobData)
    assert [source_record_id(s) for s in batch.samples] == ["inf-1", "inf-2"]
    assert [recorded_payload(s) for s in batch.samples] == [first, second]
    assert {s.metadata.get("reward") for s in batch.samples} == {None}

    # The stored samples consume their records; the job data consumes the samples.
    retention = processor.releasable_record_ids()
    assert retention == frozenset({"inf-1", "inf-2"})

    consumed = processor.acknowledge(batch.data_id)
    assert consumed == frozenset({"sample:inf-1:0", "sample:inf-2:0"})
    assert not processor.ready(2)


def test_record_driven_processor_releases_reports_untouched() -> None:
    processor = _record_processor()
    processor.push(_report("rep-1", 0.0, ["inf-0"], feedback="ignored"))
    assert not processor.ready(1)
    retention = processor.releasable_record_ids()
    assert retention == frozenset({"rep-1"})

    processor.push(_inference("inf-1", {"messages": []}))
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert sample.metadata.get("reward") is None
    assert sample.metadata.get("feedback") is None


def test_record_driven_processor_overflow_stays_pending_for_the_next_batch() -> None:
    processor = harness_stages(inference=True, job_samples=2).build(PipelineContext("s"))
    for index in range(3):
        processor.push(_inference(f"inf-{index}", {"messages": []}))
    batch = processor.build_job_data(2)
    assert [source_record_id(s) for s in batch.samples] == ["inf-0", "inf-1"]
    assert processor.acknowledge(batch.data_id) == {"sample:inf-0:0", "sample:inf-1:0"}
    # The overflow sample waits for the next job data.
    assert processor.status()["ready_samples"] == 1
    assert not processor.ready(2)
    processor.push(_inference("inf-3", {"messages": []}))
    assert [source_record_id(s) for s in processor.build_job_data(2).samples] == ["inf-2", "inf-3"]


def add_tasks(samples: SampleStore, root: Path, split: str, *names: str) -> None:
    """Add one Harbor task per name to ``split`` of the scenario's imported tasks, each with its own files."""
    directories = tuple(
        write_harbor_task(task_for(name, "record-1", 9000 + sum(name.encode())), root / split) for name in names
    )
    samples.add_tasks(directories, split)


def offline_pipeline(samples: SampleStore, *, job_samples: int, buffer_size: int | None = None) -> TrainingPipeline:
    """An offline pipeline of the scenario whose sample store is ``samples``, in ``auto``."""
    context = PipelineContext("offline", samples=TrainerSamples(samples))
    return TrainingPipeline(context, sample2job=TrainTaskJobData(context, buffer_size, job_samples))


def task_names(data: JobData) -> list[str]:
    """The names of the imported tasks in ``data``, in job order."""
    return [sample.metadata["task"]["name"] for sample in data.samples if isinstance(sample, TaskSample)]


def test_imported_train_tasks_reach_the_job_data_through_the_buffer_and_each_trains_once(tmp_path: Path) -> None:
    session = SQLiteScenarioStorage(tmp_path / "records").open("offline")
    add_tasks(session.samples, tmp_path / "tasks", "train", "task-a", "task-b", "task-c")
    add_tasks(session.samples, tmp_path / "tasks", "eval", "held-out")
    pipeline = offline_pipeline(session.samples, job_samples=2)
    assert pipeline.accepts_imported_tasks and not TraceJobData.accepts_imported_tasks

    data = pipeline.build_job_data(2)
    assert task_names(data) == ["task-a", "task-b"]
    assert all(sample.task_path.is_dir() for sample in data.samples if isinstance(sample, TaskSample))
    assert pipeline.acknowledge(data.data_id) == {consumed_sample_id("task-a"), consumed_sample_id("task-b")}
    # One train task is left; an eval task never trains, since an evaluation runs it.
    assert not pipeline.ready(2)

    # A later upload joins the buffer; a task uploaded again with the same files does not train again.
    add_tasks(session.samples, tmp_path / "tasks", "train", "task-a", "task-d")
    data = pipeline.build_job_data(2)
    assert task_names(data) == ["task-c", "task-d"]
    pipeline.acknowledge(data.data_id)
    assert not pipeline.ready(1)
    pipeline.close()
    session.close()


def test_one_job_holds_traces_or_train_tasks_never_both(tmp_path: Path) -> None:
    session = SQLiteScenarioStorage(tmp_path / "records").open("s")
    context = PipelineContext("s", samples=TrainerSamples(session.samples))
    pipeline = TrainingPipeline(
        context,
        record2sample=TraceSamples(context),
        sample2job=TrainTaskJobData(context, 4, 1),
    )
    pipeline.push(_inference("inf-1", {"messages": []}))
    pipeline.push(_report("rep-1", 0.0, ["inf-1"]))
    add_tasks(session.samples, tmp_path / "tasks", "train", "task-a", "task-b")

    # The buffer takes up to four samples, but the trace came first, so its job holds the trace alone.
    traces = pipeline.build_job_data(1)
    assert [source_record_id(sample) for sample in traces.samples] == ["inf-1"]
    pipeline.acknowledge(traces.data_id)
    tasks = pipeline.build_job_data(1)
    assert task_names(tasks) == ["task-a", "task-b"]
    pipeline.close()
    session.close()


def test_a_restart_reads_the_store_back_and_trains_the_tasks_that_no_durable_commit_trained(tmp_path: Path) -> None:
    session = SQLiteScenarioStorage(tmp_path / "records").open("offline")
    add_tasks(session.samples, tmp_path / "tasks", "train", "task-a", "task-b", "task-c")
    first = offline_pipeline(session.samples, job_samples=1)
    durable = first.build_job_data(1)
    session.commit_step(
        expected_step=0,
        commit=CommitRecord(
            scenario="offline",
            step=1,
            artifact_ref=ArtifactRef("content:1", "release:1", "base"),
            checkpoint=True,
            algorithm_state={},
            high_water_sequence=0,
            high_water_offset=0,
            consumed_ids=first.acknowledge(durable.data_id),
        ),
    )
    # The second step's commit never becomes durable.
    lost = first.build_job_data(1)
    first.acknowledge(lost.data_id)
    first.close()

    (trained,) = session.samples.get(("task-a",))
    assert task_names(durable) == ["task-a"] and trained.train_count("") == 1
    restarted = offline_pipeline(session.samples, job_samples=1)
    rest = []
    while restarted.ready(1):
        data = restarted.build_job_data(1)
        rest.append(task_names(data))
        restarted.acknowledge(data.data_id)
    # The step whose commit never became durable trains its task again.
    assert rest == [task_names(lost), ["task-c"]] == [["task-b"], ["task-c"]]
    restarted.close()
    session.close()
