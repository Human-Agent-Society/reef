"""Experiment events follow successful storage notifications, including retries."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from reef.core.artifact_ref import ArtifactRef
from reef.observability.base import (
    ExperimentTracker,
    TrainFromExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.observability.commits import CommitExperimentObserver, ExperimentReportQueue
from reef.storage.commits import CommitRecord
from reef.storage.observer import ObservedScenarioStore, RecordObserver
from reef.storage.sqlite import SQLiteScenarioStorage


class RecordingTracker(ExperimentTracker):
    def __init__(self, *, failing: bool = False) -> None:
        self.events: list[TrainingExperimentEvent] = []
        self.starts: list[TrainFromExperimentEvent] = []
        self.failing = failing
        self.close_count = 0

    def correlation_metrics(self, context: TrainingExperimentContext) -> Mapping[str, object]:
        if self.failing:
            raise ConnectionError("tracker unavailable")
        return {"experiment/run_id": f"run-{context.run_segment}"}

    def record(self, event: TrainingExperimentEvent) -> None:
        self.events.append(event)
        if self.failing:
            raise ConnectionError("tracker unavailable")

    def record_train_from(self, event: TrainFromExperimentEvent) -> None:
        self.starts.append(event)
        if self.failing:
            raise ConnectionError("tracker unavailable")

    def close(self) -> None:
        self.close_count += 1


class FailingTraceObserver(RecordObserver):
    def record_committed(self, commit: CommitRecord) -> None:
        raise ConnectionError("trace exporter unavailable")


def training_record(step: int = 1) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        artifact_ref=ArtifactRef(f"content-{step}", f"release-{step}", "seed"),
        checkpoint=True,
        algorithm_state={"steps": step},
        high_water_sequence=0,
        high_water_offset=0,
        component="weights",
        metrics={"score": 0.8},
        training_job_id=f"job-{step}",
    )


@pytest.mark.unit
@pytest.mark.parametrize("lost_acknowledgment", [False, True])
def test_only_successful_storage_notifications_report_and_retries_preserve_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_acknowledgment: bool
) -> None:
    storage = SQLiteScenarioStorage(tmp_path)
    inner = storage.open("math")
    tracker = RecordingTracker()
    seed = ArtifactRef("seed-content", "seed", None)
    observer = CommitExperimentObserver(
        tracker,
        scenario="math",
        recipe="recipe",
        source_artifact_ref=seed,
        scenario_step=0,
        store=ObservedScenarioStore(inner, FailingTraceObserver()),
    )
    store = observer.store
    context = TrainingExperimentContext(
        scenario="math",
        recipe="recipe",
        step=1,
        source_artifact_ref=seed,
        component="weights",
        backend="stub",
        backend_config={"objective": "policy"},
    )
    correlation = observer.prepare(context, checkpoint_path="/checkpoint/1")
    record = replace(training_record(), metrics={**correlation, "score": 0.8})
    commit_step = inner.commit_step

    def fail_commit(*, expected_step: int, commit: CommitRecord) -> CommitRecord:
        if lost_acknowledgment:
            commit_step(expected_step=expected_step, commit=commit)
        raise OSError("lost storage acknowledgment")

    try:
        monkeypatch.setattr(inner, "commit_step", fail_commit)
        with pytest.raises(OSError, match="lost storage acknowledgment"):
            store.commit_step(expected_step=0, commit=record)
        observer.reports.flush()
        assert tracker.events == []
        monkeypatch.setattr(inner, "commit_step", commit_step)
        landed = store.commit_step(expected_step=0, commit=record)
        store.commit_step(expected_step=0, commit=record)
        observer.reports.flush()
        (event,) = tracker.events
        assert event.context == context
        assert event.metrics == landed.metrics
        assert event.metrics["experiment/run_id"] == "run-0"
        assert event.produced_artifact_ref == landed.artifact_ref
        assert event.training_job_id == "job-1"
        assert event.checkpoint_path == "/checkpoint/1"
        assert observer.run_step == 1
    finally:
        store.close()
        storage.close()
    assert tracker.close_count == 0  # The process owns the tracker, not individual scenario sessions.


@pytest.mark.unit
@pytest.mark.parametrize("failing", [False, True])
def test_train_from_retries_rotate_once_and_tracker_errors_do_not_fail_storage(tmp_path: Path, failing: bool) -> None:
    tracker = RecordingTracker(failing=failing)
    storage = SQLiteScenarioStorage(tmp_path)
    seed = ArtifactRef("seed-content", "seed", None)
    observer = CommitExperimentObserver(
        tracker,
        scenario="math",
        recipe="recipe",
        source_artifact_ref=seed,
        scenario_step=0,
        store=storage.open("math"),
    )
    store = observer.store
    context = TrainingExperimentContext(scenario="math", recipe="recipe", step=1, source_artifact_ref=seed)
    correlation = observer.prepare(context, checkpoint_path=None)
    assert correlation == ({} if failing else {"experiment/run_id": "run-0"})
    first = training_record()

    def operation_record(step: int, operation: str, artifact_ref: ArtifactRef) -> CommitRecord:
        return CommitRecord(
            scenario="math",
            step=step,
            artifact_ref=artifact_ref,
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
            operation=operation,
        )

    start = operation_record(2, "train_from", seed)
    try:
        store.commit_step(expected_step=0, commit=first)
        store.commit_step(expected_step=1, commit=start)
        store.commit_step(expected_step=1, commit=start)
        # An activation of another artifact moves what serves and leaves the training run and its start alone.
        store.commit_step(expected_step=2, commit=operation_record(3, "activate", first.artifact_ref))
        store.commit_step(expected_step=3, commit=training_record(4))
        # Retrying an earlier step after a later one must not move tracking backwards.
        store.commit_step(expected_step=0, commit=first)
        observer.reports.flush()
        assert len(store.history()) == 4
        assert [(event.context.run_segment, event.context.run_step) for event in tracker.events] == [(0, 0), (2, 0)]
        (event,) = tracker.starts
        assert event.run_segment == 0 and event.step == 2
        assert event.start_artifact_ref == seed
        # The step after the new start names the training start as its source, not the active release.
        assert tracker.events[-1].context.source_artifact_ref == seed
    finally:
        store.close()
        storage.close()


@pytest.mark.unit
def test_an_inactive_artifact_moves_the_training_start_and_a_rejected_step_keeps_it(tmp_path: Path) -> None:
    """Training goes on from its own output, active or not; a step that made nothing leaves the start alone."""
    storage = SQLiteScenarioStorage(tmp_path)
    tracker = RecordingTracker()
    seed = ArtifactRef("seed-content", "seed", None)
    observer = CommitExperimentObserver(
        tracker,
        scenario="math",
        recipe="recipe",
        source_artifact_ref=seed,
        scenario_step=0,
        store=storage.open("math"),
    )
    store = observer.store
    # The activation rule did not activate the artifact, but the next step trains from it.
    held = training_record()
    rejected = replace(training_record(2), checkpoint=False, artifact_ref=seed, metrics={"selected": False})
    try:
        store.commit_step(expected_step=0, commit=held)
        store.commit_step(expected_step=1, commit=rejected)
        observer.reports.flush()
        assert [event.produced_artifact_ref for event in tracker.events] == [held.artifact_ref, held.artifact_ref]
        assert tracker.events[-1].outcome == "rejected"
        assert tracker.events[-1].context.source_artifact_ref == held.artifact_ref
    finally:
        store.close()
    inner = storage.open("math")
    resumed = CommitExperimentObserver(
        tracker,
        scenario="math",
        recipe="recipe",
        source_artifact_ref=seed,
        scenario_step=2,
        store=inner,
    )
    store = resumed.store
    try:
        store.commit_step(expected_step=1, commit=rejected)
        resumed.reports.flush()
        assert len(tracker.events) == 2
        store.commit_step(expected_step=2, commit=training_record(3))
        resumed.reports.flush()
        assert tracker.events[-1].context.run_step == 2
    finally:
        store.close()
        storage.close()


def training_event(step: int) -> TrainingExperimentEvent:
    ref = training_record(step).artifact_ref
    return TrainingExperimentEvent(
        context=TrainingExperimentContext(scenario="math", recipe="recipe", step=step, source_artifact_ref=ref),
        produced_artifact_ref=ref,
        metrics={},
        outcome="committed",
    )


@pytest.mark.unit
def test_the_report_queue_sends_in_order_on_its_own_thread_and_goes_on_after_a_failure(monkeypatch):
    from threading import current_thread

    tracker = RecordingTracker()
    threads: list[str] = []

    def record(event):
        threads.append(current_thread().name)
        tracker.events.append(event)
        if event.context.step == 1:
            raise ConnectionError("tracker unavailable")

    monkeypatch.setattr(tracker, "record", record)
    reports = ExperimentReportQueue(tracker)
    try:
        reports.put(training_event(1))
        reports.put(training_event(2))
        reports.flush()
        assert [event.context.step for event in tracker.events] == [1, 2]
        assert threads == ["reef-experiment-reports"] * 2
    finally:
        reports.close()


@pytest.mark.unit
def test_close_sends_the_queued_events_and_a_later_event_goes_at_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    tracker = RecordingTracker()
    entered, release = Event(), Event()

    def record(event):
        if event.context.step == 1:
            entered.set()
            assert release.wait(10)
        tracker.events.append(event)

    monkeypatch.setattr(tracker, "record", record)
    reports = ExperimentReportQueue(tracker)
    reports.put(training_event(1))
    reports.put(training_event(2))
    assert entered.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(reports.close)
        try:
            assert not closing.done()
        finally:
            release.set()
        closing.result(timeout=5)
    assert [event.context.step for event in tracker.events] == [1, 2]
    reports.put(training_event(3))
    assert [event.context.step for event in tracker.events] == [1, 2, 3]
