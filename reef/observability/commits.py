"""Translate landed scenario commits into experiment events, and deliver them.

Preparation preserves transient runtime metadata and adds correlation metrics
before persistence. Storage notifications capture the final recorded metrics
and put an event on :class:`ExperimentReportQueue`. One background thread sends
the events to the tracker in the order the commits landed, so no tracker call
runs under a serving lock. Delivery is best effort: a failed call is logged,
and a process restart does not send events again.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock, Thread

from reef.core.artifact_selection import ArtifactSelection
from reef.observability.base import (
    ExperimentTracker,
    TrainFromExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.storage.commits import CommitRecord
from reef.storage.observer import ObservedScenarioStore, RecordObserver
from reef.storage.scenario import ScenarioStore

logger = logging.getLogger(__name__)

ExperimentEvent = TrainingExperimentEvent | TrainFromExperimentEvent


@dataclass(frozen=True)
class PreparedExperiment:
    context: TrainingExperimentContext
    checkpoint_path: str | None


class ExperimentReportQueue:
    """Send experiment events to one tracker on a background thread, in the order they arrive.

    The thread starts with the first event. After :meth:`close`, an event is
    sent at once on the caller's thread, because no serving lock is held then.
    """

    def __init__(self, tracker: ExperimentTracker) -> None:
        self.tracker = tracker
        self.events: queue.Queue[ExperimentEvent | None] = queue.Queue()
        self.lock = Lock()
        self.thread: Thread | None = None
        self.closed = False

    def put(self, event: ExperimentEvent) -> None:
        with self.lock:
            if not self.closed:
                if self.thread is None:
                    self.thread = Thread(target=self.run, name="reef-experiment-reports", daemon=True)
                    self.thread.start()
                self.events.put(event)
                return
        self.send(event)

    def flush(self) -> None:
        """Wait until the thread has sent every event that is on the queue."""
        self.events.join()

    def close(self) -> None:
        """Send the events that are on the queue, then stop the thread."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            thread = self.thread
        if thread is not None:
            self.events.put(None)
            thread.join()

    def run(self) -> None:
        while True:
            event = self.events.get()
            try:
                if event is None:
                    return
                self.send(event)
            finally:
                self.events.task_done()

    def send(self, event: ExperimentEvent) -> None:
        try:
            if isinstance(event, TrainFromExperimentEvent):
                self.tracker.record_train_from(event)
            else:
                self.tracker.record(event)
        except Exception:
            logger.exception("experiment tracker failed to report a committed step")


def training_starts(source: ArtifactSelection, commit: CommitRecord) -> ArtifactSelection:
    """Apply one committed training-start change to a component-to-artifact mapping.

    The record supplies its component IDs. Other operations leave every training start unchanged.
    """
    if not commit.sets_training_start:
        return source
    if commit.artifact_ids is None:
        raise ValueError("a training-start change requires component artifact IDs")
    components = dict(source.components)
    if commit.component is not None:
        name = commit.component
    elif len(components) == 1:
        name = next(iter(components))
    else:
        # A single trainer can train a complete multi-component snapshot.
        # Its records have no trainer component and replace the whole start.
        return ArtifactSelection(commit.artifact_ids)
    if name not in components:
        raise ValueError(f"unknown training component {name!r}")
    components[name] = commit.artifact_ids[name]
    return ArtifactSelection(components)


class CommitExperimentObserver(RecordObserver):
    """Observe serialized commits for one recovered scenario storage session."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        *,
        scenario: str,
        recipe: str,
        source_artifacts: ArtifactSelection,
        scenario_step: int,
        store: ScenarioStore,
        reports: ExperimentReportQueue | None = None,
    ) -> None:
        """``reports`` delivers the events; the dispatcher shares one queue across its scenarios.

        ``start_artifacts`` is the training start: the newest artifact a
        record made or a ``train_from`` record named, else
        ``source_artifacts``. Each event names it as its source.
        """
        self.tracker = tracker
        self.scenario = scenario
        self.recipe = recipe
        records = store.history() if store.durable else ()
        self.start_artifacts = source_artifacts
        for record in records:
            self.start_artifacts = training_starts(self.start_artifacts, record)
        self.last_reported_step = scenario_step
        self.run_segment, self.run_step = store.training_run_position() if store.durable else (0, 0)
        self.durable = store.durable
        self.store: ScenarioStore = ObservedScenarioStore(store, self)
        self.prepared: PreparedExperiment | None = None
        self.reports = reports if reports is not None else ExperimentReportQueue(tracker)

    def prepare(
        self,
        context: TrainingExperimentContext,
        *,
        checkpoint_path: str | None,
    ) -> Mapping[str, object]:
        """Keep the checkpoint's local path until the commit lands; tracking failures leave metrics alone."""
        self.prepared = PreparedExperiment(context, checkpoint_path)
        try:
            return dict(self.tracker.correlation_metrics(context))
        except Exception:
            logger.exception("experiment tracker failed to prepare correlation metadata")
            return {}

    def record_committed(self, commit: CommitRecord) -> None:
        if commit.step <= self.last_reported_step:
            return
        prepared = self.prepared
        source = self.start_artifacts
        run_segment, run_step = self.run_segment, self.run_step
        # Advance even when the provider fails: a storage retry must not rotate
        # the same run twice or report a training row twice in this session.
        self.last_reported_step = commit.step
        self.prepared = None
        if commit.sets_training_start:
            self.start_artifacts = training_starts(self.start_artifacts, commit)
        if self.durable:
            if commit.operation == "train_from":
                self.run_segment, self.run_step = commit.step, 0
            elif commit.operation == "training":
                self.run_step += 1
        if commit.operation in ("activate", "import", "sync", "serve", "stop_serving"):
            # These change what serves or what exists, not the training run: no tracker event.
            return
        if commit.operation == "train_from":
            self.reports.put(
                TrainFromExperimentEvent(
                    scenario=commit.scenario,
                    recipe=self.recipe,
                    step=commit.step,
                    run_segment=run_segment,
                    start_artifacts=self.start_artifacts,
                    component=commit.component,
                )
            )
            return
        # Storage can also receive a shipped-content update without a training
        # result. Its record supplies the event; runtime-only fields stay absent.
        if prepared is not None and prepared.context.step == commit.step:
            context = prepared.context
        else:
            prepared = None
            context = TrainingExperimentContext(
                scenario=self.scenario,
                recipe=self.recipe,
                step=commit.step,
                source_artifacts=source,
                run_segment=run_segment,
                run_step=run_step,
                component=commit.component,
            )
        metrics = dict(commit.metrics or {})
        self.reports.put(
            TrainingExperimentEvent(
                context=context,
                produced_artifacts=self.start_artifacts,
                metrics=metrics,
                outcome="rejected" if metrics.get("selected") is False else "committed",
                training_job_id=commit.training_job_id,
                checkpoint_path=None if prepared is None else prepared.checkpoint_path,
            )
        )
