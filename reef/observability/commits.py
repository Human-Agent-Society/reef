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

from reef.core.artifact_ref import ArtifactRef
from reef.observability.base import (
    ExperimentTracker,
    RollbackExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.storage.commits import CommitRecord
from reef.storage.observer import ObservedScenarioStore, RecordObserver
from reef.storage.scenario import ScenarioStore

logger = logging.getLogger(__name__)

ExperimentEvent = TrainingExperimentEvent | RollbackExperimentEvent


@dataclass(frozen=True)
class PreparedExperiment:
    context: TrainingExperimentContext
    source_runtime_load_id: str | None
    produced_runtime_load_id: str | None
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
            if isinstance(event, RollbackExperimentEvent):
                self.tracker.record_rollback(event)
            else:
                self.tracker.record(event)
        except Exception:
            logger.exception("experiment tracker failed to report a committed step")


class CommitExperimentObserver(RecordObserver):
    """Observe serialized commits for one recovered scenario storage session."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        *,
        scenario: str,
        recipe: str,
        source_artifact_ref: ArtifactRef,
        scenario_step: int,
        store: ScenarioStore,
        reports: ExperimentReportQueue | None = None,
    ) -> None:
        """``reports`` delivers the events; the dispatcher shares one queue across its scenarios."""
        self.tracker = tracker
        self.scenario = scenario
        self.recipe = recipe
        self.source_artifact_ref = source_artifact_ref
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
        source_runtime_load_id: str | None,
        produced_runtime_load_id: str | None,
        checkpoint_path: str | None,
    ) -> Mapping[str, object]:
        """Keep runtime-only fields until the commit lands; tracking failures leave metrics alone."""
        self.prepared = PreparedExperiment(context, source_runtime_load_id, produced_runtime_load_id, checkpoint_path)
        try:
            return dict(self.tracker.correlation_metrics(context))
        except Exception:
            logger.exception("experiment tracker failed to prepare correlation metadata")
            return {}

    def record_committed(self, commit: CommitRecord) -> None:
        if commit.step <= self.last_reported_step:
            return
        prepared = self.prepared
        source = self.source_artifact_ref
        run_segment, run_step = self.run_segment, self.run_step
        # Advance even when the provider fails: a storage retry must not rotate
        # the same run twice or report a training row twice in this session.
        self.last_reported_step = commit.step
        self.prepared = None
        if not commit.pending:
            self.source_artifact_ref = commit.artifact_ref
        if self.durable:
            if commit.operation == "rollback":
                self.run_segment, self.run_step = commit.step, 0
            elif commit.operation == "training":
                self.run_step += 1
        if commit.operation in ("rollback", "promote"):
            # Promote already uses the rollback tracker event. Keep that contract.
            target = commit.rollback_target_release_id
            if target is None:
                raise ValueError("a rollback or promote commit must name its target")
            self.reports.put(
                RollbackExperimentEvent(
                    scenario=commit.scenario,
                    recipe=self.recipe,
                    step=commit.step,
                    run_segment=run_segment,
                    source_artifact_ref=source,
                    produced_artifact_ref=commit.artifact_ref,
                    target_release_id=target,
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
                source_artifact_ref=source,
                run_segment=run_segment,
                run_step=run_step,
                component=commit.component,
            )
        metrics = dict(commit.metrics or {})
        self.reports.put(
            TrainingExperimentEvent(
                context=context,
                produced_artifact_ref=self.source_artifact_ref,
                metrics=metrics,
                outcome="rejected" if metrics.get("selected") is False else "committed",
                training_job_id=commit.training_job_id,
                source_runtime_load_id=None if prepared is None else prepared.source_runtime_load_id,
                produced_runtime_load_id=None if prepared is None else prepared.produced_runtime_load_id,
                checkpoint_path=None if prepared is None else prepared.checkpoint_path,
            )
        )
