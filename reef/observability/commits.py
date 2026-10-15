"""Translate landed scenario commits into experiment events.

Preparation preserves transient runtime metadata and adds correlation metrics
before persistence. Storage notifications capture the final recorded metrics;
provider calls wait until the committer has released its publication locks.
Delivery is best effort: retries in this session are suppressed, but commits
are not replayed to the tracker after a process restart.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Event, Lock, RLock
from weakref import WeakValueDictionary

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


@dataclass(frozen=True)
class PreparedExperiment:
    context: TrainingExperimentContext
    source_runtime_load_id: str | None
    produced_runtime_load_id: str | None
    checkpoint_path: str | None


class ExperimentReportDelivery:
    """One scenario's in-process FIFO, shared by observers across reloads.

    Storage callbacks enqueue without waiting for the provider. Each event's
    readiness signal belongs to the outer reporting scope, which releases its
    serving locks before setting it. Only the delivery lock spans provider I/O.
    """

    def __init__(self, tracker: ExperimentTracker) -> None:
        self.tracker = tracker
        self.lock = RLock()
        self.queue_lock = Lock()
        self.events: deque[tuple[TrainingExperimentEvent | RollbackExperimentEvent, Event]] = deque()

    def enqueue(self, event: TrainingExperimentEvent | RollbackExperimentEvent, ready: Event) -> None:
        with self.queue_lock:
            self.events.append((event, ready))

    def flush(self) -> None:
        """Deliver the ready prefix outside serving locks; an unfinished scope flushes on exit."""
        with self.lock:
            while True:
                with self.queue_lock:
                    if not self.events:
                        return
                    event, ready = self.events[0]
                    if not ready.is_set():
                        # Recovery may load an observer inside an older reporting
                        # scope. Never wait for that same caller to exit itself.
                        return
                    self.events.popleft()
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
        delivery: ExperimentReportDelivery | None = None,
    ) -> None:
        self.tracker = tracker
        self.scenario = scenario
        self.recipe = recipe
        self.source_artifact_ref = source_artifact_ref
        self.last_reported_step = scenario_step
        self.run_segment, self.run_step = store.training_run_position() if store.durable else (0, 0)
        self.durable = store.durable
        self.store: ScenarioStore = ObservedScenarioStore(store, self)
        self.prepared: PreparedExperiment | None = None
        self.reporting_lock = RLock()
        self.delivery = delivery if delivery is not None else ExperimentReportDelivery(tracker)
        self.report_ready: Event | None = None

    @contextmanager
    def reporting(self, *, loading: bool = False) -> Iterator[None]:
        """Collect events until publication, admission and registry locks release.

        Ordinary operations wait for previous observers before taking serving
        locks. A factory is already under the registry lock, so loading only
        enqueues; the registry's outer scope flushes after all loads finish.
        """
        with self.reporting_lock:
            if self.report_ready is not None:
                yield
                return
            with nullcontext() if loading else self.delivery.lock:
                if not loading:
                    self.delivery.flush()
                ready = Event()
                self.report_ready = ready
                try:
                    yield
                finally:
                    self.report_ready = None
                    ready.set()
                    if not loading:
                        self.delivery.flush()

    def deliver(self, event: TrainingExperimentEvent | RollbackExperimentEvent) -> None:
        ready = self.report_ready
        if ready is None:
            ready = Event()
            ready.set()
            self.delivery.enqueue(event, ready)
            self.delivery.flush()
        else:
            self.delivery.enqueue(event, ready)

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
            self.deliver(
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
        self.deliver(
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


class LoadedExperimentReports:
    """Keep per-scenario delivery order and defer loads past the outer registry lock."""

    def __init__(self) -> None:
        self.active_scope: ContextVar[tuple[ExitStack, list[tuple[CommitExperimentObserver, int]]] | None] = (
            ContextVar("loaded_experiment_reports", default=None)
        )
        # Observers and active delivery calls own queues. Keep the same queue
        # through reloads while either exists, then release unused scenario names.
        self.deliveries: WeakValueDictionary[str, ExperimentReportDelivery] = WeakValueDictionary()
        self.lock = Lock()

    def delivery_for(self, scenario: str, tracker: ExperimentTracker) -> ExperimentReportDelivery:
        with self.lock:
            delivery = self.deliveries.get(scenario)
            if delivery is None:
                delivery = ExperimentReportDelivery(tracker)
                self.deliveries[scenario] = delivery
            return delivery

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self.active_scope.get() is not None:
            yield
            return
        observers: list[tuple[CommitExperimentObserver, int]] = []
        try:
            with ExitStack() as reports:
                token = self.active_scope.set((reports, observers))
                try:
                    yield
                finally:
                    self.active_scope.reset(token)
        finally:
            # ExitStack marks every load ready before any flush waits, including
            # multiple reloads in one outer registry scope and failed loads.
            for observer, initial_step in observers:
                if observer.last_reported_step > initial_step:
                    observer.delivery.flush()

    def defer(self, observer: CommitExperimentObserver) -> None:
        """Retain a new observer through registration, including failed loads."""
        scope = self.active_scope.get()
        if scope is not None:
            reports, observers = scope
            reports.enter_context(observer.reporting(loading=True))
            observers.append((observer, observer.last_reported_step))
