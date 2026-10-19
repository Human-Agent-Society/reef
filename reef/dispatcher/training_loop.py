"""The training thread: one serial thread that runs the dispatched training jobs of every training scenario.

A dispatched backend runs its job on the deployment's ``TrainingRuntime``.
The thread reserves a batch, runs the job, commits the result and
acknowledges it to the backend. A failed turn reloads its scenario from
durable state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from threading import Event, Lock, Thread
from typing import Any

from reef.dispatcher.errors import TrainingErrors, error_text
from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.runtime.interfaces import RuntimeContractError, TrainingRuntime
from reef.runtime.recovery import marker_in_flight
from reef.scenario.registry import InstanceState, ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.train.backend import CandidateBackend
from reef.train.types import TrainingBatch

logger = logging.getLogger(__name__)


def in_flight_job_marker(runtime: TrainingRuntime | None) -> Mapping[str, Any] | None:
    """The training runtime's durable job marker while it names a job still out; runtime failures propagate."""
    if runtime is None:
        return None
    marker = runtime.training_job_status()
    return marker if marker_in_flight(marker) else None


class ScenarioTrainingError(Exception):
    """One scenario's training turn failed; the others keep their state."""

    def __init__(self, scenario: str, cause: Exception) -> None:
        super().__init__(f"{scenario}: {cause}")
        self.scenario = scenario
        self.cause = cause


class DispatchedTrainingLoop:
    """Run the training thread and keep the state that the training status reports about it.

    Preparation, remote execution and the commit run in one serial thread.
    An accepted record can wake the thread while a slow commit runs. Because
    the thread is serial, two wakes cannot commit the same pending result as
    two steps.
    """

    storage_retry_seconds: float = 5.0
    # Poll interval while a pipeline has background jobs that run (see
    # TrainingPipeline.background_jobs_running). A finished job, such as a
    # judgment, can make a batch ready, but no new record sets the ready
    # event. Thus the thread checks for a batch at this interval. The thread
    # also polls at this interval after it skipped a scenario whose cycle lock
    # was busy.
    background_job_poll_seconds: float = 1.0
    # One drain, plus one more after reloading the scenario from durable state.
    drain_attempts: int = 2
    # A ready batch should be reserved by the next drain; one that sits longer
    # means the training thread is not waking. Status reads perform the check,
    # so the alarm rides the health polling that already watches the service.
    undrained_warning_seconds: float = 60.0

    def __init__(
        self, registry: ScenarioRegistry, errors: TrainingErrors, local_workers: LocalTrainingWorkers, ready: Event
    ) -> None:
        """``ready`` wakes the thread. The local workers set it too, when a weight-training scenario wakes."""
        self.registry = registry
        self.errors = errors
        self.local_workers = local_workers
        self.ready = ready
        self.lock = Lock()
        self.storage_status: Mapping[str, Any] | None = None
        self.last_drain: float | None = None
        self.undrained_warned = False
        self.thread: Thread | None = None
        #: Whether the last drain skipped a scenario because another holder had its cycle lock.
        self.skipped_busy_scenario = False

    def start(self) -> None:
        """Start the training thread if it hasn't been started yet, and wake it."""
        if self.local_workers.closed.is_set():
            return
        with self.lock:
            if self.thread is not None:
                return
            self.thread = Thread(target=self.run, name="reef-training", daemon=True)
            self.thread.start()
        self.ready.set()

    def drain_state(self) -> tuple[Mapping[str, Any] | None, float | None]:
        """The checkpoint storage block, or ``None``, and the time of the last drain."""
        with self.lock:
            return self.storage_status, self.last_drain

    def join(self) -> None:
        """Wait for the training thread to end. Close the local workers and set ``ready`` first."""
        if self.thread is not None:
            self.thread.join()

    def stopped_unexpectedly(self) -> bool:
        """Whether the thread ended while the dispatcher is still open."""
        with self.lock:
            thread = self.thread
        return thread is not None and not thread.is_alive() and not self.local_workers.closed.is_set()

    def run(self) -> None:
        try:
            while True:
                self.ready.wait(self.wait_timeout())
                self.ready.clear()
                if self.local_workers.closed.is_set():
                    return
                self.drain()
                self.record_drain()
        except Exception as exc:
            name = self.registry.training_scenario_name or "<unbound>"
            logger.exception("training thread stopped unexpectedly for scenario %r", name)
            self.errors.record(name, error_text(exc))

    def wait_timeout(self) -> float | None:
        """How long the training thread may sleep before re-checking.

        Blocked checkpoint storage retries on its own interval. The thread
        polls at ``background_job_poll_seconds`` in two cases. First, while a
        pipeline has background jobs that run, because a finished job can make
        a batch ready without a new record. Second, after the last drain
        skipped a scenario whose cycle lock was busy, because only the end of
        a local cycle wakes the thread. Otherwise it sleeps until the next
        accepted record.
        """
        timeout: float | None = self.storage_retry_seconds if self.storage_status is not None else None
        poll = self.skipped_busy_scenario
        for name in self.registry.training_scenario_names:
            current = self.registry.get_optional(name)
            if current is not None and any(
                bound.trainer.pipeline.background_jobs_running() for bound in current.trainers
            ):
                poll = True
        if poll:
            timeout = (
                self.background_job_poll_seconds if timeout is None else min(timeout, self.background_job_poll_seconds)
            )
        return timeout

    def drain(self) -> None:
        # One recovery attempt, not a retry loop. If a drain fails, we reload
        # the scenario from durable state and drain once more. This repairs a
        # crash between a durable commit and the processor cleanup at once.
        # A second failure also reloads, so the next wake finds a clean
        # scenario, but the thread does not try again. The training status
        # reports the cause.
        self.skipped_busy_scenario = False
        for _ in range(self.drain_attempts):
            try:
                while self.process():
                    pass
                return
            except ScenarioTrainingError as failure:
                name = failure.scenario
                logger.exception("training thread failed to commit for scenario %r", name)
                self.errors.record(name, error_text(failure.cause))
                self.reload_after_failure(name, failure.cause)
            except Exception as exc:
                name = self.registry.training_scenario_name or "<unbound>"
                logger.exception("training thread failed to commit for scenario %r", name)
                self.errors.record(name, error_text(exc))
                self.reload_after_failure(name, exc)

    def process(self) -> bool:
        """Give every training scenario one turn; True when any of them progressed.

        Several scenarios share one training runtime only when the runtime
        time-slices its adapter slot between them. The round-robin keeps the
        schedule deterministic and fair. A failed turn reloads only its own
        scenario.
        """
        names = self.registry.training_scenario_names
        if not names:
            # The last training scenario was deleted: the thread idles until a scenario binds again.
            return False
        progressed = False
        for name in names:
            try:
                progressed = self.process_scenario(name) or progressed
            except Exception as exc:
                raise ScenarioTrainingError(name, exc) from exc
        return progressed

    def process_scenario(self, name: str) -> bool:
        current = self.registry.get_optional(name)
        if current is None:
            raise RuntimeContractError(f"training thread is not bound to scenario {name!r}")
        runtime = current.training_runtime
        if not isinstance(runtime, TrainingRuntime):
            raise RuntimeContractError(
                f"training thread requires a TrainingRuntime for scenario {current.name!r}, got {type(runtime).__name__}"
            )
        component = current.dispatched_component
        backend = current.trainer_for(component).candidate_backend
        if backend is None or not backend.dispatched:
            raise RuntimeContractError(f"training scenario {current.name!r} has no dispatched training backend")
        # A scenario runs one step at a time. While another holder has the cycle lock,
        # skip the scenario. The end of a local cycle wakes this thread. Any other
        # holder, such as an idle check, lets go at once, and the thread polls for it.
        cycle_lock = self.local_workers.cycle_lock(name)
        if not cycle_lock.acquire(blocking=False):
            self.skipped_busy_scenario = True
            return False
        try:
            self.errors.clear(current.name)
            # A crash may leave remote serving updated but paused after Reef's
            # commit, or checkpointed before the weight update. Recover that
            # pending step before deciding whether another batch is available.
            backend.recover_pending_step(
                current.scenario_step,
                committed_training_job_id=current.committed_training_job_id,
                committed_training_without_job_id=current.committed_training_without_job_id,
            )
            with self.registry.lock_instance(current) as state:
                if state is not InstanceState.CURRENT:
                    # Reloaded since the turn began: the next turn reserves on the rebuilt instance.
                    return True
                batch = current.reserve_training_batch(component)
            if batch is None:
                return False
            try:
                return self.run_dispatched_turn(current, component, backend, batch)
            finally:
                # Whatever the outcome, the local cycles that waited for the job run now, on every scenario.
                self.local_workers.wake_all()
        finally:
            cycle_lock.release()
            self.local_workers.close_replaced_instances(name)

    def run_dispatched_turn(
        self, current: Scenario, component: str | None, backend: CandidateBackend, batch: TrainingBatch
    ) -> bool:
        with self.local_workers.dispatched_turn(current, component):
            execution = current.execute_reserved_training_step(component)
            if execution.outcome == "retry":
                if execution.storage is None:
                    raise RuntimeContractError("retry execution must carry storage status")
                self.set_storage_status(dict(execution.storage))
                return False
            self.set_storage_status(None)
            if execution.outcome not in ("drop", "commit"):
                raise RuntimeContractError(f"training backend returned unsupported outcome: {execution.outcome!r}")
            # The registry lock keeps a local worker's reload out between the instance check and the commit.
            with self.registry.lock_instance(current) as state:
                if state is InstanceState.REMOVED:
                    raise RuntimeContractError(
                        f"scenario {current.name!r} was deleted under its training job; the backend's job needs "
                        "operator recovery"
                    )
                if state is InstanceState.REPLACED:
                    # A local worker's failure reloaded the scenario while the job ran: its result belongs to the
                    # instance that reserved the batch. The rebuilt trainer reserves the same rows and the
                    # backend replays the job under its marker, so the commit lands with the job's identity.
                    logger.warning("scenario %r was reloaded under its training job; the job replays", current.name)
                    return True
                if execution.outcome == "drop":
                    logger.warning("dropping stale training batch %r for scenario %r", batch.batch_id, current.name)
                    current.reject_pending(execution.metrics, component=component)
                    return True
                result = execution.result
                if result is None:
                    raise RuntimeContractError("commit execution must carry a training result")
                # Another component can move the head while the job runs. The
                # committer merges a dispatched result and does not refuse it. The
                # backend has published the weights already, and the job can only
                # finish; it cannot go back.
                current.commit(result, component=component)
                if result.training_job_id is not None:
                    backend.acknowledge_commit(current.scenario_step, result.training_job_id)
        return True

    def reload_after_failure(self, scenario: str, cause: Exception) -> None:
        current = self.registry.get_optional(scenario)
        if current is None:
            if not self.registry.has(scenario):
                # Deleted while its step was in flight: nothing durable to reload.
                return
            self.local_workers.reload(scenario, None)
            return
        self.local_workers.fail_instruction(current, cause, current.dispatched_component)
        self.local_workers.reload(scenario, current)
        # The rebuilt instance holds the local components' rows unread; their workers look again.
        self.local_workers.wake_all()

    def record_drain(self) -> None:
        with self.lock:
            self.last_drain = time.time()
            self.undrained_warned = False

    def warn_if_undrained(self, scenario: str, last_drain: float | None) -> None:
        if last_drain is None or time.time() - last_drain < self.undrained_warning_seconds:
            return
        with self.lock:
            first = not self.undrained_warned
            self.undrained_warned = True
        if first:
            logger.warning(
                "scenario %r has a ready training batch undrained for %.0f seconds",
                scenario,
                time.time() - last_drain,
            )

    def set_storage_status(self, value: Mapping[str, Any] | None) -> None:
        with self.lock:
            was_blocked = self.storage_status is not None
            self.storage_status = value
        if value is not None and not was_blocked:
            logger.warning(
                "training is blocked on checkpoint storage: %s; retrying every %.0f seconds until it clears",
                "; ".join(str(reason) for reason in value.get("reasons", ())) or "no reason reported",
                self.storage_retry_seconds,
            )
        elif value is None and was_blocked:
            logger.info("checkpoint storage block cleared; training resumes")
