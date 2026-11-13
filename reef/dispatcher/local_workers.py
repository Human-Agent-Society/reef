"""Training workers: one thread per scenario component that trains.

A cycle prepares the trainer's next commit outside the scenario's registry
lock, because the preparation can take a while (a job's submission, or the
read of a job's checkpoints). Then it commits under that lock. A
worker whose trainer has a training job or a pipeline background job (such
as a judge call) running looks again every ``POLL_SECONDS``, since a job's
checkpoints and a judgment arrive without a new record.
A cycle that fails waits before the next one, longer after each failure in a
row (``failed_cycle_delay_seconds``); a cycle that ends without an error
clears the wait. A reload after a failure starts a new worker, which keeps the
wait, so a cycle that always fails does not run again at once.
The workers also close the instances that a reload replaced.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Literal

from reef.dispatcher.errors import TrainingErrors, error_text, local_error_source
from reef.inference.interfaces import RuntimeContractError
from reef.scenario.registry import InstanceState, ScenarioRegistry
from reef.scenario.scenario import Scenario

logger = logging.getLogger(__name__)

#: How soon a worker whose trainer has a training job or a pipeline background job running looks again.
POLL_SECONDS = 2.0
#: How long a worker waits after the first failed cycle in a row; each further failure doubles the wait.
FIRST_RETRY_SECONDS = 2.0
#: The longest wait after failed cycles.
MAX_RETRY_SECONDS = 60.0


def failed_cycle_delay_seconds(failures: int) -> float:
    """The wait after ``failures`` failed cycles in a row, a positive int: ``FIRST_RETRY_SECONDS`` doubled for each
    failure after the first, at most ``MAX_RETRY_SECONDS``."""
    if failures < 1:
        raise ValueError(f"failures must be a positive int, got {failures!r}")
    return min(MAX_RETRY_SECONDS, FIRST_RETRY_SECONDS * 2 ** min(failures - 1, 16))


@dataclass(frozen=True)
class LocalBackendWorker:
    ready: Event
    thread: Thread


@dataclass
class ScenarioCycles:
    lock: Lock = field(default_factory=Lock)
    replaced: list[Scenario] = field(default_factory=list)


class LocalTrainingWorkers:
    """Own worker lifetime and cycles, and scenario replacement.

    Service starts held or running. Opening releases held workers; stopping and closed never reopen.
    Each scenario runs one cycle at a time under its cycle lock. Replaced
    instances close only after the cycle that uses them releases its lock.

    Lock order: cycle locks -> registry scenario lock -> coordination lock -> training errors lock.
    Never wait for a cycle lock, join a worker, or close a scenario while holding the coordination lock.
    Registry reloads close the replaced instance while holding their scenario lock:
    closure therefore tries the cycle lock without waiting.
    """

    def __init__(
        self, registry: ScenarioRegistry, errors: TrainingErrors, *, held: bool, automatic_jobs: bool = True
    ) -> None:
        self.registry = registry
        self.errors = errors
        #: Whether a trainer in ``auto`` mode starts jobs by Reef's own rule; false when a training manager decides.
        self.automatic_jobs = automatic_jobs
        #: The coordination lock.
        self.lock = Lock()
        self.phase: Literal["held", "running", "stopping", "closed"] = "held" if held else "running"
        self.closed = Event()
        self.workers: dict[tuple[str, str | None], LocalBackendWorker] = {}
        self.scenarios: dict[str, ScenarioCycles] = {}
        #: For each worker key whose cycles failed in a row: the count of failures, and the monotonic time before
        #: which its next cycle does not run. A cycle that ends without an error removes the key.
        self.failed_cycles: dict[tuple[str, str | None], tuple[int, float]] = {}

    # -- Worker lifetime -------------------------------------------------

    def start(self, scenario: str, component: str | None = None) -> None:
        """Start the worker of ``scenario``'s ``component``, or wake it if it runs already."""
        with self.lock:
            if self.phase == "closed":
                return
            key = (scenario, component)
            worker = self.workers.get(key)
            if worker is None:
                ready = Event()
                thread = Thread(
                    target=self.run_worker,
                    args=(scenario, component, ready),
                    name=f"reef-local-backend-{scenario}" + ("" if component is None else f"/{component}"),
                    daemon=True,
                )
                worker = LocalBackendWorker(ready, thread)
                self.workers[key] = worker
                thread.start()
        worker.ready.set()

    def wake(self, current: Scenario) -> None:
        """Wake the worker of each trainer of ``current`` that trains."""
        for component_trainer in current.trainers:
            if component_trainer.trainer.trains:
                self.start(current.name, component_trainer.component)

    def resume(self, current: Scenario) -> None:
        """Start the worker of each weight trainer of ``current``.

        The worker's first cycle resumes the job a restart left running, trains the records that arrived before
        the load, and starts the pipeline's own work, such as a round of generated tasks. No record would wake
        the worker for these.
        """
        for component_trainer in current.trainers:
            if component_trainer.trainer.job_loop is not None:
                self.start(current.name, component_trainer.component)

    def remove(self, scenario: str) -> None:
        """Unregister workers without joining them under the registry lock."""
        with self.lock:
            workers = [self.workers.pop(key) for key in tuple(self.workers) if key[0] == scenario]
            for key in [key for key in self.failed_cycles if key[0] == scenario]:
                self.failed_cycles.pop(key)
        for worker in workers:
            worker.ready.set()

    def retry_wait_seconds(self, scenario: str, component: str | None) -> float:
        """How long the worker of ``scenario`` and ``component`` still waits after failed cycles; 0 when none."""
        with self.lock:
            failed = self.failed_cycles.get((scenario, component))
        return 0.0 if failed is None else max(0.0, failed[1] - time.monotonic())

    def wait(self, scenario: str, component: str | None, ready: Event) -> bool:
        current = self.registry.get_optional(scenario)
        trainer = None if current is None else current.trainer_for(component)
        polling = trainer is not None and trainer.needs_polling()
        timeout = POLL_SECONDS if polling else None
        retry = self.retry_wait_seconds(scenario, component)
        if retry > 0:
            # A wake during the wait after failed cycles does not run a cycle sooner; only the close of the workers
            # ends the wait early. Then one cycle runs.
            self.closed.wait(timeout=retry)
            ready.set()
        ready.wait(timeout=timeout)
        ready.clear()
        with self.lock:
            # A worker for a reloaded instance with the same key replaces this worker. Then this worker stops.
            worker = self.workers.get((scenario, component))
            return self.phase != "closed" and worker is not None and worker.ready is ready

    def open(self) -> None:
        with self.lock:
            if self.phase == "held":
                self.phase = "running"

    def stop(self) -> None:
        with self.lock:
            if self.phase != "closed":
                self.phase = "stopping"

    def begin_close(self) -> tuple[LocalBackendWorker, ...]:
        """Freeze registration and wake the exact workers shutdown must join."""
        with self.lock:
            self.phase = "closed"
            self.closed.set()
            workers = tuple(self.workers.values())
        for worker in workers:
            worker.ready.set()
        return workers

    def cycles_run(self) -> bool:
        with self.lock:
            return self.phase == "running"

    # -- Local cycles ----------------------------------------------------

    def run_worker(self, scenario: str, component: str | None, ready: Event) -> None:
        try:
            while self.wait(scenario, component, ready):
                self.drain(scenario, component)
        except Exception as exc:
            logger.exception("local backend worker stopped unexpectedly for scenario %r", scenario)
            self.errors.record(scenario, error_text(exc), local_error_source(component))

    def drain(self, scenario: str, component: str | None) -> None:
        key = (scenario, component)
        try:
            while self.cycles_run() and self.process_step(scenario, component):
                pass
        except Exception as exc:
            logger.exception("local backend failed to commit for scenario %r", scenario)
            self.record_cycle_error(scenario, component, error_text(exc))
            with self.lock:
                failures = self.failed_cycles.get(key, (0, 0.0))[0] + 1
                self.failed_cycles[key] = (failures, time.monotonic() + failed_cycle_delay_seconds(failures))
            return
        with self.lock:
            self.failed_cycles.pop(key, None)

    def record_cycle_error(self, scenario: str, component: str | None, error: str) -> None:
        """Record a local cycle's failure, unless the scenario was deleted under it: nothing would ever clear that."""
        with self.lock:
            if (scenario, component) in self.workers:
                self.errors.record(scenario, error, local_error_source(component))

    def process_step(self, scenario: str, component: str | None = None) -> bool:
        """Run one local cycle of ``scenario``'s ``component``; return whether the worker should look again."""
        current = self.registry.get_optional(scenario)
        if current is None:
            # Deleted while a cycle was in flight: the cycle ends without
            # a commit and without an error, as documented in http-api.rst.
            return False
        if not current.trainer_for(component).trains:
            raise RuntimeContractError(f"scenario {scenario!r} component {component!r} does not train")
        self.errors.clear(scenario, local_error_source(component))
        # A scenario runs one cycle at a time. A cycle holds the cycle lock
        # from preparation to commit. Thus the head does not move under a job's submission.
        with self.cycle(scenario):
            if not self.cycles_run():
                return False
            state = self.registry.instance_state(current)
            if state is InstanceState.REMOVED:
                # Deleted while this worker waited for its turn.
                return False
            if state is InstanceState.REPLACED:
                # Reloaded while this worker waited: look again on the rebuilt instance, whose rows are unread.
                return True
            try:
                result = current.next_commit(component, automatic_jobs=self.automatic_jobs)
            except Exception as exc:
                self.fail_request(current, exc, component)
                self.reload_after_local_failure(current)
                raise
            if result is None:
                return False
            # Keep only the short commit and recovery window under the scenario
            # registry lock. Candidate generation above can take minutes and must
            # not block record acceptance for this scenario.
            with self.registry.lock_instance(current) as state:
                if not self.cycles_run():
                    # The service stopped under this cycle: its model calls may have failed as the routes went away.
                    logger.info(
                        "scenario %r component %r: the service is stopping; the step waits", scenario, component
                    )
                    return False
                if state is not InstanceState.CURRENT:
                    # A reload replaced the scenario, or a delete removed it.
                    # The result belongs to the old instance, so the step ends without a commit. A rebuilt
                    # instance has not read these rows: look again on it.
                    if state is InstanceState.REMOVED and result.training_job_id is not None:
                        # The delete ended before this cycle added its job to the queue, so it cancelled none:
                        # the job of the deleted scenario must not run.
                        current.trainer_for(component).job_loop_required().component.cancel(result.training_job_id)
                    return state is InstanceState.REPLACED
                try:
                    current.commit(result, component=component)
                except Exception:
                    # A record may already have crossed the fsync commit point.
                    # Reload before an activation or acceptance can observe the stale
                    # in-memory step and append the same step number again.
                    self.reload_after_local_failure(current)
                    raise
                self.registry.lifecycle.scenario_committed(current)
        return True

    def fail_request(self, current: Scenario, cause: Exception, component: str | None = None) -> None:
        """A failed request step consumes the request with a skip row on the next step; wake for it."""
        if not current.fail_pending_request(error_text(cause), component):
            return
        with self.registry.lock_instance(current) as state:
            # Wake the worker, so that the failed request settles without a new record.
            # Do not start a worker for an instance that a delete or a reload removed.
            if state is InstanceState.CURRENT:
                self.wake(current)

    # -- Cycle locks -----------------------------------------------------

    def cycle_lock(self, scenario: str) -> Lock:
        with self.lock:
            return self.scenarios.setdefault(scenario, ScenarioCycles()).lock

    @contextmanager
    def cycle(self, scenario: str) -> Iterator[None]:
        try:
            with self.cycle_lock(scenario):
                yield
        finally:
            # A cycle is a use of the scenario. Its idle time starts when its last cycle ends.
            self.registry.touch(scenario)
            self.close_replaced_instances(scenario)

    # -- Scenario replacement and reload ---------------------------------

    def close_replaced(self, instance: Scenario) -> None:
        with self.lock:
            self.scenarios.setdefault(instance.name, ScenarioCycles()).replaced.append(instance)
        self.close_replaced_instances(instance.name)

    def close_replaced_instances(self, scenario: str) -> None:
        lock = self.cycle_lock(scenario)
        while True:
            if not lock.acquire(blocking=False):
                return  # The holder drains replacements when it releases the cycle lock.
            try:
                with self.lock:
                    state = self.scenarios[scenario]
                    replaced, state.replaced = state.replaced, []
                for instance in replaced:
                    instance.close()
            finally:
                lock.release()
            with self.lock:
                if not state.replaced:
                    return

    def instance_change_pending(self, scenario: str) -> bool:
        """Whether a replaced instance of ``scenario`` waits to close."""
        with self.lock:
            state = self.scenarios.get(scenario)
            return state is not None and bool(state.replaced)

    def take_replaced(self) -> list[Scenario]:
        """Collect remaining replacements after all workers have joined."""
        with self.lock:
            replaced = [instance for state in self.scenarios.values() for instance in state.replaced]
            for state in self.scenarios.values():
                state.replaced.clear()
            return replaced

    def reload_after_local_failure(self, current: Scenario) -> None:
        """Reload durable state after a failed cycle; do not wake the failing worker."""
        if not current.store.durable:
            return
        with self.registry.lock_instance(current) as state:
            if state is InstanceState.CURRENT:
                self.registry.reload(current.name)
