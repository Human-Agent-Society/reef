"""Local backend workers: one thread per scenario component whose backend runs in this process.

A local cycle prepares a candidate outside the scenario's registry lock,
because the preparation can take minutes (for example, a harness evaluation).
Then it commits the candidate under that lock. The workers share the cycle
locks with dispatched training turns. They also close the instances that a
reload replaced.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Literal

from reef.dispatcher.errors import TrainingErrors, error_text, local_error_source
from reef.runtime.interfaces import RuntimeContractError
from reef.scenario.registry import InstanceState, ScenarioRegistry
from reef.scenario.scenario import Scenario

logger = logging.getLogger(__name__)

#: How soon a local worker that stood aside for closed admission looks again.
STOOD_ASIDE_RETRY_SECONDS = 5.0


@dataclass(frozen=True)
class LocalBackendWorker:
    ready: Event
    thread: Thread


@dataclass
class ScenarioCycles:
    lock: Lock = field(default_factory=Lock)
    waiting_for_admission: set[str | None] = field(default_factory=set)
    replaced: list[Scenario] = field(default_factory=list)


class LocalTrainingWorkers:
    """Own local worker lifetime and cycles, dispatched turns, and scenario replacement.

    Service starts held or running. Opening releases held workers; stopping and closed never reopen.
    Each scenario runs one step at a time under its cycle lock: a local cycle,
    or a dispatched turn of the training thread. A colocated job also takes
    the cycle locks of all other scenarios in name order; workers yield while
    it waits. Replaced instances close only after the step that uses them
    releases its lock.

    Lock order: cycle locks -> registry scenario lock -> coordination lock -> training errors lock.
    Never wait for a cycle lock, join a worker, or close a scenario while holding the coordination lock.
    Registry reloads close the replaced instance while holding their scenario lock:
    closure therefore tries the cycle lock without waiting.
    """

    def __init__(
        self, registry: ScenarioRegistry, errors: TrainingErrors, training_ready: Event, *, held: bool
    ) -> None:
        """``training_ready`` is the wake event of the training thread. A wake of a weight-training scenario sets it."""
        self.registry = registry
        self.errors = errors
        self.training_ready = training_ready
        #: The coordination lock.
        self.lock = Lock()
        self.phase: Literal["held", "running", "stopping", "closed"] = "held" if held else "running"
        self.closed = Event()
        self.turn_waiting = Event()
        self.workers: dict[tuple[str, str | None], LocalBackendWorker] = {}
        self.scenarios: dict[str, ScenarioCycles] = {}

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
        """Wake each trainer of ``current`` that has a backend: the training thread, or a local worker."""
        if current.training_runtime is not None:
            self.training_ready.set()
        for bound in current.trainers:
            backend = bound.trainer.candidate_backend
            if backend is not None and not backend.dispatched:
                self.start(current.name, bound.component)

    def remove(self, scenario: str) -> None:
        """Unregister workers without joining them under the registry lock."""
        with self.lock:
            workers = [self.workers.pop(key) for key in tuple(self.workers) if key[0] == scenario]
            state = self.scenarios.get(scenario)
            if state is not None:
                state.waiting_for_admission.clear()
        for worker in workers:
            worker.ready.set()

    def wait(self, scenario: str, component: str | None, ready: Event) -> bool:
        with self.lock:
            state = self.scenarios.get(scenario)
            waiting = state is not None and component in state.waiting_for_admission
        ready.wait(timeout=STOOD_ASIDE_RETRY_SECONDS if waiting else None)
        ready.clear()
        with self.lock:
            # A worker for a reloaded instance with the same key replaces this worker. Then this worker stops.
            worker = self.workers.get((scenario, component))
            return self.phase != "closed" and worker is not None and worker.ready is ready

    def wake_all(self) -> None:
        with self.lock:
            workers = tuple(self.workers.values())
        for worker in workers:
            worker.ready.set()

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

    def check_admission(self, current: Scenario, component: str | None) -> bool:
        runtime = current.runtime
        admission_open = runtime is None or runtime.inference_admission_status.get("open", True)
        with self.lock:
            state = self.scenarios.setdefault(current.name, ScenarioCycles())
            if admission_open:
                state.waiting_for_admission.discard(component)
            else:
                state.waiting_for_admission.add(component)
        return admission_open

    # -- Local cycles ----------------------------------------------------

    def run_worker(self, scenario: str, component: str | None, ready: Event) -> None:
        try:
            while self.wait(scenario, component, ready):
                self.drain(scenario, component)
        except Exception as exc:
            logger.exception("local backend worker stopped unexpectedly for scenario %r", scenario)
            self.errors.record(scenario, error_text(exc), local_error_source(component))

    def drain(self, scenario: str, component: str | None) -> None:
        try:
            # A dispatched job waiting for its turn goes before the next cycle; it wakes the workers after.
            while not self.turn_waiting.is_set() and self.cycles_run() and self.process_step(scenario, component):
                pass
        except Exception as exc:
            logger.exception("local backend failed to commit for scenario %r", scenario)
            self.record_cycle_error(scenario, component, error_text(exc))

    def record_cycle_error(self, scenario: str, component: str | None, error: str) -> None:
        """Record a local cycle's failure, unless the scenario was deleted under it: nothing would ever clear that."""
        with self.lock:
            if (scenario, component) in self.workers:
                self.errors.record(scenario, error, local_error_source(component))

    def process_step(self, scenario: str, component: str | None = None) -> bool:
        """Run one local cycle of ``scenario``'s ``component``; return whether the worker should look again."""
        current = self.registry.get_optional(scenario)
        if current is None:
            # Deleted while a harness step was in flight: the step ends without
            # a commit and without an error, as documented in http-api.rst.
            return False
        if current.trainer_for(component).candidate_backend is None:
            raise RuntimeContractError(f"scenario {scenario!r} has no local backend")
        if not self.check_admission(current, component):
            return False
        self.errors.clear(scenario, local_error_source(component))
        # A scenario runs one step at a time. A local cycle holds the cycle lock
        # from preparation to commit, and a dispatched turn holds it from
        # reservation to acknowledgement. Thus the head does not move under a step.
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
            # The job whose turn this worker waited for may have left admission closed.
            if not self.check_admission(current, component):
                return False
            if current.is_job_reserved:
                # A dispatched step is still out, for example while checkpoint storage is blocked.
                # Its turn wakes this worker when the step ends.
                return False
            try:
                result = current.prepare_training_step(component)
            except Exception as exc:
                self.fail_instruction(current, exc, component)
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
                    # The training thread rebuilt the scenario after a failed turn, or a delete removed it.
                    # The result belongs to the old instance, so the step ends without a commit. A rebuilt
                    # instance has not read these rows: look again on it.
                    return state is InstanceState.REPLACED
                try:
                    current.commit(result, component=component)
                except Exception:
                    # A record may already have crossed the fsync commit point.
                    # Reload before rollback or acceptance can observe the stale
                    # in-memory step and append the same step number again.
                    self.reload_after_local_failure(current)
                    raise
        return True

    def fail_instruction(self, current: Scenario, cause: Exception, component: str | None = None) -> None:
        """A failed instruction step consumes the instruction with a skip row on the next step; wake for it."""
        if not current.trainer_for(component).fail_pending_instruction(error_text(cause)):
            return
        with self.registry.lock_instance(current) as state:
            # Wake the worker, so that the failed instruction settles without a new record.
            # Do not start a worker for an instance that a delete or a reload removed.
            if state is InstanceState.CURRENT:
                self.wake(current)

    # -- Cycle locks and dispatched turns --------------------------------

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
            # The training thread skips a scenario while a local cycle holds its lock. Wake it now.
            current = self.registry.get_optional(scenario)
            if current is not None and current.training_runtime is not None:
                self.training_ready.set()

    def dispatched_turn(self, current: Scenario, component: str | None) -> ExitStack:
        """Hold the cycle locks of all other scenarios through a colocated job's execution, commit and acknowledgement.

        The training thread holds the cycle lock of ``current`` already.
        """
        backend = current.trainer_for(component).candidate_backend
        if backend is None or not backend.colocated:
            return ExitStack()
        names = {loaded.name for loaded in self.registry.loaded_scenarios()}
        with self.lock:
            names.update(self.scenarios)
        names.discard(current.name)
        names_in_order = sorted(names)
        self.turn_waiting.set()
        try:
            with ExitStack() as turn:
                # The callbacks run after the turn releases every lock, also when it acquired only some.
                for name in names_in_order:
                    turn.callback(self.close_replaced_instances, name)
                for name in names_in_order:
                    turn.enter_context(self.cycle_lock(name))
                return turn.pop_all()
        finally:
            self.turn_waiting.clear()

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
        """Reload durable state after a failed local cycle; do not wake the failing worker.

        The cycle holds the scenario's cycle lock, so no dispatched job is reserved.
        """
        if not current.store.durable:
            return
        with self.registry.lock_instance(current) as state:
            if state is InstanceState.CURRENT:
                self.reload(current.name, current)

    def reload(self, scenario: str, current: Scenario | None) -> Scenario:
        """Rebuild durable state, retaining failed instructions that still need skip rows."""
        failures = (
            {bound.component: bound.trainer.instruction_failures() for bound in current.trainers}
            if current is not None
            else {}
        )
        recovered = self.registry.reload(scenario)
        for component, failed in failures.items():
            if failed:
                recovered.trainer_for(component).set_instruction_failures(failed)
        return recovered
