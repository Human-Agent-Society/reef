"""Ray actor lifecycle and RPC, with no tensor transport or training semantics."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from contextlib import suppress
from time import monotonic
from typing import Any

from reef.runtime.executor.base import ExecutorConfig, ExecutorFuture, SubmittingExecutor, check_rank, resolve_class
from reef.runtime.executor.failure import ExecutorFailedError, ExecutorFailure, ExecutorFailureListener, FailureState
from reef.runtime.executor.ray_runtime import RayRuntimeLease, acquire_ray_runtime

#: How often a waiting future re-checks the executor for a terminal failure.
FAILURE_POLL_S = 0.1


def _require_ray() -> Any:
    import ray

    return ray


class RayExecutorFuture(ExecutorFuture):
    """One or more Ray object references, resolved under a deadline and failure watch."""

    def __init__(
        self,
        references: Any,
        *,
        timeout: float | None = None,
        failure_state: FailureState | None = None,
        rank: int | None = None,
    ) -> None:
        self._references = references
        self._timeout = timeout
        self._failure_state = failure_state
        self._rank = rank

    def result(self, timeout: float | None = None) -> Any:
        ray = _require_ray()
        budget = self._timeout if timeout is None else timeout
        deadline = None if budget is None else monotonic() + budget
        while True:
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            if self._failure_state is not None:
                # Wake up regularly so a failure elsewhere in the group ends the wait.
                self._failure_state.check()
                remaining = FAILURE_POLL_S if remaining is None else min(FAILURE_POLL_S, remaining)
            try:
                return ray.get(self._references, timeout=remaining)
            except ray.exceptions.GetTimeoutError as exc:
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError("executor RPC timed out; worker work may still be running") from exc
            except ray.exceptions.RayActorError as exc:
                if self._failure_state is None:
                    raise
                failure = ExecutorFailure("RayExecutor", f"Ray worker unavailable: {type(exc).__name__}", self._rank)
                self._failure_state.fail(failure)
                raise ExecutorFailedError(self._failure_state.failure or failure) from exc


class RayExecutor(SubmittingExecutor):
    """Launch Ray actors, or attach an existing group without taking ownership.

    Attached executors retain only configuration, actor handles, and lifecycle
    flags so they can travel between Ray processes with their parent objects.
    """

    def _init_executor(self) -> None:
        self._runtime_lease: RayRuntimeLease | None = None
        self._workers: tuple[Any, ...] = ()
        self._owned = True
        self._init_monitor_state()
        if not self.config.workers:
            return
        ray = _require_ray()
        workers = []
        try:
            self._runtime_lease = acquire_ray_runtime()
            for spec in self.config.workers:
                actor_class = ray.remote(resolve_class(spec.worker_cls))
                options = {"max_restarts": 0, "max_task_retries": 0, **self.config.options, **spec.options}
                if options["max_task_retries"] != 0:
                    raise ValueError("RayExecutor requires max_task_retries=0 to avoid replaying mutating RPCs")
                workers.append(actor_class.options(**options).remote(*spec.args, **dict(spec.kwargs)))
            self._workers = tuple(workers)
            # Actor constructors run asynchronously. Readiness makes constructor
            # failures part of launch, so all already-created actors are freed.
            self.check_health(timeout=self.config.launch_timeout_s)
            self._start_monitor()
        except BaseException:
            for worker in workers:
                with suppress(Exception):
                    ray.kill(worker, no_restart=True)
            if self._runtime_lease is not None:
                self._runtime_lease.close()
            raise

    @classmethod
    def from_workers(cls, workers: Sequence[Any], *, owned: bool = False) -> RayExecutor:
        executor = cls(ExecutorConfig(backend=cls))
        executor._workers = tuple(workers)
        executor._owned = owned
        return executor

    @property
    def workers(self) -> tuple[Any, ...]:
        """Actor handles for integrations that must pass them to a Ray library."""
        return self._workers

    # -- Serialization -----------------------------------------------------

    def _init_monitor_state(self) -> None:
        self._monitor_stop = threading.Event()
        self._monitor_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None

    def __getstate__(self):
        # Threads and the runtime lease belong to the process that created them.
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("_monitor_stop", "_monitor_lock", "_monitor_thread", "_runtime_lease")
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._runtime_lease = None
        self._init_monitor_state()

    # -- Failure monitoring ------------------------------------------------

    def register_failure_listener(self, listener: ExecutorFailureListener) -> None:
        super().register_failure_listener(listener)
        self._start_monitor()

    def _start_monitor(self) -> None:
        with self._monitor_lock:
            if self._closed or not self._workers or self._monitor_thread is not None:
                return
            self._monitor_thread = threading.Thread(target=self._monitor_workers, daemon=True, name="reef-ray-monitor")
            self._monitor_thread.start()

    def _monitor_workers(self) -> None:
        ray = _require_ray()
        try:
            pending = {worker.__ray_ready__.remote(): rank for rank, worker in enumerate(self._workers)}
            while not self._monitor_stop.wait(0.2):
                if self.failure is not None:
                    self.shutdown()
                    return
                ready, _ = ray.wait(list(pending), num_returns=len(pending), timeout=0.2)
                for ref in ready:
                    rank = pending.pop(ref)
                    try:
                        ray.get(ref)
                    except Exception as exc:
                        # A queued health RPC timing out is NOT worker death.
                        # Completed health requests only fail on actor/runtime failure.
                        self._fail(f"Ray worker unavailable: {type(exc).__name__}", rank=rank)
                        self.shutdown()
                        return
                    if not self._monitor_stop.is_set():
                        pending[self._workers[rank].__ray_ready__.remote()] = rank
        except Exception as exc:
            if not self._monitor_stop.is_set():
                self._fail(f"Ray monitor unavailable: {type(exc).__name__}")
                self.shutdown()

    # -- RPC ----------------------------------------------------------------

    def _submit(
        self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        check_rank(rank, len(self._workers))
        # Ray's actor-method protocol is dynamic; keep .remote() at this edge.
        reference = getattr(self._workers[rank], method).remote(*args, **kwargs)
        return RayExecutorFuture(reference, timeout=timeout, failure_state=self._failure_state, rank=rank)

    def _submit_all(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        references = [getattr(worker, method).remote(*args, **kwargs) for worker in self._workers]
        return RayExecutorFuture(references, timeout=timeout, failure_state=self._failure_state)

    def check_health(self, timeout: float | None = None) -> None:
        self.collective_rpc("__ray_ready__", timeout=timeout)

    def shutdown(self) -> None:
        with self._monitor_lock:
            if self._closed:
                return
            self._closed = True
            self._monitor_stop.set()
            self._failure_state.close()
        if self._monitor_thread is not None and self._monitor_thread is not threading.current_thread():
            self._monitor_thread.join(timeout=2)
        try:
            if self._owned and self._workers:
                self._kill_workers()
        finally:
            if self._runtime_lease is not None:
                self._runtime_lease.close()

    def _kill_workers(self) -> None:
        ray = _require_ray()
        errors = [error for worker in self._workers if (error := _kill_error(ray, worker)) is not None]
        if errors:
            raise errors[0]


def _kill_error(ray: Any, worker: Any) -> Exception | None:
    try:
        ray.kill(worker, no_restart=True)
    except Exception as exc:
        return exc
    return None
