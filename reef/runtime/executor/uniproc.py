"""One worker in the caller's process; multiple workers require multiprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from threading import Lock
from time import monotonic
from typing import Any

from reef.runtime.executor.base import (
    ExecutorConfig,
    ExecutorFuture,
    SubmittingExecutor,
    check_rank,
    remaining_time,
    resolve_class,
)


class ConcurrentExecutorFuture(ExecutorFuture):
    """Rank-ordered results of ``concurrent.futures`` submissions, under one deadline."""

    def __init__(self, futures: Sequence[Future], *, single: bool = False, timeout: float | None = None) -> None:
        self._futures = tuple(futures)
        self._single = single
        self._timeout = timeout

    def result(self, timeout: float | None = None) -> Any:
        budget = self._timeout if timeout is None else timeout
        deadline = None if budget is None else monotonic() + budget
        results = []
        for future in self._futures:
            try:
                # Wait without raising a worker's own exception, so only a
                # transport timeout is normalized across Python 3.10/3.11+.
                future.exception(timeout=remaining_time(deadline))
            except FutureTimeoutError as exc:
                raise TimeoutError("executor RPC timed out; worker work may still be running") from exc
            results.append(future.result())
        return results[0] if self._single else results


def shutdown_worker(worker: Any) -> None:
    """Call a worker's optional ``shutdown`` hook."""
    shutdown = getattr(worker, "shutdown", None)
    if shutdown is not None:
        shutdown()


def _shutdown_error(worker: Any) -> Exception | None:
    try:
        shutdown_worker(worker)
    except Exception as exc:
        return exc
    return None


class UniProcExecutor(SubmittingExecutor):
    """Run at most one worker in this process, each RPC on a dedicated thread."""

    def _init_executor(self) -> None:
        if len(self.config.workers) > 1:
            raise ValueError("UniProcExecutor accepts at most one worker; use mp for multiple workers")
        if self.config.options or any(spec.options for spec in self.config.workers):
            raise ValueError("UniProcExecutor does not accept worker resource or backend options")
        if self.config.node_id is not None:
            raise ValueError("UniProcExecutor runs in this process and cannot place its worker on a cluster node")
        self._workers: tuple[Any, ...] = ()
        self._owned = True
        self._pools: dict[int, ThreadPoolExecutor] = {}
        self._pool_lock = Lock()
        workers: list[Any] = []
        try:
            workers.extend(
                resolve_class(spec.worker_cls)(*spec.args, **dict(spec.kwargs)) for spec in self.config.workers
            )
            self._workers = tuple(workers)
        except BaseException:
            for worker in workers:
                with suppress(Exception):
                    shutdown_worker(worker)
            raise

    @classmethod
    def from_workers(cls, workers: Sequence[Any], *, owned: bool = False) -> UniProcExecutor:
        if len(workers) > 1:
            raise ValueError("UniProcExecutor accepts at most one worker; use mp for multiple workers")
        executor = cls(ExecutorConfig(backend=cls))
        executor._workers = tuple(workers)
        executor._owned = owned
        return executor

    @property
    def workers(self) -> tuple[Any, ...]:
        return self._workers

    def _thread_pool(self, rank: int) -> ThreadPoolExecutor:
        with self._pool_lock:
            self._ensure_open()
            if rank not in self._pools:
                self._pools[rank] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"reef-executor-{rank}")
            return self._pools[rank]

    def _call(self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._ensure_open()
        return getattr(self._workers[rank], method)(*args, **kwargs)

    def _pending(self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Future:
        return self._failure_state.track(self._thread_pool(rank).submit(self._call, rank, method, args, kwargs))

    def _submit(
        self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        check_rank(rank, len(self._workers))
        return ConcurrentExecutorFuture([self._pending(rank, method, args, kwargs)], single=True, timeout=timeout)

    def _submit_all(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        # Match the remote worker RPC contract at this local transport boundary.
        pending = [self._pending(rank, method, args, kwargs) for rank in range(len(self._workers))]
        return ConcurrentExecutorFuture(pending, timeout=timeout)

    def check_health(self, timeout: float | None = None) -> None:
        self._ensure_open()

    def shutdown(self) -> None:
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
        self._failure_state.close()
        for pool in self._pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
        if not self._owned:
            return
        errors = [error for worker in self._workers if (error := _shutdown_error(worker)) is not None]
        if errors:
            raise errors[0]
