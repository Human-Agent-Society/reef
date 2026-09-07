"""One worker in the caller's process; multiple workers require multiprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from threading import Lock
from time import monotonic
from typing import Any, Literal, overload

from reef.runtime.executor.base import Executor, ExecutorConfig, ExecutorFuture, resolve_class


class ConcurrentExecutorFuture(ExecutorFuture):
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
                future.exception(timeout=None if deadline is None else max(0.0, deadline - monotonic()))
            except FutureTimeoutError as exc:
                raise TimeoutError("executor RPC timed out; worker work may still be running") from exc
            results.append(future.result())
        return results[0] if self._single else results


class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        if len(self.config.workers) > 1:
            raise ValueError("UniProcExecutor accepts at most one worker; use mp for multiple workers")
        self._workers: tuple[Any, ...] = ()
        self._owned = True
        self._closed = False
        self._pools: dict[int, ThreadPoolExecutor] = {}
        self._pool_lock = Lock()
        if self.config.options or any(spec.options for spec in self.config.workers):
            raise ValueError("UniProcExecutor does not accept worker resource or backend options")
        workers: list[Any] = []
        try:
            workers.extend(
                resolve_class(spec.worker_cls)(*spec.args, **dict(spec.kwargs)) for spec in self.config.workers
            )
            self._workers = tuple(workers)
        except BaseException:
            for worker in workers:
                with suppress(Exception):
                    self._shutdown_worker(worker)
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

    def _ensure_open(self) -> None:
        self._failure_state.check()
        if self._closed:
            raise RuntimeError("executor is shut down")

    def _thread_pool(self, rank: int) -> ThreadPoolExecutor:
        with self._pool_lock:
            self._ensure_open()
            if rank not in self._pools:
                self._pools[rank] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"reef-executor-{rank}")
            return self._pools[rank]

    def _call(self, rank, method, args, kwargs):
        self._ensure_open()
        return getattr(self._workers[rank], method)(*args, **dict(kwargs or {}))

    @overload
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: Literal[False] = False,
    ) -> list[Any]: ...

    @overload
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: Literal[True],
    ) -> ExecutorFuture: ...

    @overload
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool,
    ) -> list[Any] | ExecutorFuture: ...

    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool = False,
    ) -> list[Any] | ExecutorFuture:
        self._ensure_open()
        # Match the remote worker RPC contract at this local transport boundary.
        futures = [
            self._failure_state.track(self._thread_pool(rank).submit(self._call, rank, method, args, kwargs))
            for rank in range(len(self._workers))
        ]
        future = ConcurrentExecutorFuture(futures, timeout=timeout)
        return future if non_block else future.result()

    def rpc(
        self,
        rank: int,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool = False,
    ) -> Any:
        self._ensure_open()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank >= len(self._workers):
            raise IndexError(f"worker rank {rank!r} is outside the executor group")
        pending = self._failure_state.track(self._thread_pool(rank).submit(self._call, rank, method, args, kwargs))
        future = ConcurrentExecutorFuture([pending], single=True, timeout=timeout)
        return future if non_block else future.result()

    def check_health(self, timeout: float | None = None) -> None:
        self._ensure_open()

    @staticmethod
    def _shutdown_worker(worker: Any) -> None:
        # Plain CPU worker plugins may provide a shutdown hook; it is optional.
        shutdown = getattr(worker, "shutdown", None)
        if shutdown is not None:
            shutdown()

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
        errors = [error for worker in self._workers if (error := self._try_shutdown_worker(worker)) is not None]
        if errors:
            raise errors[0]

    def _try_shutdown_worker(self, worker: Any) -> Exception | None:
        try:
            self._shutdown_worker(worker)
        except Exception as exc:
            return exc
        return None
