"""In-process CPU executor for ordinary workers, tests, and custom backends.

This executor uses threads to fan out RPC; it is not a multiprocessing launcher
and must not be used to create multiple distributed GPU ranks in one process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from threading import Lock
from time import monotonic
from typing import Any, Literal, overload

from reef.runtime.executor.base import Executor, ExecutorConfig, ExecutorFuture, resolve_class


class LocalExecutorFuture(ExecutorFuture):
    def __init__(self, futures: Sequence[Future], *, single: bool = False, timeout: float | None = None) -> None:
        self._futures = tuple(futures)
        self._single = single
        self._timeout = timeout

    def result(self, timeout: float | None = None) -> Any:
        budget = self._timeout if timeout is None else timeout
        deadline = None if budget is None else monotonic() + budget
        results = [
            future.result(timeout=None if deadline is None else max(0.0, deadline - monotonic()))
            for future in self._futures
        ]
        return results[0] if self._single else results


class LocalExecutor(Executor):
    def _init_executor(self) -> None:
        self._workers: tuple[Any, ...] = ()
        self._owned = True
        self._closed = False
        self._pools: dict[int, ThreadPoolExecutor] = {}
        self._pool_lock = Lock()
        if self.config.options or any(spec.options for spec in self.config.workers):
            raise ValueError("LocalExecutor does not accept worker resource or backend options")
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
    def from_workers(cls, workers: Sequence[Any], *, owned: bool = False) -> LocalExecutor:
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
        future = LocalExecutorFuture(futures, timeout=timeout)
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
        future = LocalExecutorFuture([pending], single=True, timeout=timeout)
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
