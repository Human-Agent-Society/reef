"""Delegation for domain launchers that reuse an existing RPC transport."""

from __future__ import annotations

from typing import Any

from reef.runtime.executor.base import Executor, ExecutorFuture, SubmittingExecutor
from reef.runtime.executor.failure import ExecutorFailure, ExecutorFailureListener


class DelegatingExecutor(SubmittingExecutor):
    """Subclasses launch domain workers and set their transport in ``_rpc``.

    Every RPC is forwarded to that transport, which validates ranks and
    tracks failure; this class only adds the domain launch and release.
    """

    _rpc: Executor

    @property
    def failure(self) -> ExecutorFailure | None:
        return self._rpc.failure

    def register_failure_listener(self, listener: ExecutorFailureListener) -> None:
        self._rpc.register_failure_listener(listener)

    def _submit(
        self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        return self._rpc.rpc(rank, method, args=args, kwargs=kwargs, timeout=timeout, non_block=True)

    def _submit_all(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        return self._rpc.collective_rpc(method, args=args, kwargs=kwargs, timeout=timeout, non_block=True)

    def check_health(self, timeout: float | None = None) -> None:
        self._rpc.check_health(timeout=timeout)

    def shutdown(self) -> None:
        self._rpc.shutdown()
