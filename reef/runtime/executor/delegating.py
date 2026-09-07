"""Delegation for domain launchers that reuse an existing RPC transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, overload

from reef.runtime.executor.base import Executor, ExecutorFuture
from reef.runtime.executor.failure import ExecutorFailure, ExecutorFailureListener


class DelegatingExecutor(Executor):
    """Subclasses launch domain workers and set their transport in _rpc."""

    _rpc: Executor

    @property
    def failure(self) -> ExecutorFailure | None:
        return self._rpc.failure

    def register_failure_listener(self, listener: ExecutorFailureListener) -> None:
        self._rpc.register_failure_listener(listener)

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
        return self._rpc.collective_rpc(method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block)

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
        return self._rpc.rpc(rank, method, args=args, kwargs=kwargs, timeout=timeout, non_block=non_block)

    def check_health(self, timeout: float | None = None) -> None:
        self._rpc.check_health(timeout=timeout)

    def shutdown(self) -> None:
        self._rpc.shutdown()
