"""The executor contract shared by training and serving integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from time import monotonic
from typing import Any, Literal, overload

from reef.runtime.executor.failure import ExecutorFailure, ExecutorFailureListener, FailureState


@dataclass(frozen=True)
class WorkerSpec:
    """One worker in rank order, including backend-specific launch options."""

    worker_cls: type | str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.worker_cls, (str, type)) or not self.worker_cls:
            raise TypeError("worker_cls must be a class or a non-empty Python import path")
        if not isinstance(self.args, Sequence) or isinstance(self.args, (str, bytes)):
            raise TypeError("worker args must be a sequence")
        if not isinstance(self.kwargs, Mapping) or not isinstance(self.options, Mapping):
            raise TypeError("worker kwargs and options must be mappings")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class ExecutorConfig:
    """Worker topology and executor selection; workers define rank order.

    ``options`` contains backend-specific defaults. For Ray these are actor
    options, overridden per worker by ``WorkerSpec.options``.
    """

    backend: str | type[Executor] = "ray"
    workers: tuple[WorkerSpec, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    launch_timeout_s: float | None = 300.0

    def __post_init__(self) -> None:
        if not isinstance(self.backend, (str, type)) or not self.backend:
            raise TypeError("executor backend must be a class or a non-empty name/import path")
        if not isinstance(self.workers, Sequence) or isinstance(self.workers, (str, bytes)):
            raise TypeError("executor workers must be a sequence of WorkerSpec objects")
        if any(not isinstance(worker, WorkerSpec) for worker in self.workers):
            raise TypeError("executor workers must contain WorkerSpec objects")
        if not isinstance(self.options, Mapping):
            raise TypeError("executor options must be a mapping")
        if self.launch_timeout_s is not None and self.launch_timeout_s <= 0:
            raise ValueError("executor launch_timeout_s must be positive or None")
        object.__setattr__(self, "workers", tuple(self.workers))
        object.__setattr__(self, "options", dict(self.options))


class ExecutorFuture(ABC):
    """An outstanding RPC whose result never exposes a backend object reference.

    A timeout bounds waiting only: it neither cancels nor retries worker work.
    """

    @abstractmethod
    def result(self, timeout: float | None = None) -> Any:
        """Wait for the result, preserving any exception raised by the worker."""


def resolve(value: Any, *, timeout: float | None = None) -> Any:
    """Resolve executor futures in a list/tuple, preserving ordinary values.

    Raw backend references are deliberately opaque; only the executor that
    created an RPC knows whether its result is a reference or ordinary data.
    """
    deadline = None if timeout is None else monotonic() + timeout
    return _resolve(value, deadline)


def _resolve(value: Any, deadline: float | None) -> Any:
    if isinstance(value, ExecutorFuture):
        return value.result(timeout=None if deadline is None else max(0.0, deadline - monotonic()))
    if isinstance(value, list):
        return [_resolve(item, deadline) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve(item, deadline) for item in value)
    return value


def resolve_class(value: type | str) -> type:
    """Resolve a plugin class at the import boundary, without loading backends."""
    if isinstance(value, type):
        return value
    if not isinstance(value, str) or not value:
        raise TypeError("expected a class or a non-empty Python import path")
    module_name, separator, class_name = value.partition(":")
    if not separator:
        module_name, separator, class_name = value.rpartition(".")
    if not module_name or not separator or not class_name:
        raise ValueError(f"invalid class import path {value!r}; use 'module:Class' or 'module.Class'")
    # Plugin import paths necessarily cross a dynamic boundary. Worker RPC
    # dispatch is similarly confined to the concrete executor implementations.
    candidate = getattr(import_module(module_name), class_name)
    if not isinstance(candidate, type):
        raise TypeError(f"{value!r} does not name a class")
    return candidate


class Executor(ABC):
    """Launch and control an ordered worker group, independently of its transport.

    Collective RPC dispatches to every rank before waiting. It preserves rank
    order and must never retry mutating operations automatically. This interface
    does not promise live changes to the model's distributed process groups.
    """

    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config
        self._failure_state = FailureState()
        self._init_executor()

    @property
    def failure(self) -> ExecutorFailure | None:
        return self._failure_state.failure

    def register_failure_listener(self, listener: ExecutorFailureListener) -> None:
        """Register an observer; late registration reports an existing failure."""
        self._failure_state.register(listener)

    def _fail(self, reason: str, *, rank: int | None = None) -> bool:
        return self._failure_state.fail(ExecutorFailure(type(self).__name__, reason, rank))

    @staticmethod
    def get_class(backend: str | type[Executor]) -> type[Executor]:
        if backend == "uni":
            from reef.runtime.executor.uniproc import UniProcExecutor

            return UniProcExecutor
        if backend == "mp":
            from reef.runtime.executor.multiproc import MultiprocExecutor

            return MultiprocExecutor
        if backend == "ray":
            from reef.runtime.executor.ray import RayExecutor

            return RayExecutor
        if backend == "local":
            from reef.runtime.executor.local import LocalExecutor

            return LocalExecutor
        candidate = resolve_class(backend)
        if not issubclass(candidate, Executor):
            raise TypeError("executor backend must be an Executor subclass")
        return candidate

    @classmethod
    def create(cls, config: ExecutorConfig) -> Executor:
        return cls.get_class(config.backend)(config)

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize the worker group from ``self.config``."""

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

    @abstractmethod
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool = False,
    ) -> list[Any] | ExecutorFuture:
        """Invoke a method on all workers, returning results in rank order."""

    @abstractmethod
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
        """Invoke a method on one rank, optionally returning an executor future."""

    @abstractmethod
    def check_health(self, timeout: float | None = None) -> None:
        """Raise when the executor or a worker is unavailable."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release owned workers; never terminate workers attached as borrowed."""

    def reinitialize_distributed(self, topology: Mapping[str, Any]) -> None:
        """Optional capability: ordinary executors do not imply elastic training."""
        raise NotImplementedError(f"{type(self).__name__} does not support distributed reinitialization")
