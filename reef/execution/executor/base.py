"""The executor contract shared by training and serving integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any, Literal, overload

from reef.core.plugins import import_plugin_class, resolve_plugin_class
from reef.execution.executor.config import ExecutorSettings, select_worker_executor
from reef.execution.executor.failure import ExecutorFailure, ExecutorFailureListener, FailureState
from reef.execution.executor.requirements import ExecutionRequirements

#: The bundled executors by their short config names, as references so that no backend is imported here.
BUILTIN_BACKENDS = {
    "uni": "reef.execution.executor.uniproc:UniProcExecutor",
    "mp": "reef.execution.executor.multiproc:MultiprocExecutor",
    "ray": "reef.execution.executor.ray:RayExecutor",
}


@dataclass(frozen=True)
class WorkerSpec:
    """One worker in rank order, including backend-specific launch options."""

    worker_cls: type | str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.worker_cls, (str, type)) or not self.worker_cls:
            raise TypeError("worker_cls must be a class or a non-empty 'package.module:ClassName' reference")
        if not isinstance(self.args, Sequence) or isinstance(self.args, (str, bytes)):
            raise TypeError("worker args must be a sequence")
        if not isinstance(self.kwargs, Mapping) or not isinstance(self.options, Mapping):
            raise TypeError("worker kwargs and options must be mappings")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class ExecutorLaunch:
    """What one executor launches: its backend and its workers, which define rank order.

    Values: ``backend`` is an ``Executor`` class or a non-empty built-in name or import path; ``workers`` is a
    tuple of ``WorkerSpec``; ``options`` is a mapping; ``launch_timeout_s`` is positive or ``None``; ``node_id``
    is a non-empty string or ``None``. Construction refuses other values with ``ValueError`` and copies
    ``workers`` and ``options``.
    Purpose: the code that needs a worker group makes it, from ``ExecutorSettings`` where a deployment selects
    the backend; ``Executor.create`` and the executor read it.

    ``options`` contains backend-specific defaults. For Ray these are actor
    options, overridden per worker by ``WorkerSpec.options``. ``node_id``
    names the cluster node every worker must be placed on, in the backend's
    own node identity; executors without cluster placement reject it.
    """

    backend: str | type[Executor] = "auto"
    workers: tuple[WorkerSpec, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    launch_timeout_s: float | None = 300.0
    node_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, (str, type)) or not self.backend:
            raise ValueError("executor backend must be a class, a built-in name or 'package.module:ClassName'")
        if not isinstance(self.workers, Sequence) or isinstance(self.workers, (str, bytes)):
            raise ValueError("executor workers must be a sequence of WorkerSpec objects")
        if any(not isinstance(worker, WorkerSpec) for worker in self.workers):
            raise ValueError("executor workers must contain WorkerSpec objects")
        if not isinstance(self.options, Mapping):
            raise ValueError("executor options must be a mapping")
        if self.launch_timeout_s is not None and self.launch_timeout_s <= 0:
            raise ValueError("executor launch_timeout_s must be positive or None")
        if self.node_id is not None and (not isinstance(self.node_id, str) or not self.node_id):
            raise ValueError("executor node_id must be a non-empty string or None")
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
        return value.result(timeout=remaining_time(deadline))
    if isinstance(value, list):
        return [_resolve(item, deadline) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve(item, deadline) for item in value)
    return value


def remaining_time(deadline: float | None) -> float | None:
    """Seconds left until a ``monotonic()`` deadline, clamped at zero; ``None`` never expires."""
    return None if deadline is None else max(0.0, deadline - monotonic())


def check_rank(rank: Any, count: int) -> None:
    """Reject a rank that does not address one of ``count`` workers."""
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < count:
        raise IndexError(f"worker rank {rank!r} is outside the executor group")


def resolve_class(value: type | str) -> type:
    """The worker class that ``value`` is, or that it names as ``package.module:ClassName``.

    Raise ``PluginConfigError`` for a string that names no class.
    """
    if isinstance(value, type):
        return value
    return import_plugin_class(value, object, setting="worker class")


class Executor(ABC):
    """Launch and control an ordered worker group, independently of its transport.

    Collective RPC dispatches to every rank before waiting. It preserves rank
    order and must never retry mutating operations automatically. This interface
    does not promise live changes to the model's distributed process groups.
    """

    def __init__(self, config: ExecutorLaunch) -> None:
        self.config = config
        self._failure_state = FailureState()
        self._closed = False
        self._init_executor()

    @property
    def failure(self) -> ExecutorFailure | None:
        return self._failure_state.failure

    def register_failure_listener(self, listener: ExecutorFailureListener) -> None:
        """Register an observer; late registration reports an existing failure."""
        self._failure_state.register(listener)

    def _fail(self, reason: str, *, rank: int | None = None) -> bool:
        return self._failure_state.fail(ExecutorFailure(type(self).__name__, reason, rank))

    def _ensure_open(self) -> None:
        """Reject work after a terminal failure or shutdown."""
        self._failure_state.check()
        if self._closed:
            raise RuntimeError("executor is shut down")

    @staticmethod
    def get_class(backend: str | type[Executor]) -> type[Executor]:
        """The executor class that ``backend`` is, or that it names: a built-in name or ``package.module:ClassName``.

        Raise ``TypeError`` for a class that is not an ``Executor`` subclass, and ``PluginConfigError`` for a name
        that selects no such class.
        """
        if isinstance(backend, type):
            if not issubclass(backend, Executor):
                raise TypeError("executor backend must be an Executor subclass")
            return backend
        return resolve_plugin_class(
            backend,
            builtins=BUILTIN_BACKENDS,
            base=Executor,  # type: ignore[type-abstract]  # the base is a value; the result is concrete
            setting="executor backend",
        )

    @classmethod
    def create(cls, config: ExecutorLaunch) -> Executor:
        if config.backend == "auto":
            options = bool(config.options or any(worker.options for worker in config.workers))
            selected = select_worker_executor(
                ExecutorSettings(), ExecutionRequirements(workers=max(1, len(config.workers)), cluster=options)
            )
            config = replace(config, backend=selected.settings.backend)
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


class SubmittingExecutor(Executor):
    """An executor whose transport submits one RPC and hands back a future.

    The bundled executors differ only in how a call reaches a worker. They
    implement :meth:`_submit` and :meth:`_submit_all`; the blocking and
    non-blocking public forms and the open check are shared here so each
    transport states only its own mechanics.
    """

    @abstractmethod
    def _submit(
        self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        """Dispatch ``method`` to one rank; the future waits at most ``timeout`` by default."""

    @abstractmethod
    def _submit_all(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        """Dispatch ``method`` to every rank; results keep rank order."""

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
        future = self._submit_all(method, tuple(args), dict(kwargs or {}), timeout)
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
        future = self._submit(rank, method, tuple(args), dict(kwargs or {}), timeout)
        return future if non_block else future.result()
