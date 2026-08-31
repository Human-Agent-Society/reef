"""Internal base and decorator registration for step preparers.

Subclass :class:`StepPreparer`, set the class attribute ``name``, implement
:meth:`StepPreparer.__call__`, and register with ``@register_step_preparer``::

    @register_step_preparer
    class MyPreparer(StepPreparer):
        name = "my-preparer"

        def __call__(self, batch, state):
            ...

Public callers import the signal contract from :mod:`reef.train.algos`,
optional helpers from :mod:`reef.train.algos.helpers`, and registered-preparer
APIs from :mod:`reef.train.algos.registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch


# ---------------------------------------------------------------------------
# StepPreparer base class
# ---------------------------------------------------------------------------


class StepPreparer(ABC):
    """Base class for backend-agnostic step preparers.

    Required (subclass must set):
        ``name`` — canonical preparer name used by recipes and the registry.

    Implement:
        ``__call__`` — turn a reserved batch and algorithm state into a
        :class:`StepSignal`.

    Register a preparer with the ``@register_step_preparer`` class decorator;
    callers may alternatively register an instance at runtime via
    :func:`reef.train.algos.registry.register_preparer` or named as a dotted
    ``"module:callable"`` reference resolved by
    :func:`reef.train.algos.registry.resolve_preparer`.
    """

    name: str

    @abstractmethod
    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        """Return the training signal for one reserved batch."""


class _CallableStepPreparer(StepPreparer):
    """Adapter wrapping a plain callable as a :class:`StepPreparer` instance.

    Used when a dotted ``"module:callable"`` reference resolves to a function
    (or any callable) rather than a ``StepPreparer`` subclass instance, so
    :func:`resolve_preparer` always returns a uniform type.
    """

    name = ""

    def __init__(self, fn: Callable[[TrainingBatch, Mapping[str, Any]], StepSignal]) -> None:
        self._fn = fn

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        return self._fn(batch, state)


# ---------------------------------------------------------------------------
# decorator registration
# ---------------------------------------------------------------------------

_decorated_preparers: dict[str, StepPreparer] = {}
"""Preparers registered at import time by :func:`register_step_preparer`.

Read by :class:`~reef.train.algos.registry.StepPreparerRegistry`; instances
may also be registered at runtime through
:func:`~reef.train.algos.registry.register_preparer`.
"""


def register_step_preparer(cls: type[StepPreparer]) -> type[StepPreparer]:
    """Class decorator: instantiate and register a step preparer.

    Each preparer module applies this to its ``StepPreparer`` subclass so
    the registry is populated at import time — no explicit dict entry.
    """
    instance = cls()
    name = instance.name
    if name in _decorated_preparers:
        raise ValueError(f"step preparer {name!r} is already registered")
    _decorated_preparers[name] = instance
    return cls
