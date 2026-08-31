"""Registries a method fills at import time: step preparers and loss-family references.

The bundled preparers register themselves at import time via the
``@register_step_preparer`` class decorator in their module, each from its
method package (``recipes.tttd.preparer`` and the like, imported by
``reef``). This registry never imports a method package — a method
depends on the machinery, not the other way round.  The table is open for external preparers the same way recipe
kinds and loss families are: an external preparer registers its
:class:`StepPreparer` with :func:`register_preparer` from a module imported at
boot, or is named as a dotted ``"package.module:callable"`` reference wherever
a preparer is named (a recipe's ``training_spec().step_preparer``).
Resolving a dotted reference imports the module and wraps the callable in a
:class:`_CallableStepPreparer` if it is not already a :class:`StepPreparer`.

Loss families register here too, but only as a dotted ``"package.module:SPEC"``
reference (:func:`register_loss_family_ref`): the service process never
loads the Slime side, and the Slime registry
(``slime_backend/loss_families.py``) imports the reference the first time the
name is resolved.
"""

from __future__ import annotations

import importlib

from reef.train.algos.base import StepPreparer, _builtin_preparers, _CallableStepPreparer, register_step_preparer

__all__ = [
    "StepPreparer",
    "loss_family_refs",
    "register_loss_family_ref",
    "register_preparer",
    "register_step_preparer",
    "resolve_preparer",
    "unregister_preparer",
]


class StepPreparerRegistry:
    """Reads bundled preparers from ``_builtin_preparers`` (populated by decorators).

    External preparers are tracked separately so ``unregister`` can refuse
    bundled ones while allowing plugin teardown. The bundled set is read live:
    a method package (``recipes.tttd.preparer``) registers its preparer
    when the recipe package imports it, which may happen after this registry
    is constructed.
    """

    def __init__(self) -> None:
        self._external: dict[str, StepPreparer] = {}

    @property
    def _bundled(self) -> frozenset[str]:
        return frozenset(_builtin_preparers)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(_builtin_preparers) | set(self._external)))

    def register(self, preparer: StepPreparer) -> StepPreparer:
        """Register an external preparer under its ``name``."""
        if not isinstance(preparer, StepPreparer):
            raise TypeError(f"a step preparer registers a StepPreparer, got {type(preparer).__name__}")
        name = preparer.name
        existing = self._external.get(name)
        if existing is preparer:
            return preparer
        bundled = _builtin_preparers.get(name)
        if bundled is preparer:
            return preparer
        if bundled is not None or existing is not None:
            available = ", ".join(self.names)
            raise ValueError(f"step preparer {name!r} is already registered; registered preparers: {available}")
        self._external[name] = preparer
        return preparer

    def unregister(self, name: str) -> None:
        """Remove a previously registered external preparer (test/plugin teardown)."""
        if name in self._bundled:
            raise ValueError(f"step preparer {name!r} is bundled and cannot be unregistered")
        if name not in self._external:
            registered = ", ".join(sorted(self._external)) or "none"
            raise KeyError(f"step preparer {name!r} is not registered; registered external preparers: {registered}")
        self._external.pop(name)

    def resolve(self, preparer_id: str) -> StepPreparer:
        """Resolve a registered preparer name or a dotted ``"module:callable"`` reference."""
        preparer = _builtin_preparers.get(preparer_id) or self._external.get(preparer_id)
        if preparer is not None:
            return preparer
        if ":" in preparer_id:
            return self._resolve_dotted(preparer_id)
        available = ", ".join(self.names)
        raise ValueError(f"unknown step preparer {preparer_id!r}; available preparers: {available}")

    def _resolve_dotted(self, reference: str) -> StepPreparer:
        """Import a dotted preparer reference, wrapping plain callables."""
        module_name, separator, attribute = reference.partition(":")
        if not separator or not module_name or not attribute:
            available = ", ".join(self.names)
            raise ValueError(f"unknown step preparer {reference!r}; available preparers: {available}")
        candidate = getattr(importlib.import_module(module_name), attribute)
        if isinstance(candidate, StepPreparer):
            return candidate
        if callable(candidate):
            return _CallableStepPreparer(candidate)
        raise TypeError(f"step preparer {reference!r} is not callable")


PREPARERS = StepPreparerRegistry()


def register_preparer(preparer: StepPreparer) -> StepPreparer:
    """Register an external preparer (module-level convenience)."""
    return PREPARERS.register(preparer)


def unregister_preparer(name: str) -> None:
    """Remove an external preparer registered by :func:`register_preparer`."""
    PREPARERS.unregister(name)


def resolve_preparer(preparer_id: str) -> StepPreparer:
    """Resolve a registered preparer name or a dotted ``"module:callable"`` reference."""
    return PREPARERS.resolve(preparer_id)


_loss_family_refs: dict[str, str] = {}


def register_loss_family_ref(loss_family: str, reference: str) -> None:
    """Record that ``loss_family`` is implemented by the dotted ``reference``."""
    if ":" not in reference:
        raise ValueError(f"loss family reference {reference!r} must be 'package.module:SPEC'")
    existing = _loss_family_refs.get(loss_family)
    if existing is not None and existing != reference:
        raise ValueError(f"loss family {loss_family!r} is already registered to {existing!r}")
    _loss_family_refs[loss_family] = reference


def loss_family_refs() -> dict[str, str]:
    """The registered ``{loss_family: reference}`` table (a copy)."""
    return dict(_loss_family_refs)
