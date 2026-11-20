"""Tinker's loss families: their registration and their resolution.

A loss family is a :class:`~reef.train.tinker_backend.losses.TinkerLoss` with a
``loss_family`` name. A family registers with :func:`register_loss_family`: the
decorator of its class, in the module that defines it. A method package defines
its family beside its objective, so the import of the objective's module
registers the family too; the package's public half registers only the
objective, by reference (``reef.train.tinker_backend.objectives``). The
built-in families are ``importance_sampling`` and ``sft``. This module imports
no Tinker SDK.
"""

from __future__ import annotations

from typing import overload

from reef.train.tinker_backend.losses import CrossEntropyLoss, ImportanceSamplingLoss, TinkerLoss

#: Tinker's loss families by name, the one table of this backend's families. It holds the built-in families from
#: the start; ``register_loss_family`` adds a method's family when the method's module is imported.
LOSS_FAMILIES: dict[str, TinkerLoss] = {
    loss.loss_family: loss for loss in (ImportanceSamplingLoss(), CrossEntropyLoss())
}


@overload
def register_loss_family(family: type[TinkerLoss]) -> type[TinkerLoss]: ...


@overload
def register_loss_family(family: TinkerLoss) -> TinkerLoss: ...


def register_loss_family(family: TinkerLoss | type[TinkerLoss]) -> TinkerLoss | type[TinkerLoss]:
    """Register a family under its ``loss_family`` name: decorate its zero-argument class, or pass an instance.

    Registering the same instance twice changes nothing. ``TypeError``: the value is no ``TinkerLoss``.
    ``ValueError``: the family declares no name, or its name holds another family.
    """
    if isinstance(family, type):
        if not issubclass(family, TinkerLoss):
            raise TypeError(f"a loss family registers a TinkerLoss, got {family.__name__}")
        loss = family()
    elif isinstance(family, TinkerLoss):
        loss = family
    else:
        raise TypeError(f"a loss family registers a TinkerLoss, got {type(family).__name__}")
    name = loss.loss_family
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a loss family must declare a non-empty loss_family")
    existing = LOSS_FAMILIES.setdefault(name, loss)
    if existing is not loss:
        raise ValueError(f"Tinker loss family {name!r} is already registered")
    return family


def unregister_loss_family(loss_family: str) -> None:
    """Remove a loss family, for extension teardown. ``KeyError``: no family has the name."""
    if LOSS_FAMILIES.pop(loss_family, None) is None:
        raise KeyError(f"Tinker loss family {loss_family!r} is not registered")


def resolve_loss_family(loss_family: str) -> TinkerLoss:
    """The family of a registered name. ``ValueError``: no family has the name in this process."""
    loss = LOSS_FAMILIES.get(loss_family)
    if loss is None:
        raise ValueError(
            f"unsupported Tinker loss family {loss_family!r}; registered: {', '.join(sorted(LOSS_FAMILIES))}. "
            "A family registers with register_loss_family() in the module that its objective names."
        )
    return loss
