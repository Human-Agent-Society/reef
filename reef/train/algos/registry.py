"""Method-owned objectives and lazy references to their backend loss implementations.

Recipes select an objective by registered name or dotted class/instance reference.
Only the selected method is imported; core never imports cookbook packages.
Backend loss references stay lazy so service-side validation needs no GPU stack.
"""

from __future__ import annotations

import importlib
from typing import TypeVar

from reef.train.algos.objective import TrainingObjective

ObjectiveRegistration = TypeVar("ObjectiveRegistration", bound=TrainingObjective | type[TrainingObjective])


class ObjectiveRegistry:
    """Resolve explicit objective implementations by name or dotted reference."""

    def __init__(self) -> None:
        self.objectives: dict[str, TrainingObjective] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.objectives))

    def register(self, objective: TrainingObjective) -> TrainingObjective:
        if not isinstance(objective, TrainingObjective):
            raise TypeError("an objective must inherit TrainingObjective")
        if not isinstance(objective.name, str) or not objective.name.strip():
            raise ValueError("an objective must declare a non-empty name")
        if not isinstance(objective.loss_family, str) or not objective.loss_family.strip():
            raise ValueError("an objective must declare a non-empty loss_family")
        if not isinstance(objective.supports_multiple_epochs, bool):
            raise ValueError("an objective must declare supports_multiple_epochs as a bool")
        existing = self.objectives.get(objective.name)
        if existing is objective:
            return objective
        if existing is not None:
            raise ValueError(f"objective {objective.name!r} is already registered")
        self.objectives[objective.name] = objective
        return objective

    def unregister(self, name: str) -> None:
        if self.objectives.pop(name, None) is None:
            raise KeyError(f"objective {name!r} is not registered")

    def resolve(self, reference: str) -> TrainingObjective:
        objective = self.objectives.get(reference)
        if objective is not None:
            return objective
        if ":" not in reference:
            raise ValueError(f"unknown objective {reference!r}; available objectives: {', '.join(self.names)}")
        module_name, _, attribute = reference.partition(":")
        if not module_name or not attribute:
            raise ValueError(f"objective reference {reference!r} must be 'package.module:Objective'")
        candidate = vars(importlib.import_module(module_name)).get(attribute)
        if isinstance(candidate, type) and issubclass(candidate, TrainingObjective):
            registered = self.objectives.get(candidate.name)
            if registered is not None and type(registered) is candidate:
                return registered
            candidate = candidate()
        if not isinstance(candidate, TrainingObjective):
            raise TypeError(f"objective {reference!r} must name a TrainingObjective class or instance")
        return self.register(candidate)


OBJECTIVES = ObjectiveRegistry()


def register_objective(value: ObjectiveRegistration) -> ObjectiveRegistration:
    """Register an instance, or decorate a zero-argument objective class."""
    if isinstance(value, type):
        if not issubclass(value, TrainingObjective):
            raise TypeError("an objective must inherit TrainingObjective")
        OBJECTIVES.register(value())
    else:
        OBJECTIVES.register(value)
    return value


def unregister_objective(name: str) -> None:
    """Remove an objective registration, for extension teardown."""
    OBJECTIVES.unregister(name)


def resolve_objective(reference: str) -> TrainingObjective:
    """Resolve a registered name or a dotted objective class/instance reference."""
    return OBJECTIVES.resolve(reference)


#: ``{backend: {loss_family: reference}}``; Slime is the default backend.
_loss_family_refs: dict[str, dict[str, str]] = {}
_DEFAULT_BACKEND = "slime"


def register_loss_family_ref(loss_family: str, reference: str, *, backend: str = _DEFAULT_BACKEND) -> None:
    """Record that ``backend`` implements ``loss_family`` at the dotted ``reference``.

    A method registers one reference per backend it supports; the same family
    name selects each backend's implementation.
    """
    if ":" not in reference:
        raise ValueError(f"loss family reference {reference!r} must be 'package.module:SPEC'")
    if not backend:
        raise ValueError("loss family references need a non-empty backend name")
    table = _loss_family_refs.setdefault(backend, {})
    existing = table.get(loss_family)
    if existing is not None and existing != reference:
        raise ValueError(f"loss family {loss_family!r} is already registered to {existing!r} for {backend}")
    table[loss_family] = reference


def loss_family_refs(backend: str = _DEFAULT_BACKEND) -> dict[str, str]:
    """The registered ``{loss_family: reference}`` table of one backend (a copy)."""
    return dict(_loss_family_refs.get(backend, {}))


def _unregister_loss_family_ref(loss_family: str, *, backend: str = _DEFAULT_BACKEND) -> bool:
    """Remove a registered family reference, returning whether it existed."""
    return _loss_family_refs.get(backend, {}).pop(loss_family, None) is not None
