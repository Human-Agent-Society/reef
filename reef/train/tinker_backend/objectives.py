"""Tinker's training objectives: what a job's ``objective`` parameter names.

A ``TinkerObjective`` owns a method's data preparation and its choice of Tinker
loss family. A job names one by a registered name or by a dotted
``package.module:Objective`` reference. Tinker's built-in objective is ``sft``. A method package refers to its own in its public half
(``reef.train.algos.registry.register_objective_ref``). This module imports no
Tinker SDK.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, overload

from reef.core.job_data import JobData, trajectories
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import objective_refs
from reef.train.algos.schedule import RolloutUnit, StepScheduling
from reef.train.algos.signals import TrainingSignal


class TinkerObjective(ABC):
    """A method's data preparation and its Tinker loss family.

    Values: a subclass declares a non-empty ``name`` and ``loss_family``, the
    name of a loss family that ``resolve_tinker_loss`` resolves.
    ``supports_multiple_epochs`` is whether the loss stays valid when a job
    trains its data more than once. ``rollout_unit`` is the rows that one
    optimizer step keeps together: ``"sample"`` or ``"comparison_set"``.

    Purpose: ``prepare_tinker_job`` calls ``prepare`` with a job's complete
    data. ``prepare`` returns the proposed objective state; the component
    commits it with the job's last checkpoint.
    """

    name: str = ""
    loss_family: str = ""
    supports_multiple_epochs: bool = False
    rollout_unit: RolloutUnit = "sample"

    def validate_scheduling(self, scheduling: StepScheduling) -> None:
        """Raise ``ValueError`` for a job schedule whose passes this objective's loss cannot train correctly."""
        if scheduling.epochs > 1 and not self.supports_multiple_epochs:
            raise ValueError(
                f"objective {self.name!r} does not support {scheduling.epochs} epochs per job: "
                "its loss is not clipped for off-policy passes"
            )

    @abstractmethod
    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        """Prepare a job's full data without mutating it or committing objective state."""


#: Tinker's objectives by name. ``register_tinker_objective`` and ``resolve_tinker_objective`` fill it.
TINKER_OBJECTIVES: dict[str, TinkerObjective] = {}


def add_tinker_objective(objective: TinkerObjective) -> TinkerObjective:
    """Register ``objective`` under its name; registering it twice changes nothing. ``ValueError``: another holds it."""
    if not isinstance(objective, TinkerObjective):
        raise TypeError("a Tinker objective must inherit TinkerObjective")
    if not isinstance(objective.name, str) or not objective.name.strip():
        raise ValueError("an objective must declare a non-empty name")
    if not isinstance(objective.loss_family, str) or not objective.loss_family.strip():
        raise ValueError("an objective must declare a non-empty loss_family")
    if not isinstance(objective.supports_multiple_epochs, bool):
        raise ValueError("an objective must declare supports_multiple_epochs as a bool")
    existing = TINKER_OBJECTIVES.get(objective.name)
    if existing is objective:
        return objective
    if existing is not None:
        raise ValueError(f"Tinker objective {objective.name!r} is already registered")
    TINKER_OBJECTIVES[objective.name] = objective
    return objective


@overload
def register_tinker_objective(value: type[TinkerObjective]) -> type[TinkerObjective]: ...


@overload
def register_tinker_objective(value: TinkerObjective) -> TinkerObjective: ...


def register_tinker_objective(
    value: TinkerObjective | type[TinkerObjective],
) -> TinkerObjective | type[TinkerObjective]:
    """Register an instance, or decorate a zero-argument objective class."""
    if isinstance(value, type):
        if not issubclass(value, TinkerObjective):
            raise TypeError("a Tinker objective must inherit TinkerObjective")
        add_tinker_objective(value())
    else:
        add_tinker_objective(value)
    return value


def unregister_tinker_objective(name: str) -> None:
    """Remove an objective registration, for extension teardown. ``KeyError``: no objective has ``name``."""
    if TINKER_OBJECTIVES.pop(name, None) is None:
        raise KeyError(f"Tinker objective {name!r} is not registered")


def resolve_tinker_objective(reference: str) -> TinkerObjective:
    """The objective of a registered name, a name that a method package refers to (``objective_refs``), or a dotted
    ``package.module:Objective`` class or instance reference.

    ``ValueError`` for an unknown name, a malformed reference, a module that does not import, or anything else that
    the reference names.
    """
    objective = TINKER_OBJECTIVES.get(reference)
    if objective is not None:
        return objective
    references = objective_refs("tinker")
    dotted = references.get(reference, reference)
    if ":" not in dotted:
        names = ", ".join(sorted({*TINKER_OBJECTIVES, *references}))
        raise ValueError(f"unknown Tinker objective {reference!r}; available objectives: {names}")
    module_name, _, attribute = dotted.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"objective reference {reference!r} must be 'package.module:Objective'")
    try:
        candidate = vars(importlib.import_module(module_name)).get(attribute)
    except ImportError as exc:
        raise ValueError(f"cannot import objective {dotted!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, TinkerObjective):
        registered = TINKER_OBJECTIVES.get(candidate.name)
        if registered is not None and type(registered) is candidate:
            return registered
        candidate = candidate()
    if not isinstance(candidate, TinkerObjective):
        raise ValueError(f"objective {reference!r} must name a TinkerObjective class or instance")
    return add_tinker_objective(candidate)


@register_tinker_objective
class SftObjective(TinkerObjective):
    """Train all samples in the data with the same weight, on Tinker's ``cross_entropy``."""

    name = "sft"
    loss_family = "sft"
    # The target is fixed and the loss reads no old-policy log probs. Thus each pass trains the same loss.
    supports_multiple_epochs = True

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(trajectories(data))})
