"""Slime's training objectives: what a job's ``objective`` parameter names.

A ``SlimeObjective`` owns a method's data preparation and its choice of Slime
loss family. A job names one by a registered name or by a dotted
``package.module:Objective`` reference. Slime's built-in objectives are ``sft``
and ``slime_dataset``. A method package
registers its own by reference in its public half, with
``register_objective(name, "package.module:Objective")``, and defines it in a
module that only this backend imports. This module imports no torch, Ray or Slime, so a
service validates a job without the GPU stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.job_data import JobData, trajectories
from reef.core.plugins import import_plugin_class
from reef.train.slime_backend.schedule import RolloutUnit, StepScheduling, next_steps
from reef.train.slime_backend.signals import TrainingSignal


class SlimeObjective(ABC):
    """A method's data preparation and its Slime loss family.

    Values: a subclass declares a non-empty ``name`` and ``loss_family``, the
    name of a loss family that ``resolve_loss_family`` resolves.
    ``supports_multiple_epochs`` is whether the loss stays valid when a job
    trains its data more than once. Old-policy log-probs are computed once
    before the first pass, so later passes are off-policy: a clipped PPO-style
    ratio tolerates that, an unclipped one does not. ``rollout_unit`` is the
    rows that one optimizer step keeps together: ``"sample"`` or
    ``"comparison_set"``.

    Purpose: ``prepare_slime_job`` calls ``prepare`` with a job's complete data, so
    group-relative statistics do not depend on the data-parallel partitioning.
    ``prepare`` returns the proposed objective state; the component commits it
    with the job's last checkpoint. Model-dependent terms stay in the loss family.
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


#: Slime's objectives by name, the one table of this backend's objectives. A value is the objective, or the
#: ``package.module:Objective`` reference of one that this process did not import yet: ``resolve_objective`` replaces
#: the reference with the objective that it names.
OBJECTIVES: dict[str, SlimeObjective | str] = {}


def register_objective(name: str, objective: SlimeObjective | str) -> None:
    """Register ``objective`` under ``name``: an objective, or the ``package.module:Objective`` reference of one.

    A reference is not imported here, so a method package registers its objective without importing its code.
    Registering the same value twice changes nothing. ``TypeError``: the value is neither. ``ValueError``: the
    reference is malformed, the objective does not declare ``name`` with a loss family, or ``name`` holds
    another value.
    """
    if isinstance(objective, str):
        module_name, _, attribute = objective.partition(":")
        if not module_name or not attribute:
            raise ValueError(f"objective reference {objective!r} must be 'package.module:Objective'")
    elif not isinstance(objective, SlimeObjective):
        raise TypeError("a Slime objective must inherit SlimeObjective")
    else:
        if not isinstance(objective.name, str) or not objective.name.strip():
            raise ValueError("an objective must declare a non-empty name")
        if not isinstance(objective.loss_family, str) or not objective.loss_family.strip():
            raise ValueError("an objective must declare a non-empty loss_family")
        if not isinstance(objective.supports_multiple_epochs, bool):
            raise ValueError("an objective must declare supports_multiple_epochs as a bool")
        if objective.name != name:
            raise ValueError(f"objective {objective.name!r} cannot register under the name {name!r}")
    existing = OBJECTIVES.get(name)
    if existing is None:
        OBJECTIVES[name] = objective
    elif existing is not objective and existing != objective:
        raise ValueError(f"Slime objective {name!r} is already registered")


def unregister_objective(name: str) -> None:
    """Remove an objective registration, for extension teardown. ``KeyError``: no objective has ``name``."""
    if OBJECTIVES.pop(name, None) is None:
        raise KeyError(f"Slime objective {name!r} is not registered")


def resolve_objective(reference: str) -> SlimeObjective:
    """The objective of a registered name, or of a dotted ``package.module:Objective`` class reference.

    The first resolution of a name that holds a reference imports the reference
    (``reef.core.plugins.import_plugin_class``) and builds the class without arguments. An unregistered dotted
    reference registers its objective under the objective's name. ``ValueError``: the name is unknown, or the
    objective's name holds another objective. ``PluginConfigError``, a ``ValueError``: the reference is malformed,
    its module does not import, or it names no concrete ``SlimeObjective`` class.
    """
    registered = OBJECTIVES.get(reference)
    if isinstance(registered, SlimeObjective):
        return registered
    if registered is None and ":" not in reference:
        raise ValueError(
            f"unknown Slime objective {reference!r}; available objectives: {', '.join(sorted(OBJECTIVES))}"
        )
    dotted = reference if registered is None else registered
    objective_class = import_plugin_class(
        dotted,
        SlimeObjective,  # type: ignore[type-abstract]  # the base is a value; the result is concrete
        setting="Slime objective",
    )
    candidate = objective_class()
    if registered is None:
        # A job named the objective by its reference: the name may already hold this class.
        named = resolve_objective(candidate.name) if candidate.name in OBJECTIVES else None
        if named is not None and type(named) is type(candidate):
            return named
        register_objective(candidate.name, candidate)
    else:
        # The loaded objective takes the place of its reference, unless it breaks the objective contract.
        OBJECTIVES.pop(reference)
        try:
            register_objective(reference, candidate)
        except ValueError:
            OBJECTIVES[reference] = registered
            raise
    return candidate


class SftObjective(SlimeObjective):
    """Train all samples in the data with the same weight. A sample has no advantage."""

    name = "sft"
    loss_family = "sft"
    # The target is fixed and the loss reads no old-policy log probs. Thus each pass trains the same loss.
    supports_multiple_epochs = True

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(trajectories(data))})


register_objective("sft", SftObjective())


class SlimeDatasetObjective(SlimeObjective):
    """Leave a dataset job's or a task job's advantages and loss to Slime (the ``slime`` loss family)."""

    name = "slime_dataset"
    loss_family = "slime"

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        raise RuntimeError("a dataset job trains on Slime's own rollouts; Reef prepares no training signal for it")


register_objective("slime_dataset", SlimeDatasetObjective())
