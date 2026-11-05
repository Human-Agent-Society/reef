"""Slime's training objectives: what a job's ``objective`` parameter names.

A ``SlimeObjective`` owns a method's data preparation and its choice of Slime
loss family. A job names one by a registered name or by a dotted
``package.module:Objective`` reference. Slime's built-in objectives are ``sft``
and ``slime_dataset``. A method package refers to its own in its
public half (``reef.train.algos.registry.register_objective_ref``) and defines it
in its ``slime`` subpackage. This module imports no torch, Ray or Slime, so a
service validates a job without the GPU stack.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, overload

from reef.core.job_data import JobData, trajectories
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import objective_refs, register_loss_family_ref
from reef.train.algos.schedule import RolloutUnit, StepScheduling
from reef.train.algos.signals import TrainingSignal


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

    Purpose: ``SlimeJobRunner`` calls ``prepare`` with a job's complete data, so
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


#: Slime's objectives by name. ``register_slime_objective`` and ``resolve_slime_objective`` fill it.
SLIME_OBJECTIVES: dict[str, SlimeObjective] = {}


def add_slime_objective(objective: SlimeObjective) -> SlimeObjective:
    """Register ``objective`` under its name; registering it twice changes nothing. ``ValueError``: another holds it."""
    if not isinstance(objective, SlimeObjective):
        raise TypeError("a Slime objective must inherit SlimeObjective")
    if not isinstance(objective.name, str) or not objective.name.strip():
        raise ValueError("an objective must declare a non-empty name")
    if not isinstance(objective.loss_family, str) or not objective.loss_family.strip():
        raise ValueError("an objective must declare a non-empty loss_family")
    if not isinstance(objective.supports_multiple_epochs, bool):
        raise ValueError("an objective must declare supports_multiple_epochs as a bool")
    existing = SLIME_OBJECTIVES.get(objective.name)
    if existing is objective:
        return objective
    if existing is not None:
        raise ValueError(f"Slime objective {objective.name!r} is already registered")
    SLIME_OBJECTIVES[objective.name] = objective
    return objective


@overload
def register_slime_objective(value: type[SlimeObjective]) -> type[SlimeObjective]: ...


@overload
def register_slime_objective(value: SlimeObjective) -> SlimeObjective: ...


def register_slime_objective(value: SlimeObjective | type[SlimeObjective]) -> SlimeObjective | type[SlimeObjective]:
    """Register an instance, or decorate a zero-argument objective class."""
    if isinstance(value, type):
        if not issubclass(value, SlimeObjective):
            raise TypeError("a Slime objective must inherit SlimeObjective")
        add_slime_objective(value())
    else:
        add_slime_objective(value)
    return value


def unregister_slime_objective(name: str) -> None:
    """Remove an objective registration, for extension teardown. ``KeyError``: no objective has ``name``."""
    if SLIME_OBJECTIVES.pop(name, None) is None:
        raise KeyError(f"Slime objective {name!r} is not registered")


def resolve_slime_objective(reference: str) -> SlimeObjective:
    """The objective of a registered name, a name that a method package refers to (``objective_refs``), or a dotted
    ``package.module:Objective`` class or instance reference.

    ``ValueError`` for an unknown name, a malformed reference, a module that does not import, or anything else that
    the reference names.
    """
    objective = SLIME_OBJECTIVES.get(reference)
    if objective is not None:
        return objective
    references = objective_refs("slime")
    dotted = references.get(reference, reference)
    if ":" not in dotted:
        names = ", ".join(sorted({*SLIME_OBJECTIVES, *references}))
        raise ValueError(f"unknown Slime objective {reference!r}; available objectives: {names}")
    module_name, _, attribute = dotted.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"objective reference {reference!r} must be 'package.module:Objective'")
    try:
        candidate = vars(importlib.import_module(module_name)).get(attribute)
    except ImportError as exc:
        raise ValueError(f"cannot import objective {dotted!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, SlimeObjective):
        registered = SLIME_OBJECTIVES.get(candidate.name)
        if registered is not None and type(registered) is candidate:
            return registered
        candidate = candidate()
    if not isinstance(candidate, SlimeObjective):
        raise ValueError(f"objective {reference!r} must name a SlimeObjective class or instance")
    return add_slime_objective(candidate)


@register_slime_objective
class SftObjective(SlimeObjective):
    """Train all samples in the data with the same weight. A sample has no advantage."""

    name = "sft"
    loss_family = "sft"
    # The target is fixed and the loss reads no old-policy log probs. Thus each pass trains the same loss.
    supports_multiple_epochs = True

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(trajectories(data))})


@register_slime_objective
class SlimeDatasetObjective(SlimeObjective):
    """Leave a dataset job's or a task job's advantages and loss to Slime (the ``slime`` loss family)."""

    name = "slime_dataset"
    loss_family = "slime"

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        raise RuntimeError("a dataset job trains on Slime's own rollouts; Reef prepares no training signal for it")


# By reference: only the training driver and the workers import the Slime code. No built-in objective trains
# ``importance_sampling``: a method that a deployment provides can.
register_loss_family_ref("sft", "reef.train.slime_backend.sft:SftAlgorithm")
register_loss_family_ref(
    "importance_sampling", "reef.train.slime_backend.importance_sampling:ImportanceSamplingAlgorithm"
)
register_loss_family_ref("slime", "reef.train.slime_backend.native:SlimeNativeAlgorithm")
