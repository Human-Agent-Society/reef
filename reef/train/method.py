"""The training method of a job: its objective and its learning-rate schedule.

A recipe sets the method in each job's parameters, and every training component reads it.
The Tinker training component also keeps the schedule's progress in its checkpoint manifests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal


LearningRateDecayStyle = Literal["constant", "linear", "cosine"]


LEARNING_RATE_DECAY_STYLES: tuple[LearningRateDecayStyle, ...] = ("constant", "linear", "cosine")


def checked_learning_rate(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


def checked_step_count(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class LearningRateSchedule:
    """Linear warmup to ``peak_learning_rate``, then a constant, linear or cosine decay.

    Step ``s`` is the number of optimizer steps this schedule has already
    completed, so its first step uses ``s = 0``. During warmup
    (``s <= warmup_steps``) the rate rises linearly from
    ``initial_learning_rate`` to the peak. ``decay_style="constant"`` then holds
    the peak; ``"linear"`` and ``"cosine"`` decay it to ``min_learning_rate``
    over the next ``decay_steps`` steps and hold that minimum afterwards. This
    is Megatron's ``OptimizerParamScheduler`` curve counted in optimizer steps,
    so Slime and Tinker train a schedule the same way.

    ``name`` identifies the schedule together with its values. The backend
    keeps the active schedule and its progress with its training state
    (:class:`LearningRateScheduleState`): a job that selects the active
    schedule continues it, a job that selects another one starts that one at
    step 0, and a job that selects none keeps the active one. To start the same
    curve again, give it another ``name``.
    """

    name: str
    peak_learning_rate: float
    warmup_steps: int = 0
    decay_style: LearningRateDecayStyle = "constant"
    decay_steps: int = 0
    min_learning_rate: float = 0.0
    initial_learning_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LearningRateSchedule.name must be a non-empty string")
        peak_learning_rate = checked_learning_rate(self.peak_learning_rate, "LearningRateSchedule.peak_learning_rate")
        if peak_learning_rate <= 0:
            raise ValueError("LearningRateSchedule.peak_learning_rate must be positive")
        min_learning_rate = checked_learning_rate(self.min_learning_rate, "LearningRateSchedule.min_learning_rate")
        initial_learning_rate = checked_learning_rate(
            self.initial_learning_rate, "LearningRateSchedule.initial_learning_rate"
        )
        if min_learning_rate > peak_learning_rate or initial_learning_rate > peak_learning_rate:
            raise ValueError(
                "LearningRateSchedule min_learning_rate and initial_learning_rate must not exceed the peak"
            )
        warmup_steps = checked_step_count(self.warmup_steps, "LearningRateSchedule.warmup_steps")
        decay_steps = checked_step_count(self.decay_steps, "LearningRateSchedule.decay_steps")
        if self.decay_style not in LEARNING_RATE_DECAY_STYLES:
            raise ValueError(
                f"LearningRateSchedule.decay_style must be one of {', '.join(LEARNING_RATE_DECAY_STYLES)}, "
                f"got {self.decay_style!r}"
            )
        if self.decay_style == "constant" and decay_steps > 0:
            raise ValueError("a constant LearningRateSchedule takes no decay_steps")
        if self.decay_style != "constant" and decay_steps <= 0:
            raise ValueError(f"a {self.decay_style} LearningRateSchedule needs positive decay_steps")
        object.__setattr__(self, "peak_learning_rate", peak_learning_rate)
        object.__setattr__(self, "min_learning_rate", min_learning_rate)
        object.__setattr__(self, "initial_learning_rate", initial_learning_rate)
        object.__setattr__(self, "warmup_steps", warmup_steps)
        object.__setattr__(self, "decay_steps", decay_steps)

    def learning_rate(self, completed_steps: int) -> float:
        """The rate of the optimizer step that follows ``completed_steps`` steps of this schedule."""
        if self.warmup_steps and completed_steps <= self.warmup_steps:
            warmup_fraction = completed_steps / self.warmup_steps
            learning_rate_range = self.peak_learning_rate - self.initial_learning_rate
            return self.initial_learning_rate + learning_rate_range * warmup_fraction
        if self.decay_style == "constant":
            return self.peak_learning_rate
        steps_since_warmup = completed_steps - self.warmup_steps
        if steps_since_warmup >= self.decay_steps:
            return self.min_learning_rate
        decay_fraction = steps_since_warmup / self.decay_steps
        if self.decay_style == "linear":
            decay_factor = 1.0 - decay_fraction
        else:
            decay_factor = 0.5 * (math.cos(math.pi * decay_fraction) + 1.0)
        return self.min_learning_rate + decay_factor * (self.peak_learning_rate - self.min_learning_rate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearningRateSchedule:
        if not isinstance(value, Mapping):
            raise ValueError("a learning-rate schedule must be an object")
        return cls(**dict(value))


@dataclass(frozen=True)
class TrainingMethod:
    """The objective one training job trains with, and the learning-rate schedule it selects.

    A recipe selects one for every job, by the kind of the job's samples
    (``WeightTrainingRecipe.parameters_by_kind``); the
    job carries it to the training component beside the job data, and the job's payload
    records it, so it is part of the job's identity and stays fixed across the
    job's retries. ``objective`` is a registered objective name or a dotted
    ``package.module:Objective`` reference; the backend resolves it in its own
    process, so a name another package registers is only known where that
    package was imported. ``learning_rate_schedule`` ``None`` keeps the active
    schedule.
    """

    objective: str
    learning_rate_schedule: LearningRateSchedule | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("TrainingMethod.objective must be a non-empty objective reference")
        if self.learning_rate_schedule is not None and not isinstance(
            self.learning_rate_schedule, LearningRateSchedule
        ):
            raise TypeError("TrainingMethod.learning_rate_schedule must be a LearningRateSchedule or None")

    def to_dict(self) -> dict[str, Any]:
        """The job payload's record of this method; it is part of the job's identity."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingMethod:
        if not isinstance(value, Mapping):
            raise ValueError("a training method must be an object")
        schedule = value.get("learning_rate_schedule")
        return cls(value["objective"], None if schedule is None else LearningRateSchedule.from_dict(schedule))


@dataclass(frozen=True)
class LearningRateScheduleState:
    """The active schedule and the optimizer steps it has completed, kept with the training state."""

    schedule: LearningRateSchedule
    completed_steps: int = 0

    def __post_init__(self) -> None:
        checked_step_count(self.completed_steps, "LearningRateScheduleState.completed_steps")

    def learning_rates(self, optimizer_steps: int) -> tuple[float, ...]:
        """The rates of the next ``optimizer_steps`` steps, in order."""
        return tuple(self.schedule.learning_rate(self.completed_steps + step) for step in range(optimizer_steps))

    def advanced(self, optimizer_steps: int) -> LearningRateScheduleState:
        """This state after ``optimizer_steps`` more steps."""
        return replace(self, completed_steps=self.completed_steps + optimizer_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearningRateScheduleState:
        if not isinstance(value, Mapping):
            raise ValueError("a learning-rate schedule state must be an object")
        return cls(LearningRateSchedule.from_dict(value["schedule"]), value["completed_steps"])


def resolve_learning_rate_schedule_state(
    active: LearningRateScheduleState | None, requested: LearningRateSchedule | None
) -> LearningRateScheduleState | None:
    """The schedule state a job trains with, given the active state and the job's request.

    No request keeps the active state (``None``: the backend's configured
    rate); the active schedule continues; any other schedule starts at step 0.
    """
    if requested is None or (active is not None and active.schedule == requested):
        return active
    return LearningRateScheduleState(requested)


def learning_rate_metrics(
    learning_rates: Sequence[float], schedule_state: LearningRateScheduleState | None
) -> dict[str, Any]:
    """The metrics every backend reports for a job's rate: its last optimizer step's, and the schedule's progress."""
    metrics: dict[str, Any] = {"learning_rate": learning_rates[-1]}
    if schedule_state is not None:
        metrics["learning_rate_schedule"] = {
            "name": schedule_state.schedule.name,
            "completed_steps": schedule_state.completed_steps,
        }
    return metrics
