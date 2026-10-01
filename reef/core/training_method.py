"""The training method of one job: its objective and, optionally, a learning-rate schedule.

A recipe selects a :class:`TrainingMethod` for every training job; the runtime
carries it to the backend beside the batch, and the job's payload records it,
so the selection is part of the job's identity and stays fixed across its
retries. The backend resolves the objective and applies the schedule.

A :class:`LearningRateSchedule` counts optimizer steps, including every
update inside one job. The backend keeps the active schedule and the steps it
has completed with its training state (:class:`LearningRateScheduleState`):
a job that names the active schedule continues it, a job that names another
one starts that one at step 0, and a job that names none keeps the active
schedule, or the backend's configured learning rate when none was ever
selected. A retry or a restart therefore never repeats a warmup; to start the
same curve again, give it another ``name``. Starting a schedule does not
reset optimizer moments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal

LearningRateDecayStyle = Literal["constant", "linear", "cosine"]
LEARNING_RATE_DECAY_STYLES: tuple[LearningRateDecayStyle, ...] = ("constant", "linear", "cosine")


def checked_learning_rate(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"LearningRateSchedule.{name} must be a finite number >= 0, got {value!r}")
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

    ``name`` identifies the schedule together with its values: a job that
    selects the schedule already active continues it.
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
        peak = checked_learning_rate(self.peak_learning_rate, "peak_learning_rate")
        if peak <= 0:
            raise ValueError("LearningRateSchedule.peak_learning_rate must be positive")
        minimum = checked_learning_rate(self.min_learning_rate, "min_learning_rate")
        initial = checked_learning_rate(self.initial_learning_rate, "initial_learning_rate")
        if minimum > peak or initial > peak:
            raise ValueError(
                "LearningRateSchedule min_learning_rate and initial_learning_rate must not exceed the peak"
            )
        warmup = checked_step_count(self.warmup_steps, "LearningRateSchedule.warmup_steps")
        decay = checked_step_count(self.decay_steps, "LearningRateSchedule.decay_steps")
        if self.decay_style not in LEARNING_RATE_DECAY_STYLES:
            raise ValueError(
                f"LearningRateSchedule.decay_style must be one of {', '.join(LEARNING_RATE_DECAY_STYLES)}, "
                f"got {self.decay_style!r}"
            )
        if self.decay_style == "constant" and decay:
            raise ValueError("a constant LearningRateSchedule takes no decay_steps")
        if self.decay_style != "constant" and decay <= 0:
            raise ValueError(f"a {self.decay_style} LearningRateSchedule needs positive decay_steps")
        object.__setattr__(self, "peak_learning_rate", peak)
        object.__setattr__(self, "min_learning_rate", minimum)
        object.__setattr__(self, "initial_learning_rate", initial)
        object.__setattr__(self, "warmup_steps", warmup)
        object.__setattr__(self, "decay_steps", decay)

    def learning_rate(self, step: int) -> float:
        """The rate of the optimizer step that follows ``step`` completed steps of this schedule."""
        if self.warmup_steps and step <= self.warmup_steps:
            fraction = step / self.warmup_steps
            return self.initial_learning_rate + (self.peak_learning_rate - self.initial_learning_rate) * fraction
        if self.decay_style == "constant":
            return self.peak_learning_rate
        decayed = step - self.warmup_steps
        if decayed >= self.decay_steps:
            return self.min_learning_rate
        ratio = decayed / self.decay_steps
        coefficient = 1.0 - ratio if self.decay_style == "linear" else 0.5 * (math.cos(math.pi * ratio) + 1.0)
        return self.min_learning_rate + coefficient * (self.peak_learning_rate - self.min_learning_rate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearningRateSchedule:
        if not isinstance(value, Mapping):
            raise ValueError("a learning-rate schedule must be an object")
        return cls(**dict(value))


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


def resolve_learning_rate_schedule(
    active: LearningRateScheduleState | None, requested: LearningRateSchedule | None
) -> LearningRateScheduleState | None:
    """The schedule state a job trains with, given the active state and the job's request.

    No request keeps the active state (``None``: the backend's configured
    rate); the active schedule continues; any other schedule starts at step 0.
    """
    if requested is None or (active is not None and active.schedule == requested):
        return active
    return LearningRateScheduleState(requested)


@dataclass(frozen=True)
class TrainingMethod:
    """The objective one training job trains with, and the learning-rate schedule it selects.

    ``objective`` is a registered objective name or a dotted
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
        """The job payload's record of this method; it takes part in the job's identity."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingMethod:
        if not isinstance(value, Mapping):
            raise ValueError("a training method must be an object")
        schedule = value.get("learning_rate_schedule")
        return cls(value["objective"], None if schedule is None else LearningRateSchedule.from_dict(schedule))


__all__ = [
    "LEARNING_RATE_DECAY_STYLES",
    "LearningRateDecayStyle",
    "LearningRateSchedule",
    "LearningRateScheduleState",
    "TrainingMethod",
    "resolve_learning_rate_schedule",
]
