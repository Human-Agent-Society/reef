"""Backend-neutral output contracts for one prepared training step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class StepScheduling:
    """How one reserved batch is cut into optimizer steps by the training runtime.

    A processor hands the preparer one batch; the runtime may run several
    optimizer steps over it. The *rollout* is the unit that is never split
    across steps — a comparison set by default, or one sample with
    ``unit="sample"``.

    ``batch_size`` is rollouts per optimizer step:

    * ``"configured"`` — the backend's own setting (Slime ``--global-batch-size``);
    * ``"actual"`` — every rollout in the batch, i.e. one step per batch;
    * an ``int`` — an explicit step size, so a batch of 1024 rollouts with
      ``batch_size=64`` trains sixteen steps.

    ``epochs`` repeats the whole batch that many passes; every pass is its own
    run of steps, and the reference / old-policy log-probs are computed once
    before the first, so multi-epoch training is PPO-style off-policy after
    the first pass — pair it with a clipped loss family. ``shuffle`` reorders
    rollouts independently in every epoch, deterministically from the batch id.

    ``remainder`` decides what happens when an epoch's rollout count is not
    a multiple of the step size: ``"partial"`` (default) trains the tail as
    one smaller final step — what SFT trainers do with ``drop_last=False`` —
    ``"drop"`` leaves the tail out, ``"error"`` refuses the batch. A partial
    step needs at least one sample per data-parallel rank and, under static
    micro-batching, a sample count divisible by ``dp_size * micro_batch_size``;
    dynamic batching splits its bins to fit.
    """

    unit: Literal["comparison_set", "sample"] = "comparison_set"
    batch_size: Literal["configured", "actual"] | int = "configured"
    epochs: int = 1
    shuffle: bool = False
    remainder: Literal["partial", "drop", "error"] = "partial"

    def __post_init__(self) -> None:
        if self.unit not in ("comparison_set", "sample"):
            raise ValueError(f"StepScheduling.unit must be 'comparison_set' or 'sample', got {self.unit!r}")
        if isinstance(self.batch_size, bool) or (
            not isinstance(self.batch_size, int) and self.batch_size not in ("configured", "actual")
        ):
            raise ValueError(
                f"StepScheduling.batch_size must be 'configured', 'actual' or a positive int, got {self.batch_size!r}"
            )
        if isinstance(self.batch_size, int) and self.batch_size <= 0:
            raise ValueError(f"StepScheduling.batch_size must be positive, got {self.batch_size}")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"StepScheduling.epochs must be a positive int, got {self.epochs!r}")
        if not isinstance(self.shuffle, bool):
            raise ValueError(f"StepScheduling.shuffle must be a bool, got {self.shuffle!r}")
        if self.remainder not in ("partial", "drop", "error"):
            raise ValueError(f"StepScheduling.remainder must be 'partial', 'drop' or 'error', got {self.remainder!r}")
        if self.remainder == "drop" and not isinstance(self.batch_size, int):
            raise ValueError("StepScheduling.remainder='drop' requires an explicit integer batch_size")


@dataclass(frozen=True, slots=True)
class StepSignal:
    """Algorithm output before any backend's wire payload is materialized."""

    action: Literal["train", "skip"]
    loss_family: str
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    advantages: tuple[float, ...] | None = None
    scheduling: StepScheduling = field(default_factory=StepScheduling)
