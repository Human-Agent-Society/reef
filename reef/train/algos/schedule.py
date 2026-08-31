"""Materialize a :class:`StepScheduling` into a concrete row order.

Backend-neutral: the input is one rollout id per batch row, the output is the
row order the backend should train in, the rollout id of every emitted row,
and the rollouts-per-step layout when the scheduling fixes one. A backend
turns the row order into its own wire rows; Slime's ``_build_payload`` is the
one caller today.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from reef.train.algos.signals import StepScheduling


@dataclass(frozen=True, slots=True)
class MaterializedSchedule:
    """Row order and step layout for one batch.

    ``row_indices[i]`` is the source row of emitted row ``i``; ``rollout_ids[i]``
    its step-grouping id, distinct per epoch so every epoch's copy of a rollout
    is scheduled on its own. ``step_sizes`` is the rollout count of every
    optimizer step in order (a partial tail step is simply smaller), or
    ``None`` when the backend's configured step size applies.
    """

    row_indices: tuple[int, ...]
    rollout_ids: tuple[int, ...]
    step_sizes: tuple[int, ...] | None
    epochs: int
    rollouts_per_epoch: int
    dropped_rollouts: int

    @property
    def optimizer_steps(self) -> int | None:
        """Steps the schedule fixes, or ``None`` when the backend decides."""
        return None if self.step_sizes is None else len(self.step_sizes)

    @property
    def tail_rollouts(self) -> int:
        """Rollouts in the partial final step of each epoch, 0 when every step is full."""
        if not self.step_sizes:
            return 0
        return self.step_sizes[-1] if self.step_sizes[-1] != self.step_sizes[0] else 0


def schedule_seed(batch_id: str) -> int:
    """Deterministic shuffle seed for one batch, stable across replays."""
    return zlib.crc32(batch_id.encode("utf-8"))


def materialize_schedule(
    rollout_ids: Sequence[int],
    scheduling: StepScheduling,
    *,
    seed: int = 0,
) -> MaterializedSchedule:
    """Expand ``scheduling`` over one batch whose rows carry ``rollout_ids``.

    Rows sharing a rollout id stay contiguous and in their source order.
    Rollouts keep first-occurrence order unless ``scheduling.shuffle``; epochs
    are laid out back to back.
    """
    groups: dict[int, list[int]] = {}
    for row, rollout_id in enumerate(rollout_ids):
        groups.setdefault(rollout_id, []).append(row)
    ordered_groups = list(groups.values())
    total = len(ordered_groups)
    if total == 0:
        raise ValueError("cannot schedule an empty batch")

    if scheduling.batch_size == "configured":
        step_size: int | None = None
    elif scheduling.batch_size == "actual":
        step_size = total
    else:
        step_size = scheduling.batch_size
        if step_size > total:
            raise ValueError(f"StepScheduling.batch_size={step_size} exceeds the {total} rollouts in the batch")

    dropped = 0
    tail = 0
    if step_size is not None and total % step_size:
        if scheduling.remainder == "error":
            raise ValueError(
                f"{total} rollouts do not form complete optimizer steps of {step_size}; "
                "size the processor batch as a multiple, or set StepScheduling(remainder='partial'|'drop')"
            )
        if scheduling.remainder == "drop":
            dropped = total % step_size
            ordered_groups = ordered_groups[: total - dropped]
        else:
            tail = total % step_size
    kept = len(ordered_groups)
    if step_size is None:
        step_sizes: tuple[int, ...] | None = None
    else:
        epoch_steps = [step_size] * (kept // step_size) + ([tail] if tail else [])
        step_sizes = tuple(epoch_steps * scheduling.epochs)

    row_indices: list[int] = []
    emitted_ids: list[int] = []
    for epoch in range(scheduling.epochs):
        order = list(range(kept))
        if scheduling.shuffle:
            random.Random(f"{seed}:{epoch}").shuffle(order)
        for position, group_index in enumerate(order):
            rollout_id = epoch * kept + position
            for row in ordered_groups[group_index]:
                row_indices.append(row)
                emitted_ids.append(rollout_id)

    return MaterializedSchedule(
        row_indices=tuple(row_indices),
        rollout_ids=tuple(emitted_ids),
        step_sizes=step_sizes,
        epochs=scheduling.epochs,
        rollouts_per_epoch=kept,
        dropped_rollouts=dropped,
    )


__all__ = ["MaterializedSchedule", "materialize_schedule", "schedule_seed"]
