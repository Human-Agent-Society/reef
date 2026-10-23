"""Materialize a :class:`StepScheduling` into a concrete row order.

Backend-neutral: the input is one rollout id per batch row and the backend's
step size, the output is the row order the backend should train in, the
rollout id of every emitted row, and the rollouts-per-step layout. A backend
turns the row order into its own wire rows.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from reef.core.batches import StepScheduling, TrainingBatch, trajectories
from reef.core.trajectories import source_record_id


@dataclass(frozen=True, slots=True)
class MaterializedSchedule:
    """Row order and step layout for one batch.

    ``row_indices[i]`` is the source row of emitted row ``i``; ``rollout_ids[i]``
    its step-grouping id, distinct per epoch so every epoch's copy of a rollout
    is scheduled on its own. ``step_sizes`` is the rollout count of every
    optimizer step in order (a partial tail step is simply smaller); it is not
    empty, and ``sum(step_sizes) == epochs * rollouts_per_epoch``.
    ``dropped_rollouts`` is the tail that ``remainder="drop"`` left out of
    each epoch.
    """

    row_indices: tuple[int, ...]
    rollout_ids: tuple[int, ...]
    step_sizes: tuple[int, ...]
    epochs: int
    rollouts_per_epoch: int
    dropped_rollouts: int

    @property
    def optimizer_steps(self) -> int:
        """The optimizer steps of the schedule, over every epoch."""
        return len(self.step_sizes)

    @property
    def tail_rollouts(self) -> int:
        """Rollouts in the partial final step of each epoch, 0 when every step is full."""
        return self.step_sizes[-1] if self.step_sizes[-1] != self.step_sizes[0] else 0


def schedule_seed(batch_id: str) -> int:
    """Deterministic shuffle seed for one batch, stable across replays."""
    return zlib.crc32(batch_id.encode("utf-8"))


def batch_schedule_seed(batch: TrainingBatch) -> int:
    """The shuffle seed of one batch from its rows: a reload numbers batches again, the rows stay the same."""
    ids = [source_record_id(item) for item in trajectories(batch)]
    return schedule_seed("|".join(ids) if ids else batch.batch_id)


def materialize_schedule(
    rollout_ids: Sequence[int],
    scheduling: StepScheduling,
    *,
    step_size: int,
    seed: int = 0,
) -> MaterializedSchedule:
    """Expand ``scheduling`` over one batch whose rows carry ``rollout_ids``.

    ``step_size`` is the backend's rollouts per optimizer step. A batch with
    fewer rollouts trains them as one partial step under
    ``remainder="partial"``. Rows sharing a rollout id stay contiguous and in
    their source order. Rollouts keep first-occurrence order unless
    ``scheduling.shuffle``; epochs are laid out back to back.
    """
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise ValueError(f"step_size must be a positive int, got {step_size!r}")
    groups: dict[int, list[int]] = {}
    for row, rollout_id in enumerate(rollout_ids):
        groups.setdefault(rollout_id, []).append(row)
    ordered_groups = list(groups.values())
    total = len(ordered_groups)
    if total == 0:
        raise ValueError("cannot schedule an empty batch")

    dropped = 0
    tail = 0
    if total % step_size:
        if scheduling.remainder == "error":
            raise ValueError(
                f"{total} rollouts do not form complete optimizer steps of {step_size}; "
                "set the backend's batch size to a divisor, or set StepScheduling(remainder='partial'|'drop')"
            )
        if scheduling.remainder == "drop":
            dropped = total % step_size
            if dropped == total:
                raise ValueError(f"{total} rollouts do not form one complete optimizer step of {step_size}")
            ordered_groups = ordered_groups[: total - dropped]
        else:
            tail = total % step_size
    kept = len(ordered_groups)
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
