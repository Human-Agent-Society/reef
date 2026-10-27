"""How a training component cuts one job's data into optimizer steps.

``StepScheduling`` is a job's optional ``scheduling`` parameter: the passes
over the job data, the packing of its rollouts into steps by token count, and
what happens to a partial last step. The objective's
``rollout_unit`` says which rows one step keeps together. A component expands
both into a row order and a step layout (``schedule_job_data``) and turns that
order into its own wire rows. Reef's control plane reads none of it.
"""

from __future__ import annotations

import bisect
import random
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from reef.core.job_data import JobData, trajectories
from reef.core.trajectories import source_record_id

#: The rows one optimizer step keeps together: each sample alone, or the samples of one comparison set.
RolloutUnit = Literal["sample", "comparison_set"]


@dataclass(frozen=True, slots=True)
class StepScheduling:
    """The passes over one job's data, the packing of its rollouts, and the handling of a partial last step.

    ``epochs`` is a positive int: the job trains all of its data that many
    times, each pass its own run of optimizer steps. Only an objective with
    ``supports_multiple_epochs``, whose loss stays valid when it trains the
    same rollouts again, accepts more than one.
    ``shuffle`` reorders the rollouts in every pass, deterministically from the
    data's rows. ``remainder`` decides what happens when a pass's rollout
    count is not a multiple of the step size: ``"partial"`` trains the tail as
    one smaller last step, ``"drop"`` leaves it out, and ``"error"`` refuses the
    job data. ``packing_length`` is None or a positive int. When it is set,
    the job's rollouts are first packed into packs of at most that many
    tokens, and one optimizer step trains step-size packs instead of
    step-size rollouts; ``shuffle`` and ``remainder`` then apply to packs (see
    ``pack_rollouts``). The packs decide only which rollouts share a step: the
    loss and the micro-batches stay the component's. A job's ``scheduling``
    parameter holds the value as a dict; a training component reads it with
    ``from_parameters``.
    """

    epochs: int = 1
    shuffle: bool = False
    remainder: Literal["partial", "drop", "error"] = "partial"
    packing_length: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"StepScheduling.epochs must be a positive int, got {self.epochs!r}")
        if not isinstance(self.shuffle, bool):
            raise ValueError(f"StepScheduling.shuffle must be a bool, got {self.shuffle!r}")
        if self.remainder not in ("partial", "drop", "error"):
            raise ValueError(f"StepScheduling.remainder must be 'partial', 'drop' or 'error', got {self.remainder!r}")
        if self.packing_length is not None and (
            isinstance(self.packing_length, bool)
            or not isinstance(self.packing_length, int)
            or self.packing_length <= 0
        ):
            raise ValueError(
                f"StepScheduling.packing_length must be None or a positive int, got {self.packing_length!r}"
            )

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> StepScheduling:
        """The ``scheduling`` parameter of a job, or the defaults when the job has none.

        Raise ``ValueError`` for a value that is not an object of the fields above.
        """
        value = parameters.get("scheduling", {})
        if not isinstance(value, Mapping) or set(value) - {"epochs", "shuffle", "remainder", "packing_length"}:
            raise ValueError(
                "the scheduling parameter must be an object of epochs, shuffle, remainder and packing_length"
            )
        return cls(**value)


def rollout_ids(data: JobData, unit: RolloutUnit) -> list[int]:
    """One rollout id per trajectory of ``data``: its own index, or its comparison set's.

    A trajectory without a comparison set is its own rollout under either unit.
    """
    samples = trajectories(data)
    if unit == "sample":
        return list(range(len(samples)))
    groups: dict[tuple[str, str | int], int] = {}
    return [
        groups.setdefault(("group", item.group_id) if item.group_id is not None else ("sample", index), len(groups))
        for index, item in enumerate(samples)
    ]


@dataclass(frozen=True, slots=True)
class MaterializedSchedule:
    """Row order and step layout for one job's data.

    ``row_indices[i]`` is the source row of emitted row ``i``. ``rollout_ids[i]``
    is its step-grouping id: the rollouts are numbered 0, 1, ... in emission
    order, distinct per epoch, so every epoch's copy of a rollout is scheduled
    on its own. ``step_sizes`` is the rollout count of every optimizer step in
    order; it is not empty, each entry is positive, and step ``k`` holds the
    rollouts with ids from ``sum(step_sizes[:k])`` up to
    ``sum(step_sizes[:k + 1])``. Its sum is ``epochs * rollouts_per_epoch``,
    with ``epochs`` and ``rollouts_per_epoch`` positive. ``row_indices`` and
    ``rollout_ids`` have one non-negative entry per emitted row, and
    ``rollout_ids`` starts at 0 and grows by 0 or 1 per row up to the last
    rollout, so every rollout has a row. ``dropped_rollouts`` is the
    non-negative tail that ``remainder="drop"`` left out of each epoch, and
    ``tail_rollouts`` the rollouts of each epoch's partial last step under
    ``remainder="partial"``: the last step's size, or 0 when every step is
    full. The constructor checks these conditions. ``materialize_schedule``
    makes the value; a component turns it into its wire rows.
    """

    row_indices: tuple[int, ...]
    rollout_ids: tuple[int, ...]
    step_sizes: tuple[int, ...]
    epochs: int
    rollouts_per_epoch: int
    dropped_rollouts: int
    tail_rollouts: int

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.rollouts_per_epoch <= 0 or self.dropped_rollouts < 0:
            raise ValueError("a schedule has positive epochs and rollouts per epoch, and non-negative drops")
        if not self.step_sizes or any(size <= 0 for size in self.step_sizes):
            raise ValueError("a schedule has one or more optimizer steps, each of one or more rollouts")
        rollouts = sum(self.step_sizes)
        if rollouts != self.epochs * self.rollouts_per_epoch:
            raise ValueError("a schedule's steps hold every rollout of every epoch")
        if self.tail_rollouts not in (0, self.step_sizes[-1]):
            raise ValueError("a schedule's tail is its last step, or 0")
        ids = self.rollout_ids
        if len(self.row_indices) != len(ids) or any(row < 0 for row in self.row_indices):
            raise ValueError("a schedule has one non-negative source row per rollout id")
        if (
            not ids
            or ids[0] != 0
            or ids[-1] != rollouts - 1
            or any(later - earlier not in (0, 1) for earlier, later in pairwise(ids))
        ):
            raise ValueError("a schedule numbers its rollouts 0, 1, ... in emission order, each with a row")

    @property
    def optimizer_steps(self) -> int:
        """The optimizer steps of the schedule, over every epoch."""
        return len(self.step_sizes)


def pack_rollouts(token_counts: Sequence[int], packing_length: int) -> list[list[int]]:
    """Pack rollouts of ``token_counts`` tokens into packs of at most ``packing_length`` tokens.

    The result lists every rollout index exactly once. Best-fit decreasing
    fills the packs: each rollout, longest first, goes into the pack whose
    free tokens it fills most closely, or into a new pack. A rollout longer
    than ``packing_length`` is a pack of its own. Packs are ordered by their
    first rollout, and the rollouts in a pack keep their order.
    """
    packs: list[list[int]] = []
    # (free tokens, pack index) of the packs that can take more, in ascending order.
    free_tokens: list[tuple[int, int]] = []
    for rollout in sorted(range(len(token_counts)), key=lambda index: (-token_counts[index], index)):
        position = bisect.bisect_left(free_tokens, (token_counts[rollout], -1))
        if position == len(free_tokens):
            packs.append([rollout])
            free, pack = packing_length - token_counts[rollout], len(packs) - 1
        else:
            free, pack = free_tokens.pop(position)
            packs[pack].append(rollout)
            free -= token_counts[rollout]
        if free > 0:
            bisect.insort(free_tokens, (free, pack))
    return sorted((sorted(pack) for pack in packs), key=lambda pack: pack[0])


def schedule_seed(data_id: str) -> int:
    """Deterministic shuffle seed for one job's data, stable across replays."""
    return zlib.crc32(data_id.encode("utf-8"))


def job_data_schedule_seed(data: JobData) -> int:
    """The shuffle seed of one job's data from its rows: a reload numbers the data again, the rows stay the same."""
    ids = [source_record_id(item) for item in trajectories(data)]
    return schedule_seed("|".join(ids) if ids else data.data_id)


def materialize_schedule(
    rollout_ids: Sequence[int],
    scheduling: StepScheduling,
    *,
    step_size: int,
    seed: int = 0,
    token_counts: Sequence[int] | None = None,
) -> MaterializedSchedule:
    """Expand ``scheduling`` over one job's data whose rows carry ``rollout_ids``.

    ``step_size`` is the backend's units per optimizer step: rollouts, or
    packs when ``scheduling.packing_length`` is set. ``token_counts`` has one
    token count per row; packing requires it. Job data with fewer units
    trains them as one partial step under ``remainder="partial"``. Rows
    sharing a rollout id stay contiguous and in their source order. Units
    keep first-occurrence order unless ``scheduling.shuffle``; epochs are laid
    out back to back.
    """
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise ValueError(f"step_size must be a positive int, got {step_size!r}")
    groups: dict[int, list[int]] = {}
    for row, rollout_id in enumerate(rollout_ids):
        groups.setdefault(rollout_id, []).append(row)
    ordered_groups = list(groups.values())
    if not ordered_groups:
        raise ValueError("cannot schedule empty job data")
    if scheduling.packing_length is None:
        unit_name = "rollouts"
        units = [[group] for group in range(len(ordered_groups))]
    else:
        if token_counts is None or len(token_counts) != len(rollout_ids):
            raise ValueError("packing needs the token count of every row")
        unit_name = "packs"
        group_tokens = [sum(token_counts[row] for row in group) for group in ordered_groups]
        units = pack_rollouts(group_tokens, scheduling.packing_length)

    total = len(units)
    dropped = 0
    tail_units = 0
    if total % step_size:
        if scheduling.remainder == "error":
            raise ValueError(
                f"{total} {unit_name} do not form complete optimizer steps of {step_size}; "
                "set the backend's batch size to a divisor, or set the scheduling parameter's remainder to partial or drop"
            )
        if scheduling.remainder == "drop":
            if total < step_size:
                raise ValueError(f"{total} {unit_name} do not form one complete optimizer step of {step_size}")
            dropped = sum(len(unit) for unit in units[total - total % step_size :])
            units = units[: total - total % step_size]
        else:
            tail_units = total % step_size
    kept = sum(len(unit) for unit in units)

    row_indices: list[int] = []
    emitted_ids: list[int] = []
    step_sizes: list[int] = []
    # The rollouts are numbered in emission order, over every epoch.
    next_rollout_id = 0
    for epoch in range(scheduling.epochs):
        order = list(range(len(units)))
        if scheduling.shuffle:
            random.Random(f"{seed}:{epoch}").shuffle(order)
        for first in range(0, len(order), step_size):
            step_groups = [group for unit in order[first : first + step_size] for group in units[unit]]
            for group in step_groups:
                for row in ordered_groups[group]:
                    row_indices.append(row)
                    emitted_ids.append(next_rollout_id)
                next_rollout_id += 1
            step_sizes.append(len(step_groups))

    return MaterializedSchedule(
        row_indices=tuple(row_indices),
        rollout_ids=tuple(emitted_ids),
        step_sizes=tuple(step_sizes),
        epochs=scheduling.epochs,
        rollouts_per_epoch=kept,
        dropped_rollouts=dropped,
        tail_rollouts=step_sizes[-1] if tail_units else 0,
    )


def schedule_job_data(
    data: JobData, unit: RolloutUnit, scheduling: StepScheduling, *, step_size: int
) -> MaterializedSchedule:
    """The row order and step layout of ``data``: its rollouts under ``unit``, expanded by ``scheduling``.

    Packing counts each trajectory's ``training["tokens"]``.
    """
    token_counts = None
    if scheduling.packing_length is not None:
        token_counts = [len(item.training["tokens"]) for item in trajectories(data)]
    return materialize_schedule(
        rollout_ids(data, unit),
        scheduling,
        step_size=step_size,
        seed=job_data_schedule_seed(data),
        token_counts=token_counts,
    )


__all__ = [
    "MaterializedSchedule",
    "RolloutUnit",
    "StepScheduling",
    "job_data_schedule_seed",
    "materialize_schedule",
    "pack_rollouts",
    "rollout_ids",
    "schedule_job_data",
    "schedule_seed",
]
