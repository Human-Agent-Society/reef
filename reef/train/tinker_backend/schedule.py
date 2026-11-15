"""How the Tinker component cuts one job's data into optimizer steps.

``StepScheduling`` is the passes over the job data, the packing of its rollouts
into steps by token count, and what happens to a partial last step. The
component parses it from its job parameters. The objective's ``rollout_unit``
says which rows one step keeps together. ``schedule_job_data`` expands both
into the source rows of each optimizer step, which ``prepare_tinker_job`` turns
into the rows of a Tinker loss. ``next_steps`` is the step counter that an
objective keeps in its state. Reef's control plane reads none of it.
"""

from __future__ import annotations

import bisect
import random
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from reef.core.job_data import JobData, trajectories
from reef.core.trajectories import source_record_id

#: The rows one optimizer step keeps together: each sample alone, or the samples of one comparison set.
RolloutUnit = Literal["sample", "comparison_set"]


@dataclass(frozen=True, slots=True)
class StepScheduling:
    """The passes over one job's data, the packing of its rollouts, and the handling of a partial last step.

    ``epochs`` is a positive int: the job trains all of its data that many
    times, each pass its own run of optimizer steps.
    ``shuffle`` reorders the rollouts in every pass, deterministically from the
    data's rows. ``remainder`` decides what happens when a pass's rollout
    count is not a multiple of the step size: ``"partial"`` trains the tail as
    one smaller last step, ``"drop"`` leaves it out, and ``"error"`` refuses the
    job data. ``packing_length`` is None or a positive int. When it is set,
    the job's rollouts are first packed into packs of at most that many
    tokens, and one optimizer step trains step-size packs instead of
    step-size rollouts; ``shuffle`` and ``remainder`` then apply to packs (see
    ``pack_rollouts``). ``TinkerJobParameters`` makes the value from a job's
    parameters, and ``schedule_job_data`` reads it.
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


@dataclass(frozen=True, slots=True)
class StepRows:
    """The source rows of every optimizer step of one job, in order.

    ``steps`` holds one or more steps; each is a non-empty tuple of indices into the job's trajectories, the rows
    that the step trains. A row occurs once per epoch that keeps it. ``dropped_rollouts`` is the non-negative
    count of rollouts that ``remainder="drop"`` left out of each epoch. ``schedule_job_data`` makes the value;
    ``prepare_tinker_job`` reads it.
    """

    steps: tuple[tuple[int, ...], ...]
    dropped_rollouts: int

    def __post_init__(self) -> None:
        if not self.steps or any(not step or min(step) < 0 for step in self.steps):
            raise ValueError("a job has one or more optimizer steps, each of one or more non-negative source rows")
        if self.dropped_rollouts < 0:
            raise ValueError("a job drops a non-negative count of rollouts")


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


def schedule_job_data(data: JobData, unit: RolloutUnit, scheduling: StepScheduling, *, step_size: int) -> StepRows:
    """The source rows of each optimizer step of ``data``: its rollouts under ``unit``, expanded by ``scheduling``.

    A rollout is one trajectory, or under ``unit`` ``"comparison_set"`` the trajectories of one comparison set; a
    trajectory without a comparison set is its own rollout. ``step_size`` is the units of one optimizer step:
    rollouts, or packs when ``scheduling.packing_length`` is set, which counts each trajectory's
    ``training["tokens"]``. Job data with fewer units trains them as one partial step under
    ``remainder="partial"``. The rows of a rollout stay together and in their source order. Units keep
    first-occurrence order unless ``scheduling.shuffle``, whose order depends only on the data's rows and the
    epoch. The epochs follow each other. ``ValueError``: ``step_size`` is not a positive int, the data holds no
    trajectory, or ``scheduling.remainder`` refuses the unit count.
    """
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise ValueError(f"step_size must be a positive int, got {step_size!r}")
    samples = trajectories(data)
    if not samples:
        raise ValueError("cannot schedule empty job data")
    rows_by_rollout: dict[tuple[str, str | int], list[int]] = {}
    for row, item in enumerate(samples):
        if unit == "comparison_set" and item.group_id is not None:
            rows_by_rollout.setdefault(("group", item.group_id), []).append(row)
        else:
            rows_by_rollout[("sample", row)] = [row]
    rollouts = list(rows_by_rollout.values())
    if scheduling.packing_length is None:
        unit_name = "rollouts"
        units = [[rollout] for rollout in range(len(rollouts))]
    else:
        unit_name = "packs"
        token_counts = [sum(len(samples[row].training["tokens"]) for row in rollout) for rollout in rollouts]
        units = pack_rollouts(token_counts, scheduling.packing_length)

    total = len(units)
    dropped = 0
    if total % step_size and scheduling.remainder == "error":
        raise ValueError(
            f"{total} {unit_name} do not form complete optimizer steps of {step_size}; "
            "set the backend's batch size to a divisor, or set the scheduling parameter's remainder to partial or drop"
        )
    if total % step_size and scheduling.remainder == "drop":
        if total < step_size:
            raise ValueError(f"{total} {unit_name} do not form one complete optimizer step of {step_size}")
        dropped = sum(len(unit) for unit in units[total - total % step_size :])
        units = units[: total - total % step_size]

    # A reload numbers the data again and the rows stay the same, so the seed reads the rows.
    seed = zlib.crc32("|".join(source_record_id(item) for item in samples).encode("utf-8"))
    steps: list[tuple[int, ...]] = []
    for epoch in range(scheduling.epochs):
        order = list(range(len(units)))
        if scheduling.shuffle:
            random.Random(f"{seed}:{epoch}").shuffle(order)
        steps.extend(
            tuple(
                row
                for unit_index in order[first : first + step_size]
                for rollout in units[unit_index]
                for row in rollouts[rollout]
            )
            for first in range(0, len(order), step_size)
        )
    return StepRows(tuple(steps), dropped)


def next_steps(state: Mapping[str, Any]) -> int:
    """The step counter of an objective's ``state`` plus one. ``ValueError``: it is not a non-negative int."""
    steps = state.get("steps", 0)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("algorithm state 'steps' must be a non-negative integer")
    return steps + 1
