"""Backend-neutral schedule materialization: rollouts → ordered rows and steps."""

from __future__ import annotations

import pytest

from reef.train.algos import StepScheduling
from reef.train.algos.schedule import materialize_schedule, schedule_seed


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"batch_size": 0}, "must be positive"),
        ({"batch_size": True}, "'configured', 'actual' or a positive int"),
        ({"batch_size": "all"}, "'configured', 'actual' or a positive int"),
        ({"epochs": 0}, "epochs must be a positive int"),
        ({"shuffle": 1}, "shuffle must be a bool"),
        ({"remainder": "pad"}, "remainder must be"),
        ({"remainder": "drop"}, "requires an explicit integer batch_size"),
        ({"unit": "group"}, "unit must be"),
    ],
)
def test_step_scheduling_validates_fields(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        StepScheduling(**kwargs)


@pytest.mark.unit
def test_configured_schedule_is_identity() -> None:
    schedule = materialize_schedule([0, 0, 1, 2], StepScheduling())

    assert schedule.row_indices == (0, 1, 2, 3)
    assert schedule.rollout_ids == (0, 0, 1, 2)
    assert schedule.step_sizes is None
    assert schedule.optimizer_steps is None
    assert schedule.rollouts_per_epoch == 3


@pytest.mark.unit
def test_actual_schedule_fixes_one_step_per_epoch() -> None:
    schedule = materialize_schedule([0, 0, 1], StepScheduling(batch_size="actual", epochs=2))

    assert schedule.step_sizes == (2, 2)
    assert schedule.optimizer_steps == 2
    assert schedule.row_indices == (0, 1, 2, 0, 1, 2)
    assert schedule.rollout_ids == (0, 0, 1, 2, 2, 3)


@pytest.mark.unit
def test_explicit_batch_size_counts_rollouts_not_rows() -> None:
    schedule = materialize_schedule([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], StepScheduling(batch_size=2))

    assert schedule.step_sizes == (2, 2)
    assert schedule.dropped_rollouts == 0


@pytest.mark.unit
def test_remainder_drop_keeps_leading_full_steps() -> None:
    schedule = materialize_schedule([5, 5, 7, 9, 11, 13], StepScheduling(batch_size=2, remainder="drop"))

    assert schedule.row_indices == (0, 1, 2, 3, 4)
    assert schedule.rollout_ids == (0, 0, 1, 2, 3)
    assert schedule.dropped_rollouts == 1
    assert schedule.step_sizes == (2, 2)


@pytest.mark.unit
def test_remainder_partial_is_default_and_ends_every_epoch_with_a_smaller_step() -> None:
    schedule = materialize_schedule([5, 5, 7, 9, 11, 13], StepScheduling(batch_size=2, epochs=2))

    assert schedule.dropped_rollouts == 0
    assert schedule.step_sizes == (2, 2, 1, 2, 2, 1)
    assert schedule.tail_rollouts == 1
    assert schedule.optimizer_steps == 6
    assert schedule.row_indices == (0, 1, 2, 3, 4, 5) * 2
    assert schedule.rollout_ids == (0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9)


@pytest.mark.unit
def test_remainder_error_refuses_a_tail() -> None:
    with pytest.raises(ValueError, match="do not form complete optimizer steps of 2"):
        materialize_schedule([0, 1, 2], StepScheduling(batch_size=2, remainder="error"))


@pytest.mark.unit
def test_shuffle_varies_by_epoch_and_seed_but_not_between_calls() -> None:
    ids = list(range(32))
    scheduling = StepScheduling(batch_size=8, epochs=2, shuffle=True)
    first = materialize_schedule(ids, scheduling, seed=schedule_seed("batch-a"))
    again = materialize_schedule(ids, scheduling, seed=schedule_seed("batch-a"))
    other = materialize_schedule(ids, scheduling, seed=schedule_seed("batch-b"))

    assert first == again
    assert first.row_indices != other.row_indices
    epoch_0, epoch_1 = first.row_indices[:32], first.row_indices[32:]
    assert sorted(epoch_0) == sorted(epoch_1) == ids
    assert epoch_0 != epoch_1
    assert first.rollout_ids == tuple(range(64))
    assert first.step_sizes == (8,) * 8


@pytest.mark.unit
def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        materialize_schedule([], StepScheduling())
