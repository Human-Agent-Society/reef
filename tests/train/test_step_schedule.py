"""Backend-neutral schedule materialization: rollouts → ordered rows and steps."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support._trajectories import policy_trajectory

from reef.train.algos.schedule import (
    MaterializedSchedule,
    StepScheduling,
    materialize_schedule,
    pack_rollouts,
    rollout_ids,
    schedule_job_data,
    schedule_seed,
)
from reef.train.types import JobData


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"epochs": 0}, "epochs must be a positive int"),
        ({"shuffle": 1}, "shuffle must be a bool"),
        ({"remainder": "pad"}, "remainder must be"),
        ({"packing_length": 0}, "packing_length must be None or a positive int"),
        ({"packing_length": True}, "packing_length must be None or a positive int"),
    ],
)
def test_step_scheduling_validates_fields(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        StepScheduling(**kwargs)


@pytest.mark.unit
def test_a_job_without_a_scheduling_parameter_gets_the_default_schedule() -> None:
    assert StepScheduling.from_parameters({"method": {"objective": "sft"}}) == StepScheduling()
    assert StepScheduling.from_parameters({"scheduling": {"epochs": 2, "shuffle": True}}) == StepScheduling(
        epochs=2, shuffle=True
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "scheduling, error",
    [
        (None, "must be an object of epochs, shuffle, remainder and packing_length"),
        ({"unit": "sample"}, "must be an object of epochs, shuffle, remainder and packing_length"),
        ({"epochs": 0}, "epochs must be a positive int"),
    ],
)
def test_a_malformed_scheduling_parameter_is_refused(scheduling, error) -> None:
    with pytest.raises(ValueError, match=error):
        StepScheduling.from_parameters({"scheduling": scheduling})


def grouped_batch(group_ids) -> JobData:
    return JobData(
        "grouped",
        tuple(
            replace(policy_trajectory(f"r{index}", (10, 11), (1,), (-0.1,), 1.0), group_id=group_id)
            for index, group_id in enumerate(group_ids)
        ),
    )


@pytest.mark.unit
def test_the_comparison_set_unit_keeps_each_comparison_set_in_one_rollout() -> None:
    batch = grouped_batch(("a", "a", None, "b", "b"))

    assert rollout_ids(batch, "sample") == [0, 1, 2, 3, 4]
    assert rollout_ids(batch, "comparison_set") == [0, 0, 1, 2, 2]
    by_set = schedule_job_data(batch, "comparison_set", StepScheduling(), step_size=2)
    assert by_set.step_sizes == (2, 1)
    assert by_set.rollout_ids == (0, 0, 1, 2, 2)
    assert schedule_job_data(batch, "sample", StepScheduling(), step_size=2).step_sizes == (2, 2, 1)


@pytest.mark.unit
@pytest.mark.parametrize("step_size", [0, True, "4"])
def test_step_size_must_be_a_positive_int(step_size) -> None:
    with pytest.raises(ValueError, match="step_size must be a positive int"):
        materialize_schedule([0, 1], StepScheduling(), step_size=step_size)


@pytest.mark.unit
def test_step_size_of_the_whole_batch_trains_one_step_per_epoch() -> None:
    schedule = materialize_schedule([0, 0, 1], StepScheduling(epochs=2), step_size=2)

    assert schedule.step_sizes == (2, 2)
    assert schedule.optimizer_steps == 2
    assert schedule.row_indices == (0, 1, 2, 0, 1, 2)
    assert schedule.rollout_ids == (0, 0, 1, 2, 2, 3)


@pytest.mark.unit
def test_step_size_larger_than_the_batch_trains_one_partial_step() -> None:
    schedule = materialize_schedule([0, 0, 1, 2], StepScheduling(), step_size=8)

    assert schedule.row_indices == (0, 1, 2, 3)
    assert schedule.step_sizes == (3,)
    assert schedule.optimizer_steps == 1
    assert schedule.rollouts_per_epoch == 3


@pytest.mark.unit
def test_step_size_larger_than_the_batch_has_no_complete_step_to_keep() -> None:
    with pytest.raises(ValueError, match="do not form complete optimizer steps of 8"):
        materialize_schedule([0, 1, 2], StepScheduling(remainder="error"), step_size=8)
    with pytest.raises(ValueError, match="do not form one complete optimizer step of 8"):
        materialize_schedule([0, 1, 2], StepScheduling(remainder="drop"), step_size=8)


@pytest.mark.unit
def test_step_size_counts_rollouts_not_rows() -> None:
    schedule = materialize_schedule([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], StepScheduling(), step_size=2)

    assert schedule.step_sizes == (2, 2)
    assert schedule.dropped_rollouts == 0


@pytest.mark.unit
def test_remainder_drop_keeps_leading_full_steps() -> None:
    schedule = materialize_schedule([5, 5, 7, 9, 11, 13], StepScheduling(remainder="drop"), step_size=2)

    assert schedule.row_indices == (0, 1, 2, 3, 4)
    assert schedule.rollout_ids == (0, 0, 1, 2, 3)
    assert schedule.dropped_rollouts == 1
    assert schedule.step_sizes == (2, 2)


@pytest.mark.unit
def test_remainder_partial_is_default_and_ends_every_epoch_with_a_smaller_step() -> None:
    schedule = materialize_schedule([5, 5, 7, 9, 11, 13], StepScheduling(epochs=2), step_size=2)

    assert schedule.dropped_rollouts == 0
    assert schedule.step_sizes == (2, 2, 1, 2, 2, 1)
    assert schedule.tail_rollouts == 1
    assert schedule.optimizer_steps == 6
    assert schedule.row_indices == (0, 1, 2, 3, 4, 5) * 2
    assert schedule.rollout_ids == (0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9)


@pytest.mark.unit
def test_remainder_error_refuses_a_tail() -> None:
    with pytest.raises(ValueError, match="do not form complete optimizer steps of 2"):
        materialize_schedule([0, 1, 2], StepScheduling(remainder="error"), step_size=2)


@pytest.mark.unit
def test_shuffle_varies_by_epoch_and_seed_but_not_between_calls() -> None:
    ids = list(range(32))
    scheduling = StepScheduling(epochs=2, shuffle=True)
    first = materialize_schedule(ids, scheduling, step_size=8, seed=schedule_seed("batch-a"))
    again = materialize_schedule(ids, scheduling, step_size=8, seed=schedule_seed("batch-a"))
    other = materialize_schedule(ids, scheduling, step_size=8, seed=schedule_seed("batch-b"))

    assert first == again
    assert first.row_indices != other.row_indices
    epoch_0, epoch_1 = first.row_indices[:32], first.row_indices[32:]
    assert sorted(epoch_0) == sorted(epoch_1) == ids
    assert epoch_0 != epoch_1
    assert first.rollout_ids == tuple(range(64))
    assert first.step_sizes == (8,) * 8


@pytest.mark.unit
def test_empty_job_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty job data"):
        materialize_schedule([], StepScheduling(), step_size=1)


@pytest.mark.unit
def test_pack_rollouts_fills_each_pack_best_fit_decreasing_and_keeps_source_order() -> None:
    # 12 exceeds the limit and is a pack of its own; 9 takes 1, 7 takes 2, and 6 takes 4.
    assert pack_rollouts([6, 9, 2, 7, 4, 1, 12], 10) == [[0, 4], [1, 5], [2, 3], [6]]
    assert pack_rollouts([], 10) == []


@pytest.mark.unit
def test_a_packed_step_trains_step_size_packs_and_never_splits_a_rollout() -> None:
    # Rollouts 0..4 of 6, 9, 2, 7 and 4 tokens; comparison set 1 has two rows of 4 and 5 tokens.
    rollout_ids = [0, 1, 1, 2, 3, 4]
    token_counts = [6, 4, 5, 2, 7, 4]
    scheduling = StepScheduling(packing_length=10)
    schedule = materialize_schedule(rollout_ids, scheduling, step_size=2, token_counts=token_counts)

    # Packs [0, 4], [1] and [2, 3]: the first step trains three rollouts, the partial last step two.
    assert schedule.row_indices == (0, 5, 1, 2, 3, 4)
    assert schedule.rollout_ids == (0, 1, 2, 2, 3, 4)
    assert schedule.step_sizes == (3, 2)
    assert (schedule.rollouts_per_epoch, schedule.tail_rollouts) == (5, 2)
    dropped = materialize_schedule(
        rollout_ids, replace(scheduling, remainder="drop"), step_size=2, token_counts=token_counts
    )
    assert (dropped.step_sizes, dropped.dropped_rollouts) == ((3,), 2)
    with pytest.raises(ValueError, match="3 packs do not form complete optimizer steps of 2"):
        materialize_schedule(
            rollout_ids, replace(scheduling, remainder="error"), step_size=2, token_counts=token_counts
        )
    with pytest.raises(ValueError, match="packing needs the token count of every row"):
        materialize_schedule(rollout_ids, scheduling, step_size=2)


@pytest.mark.unit
def test_packing_counts_the_tokens_of_each_trajectory() -> None:
    long = policy_trajectory("long", tuple(range(8)), (1,), (-0.1,), 1.0)
    short = [policy_trajectory(f"s{index}", (10, 11), (1,), (-0.1,), 1.0) for index in range(3)]
    data = JobData("packed", (long, *short))

    schedule = schedule_job_data(data, "sample", StepScheduling(packing_length=9), step_size=1)

    # 8 tokens and 2 tokens do not fit in 9: the long trajectory is a pack alone, and the three short ones share one.
    assert schedule.step_sizes == (1, 3)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"step_sizes": ()}, "one or more optimizer steps"),
        ({"step_sizes": (2, 2)}, "every rollout of every epoch"),
        ({"tail_rollouts": 2}, "its last step, or 0"),
        ({"rollout_ids": (0, 1)}, "one non-negative source row"),
        ({"rollout_ids": (0, 2, 2)}, "numbers its rollouts"),
    ],
)
def test_a_schedule_refuses_values_outside_its_definition(changes, message) -> None:
    valid = MaterializedSchedule(
        row_indices=(0, 1, 2),
        rollout_ids=(0, 1, 2),
        step_sizes=(2, 1),
        epochs=1,
        rollouts_per_epoch=3,
        dropped_rollouts=0,
        tail_rollouts=1,
    )
    with pytest.raises(ValueError, match=message):
        replace(valid, **changes)
