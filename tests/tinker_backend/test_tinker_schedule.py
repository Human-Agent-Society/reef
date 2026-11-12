"""Tinker's step schedule: the rollouts of one job as the source rows of each optimizer step."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support._trajectories import policy_trajectory

from reef.train.tinker_backend.schedule import StepRows, StepScheduling, next_steps, pack_rollouts, schedule_job_data
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


def job_data(group_ids, token_counts=None, data_id="grouped") -> JobData:
    """One trajectory per entry of ``group_ids``, in that comparison set, with that many tokens (2 by default)."""
    counts = [2] * len(group_ids) if token_counts is None else token_counts
    return JobData(
        data_id,
        tuple(
            replace(
                policy_trajectory(f"{data_id}-{index}", tuple(range(count)), (1,), (-0.1,), 1.0), group_id=group_id
            )
            for index, (group_id, count) in enumerate(zip(group_ids, counts, strict=True))
        ),
    )


def steps(group_ids, scheduling=None, *, step_size, unit="comparison_set", token_counts=None) -> StepRows:
    return schedule_job_data(
        job_data(group_ids, token_counts), unit, scheduling or StepScheduling(), step_size=step_size
    )


@pytest.mark.unit
def test_the_comparison_set_unit_keeps_each_comparison_set_in_one_rollout() -> None:
    groups = ("a", "a", None, "b", "b")

    assert steps(groups, step_size=2).steps == ((0, 1, 2), (3, 4))
    assert steps(groups, step_size=2, unit="sample").steps == ((0, 1), (2, 3), (4,))
    # The rows of a comparison set stay together when they are not adjacent.
    assert steps(("a", "b", "a"), step_size=1).steps == ((0, 2), (1,))


@pytest.mark.unit
@pytest.mark.parametrize("step_size", [0, True, "4"])
def test_step_size_must_be_a_positive_int(step_size) -> None:
    with pytest.raises(ValueError, match="step_size must be a positive int"):
        steps(("a", "b"), step_size=step_size)


@pytest.mark.unit
def test_step_size_of_the_whole_job_trains_one_step_per_epoch() -> None:
    assert steps(("a", "a", "b"), StepScheduling(epochs=2), step_size=2).steps == ((0, 1, 2), (0, 1, 2))


@pytest.mark.unit
def test_step_size_larger_than_the_job_trains_one_partial_step() -> None:
    schedule = steps(("a", "a", "b", "c"), step_size=8)

    assert schedule == StepRows(((0, 1, 2, 3),), 0)


@pytest.mark.unit
def test_step_size_larger_than_the_job_has_no_complete_step_to_keep() -> None:
    with pytest.raises(ValueError, match="do not form complete optimizer steps of 8"):
        steps(("a", "b", "c"), StepScheduling(remainder="error"), step_size=8)
    with pytest.raises(ValueError, match="do not form one complete optimizer step of 8"):
        steps(("a", "b", "c"), StepScheduling(remainder="drop"), step_size=8)


@pytest.mark.unit
def test_step_size_counts_rollouts_not_rows() -> None:
    schedule = steps(tuple("aaabbbcccddd"), step_size=2)

    assert [len(step) for step in schedule.steps] == [6, 6]
    assert schedule.dropped_rollouts == 0


@pytest.mark.unit
def test_remainder_drop_keeps_leading_full_steps() -> None:
    schedule = steps(("a", "a", "b", "c", "d", "e"), StepScheduling(remainder="drop"), step_size=2)

    assert schedule == StepRows(((0, 1, 2), (3, 4)), 1)


@pytest.mark.unit
def test_remainder_partial_is_default_and_ends_every_epoch_with_a_smaller_step() -> None:
    schedule = steps(("a", "a", "b", "c", "d", "e"), StepScheduling(epochs=2), step_size=2)

    assert schedule == StepRows(((0, 1, 2), (3, 4), (5,)) * 2, 0)


@pytest.mark.unit
def test_remainder_error_refuses_a_tail() -> None:
    with pytest.raises(ValueError, match="do not form complete optimizer steps of 2"):
        steps(("a", "b", "c"), StepScheduling(remainder="error"), step_size=2)


@pytest.mark.unit
def test_shuffle_varies_by_epoch_and_rows_but_not_between_calls() -> None:
    rows = list(range(32))
    scheduling = StepScheduling(epochs=2, shuffle=True)
    first = schedule_job_data(job_data((None,) * 32, data_id="a"), "sample", scheduling, step_size=8)
    again = schedule_job_data(job_data((None,) * 32, data_id="a"), "sample", scheduling, step_size=8)
    other = schedule_job_data(job_data((None,) * 32, data_id="b"), "sample", scheduling, step_size=8)

    assert first == again
    assert first.steps != other.steps
    assert [len(step) for step in first.steps] == [8] * 8
    epoch_0 = [row for step in first.steps[:4] for row in step]
    epoch_1 = [row for step in first.steps[4:] for row in step]
    assert sorted(epoch_0) == sorted(epoch_1) == rows
    assert epoch_0 != epoch_1


@pytest.mark.unit
def test_empty_job_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty job data"):
        schedule_job_data(JobData("empty", ()), "sample", StepScheduling(), step_size=1)


@pytest.mark.unit
def test_pack_rollouts_fills_each_pack_best_fit_decreasing_and_keeps_source_order() -> None:
    # 12 exceeds the limit and is a pack of its own; 9 takes 1, 7 takes 2, and 6 takes 4.
    assert pack_rollouts([6, 9, 2, 7, 4, 1, 12], 10) == [[0, 4], [1, 5], [2, 3], [6]]
    assert pack_rollouts([], 10) == []


@pytest.mark.unit
def test_a_packed_step_trains_step_size_packs_and_never_splits_a_rollout() -> None:
    # Rollouts a..e of 6, 9, 2, 7 and 4 tokens; comparison set b has two rows of 4 and 5 tokens.
    groups = ("a", "b", "b", "c", "d", "e")
    token_counts = [6, 4, 5, 2, 7, 4]
    scheduling = StepScheduling(packing_length=10)

    # Packs [a, e], [b] and [c, d]: the first step trains three rollouts, the partial last step two.
    assert steps(groups, scheduling, step_size=2, token_counts=token_counts) == StepRows(((0, 5, 1, 2), (3, 4)), 0)
    dropped = steps(groups, replace(scheduling, remainder="drop"), step_size=2, token_counts=token_counts)
    assert dropped == StepRows(((0, 5, 1, 2),), 2)
    with pytest.raises(ValueError, match="3 packs do not form complete optimizer steps of 2"):
        steps(groups, replace(scheduling, remainder="error"), step_size=2, token_counts=token_counts)


@pytest.mark.unit
def test_packing_counts_the_tokens_of_each_trajectory() -> None:
    # 8 tokens and 2 tokens do not fit in 9: the long trajectory is a pack alone, and the three short ones share one.
    schedule = steps(
        (None,) * 4, StepScheduling(packing_length=9), step_size=1, unit="sample", token_counts=[8, 2, 2, 2]
    )

    assert schedule.steps == ((0,), (1, 2, 3))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "message"),
    [
        (((), 0), "one or more optimizer steps"),
        ((((0,), ()), 0), "one or more optimizer steps"),
        ((((0, -1),), 0), "non-negative source rows"),
        ((((0,),), -1), "non-negative count of rollouts"),
    ],
)
def test_step_rows_refuse_values_outside_their_definition(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        StepRows(*values)


@pytest.mark.unit
def test_next_steps_counts_from_the_state_and_refuses_a_malformed_counter() -> None:
    assert next_steps({}) == 1
    assert next_steps({"steps": 4}) == 5
    for steps_value in (-1, True, "4", 1.0):
        with pytest.raises(ValueError, match="'steps' must be a non-negative integer"):
            next_steps({"steps": steps_value})
