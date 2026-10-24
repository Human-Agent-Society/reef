"""Slime's wire data: the payload a job's batch becomes, and the rollout data the workers receive.

These are deterministic Python operations: the step schedule, rollout IDs,
advantages and log-prob shapes, without workers or Ray.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support._trajectories import policy_trajectory

from reef.runtime.interfaces import TrainingMethod
from reef.train.algos import TrainingObjective, TrainingSignal
from reef.train.algos.schedule import RolloutUnit, StepScheduling, schedule_job_data
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.reef_adapters import preparation
from reef.train.slime_backend.reef_adapters.preparation import build_slime_payload
from reef.train.types import JobData


def wire_row(
    source_id: str,
    *,
    tokens: tuple[int, ...] = (10, 11, 12),
    loss_mask: tuple[int, ...] = (1, 1),
    log_probs: tuple[float, ...] = (-0.1, -0.2),
    reward: float = 0.5,
) -> list[object]:
    """Build one framework-agnostic row emitted by Reef's remote handle."""

    return [source_id, list(tokens), list(loss_mask), list(log_probs), reward]


def wire_payload(
    *,
    log_probs: object = "default",
    advantages: object = "default",
    loss: str = "pg",
) -> dict:
    if log_probs == "default":
        rows = [wire_row("a"), wire_row("b", reward=1.0), wire_row("c", reward=-1.0)]
    else:
        rows = [
            wire_row("a", log_probs=tuple(log_probs[0])),
            wire_row("b", log_probs=tuple(log_probs[1]), reward=1.0),
            wire_row("c", log_probs=tuple(log_probs[2]), reward=-1.0),
        ]
    data: dict[str, object] = {
        "samples": rows,
        # Two samples from one comparison set and one from another.  IDs are
        # deliberately explicit: the bridge must not derive them from a hash
        # or from the order in which a dict happens to be traversed.
        "rollout_ids": [0, 0, 1],
        "loss": loss,
        "method": TrainingMethod(loss).to_dict(),
        "external_step_sizes": [2],
    }
    if advantages != "default":
        data["advantages"] = advantages
    return data


@pytest.mark.unit
def test_to_slime_rollout_data_converts_non_empty_payload_and_preserves_groups() -> None:
    converted = to_slime_rollout_data(
        {
            "samples": [wire_row("first"), wire_row("second", reward=1.0)],
            "rollout_ids": [7, 7],
            "loss": "pg",
            "method": TrainingMethod("pg").to_dict(),
            "advantages": [0.25, -0.25],
            "external_step_sizes": [1],
        }
    )

    assert converted["tokens"] == [[10, 11, 12], [10, 11, 12]]
    assert converted["loss_masks"] == [[1, 1], [1, 1]]
    assert converted["response_lengths"] == [2, 2]
    assert converted["rewards"] == [0.5, 1.0]
    assert converted["rollout_ids"] == [7, 7]
    assert converted["sample_indices"] == [0, 1]
    assert converted["loss"] == "pg"
    # Reef supplies one scalar advantage per sample; Slime consumes one value
    # per response token.
    assert converted["advantages"] == [[0.25, 0.25], [-0.25, -0.25]]


@pytest.mark.unit
def test_to_slime_rollout_data_aggregates_rollout_mask_sums() -> None:
    payload = {
        "samples": [
            wire_row("a", loss_mask=(1, 1)),
            wire_row("b", loss_mask=(1,), log_probs=(-0.1,)),
            wire_row(
                "c",
                tokens=(10, 11, 12, 13, 14),
                loss_mask=(1, 1, 1, 1),
                log_probs=(-0.1, -0.2, -0.3, -0.4),
            ),
        ],
        "rollout_ids": [0, 0, 1],
        "loss": "pg",
        "method": TrainingMethod("pg").to_dict(),
        "advantages": [0.25, -0.25, 0.0],
        "external_step_sizes": [2],
    }

    converted = to_slime_rollout_data(payload)

    # IDs [0, 0, 1], per-sample mask totals [2, 1, 4] => aggregate and
    # broadcast [3, 3, 4].
    assert converted["rollout_mask_sums"] == [3, 3, 4]


@pytest.mark.unit
def test_to_slime_rollout_data_omits_optional_empty_log_probs() -> None:
    converted = to_slime_rollout_data(wire_payload(log_probs=[[], [], []], loss="sft"))

    assert "rollout_log_probs" not in converted


@pytest.mark.unit
def test_to_slime_rollout_data_keeps_non_empty_log_probs() -> None:
    converted = to_slime_rollout_data(
        wire_payload(
            log_probs=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            loss="sft",
        )
    )

    assert converted["rollout_log_probs"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


@pytest.mark.unit
def test_to_slime_rollout_data_rejects_non_empty_log_prob_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="log_prob"):
        to_slime_rollout_data(
            wire_payload(
                log_probs=[[0.1], [0.3, 0.4], [0.5, 0.6]],
                loss="sft",
            )
        )


@pytest.mark.unit
def test_to_slime_rollout_data_rejects_mixed_empty_and_non_empty_log_probs() -> None:
    with pytest.raises(ValueError, match="log_prob"):
        to_slime_rollout_data(
            wire_payload(
                log_probs=[[], [0.3, 0.4], [0.5, 0.6]],
                loss="sft",
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload, message",
    [
        ({"samples": [], "rollout_ids": [], "loss": "pg"}, "non-empty"),
        ({"samples": [wire_row("a")], "rollout_ids": [], "loss": "pg"}, "rollout_ids"),
        (
            {"samples": [wire_row("a")], "rollout_ids": [0], "loss": "pg", "advantages": []},
            "advantages",
        ),
        (
            {
                "samples": [
                    ["a", [1], [1]],
                ],
                "rollout_ids": [0],
                "loss": "pg",
                "method": TrainingMethod("pg").to_dict(),
            },
            "sample",
        ),
    ],
)
def test_to_slime_rollout_data_validates_non_empty_and_parallel_shapes(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        to_slime_rollout_data(payload)


def slime_payload(
    batch: JobData,
    loss_family: str,
    advantages: tuple[float, ...] | None,
    scheduling: StepScheduling,
    *,
    step_size: int,
    unit: RolloutUnit = "comparison_set",
) -> dict[str, object]:
    """Slime's payload of ``batch`` for an objective whose ``rollout_unit`` is ``unit``."""
    return build_slime_payload(
        batch, loss_family, advantages, schedule_job_data(batch, unit, scheduling, step_size=step_size)
    )


@pytest.mark.unit
def test_slime_preparation_emits_stable_rollout_ids_for_comparison_sets() -> None:
    batch = JobData(
        "batch",
        tuple(
            replace(sample, group_id=str(index))
            for index, group in enumerate(
                (
                    (
                        policy_trajectory("a", (1, 2), (1, 1), (-0.1, -0.2), 0.2),
                        policy_trajectory("b", (3, 4), (1, 1), (-0.3, -0.4), 0.8),
                    ),
                    (policy_trajectory("c", (5, 6), (1, 1), (-0.5, -0.6), 0.4),),
                )
            )
            for sample in group
        ),
    )

    payload = slime_payload(batch, "pg", (0.1, -0.1, 0.0), StepScheduling(), step_size=2)

    assert payload["rollout_ids"] == [0, 0, 1]
    assert len(payload["rollout_ids"]) == len(payload["samples"]) == 3


@pytest.mark.unit
def test_slime_preparation_honors_the_sample_unit_and_the_step_size() -> None:
    batch = JobData(
        "batch",
        tuple(
            replace(sample, group_id=str(index))
            for index, group in enumerate(
                (
                    (
                        policy_trajectory("a", (1, 2), (1,), (-0.1,), 0.2),
                        policy_trajectory("b", (3, 4), (1,), (-0.2,), 0.8),
                    ),
                )
            )
            for sample in group
        ),
    )

    payload = slime_payload(batch, "tttd", (0.1, -0.1), StepScheduling(), step_size=2, unit="sample")

    assert payload["rollout_ids"] == [0, 1]
    assert payload["external_step_sizes"] == [2]


@pytest.mark.unit
def test_slime_preparation_assigns_distinct_stable_ids_to_policy_samples() -> None:
    batch = JobData(
        "batch",
        (
            policy_trajectory("a", (1,), (1,), (-0.1,), 0.2),
            policy_trajectory("b", (2,), (1,), (-0.2,), 0.8),
        ),
    )

    payload = slime_payload(batch, "sft", None, StepScheduling(), step_size=2)

    assert payload["rollout_ids"] == [0, 1]


@pytest.mark.unit
def test_assembled_multi_turn_sample_uses_existing_slime_payload_path() -> None:
    sample = policy_trajectory(
        "harbor-report",
        (10, 20, 11, 21),
        (1, 0, 1),
        (-0.1, 0.0, -0.2),
        0.75,
        "wv-1",
        turn_count=2,
    )
    assert sample.training.get("turn_count", 1) > 1
    payload = slime_payload(JobData("batch", (sample,)), "sft", None, StepScheduling(), step_size=1)

    assert payload["samples"] == [["harbor-report", [10, 20, 11, 21], [1, 0, 1], [-0.1, 0.0, -0.2], 0.75]]
    converted = to_slime_rollout_data(payload)

    assert converted["tokens"] == [[10, 20, 11, 21]]
    assert converted["loss_masks"] == [[1, 0, 1]]
    assert converted["rollout_log_probs"] == [[-0.1, 0.0, -0.2]]
    assert converted["rewards"] == [0.75]


def grouped_batch(groups: int, size: int, data_id: str = "batch") -> JobData:
    return JobData(
        data_id,
        tuple(
            replace(sample, group_id=str(index))
            for index, group in enumerate(
                tuple(
                    tuple(
                        policy_trajectory(f"g{group}-s{index}", (group, index), (1,), (-0.1,), float(index))
                        for index in range(size)
                    )
                    for group in range(groups)
                )
            )
            for sample in group
        ),
    )


@pytest.mark.unit
def test_slime_preparation_step_size_cuts_one_batch_into_steps() -> None:
    batch = grouped_batch(groups=4, size=2)

    payload = slime_payload(batch, "pg", tuple(range(8)), StepScheduling(), step_size=2)

    assert payload["rollout_ids"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert payload["external_step_sizes"] == [2, 2]
    assert payload["advantages"] == list(range(8))
    converted = to_slime_rollout_data(payload)
    assert converted["external_step_sizes"] == [2, 2]


@pytest.mark.unit
def test_slime_preparation_epochs_repeat_rollouts_with_distinct_ids() -> None:
    batch = grouped_batch(groups=2, size=2)

    payload = slime_payload(batch, "pg", (1.0, 2.0, 3.0, 4.0), StepScheduling(epochs=3), step_size=1)

    assert [row[0] for row in payload["samples"]] == ["g0-s0", "g0-s1", "g1-s0", "g1-s1"] * 3
    assert payload["rollout_ids"] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert payload["advantages"] == [1.0, 2.0, 3.0, 4.0] * 3
    assert payload["external_step_sizes"] == [1] * 6


@pytest.mark.unit
def test_slime_preparation_shuffle_is_deterministic_per_batch_and_keeps_groups_contiguous() -> None:
    scheduling = StepScheduling(epochs=2, shuffle=True)
    first = slime_payload(grouped_batch(8, 2, "b-1"), "pg", tuple(range(16)), scheduling, step_size=2)
    again = slime_payload(grouped_batch(8, 2, "b-1"), "pg", tuple(range(16)), scheduling, step_size=2)
    other = slime_payload(grouped_batch(8, 2, "b-2"), "pg", tuple(range(16)), scheduling, step_size=2)

    assert first == again
    # The order follows the rows, not the batch number a reload starts again: a retry replays the same job.
    assert other["samples"] == first["samples"]
    names = [row[0] for row in first["samples"]]
    assert names != [
        row[0]
        for row in slime_payload(grouped_batch(8, 2, "b-1"), "pg", None, StepScheduling(epochs=2), step_size=2)[
            "samples"
        ]
    ]
    for epoch in range(2):
        epoch_names = names[epoch * 16 : (epoch + 1) * 16]
        assert sorted(epoch_names) == sorted(f"g{g}-s{s}" for g in range(8) for s in range(2))
        # every comparison set stays contiguous and in source order
        for start in range(0, 16, 2):
            g0, g1 = epoch_names[start], epoch_names[start + 1]
            assert g0[:2] == g1[:2] and g0.endswith("s0") and g1.endswith("s1")
    # advantages travel with their rows
    for row, advantage in zip(first["samples"], first["advantages"], strict=True):
        g, s = int(row[0][1]), int(row[0][-1])
        assert advantage == g * 2 + s


@pytest.mark.unit
def test_slime_preparation_remainder_policy() -> None:
    batch = grouped_batch(groups=5, size=1)

    with pytest.raises(ValueError, match="5 rollouts do not form complete optimizer steps of 2"):
        slime_payload(batch, "pg", tuple(range(5)), StepScheduling(remainder="error"), step_size=2)
    # A step size over the batch's rollouts trains them as one partial step.
    assert slime_payload(batch, "pg", tuple(range(5)), StepScheduling(), step_size=8)["external_step_sizes"] == [5]

    dropped = slime_payload(batch, "pg", tuple(range(5)), StepScheduling(remainder="drop"), step_size=2)
    assert [row[0] for row in dropped["samples"]] == ["g0-s0", "g1-s0", "g2-s0", "g3-s0"]
    assert dropped["advantages"] == [0, 1, 2, 3]
    assert dropped["external_step_sizes"] == [2, 2]

    partial = slime_payload(batch, "pg", tuple(range(5)), StepScheduling(epochs=2), step_size=2)
    assert len(partial["samples"]) == 10
    assert partial["external_step_sizes"] == [2, 2, 1, 2, 2, 1]


@pytest.mark.unit
def test_slime_preparation_step_size_applies_to_policy_batches() -> None:
    batch = JobData(
        "batch",
        (
            policy_trajectory("a", (1,), (1,), (-0.1,), 0.2),
            policy_trajectory("b", (2,), (1,), (-0.2,), 0.8),
        ),
    )

    payload = slime_payload(batch, "sft", None, StepScheduling(), step_size=2)

    assert payload["external_step_sizes"] == [2]


@pytest.mark.unit
def test_to_slime_rollout_data_validates_step_layout() -> None:
    sample = ["a", [1, 2], [1], [-0.1], 0.5]
    payload = {
        "samples": [sample, sample, sample, sample],
        "rollout_ids": [0, 1, 2, 3],
        "loss": "sft",
        "method": TrainingMethod("sft").to_dict(),
    }

    assert to_slime_rollout_data({**payload, "external_step_sizes": [3, 1]})["external_step_sizes"] == [3, 1]
    with pytest.raises(ValueError, match="sum 3 must equal the 4 distinct rollout_ids"):
        to_slime_rollout_data({**payload, "external_step_sizes": [3]})
    with pytest.raises(ValueError, match="positive integers"):
        to_slime_rollout_data({**payload, "external_step_sizes": [4, 0]})
    # Reef always fixes the step layout; a payload without one is refused.
    with pytest.raises(ValueError, match="positive integers"):
        to_slime_rollout_data(payload)


@pytest.mark.unit
def test_prepare_slime_job_reports_schedule_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class ScheduleObjective(TrainingObjective):
        name = "test-schedule"
        loss_family = "pg"
        supports_multiple_epochs = True

        def prepare(self, batch, state):
            return TrainingSignal("train", {}, {"steps": 1}, tuple(range(8)))

    monkeypatch.setattr(preparation, "resolve_objective", lambda _name: ScheduleObjective())

    result = preparation.prepare_slime_job(
        grouped_batch(8, 1),
        TrainingMethod("any"),
        {},
        StepScheduling(epochs=2, remainder="drop"),
        step_size=3,
    )

    assert result.metrics == {"steps": 1, "epochs": 2, "optimizer_steps": 4, "dropped_rollouts": 2}
    assert result.payload is not None and len(result.payload["samples"]) == 12


@pytest.mark.unit
@pytest.mark.parametrize(("unit", "rollout_ids"), [("sample", [0, 1, 2, 3]), ("comparison_set", [0, 0, 1, 1])])
def test_prepare_slime_job_groups_rows_by_the_objective_rollout_unit(
    monkeypatch: pytest.MonkeyPatch, unit: RolloutUnit, rollout_ids: list[int]
) -> None:
    class GroupedObjective(TrainingObjective):
        name = "test-rollout-unit"
        loss_family = "pg"
        rollout_unit = unit

        def prepare(self, batch, state):
            return TrainingSignal("train", {}, {}, (1.0, -1.0, 1.0, -1.0))

    monkeypatch.setattr(preparation, "resolve_objective", lambda _name: GroupedObjective())

    result = preparation.prepare_slime_job(
        grouped_batch(2, 2), TrainingMethod("any"), {}, StepScheduling(), step_size=1
    )

    assert result.payload is not None
    assert result.payload["rollout_ids"] == rollout_ids
    assert len(result.payload["external_step_sizes"]) == len(set(rollout_ids))
