"""Slime job preparation: one job's data and objective into Slime's training payload rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support._grouped_pg import GROUPED_PG_OBJECTIVE
from tests.support._trajectories import policy_trajectory

from reef.core.artifact_ref import RuntimeLoadSpan
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos import TrainingObjective, TrainingSignal
from reef.train.algos.schedule import StepScheduling
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData, TaskItem, trajectories

SFT_OBJECTIVE = "tests.slime_backend.test_slime_job_preparation:SampleSftObjective"
TWO_EPOCH_OBJECTIVE = "tests.slime_backend.test_slime_job_preparation:TwoEpochObjective"


class SampleSftObjective(TrainingObjective):
    name = "test-SampleSftObjective"
    loss_family = "sft"

    def prepare(self, batch, state) -> TrainingSignal:
        """Package-test custom objective; cookbook examples are not in the wheel."""
        trajectories(batch)
        return TrainingSignal(action="train", next_algorithm_state={"steps": int(state.get("steps", 0)) + 1})


class TwoEpochObjective(TrainingObjective):
    name = "test-TwoEpochObjective"
    loss_family = "pg"
    supports_multiple_epochs = True

    def prepare(self, batch, state):
        """A cookbook-shaped objective that accepts a job of two passes over the batch."""
        del state
        return TrainingSignal(action="train", next_algorithm_state={}, advantages=(-1.0, 1.0))


def policy_batch() -> JobData:
    return JobData(
        "batch-1",
        (
            policy_trajectory(
                "i1",
                (1, 2),
                (0, 1),
                (-0.2, -0.15),
                1.0,
                "v0",
                topk_indices=((1, 2), (3, 4)),
                topk_log_probs=((-0.2, -0.9), (-0.15, -0.8)),
            ),
            policy_trajectory(
                "i2",
                (3, 4),
                (1, 1),
                (-0.1, -0.05),
                0.5,
                "v0",
                topk_indices=((5, 6), (7, 8)),
                topk_log_probs=((-0.1, -0.7), (-0.05, -0.6)),
            ),
        ),
    )


def grouped_policy_batch() -> JobData:
    return JobData(
        "batch-2",
        tuple(
            replace(sample, group_id=str(index))
            for index, group in enumerate(
                (
                    (
                        policy_trajectory("g1", (1, 2), (1,), (-0.1,), 0.2, "v0"),
                        policy_trajectory("g2", (3, 4), (1,), (-0.2,), 0.8, "v0"),
                    ),
                )
            )
            for sample in group
        ),
    )


def test_preparation_emits_framework_agnostic_sft_rows() -> None:
    for objective in ("sft", SFT_OBJECTIVE):
        prepared = prepare_slime_job(policy_batch(), TrainingMethod(objective), {}, StepScheduling(), step_size=1)
        assert prepared.payload is not None
        payload = prepared.payload
        assert payload["loss"] == "sft"
        assert "advantages" not in payload
        assert payload["method"] == TrainingMethod(objective).to_dict()
        assert payload["samples"] == [
            ["i1", [1, 2], [0, 1], [-0.2, -0.15], 1.0],
            ["i2", [3, 4], [1, 1], [-0.1, -0.05], 0.5],
        ]


@pytest.mark.parametrize(
    ("batch", "objective", "advantages", "loss"),
    [
        (policy_batch(), "openclawrl", (1.0, 0.5), "openclawrl"),
        (grouped_policy_batch(), GROUPED_PG_OBJECTIVE, (-1.0, 1.0), "pg"),
    ],
)
def test_preparation_keeps_the_objectives_policy_signals(batch, objective, advantages, loss) -> None:
    prepared = prepare_slime_job(batch, TrainingMethod(objective), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    assert prepared.payload["loss"] == loss
    assert prepared.payload["advantages"] == pytest.approx(advantages)


@pytest.mark.parametrize(
    ("batch", "objective", "error", "message"),
    [
        (JobData("task", (TaskItem(Path("tasks/example")),)), "sft", TypeError, "unsupported item"),
        (JobData("task", (TaskItem(Path("tasks/example")),)), SFT_OBJECTIVE, TypeError, "unsupported item"),
        (policy_batch(), "unknown", ValueError, "unknown objective"),
    ],
)
def test_preparation_rejects_unsupported_objective_batch_pairs(batch, objective, error, message) -> None:
    with pytest.raises(error, match=message):
        prepare_slime_job(batch, TrainingMethod(objective), {}, StepScheduling(), step_size=1)


def test_slime_backend_preparation_emits_framework_agnostic_rows() -> None:
    prepared = prepare_slime_job(policy_batch(), TrainingMethod("sft"), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == "sft"
    assert "advantages" not in payload
    assert payload["samples"] == [
        ["i1", [1, 2], [0, 1], [-0.2, -0.15], 1.0],
        ["i2", [3, 4], [1, 1], [-0.1, -0.05], 0.5],
    ]


def test_preparation_carries_each_row_token_spans() -> None:
    sample = policy_trajectory(
        "s1",
        (1, 2, 3),
        (1, 1),
        (-0.1, -0.2),
        1.0,
        "v1",
        runtime_load_spans=(RuntimeLoadSpan(0, 2, "v1"),),
    )
    prepared = prepare_slime_job(
        JobData("batch-spans", (sample,)), TrainingMethod("sft"), {}, StepScheduling(), step_size=1
    )
    assert prepared.payload is not None
    assert prepared.payload["producing_runtime_load_spans"] == [[{"start": 0, "end": 2, "runtime_load_id": "v1"}]]


def test_slime_backend_preparation_emits_sao_rows_with_action_mask_and_source_fields() -> None:
    # SAO ships an 8-element row: the action mask (for skip-observation GAE)
    # and rollout source fields (producing runtime load ID, creation time) have no
    # slot in the policy 5-tuple. Each sample is its own rollout (no grouping)
    # and advantages are never shipped — the critic computes them in-backend.
    batch = JobData(
        "math:sao:1",
        (
            policy_trajectory(
                source_agent_record_id="i1",
                tokens=(5, 1, 2, 3),
                loss_mask=(1, 0, 1),
                action_mask=(1, 0, 1),
                rollout_log_probs=(-0.1, -0.2, -0.3),
                reward=0.9,
                runtime_load_id="slime-v3",
                rollout_created_at=1234.5,
            ),
        ),
    )

    prepared = prepare_slime_job(batch, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    payload = prepared.payload

    assert payload["loss"] == "sao"
    assert "advantages" not in payload
    assert payload["rollout_ids"] == [0]
    assert payload["samples"] == [
        [
            "i1",
            [5, 1, 2, 3],
            [1, 0, 1],
            [-0.1, -0.2, -0.3],
            0.9,
            [1, 0, 1],
            "slime-v3",
            1234.5,
        ]
    ]


def test_slime_backend_preparation_sao_rows_tolerate_missing_source_fields() -> None:
    batch = JobData(
        "math:sao:1",
        (
            policy_trajectory(
                source_agent_record_id="i1",
                tokens=(5, 1),
                loss_mask=(1,),
                action_mask=(1,),
                rollout_log_probs=(-0.1,),
                reward=0.5,
            ),
        ),
    )

    prepared = prepare_slime_job(batch, TrainingMethod("sao"), {}, StepScheduling(), step_size=1)
    assert prepared.payload is not None
    payload = prepared.payload

    # A rollout served before runtime-load-ID tracking still ships; the last two
    # slots carry None rather than dropping the row.
    assert payload["samples"][0][-2:] == [None, None]


def test_preparation_repeats_every_row_for_each_scheduled_epoch() -> None:
    batch = JobData(
        "batch-epochs",
        (
            policy_trajectory("g1", (1, 2), (1,), (-0.1,), 0.2, "v0"),
            policy_trajectory("g2", (3, 4), (1,), (-0.2,), 0.8, "v1"),
        ),
    )
    prepared = prepare_slime_job(batch, TrainingMethod(TWO_EPOCH_OBJECTIVE), {}, StepScheduling(epochs=2), step_size=1)
    assert prepared.payload is not None
    assert [row[0] for row in prepared.payload["samples"]] == ["g1", "g2", "g1", "g2"]
    # Each wire row carries the producing version of the batch sample it repeats.
    assert prepared.payload["producing_runtime_load_ids"] == ["v0", "v1", "v0", "v1"]
    assert "producing_runtime_load_spans" not in prepared.payload
    assert prepared.metrics["epochs"] == 2
