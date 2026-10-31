from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.record2sample.samples import make_multi_turn_policy_trajectory, make_policy_trajectory
from reef.service.inference_service import provider_response


def _inference(payload) -> AgentRecord:
    return AgentRecord.create(scenario="s", request_type=RequestType.INFERENCE, payload=payload)


def _turn(
    agent_record_id: str,
    tokens: list[int],
    loss_mask: list[int],
    log_probs: list[float],
    *,
    artifact_id: str = "artifact-1",
    topk: list[tuple[list[int], list[float]]] | None = None,
) -> AgentRecord:
    training = {
        "tokens": tokens,
        "loss_mask": loss_mask,
        "rollout_log_probs": log_probs,
        "artifact_id": artifact_id,
    }
    if topk is not None:
        training["topk_indices"] = [ids for ids, _ in topk]
        training["topk_log_probs"] = [values for _, values in topk]
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        payload={"response": {"training": training}},
    )


@pytest.mark.unit
def test_policy_sample_uses_engine_native_response_training() -> None:
    item = _inference(
        {
            "response": {
                "training": {
                    "tokens": [10, 11, 20],
                    "loss_mask": [1],
                    "rollout_log_probs": [-0.25],
                    "artifact_id": "artifact-1",
                }
            }
        }
    )

    sample = make_policy_trajectory(item, 1.0)

    assert tuple(sample.training.get("tokens", [])) == (10, 11, 20)
    assert tuple(sample.training.get("loss_mask", [])) == (1,)
    assert tuple(sample.training.get("rollout_log_probs", [])) == (-0.25,)
    assert sample.training.get("artifact_id", None) == "artifact-1"
    assert sample.training.get("turn_count", 1) == 1
    assert (sample.training.get("turn_count", 1) > 1) is False


@pytest.mark.unit
def test_policy_sample_does_not_reconstruct_ids_from_chat_logprobs() -> None:
    item = _inference(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "response": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "logprobs": {"content": [{"token": "answer", "logprob": -0.25}]},
                    }
                ]
            },
        }
    )

    sample = make_policy_trajectory(item, 1.0)

    assert tuple(sample.training.get("tokens", [])) == ()
    assert tuple(sample.training.get("loss_mask", [])) == ()
    assert tuple(sample.training.get("rollout_log_probs", [])) == ()


@pytest.mark.unit
def test_policy_sample_accepts_exact_harness_tensors() -> None:
    item = _inference(
        {
            "tokens": [1, 2, 3],
            "loss_mask": [1],
            "rollout_log_probs": [-0.5],
            "artifact_id": "artifact-2",
        }
    )

    sample = make_policy_trajectory(item, 0.5)

    assert tuple(sample.training.get("tokens", [])) == (1, 2, 3)
    assert tuple(sample.training.get("loss_mask", [])) == (1,)
    assert tuple(sample.training.get("rollout_log_probs", [])) == (-0.5,)
    assert sample.training.get("artifact_id", None) == "artifact-2"


@pytest.mark.unit
def test_provider_response_drops_the_training_block() -> None:
    response = {
        "choices": [{"message": {"content": "answer"}}],
        "training": {
            "tokens": [10, 20],
            "loss_mask": [1],
            "rollout_log_probs": [-0.25],
        },
    }

    answered = provider_response(response)

    assert "training" not in answered
    assert response["training"]["rollout_log_probs"] == [-0.25]


@pytest.mark.unit
def test_multi_turn_policy_sample_assembles_clean_linear_history() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [10, 11, 20, 21], [1, 1], [-0.1, -0.2]),
            _turn("i2", [10, 11, 20, 21, 30, 40], [1], [-0.3]),
            _turn("i3", [10, 11, 20, 21, 30, 40, 31, 50, 51], [1, 1], [-0.4, -0.5]),
        ],
        0.75,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert source_record_id(sample) == "report-1"
    assert tuple(sample.training.get("tokens", [])) == (10, 11, 20, 21, 30, 40, 31, 50, 51)
    assert tuple(sample.training.get("loss_mask", [])) == (1, 1, 0, 1, 0, 1, 1)
    assert tuple(sample.training.get("rollout_log_probs", [])) == (-0.1, -0.2, 0.0, -0.3, 0.0, -0.4, -0.5)
    assert trajectory_reward(sample) == 0.75
    assert sample.training.get("artifact_id", None) == "artifact-1"
    assert sample.training.get("turn_count", 1) == 3
    assert (sample.training.get("turn_count", 1) > 1) is True


@pytest.mark.unit
def test_multi_turn_policy_sample_keeps_sampler_top_k_aligned_with_the_response() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn(
                "i1", [10, 11, 20, 21], [1, 1], [-0.1, -0.2], topk=[([20, 7], [-0.1, -2.0]), ([21, 8], [-0.2, -3.0])]
            ),
            _turn("i2", [10, 11, 20, 21, 30, 40], [1], [-0.3], topk=[([40, 9], [-0.3, -1.0])]),
        ],
        1.0,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert tuple(sample.training.get("loss_mask", [])) == (1, 1, 0, 1)
    # The inserted context position 30 gets an empty row; trained rows keep their own head.
    assert sample.training.get("topk_indices", []) == [[20, 7], [21, 8], [], [40, 9]]
    assert sample.training.get("topk_log_probs", []) == [[-0.1, -2.0], [-0.2, -3.0], [], [-0.3, -1.0]]


@pytest.mark.unit
def test_multi_turn_policy_sample_masks_the_top_k_of_a_realigned_response() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [1, 2, 3], [1], [-0.3], topk=[([3, 5], [-0.3, -1.0])]),
            _turn("i2", [1, 2, 30, 4], [1], [-0.4], topk=[([4, 6], [-0.4, -1.0])]),
        ],
        1.0,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert tuple(sample.training.get("loss_mask", [])) == (0, 1)
    assert sample.training.get("topk_indices", []) == [[], [4, 6]]


@pytest.mark.unit
def test_multi_turn_policy_sample_drops_top_k_unless_every_turn_recorded_it() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [10, 11, 20], [1], [-0.1], topk=[([20, 7], [-0.1, -2.0])]),
            _turn("i2", [10, 11, 20, 30, 40], [1], [-0.3]),
        ],
        1.0,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert sample.training.get("topk_indices", []) == []
    assert sample.training.get("topk_log_probs", []) == []


@pytest.mark.unit
def test_multi_turn_policy_sample_realigns_latest_response_drift() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [1, 2, 3], [1], [-0.3]),
            _turn("i2", [1, 2, 30, 4], [1], [-0.4]),
        ],
        1.0,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert tuple(sample.training.get("tokens", [])) == (1, 2, 30, 4)
    assert tuple(sample.training.get("loss_mask", [])) == (0, 1)
    assert tuple(sample.training.get("rollout_log_probs", [])) == (0.0, -0.4)


@pytest.mark.unit
def test_multi_turn_policy_sample_bounds_replaced_tokens_not_new_output() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [1, 2, 3], [1], [-0.3]),
            _turn("i2", [1, 2, 30, 4, 5, 6], [1, 1, 1], [-0.4, -0.5, -0.6]),
        ],
        1.0,
        source_agent_record_id="report-1",
        realign_threshold=1,
    )

    assert sample is not None
    assert tuple(sample.training.get("loss_mask", [])) == (0, 1, 1, 1)
    assert tuple(sample.training.get("rollout_log_probs", [])) == (0.0, -0.4, -0.5, -0.6)


@pytest.mark.unit
def test_multi_turn_policy_sample_rejects_replacement_over_threshold() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [1, 2, 3, 4, 5], [1, 1, 1], [-0.1, -0.2, -0.3]),
            _turn("i2", [1, 2, 30, 40, 50, 6], [1], [-0.4]),
        ],
        1.0,
        source_agent_record_id="report-1",
        realign_threshold=2,
    )

    assert sample is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "turns",
    [
        [
            _turn("i1", [1, 2, 3], [1], [-0.3]),
            _turn("i2", [1, 9, 3, 4], [1], [-0.4]),
        ],
        [
            _turn("i1", [1, 2, 3], [1], [-0.3], artifact_id="artifact-1"),
            _turn("i2", [1, 2, 3, 4], [1], [-0.4], artifact_id="artifact-2"),
        ],
    ],
)
def test_multi_turn_policy_sample_rejects_forks_and_mixed_weights(turns) -> None:
    assert (
        make_multi_turn_policy_trajectory(
            turns,
            1.0,
            source_agent_record_id="report-1",
        )
        is None
    )


@pytest.mark.unit
def test_multi_turn_scaffold_tolerance_realls_masked_scaffold_only() -> None:
    """A thinking template drops the generation scaffold when re-rendering
    history, so turn N+1's prompt diverges just *before* turn N's response.
    With the default strict boundary that is a fork; with a scaffold
    tolerance covering the divergence it realigns, and the trained span is
    the latest response only (the realigned region is masked)."""
    # turn 0: prompt [1, 2, 99] where 99 is generation scaffold, response [3, 4]
    # turn 1: history re-render drops 99 -> prompt [1, 2, 3, 4, 5], response [6]
    turns = [
        _turn("i1", [1, 2, 99, 3, 4], [1, 1], [-0.1, -0.2]),
        _turn("i2", [1, 2, 3, 4, 5, 6], [1], [-0.3]),
    ]

    strict = make_multi_turn_policy_trajectory(turns, 1.0, source_agent_record_id="r1")
    assert strict is None

    tolerant = make_multi_turn_policy_trajectory(turns, 1.0, source_agent_record_id="r1", scaffold_tolerance=2)
    assert tolerant is not None
    assert tuple(tolerant.training.get("tokens", [])) == (1, 2, 3, 4, 5, 6)
    # the loss mask starts after the leading prompt (3 tokens); everything
    # before the final response is masked context
    assert tuple(tolerant.training.get("loss_mask", [])) == (0, 0, 1)
    assert tuple(tolerant.training.get("rollout_log_probs", [])) == (0.0, 0.0, -0.3)
    assert tolerant.training.get("turn_count", 1) == 2

    # a genuine fork deeper than the tolerance still returns None
    forked = [
        _turn("i1", [1, 2, 3, 7, 8], [1, 1], [-0.1, -0.2]),
        _turn("i2", [9, 9, 9, 9, 9, 6], [1], [-0.3]),
    ]
    assert make_multi_turn_policy_trajectory(forked, 1.0, source_agent_record_id="r1", scaffold_tolerance=2) is None
