from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.service.request_service import client_inference_response
from reef.train.processors.common import make_multi_turn_policy_sample, make_policy_sample


def _inference(payload) -> AgentRecord:
    return AgentRecord.create(scenario="s", request_type=RequestType.INFERENCE, payload=payload)


def _turn(
    agent_record_id: str,
    tokens: list[int],
    loss_mask: list[int],
    log_probs: list[float],
    *,
    runtime_load_id: str = "wv-1",
) -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        payload={
            "response": {
                "training": {
                    "tokens": tokens,
                    "loss_mask": loss_mask,
                    "rollout_log_probs": log_probs,
                    "runtime_load_id": runtime_load_id,
                }
            }
        },
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
                    "runtime_load_id": "wv-1",
                }
            }
        }
    )

    sample = make_policy_sample(item, 1.0)

    assert sample.tokens == (10, 11, 20)
    assert sample.loss_mask == (1,)
    assert sample.rollout_log_probs == (-0.25,)
    assert sample.runtime_load_id == "wv-1"
    assert sample.turn_count == 1
    assert sample.is_multi_turn is False


@pytest.mark.unit
def test_policy_sample_preserves_mixed_token_runtime_load_ids() -> None:
    spans = [
        {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
        {"start": 1, "end": 3, "runtime_load_id": "engine:7"},
    ]
    item = _inference(
        {
            "runtime_load_spans": spans,
            "response": {
                "training": {
                    "tokens": [10, 20, 21, 22],
                    "loss_mask": [1, 1, 1],
                    "rollout_log_probs": [-0.1, -0.2, -0.3],
                    "runtime_load_id": None,
                    "runtime_load_spans": spans,
                }
            },
        }
    )

    sample = make_policy_sample(item, 1.0)

    assert sample.runtime_load_id is None
    assert [(span.start, span.end, span.runtime_load_id) for span in sample.runtime_load_spans] == [
        (0, 1, "engine:6"),
        (1, 3, "engine:7"),
    ]


@pytest.mark.unit
def test_policy_sample_rejects_conflicting_validated_and_training_runtime_load_ids() -> None:
    item = _inference(
        {
            "runtime_load_id": "engine:3",
            "response": {
                "training": {
                    "tokens": [10, 11, 20],
                    "loss_mask": [1],
                    "rollout_log_probs": [-0.25],
                    "runtime_load_id": "engine:1",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="disagrees with the validated record"):
        make_policy_sample(item, 1.0)


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

    sample = make_policy_sample(item, 1.0)

    assert sample.tokens == ()
    assert sample.loss_mask == ()
    assert sample.rollout_log_probs == ()


@pytest.mark.unit
def test_policy_sample_accepts_exact_harness_tensors() -> None:
    item = _inference(
        {
            "tokens": [1, 2, 3],
            "loss_mask": [1],
            "rollout_log_probs": [-0.5],
            "runtime_load_id": "harness-v1",
        }
    )

    sample = make_policy_sample(item, 0.5)

    assert sample.tokens == (1, 2, 3)
    assert sample.loss_mask == (1,)
    assert sample.rollout_log_probs == (-0.5,)
    assert sample.runtime_load_id == "harness-v1"


@pytest.mark.unit
def test_client_response_hides_private_training_tensors() -> None:
    response = {
        "choices": [{"message": {"content": "answer"}}],
        "training": {
            "tokens": [10, 20],
            "loss_mask": [1],
            "rollout_log_probs": [-0.25],
        },
    }

    client_response = client_inference_response(response)

    assert "training" not in client_response
    assert response["training"]["rollout_log_probs"] == [-0.25]


@pytest.mark.unit
def test_multi_turn_policy_sample_assembles_clean_linear_history() -> None:
    sample = make_multi_turn_policy_sample(
        [
            _turn("i1", [10, 11, 20, 21], [1, 1], [-0.1, -0.2]),
            _turn("i2", [10, 11, 20, 21, 30, 40], [1], [-0.3]),
            _turn("i3", [10, 11, 20, 21, 30, 40, 31, 50, 51], [1, 1], [-0.4, -0.5]),
        ],
        0.75,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert sample.source_agent_record_id == "report-1"
    assert sample.tokens == (10, 11, 20, 21, 30, 40, 31, 50, 51)
    assert sample.loss_mask == (1, 1, 0, 1, 0, 1, 1)
    assert sample.rollout_log_probs == (-0.1, -0.2, 0.0, -0.3, 0.0, -0.4, -0.5)
    assert sample.reward == 0.75
    assert sample.runtime_load_id == "wv-1"
    assert sample.turn_count == 3
    assert sample.is_multi_turn is True


@pytest.mark.unit
def test_multi_turn_policy_sample_realigns_latest_response_drift() -> None:
    sample = make_multi_turn_policy_sample(
        [
            _turn("i1", [1, 2, 3], [1], [-0.3]),
            _turn("i2", [1, 2, 30, 4], [1], [-0.4]),
        ],
        1.0,
        source_agent_record_id="report-1",
    )

    assert sample is not None
    assert sample.tokens == (1, 2, 30, 4)
    assert sample.loss_mask == (0, 1)
    assert sample.rollout_log_probs == (0.0, -0.4)


@pytest.mark.unit
def test_multi_turn_policy_sample_bounds_replaced_tokens_not_new_output() -> None:
    sample = make_multi_turn_policy_sample(
        [
            _turn("i1", [1, 2, 3], [1], [-0.3]),
            _turn("i2", [1, 2, 30, 4, 5, 6], [1, 1, 1], [-0.4, -0.5, -0.6]),
        ],
        1.0,
        source_agent_record_id="report-1",
        realign_threshold=1,
    )

    assert sample is not None
    assert sample.loss_mask == (0, 1, 1, 1)
    assert sample.rollout_log_probs == (0.0, -0.4, -0.5, -0.6)


@pytest.mark.unit
def test_multi_turn_policy_sample_rejects_replacement_over_threshold() -> None:
    sample = make_multi_turn_policy_sample(
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
            _turn("i1", [1, 2, 3], [1], [-0.3], runtime_load_id="wv-1"),
            _turn("i2", [1, 2, 3, 4], [1], [-0.4], runtime_load_id="wv-2"),
        ],
    ],
)
def test_multi_turn_policy_sample_rejects_forks_and_mixed_weights(turns) -> None:
    assert (
        make_multi_turn_policy_sample(
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

    strict = make_multi_turn_policy_sample(turns, 1.0, source_agent_record_id="r1")
    assert strict is None

    tolerant = make_multi_turn_policy_sample(turns, 1.0, source_agent_record_id="r1", scaffold_tolerance=2)
    assert tolerant is not None
    assert tolerant.tokens == (1, 2, 3, 4, 5, 6)
    # the loss mask starts after the leading prompt (3 tokens); everything
    # before the final response is masked context
    assert tolerant.loss_mask == (0, 0, 1)
    assert tolerant.rollout_log_probs == (0.0, 0.0, -0.3)
    assert tolerant.turn_count == 2

    # a genuine fork deeper than the tolerance still returns None
    forked = [
        _turn("i1", [1, 2, 3, 7, 8], [1, 1], [-0.1, -0.2]),
        _turn("i2", [9, 9, 9, 9, 9, 6], [1], [-0.3]),
    ]
    assert make_multi_turn_policy_sample(forked, 1.0, source_agent_record_id="r1", scaffold_tolerance=2) is None
