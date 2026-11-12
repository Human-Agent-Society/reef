from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.trajectories import has_image_content, make_trajectory, source_record_id, trajectory_reward
from reef.record2sample.samples import make_multi_turn_policy_trajectory, make_policy_trajectory
from reef.service.inference_service import provider_response


def _inference(payload) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload=payload, artifact_id=payload.get("artifact_id")
    )


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
def test_multi_turn_scaffold_tolerance_keeps_the_scaffold_and_trains_every_turn() -> None:
    """A thinking template drops the generation scaffold when re-rendering history, so turn N+1's prompt diverges just
    *before* turn N's response. With the default strict boundary that is a fork; with a scaffold tolerance covering
    the divergence the transcript keeps the sampled scaffold and response, and both responses train."""
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
    assert tuple(tolerant.training.get("tokens", [])) == (1, 2, 99, 3, 4, 5, 6)
    # the loss mask starts after the leading prompt (3 tokens): both responses train
    assert tuple(tolerant.training.get("loss_mask", [])) == (1, 1, 0, 1)
    assert tuple(tolerant.training.get("rollout_log_probs", [])) == (-0.1, -0.2, 0.0, -0.3)
    assert tolerant.training.get("turn_count", 1) == 2

    # a genuine fork deeper than the tolerance still returns None
    forked = [
        _turn("i1", [1, 2, 3, 7, 8], [1, 1], [-0.1, -0.2]),
        _turn("i2", [9, 9, 9, 9, 9, 6], [1], [-0.3]),
    ]
    assert make_multi_turn_policy_trajectory(forked, 1.0, source_agent_record_id="r1", scaffold_tolerance=2) is None


PNG_URL = "data:image/png;base64,AAAA"
IMAGE_URL_PART = {"type": "image_url", "image_url": {"url": PNG_URL}}
ANTHROPIC_IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": [IMAGE_URL_PART]}]},
        {"messages": [{"role": "user", "content": [{"image_url": {"url": PNG_URL}}]}]},
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/bmp;base64,A"}}]}
            ]
        },
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x.test/a"}}]}]},
        {"messages": [{"role": "user", "content": [ANTHROPIC_IMAGE]}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "file", "file_id": "f"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": [ANTHROPIC_IMAGE]}]}]},
        {"messages": [{"role": "tool", "tool_call_id": "c", "content": [IMAGE_URL_PART]}]},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": PNG_URL}]}]},
        {"input": [{"type": "computer_call_output", "output": {"type": "computer_screenshot", "image_url": PNG_URL}}]},
        {"input": [{"type": "function_call_output", "output": {"type": "input_image", "image_url": PNG_URL}}]},
        {"input": [{"type": "custom_tool_call_output", "output": [{"type": "input_image", "file_id": "f"}]}]},
        {"messages": [], "response": {"output": [{"type": "image_generation_call", "result": "AAAA"}]}},
        {"messages": [], "response": {"choices": [{"message": {"content": [IMAGE_URL_PART]}}]}},
        {"steps": [{"source": "user", "message": [{"type": "image", "source": {"path": PNG_URL}}]}]},
        {"steps": [{"source": "agent", "observation": {"results": [{"content": [{"type": "image", "source": {}}]}]}}]},
        {"system": [ANTHROPIC_IMAGE]},
        {"instructions": [IMAGE_URL_PART]},
        {"prompt": [{"role": "user", "content": [IMAGE_URL_PART]}]},
        {"messages": [{"role": "user", "content": [{"type": "document", "source": {"content": [ANTHROPIC_IMAGE]}}]}]},
        {"input": [{"type": "computer_call_output", "output": {"type": "computer_screenshot", "file_id": "f"}}]},
        {"input": [{"type": "code_interpreter_call", "outputs": [{"type": "image", "url": PNG_URL}]}]},
        {"response": {"training": {"request_messages": [{"role": "user", "content": [IMAGE_URL_PART]}]}}},
        {"response": {"training": {"response_message": {"content": [IMAGE_URL_PART]}}}},
    ],
)
def test_has_image_content_finds_an_image_in_each_shape(payload) -> None:
    assert has_image_content(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "an image_url"}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "image"}]}]},
        # A tool's definition and its call's arguments are not conversation content.
        {"messages": [], "tools": [{"type": "function", "function": {"parameters": {"type": "image"}}}]},
        {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "input": {"type": "image"}}]}]},
        {"input": "plain text"},
        {"steps": [{"source": "user", "message": "q"}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "q", "image_url": None}]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "q", "image_url": "a"}]}]},
        {"messages": [{"role": "user", "content": "q", "image_url": None}]},
        {"input": [{"type": "input_text", "text": "q", "image_url": None}]},
        {"messages": [{"role": "user", "content": [{"type": ["image"], "text": "q"}]}]},
        {},
    ],
)
def test_has_image_content_ignores_text_and_tools(payload) -> None:
    assert not has_image_content(payload)


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "computer_call_output",
            "call_id": "c",
            "output": {"type": "computer_screenshot", "image_url": PNG_URL},
        },
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/bmp;base64,AAAA"}]},
    ],
)
def test_has_image_content_finds_an_image_that_an_atif_step_does_not_keep(item) -> None:
    payload = {
        "model": "m",
        "input": [{"role": "user", "content": "q"}, item],
        "response": {
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]}]
        },
    }
    document = make_trajectory((_inference(payload),), 1.0).trajectory
    assert has_image_content(document)
    assert not has_image_content({**document, "extra": {"reef": {}}}), "only the captured records hold it"


# Qwen3.5's empty think block with thinking off: <think> \n\n </think> \n\n.
EMPTY_THINK = [248068, 271, 248069, 271]


def scaffold_dropping_rollout() -> list[AgentRecord]:
    """Three calls whose every prompt ends with the empty think block, which the next prompt's history drops."""
    return [
        _turn("i1", [10, 11, *EMPTY_THINK, 20, 21], [1, 1], [-0.1, -0.2]),
        _turn("i2", [10, 11, 20, 21, 30, *EMPTY_THINK, 40], [1], [-0.3]),
        _turn("i3", [10, 11, 20, 21, 30, 40, 50, *EMPTY_THINK, 60, 61], [1, 1], [-0.4, -0.5]),
    ]


@pytest.mark.unit
def test_a_rollout_whose_history_drops_the_scaffold_trains_every_turn() -> None:
    """Turn 3's prompt lacks both scaffolds that the transcript keeps: each prompt is compared with the previous call,
    so it still extends, and every response trains with its sampler's log probabilities."""
    turns = scaffold_dropping_rollout()

    sample = make_multi_turn_policy_trajectory(turns, 1.0, source_agent_record_id="r1", scaffold_tolerance=8)

    assert sample is not None
    assert sample.training["tokens"] == [10, 11, *EMPTY_THINK, 20, 21, 30, *EMPTY_THINK, 40, 50, *EMPTY_THINK, 60, 61]
    assert sum(sample.training["loss_mask"]) == sum(
        len(turn.payload["response"]["training"]["loss_mask"]) for turn in turns
    )
    trained = [index for index, value in enumerate(sample.training["loss_mask"]) if value]
    assert [sample.training["rollout_log_probs"][index] for index in trained] == [-0.1, -0.2, -0.3, -0.4, -0.5]
    assert sample.training["turn_count"] == 3


@pytest.mark.unit
def test_the_default_tolerance_makes_a_dropped_scaffold_a_fork() -> None:
    assert make_multi_turn_policy_trajectory(scaffold_dropping_rollout(), 1.0, source_agent_record_id="r1") is None


@pytest.mark.unit
def test_a_drifted_response_is_masked_and_the_other_turns_train() -> None:
    """The history also trimmed a trailing newline of turn 1's response: that response realigns as masked context,
    and the other turns still train."""
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [10, *EMPTY_THINK, 20, 198], [1, 1], [-0.1, -0.2]),
            _turn("i2", [10, 20, 30, *EMPTY_THINK, 40], [1], [-0.3]),
            _turn("i3", [10, 20, 30, 40, 50, *EMPTY_THINK, 60], [1], [-0.4]),
        ],
        1.0,
        source_agent_record_id="r1",
        scaffold_tolerance=8,
    )

    assert sample is not None
    assert sample.training["tokens"] == [10, 20, 30, *EMPTY_THINK, 40, 50, *EMPTY_THINK, 60]
    # The loss mask starts after turn 1's prompt (5 tokens).
    assert sample.training["loss_mask"] == [0, 0, 1, 0, 0, 0, 0, 0, 1]
    assert sample.training["rollout_log_probs"] == [0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4]


@pytest.mark.unit
def test_the_top_k_rows_stay_aligned_when_the_scaffold_is_kept() -> None:
    sample = make_multi_turn_policy_trajectory(
        [
            _turn("i1", [1, 99, 3], [1], [-0.1], topk=[([3, 5], [-0.1, -1.0])]),
            _turn("i2", [1, 3, 4, 99, 6], [1], [-0.3], topk=[([6, 7], [-0.3, -2.0])]),
        ],
        1.0,
        source_agent_record_id="r1",
        scaffold_tolerance=1,
    )

    assert sample is not None
    assert sample.training["tokens"] == [1, 99, 3, 4, 99, 6]
    assert sample.training["loss_mask"] == [1, 0, 0, 1]
    assert sample.training["topk_indices"] == [[3, 5], [], [], [6, 7]]
    assert sample.training["topk_log_probs"] == [[-0.1, -1.0], [], [], [-0.3, -2.0]]


@pytest.mark.unit
def test_drift_into_a_dropped_scaffold_is_a_fork() -> None:
    """Turn 2's prompt dropped scaffold 99, which the transcript keeps; turn 3's prompt then rewrites the history
    before it, which the transcript cannot realign."""
    turns = [
        _turn("i1", [1, 2, 99, 3], [1], [-0.1]),
        _turn("i2", [1, 2, 3, 4, 99, 5], [1], [-0.2]),
        _turn("i3", [1, 7, 3, 4, 5, 6, 99, 8], [1], [-0.3]),
    ]

    assert make_multi_turn_policy_trajectory(turns, 1.0, source_agent_record_id="r1", scaffold_tolerance=4) is None
