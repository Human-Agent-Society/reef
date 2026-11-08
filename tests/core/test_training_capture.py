"""The training block of an exact engine sample and the distribution it names."""

from __future__ import annotations

import pytest

from reef.core.training_capture import sample_training, sampling_distribution

FULL = {"temperature": 0.6, "top_p": 1.0, "top_k": -1}


@pytest.mark.parametrize(
    ("params", "distribution"),
    [
        (FULL, {"temperature": 0.6, "top_p": 1.0, "top_k": -1}),
        ({**FULL, "top_k": 0}, {"temperature": 0.6, "top_p": 1.0, "top_k": -1}),
        ({**FULL, "top_k": 20, "top_p": 0.95}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
        ({**FULL, "repetition_penalty": 1.0, "min_p": 0.0, "logit_bias": {}}, {**FULL, "temperature": 0.6}),
        ({"temperature": 0.6, "top_p": 1.0}, None),
        ({**FULL, "temperature": 0.0}, None),
        ({**FULL, "top_p": 0.0}, None),
        ({**FULL, "min_p": 0.05}, None),
        ({**FULL, "presence_penalty": 0.5}, None),
        ({**FULL, "logit_bias": {"7": 2.0}}, None),
    ],
)
def test_the_distribution_is_named_only_when_the_parameters_fix_it(params, distribution) -> None:
    assert sampling_distribution(params) == distribution


def test_a_training_block_needs_one_log_prob_per_sampled_token() -> None:
    block = sample_training(
        prompt_ids=[1, 2],
        output_ids=[3],
        rollout_log_probs=[-0.5],
        runtime_load_id="e:1",
        finish_reason="length",
        distribution=None,
    )
    assert block["tokens"] == [1, 2, 3] and block["loss_mask"] == [1] and "distribution" not in block
    with pytest.raises(ValueError, match="one log-prob per sampled token"):
        sample_training(
            prompt_ids=[1],
            output_ids=[3, 4],
            rollout_log_probs=[-0.5],
            runtime_load_id="e:1",
            finish_reason="stop",
            distribution=FULL,
        )
