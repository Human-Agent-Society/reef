"""The training block of an exact engine sample and the distribution it names."""

from __future__ import annotations

import math

import pytest

from reef.core.training_capture import SamplingDistribution, sample_training, sampling_distribution

FULL = {"temperature": 0.6, "top_p": 1.0, "top_k": -1}


@pytest.mark.parametrize(
    ("params", "distribution"),
    [
        (FULL, SamplingDistribution(0.6, 1.0, -1)),
        ({**FULL, "top_k": 0}, SamplingDistribution(0.6, 1.0, -1)),
        ({**FULL, "top_k": 20, "top_p": 0.95}, SamplingDistribution(0.6, 0.95, 20)),
        ({**FULL, "repetition_penalty": 1.0, "min_p": 0.0, "logit_bias": {}}, SamplingDistribution(0.6, 1.0, -1)),
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


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ((0.0, 1.0, -1), "positive finite temperature"),
        ((math.inf, 1.0, -1), "positive finite temperature"),
        ((True, 1.0, -1), "temperature must be a number"),
        ((0.6, 1.5, -1), "top_p in"),
        ((0.6, 1.0, 0), "top_k must be -1 or a positive int"),
        ((0.6, 1.0, 2.0), "top_k must be -1 or a positive int"),
    ],
)
def test_a_distribution_outside_its_domain_is_refused(fields, message) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingDistribution(*fields)


def test_a_distribution_reads_back_from_its_dict_and_only_from_its_dict() -> None:
    distribution = SamplingDistribution(1, 0.95, 20)
    assert distribution.to_dict() == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert SamplingDistribution.from_dict(distribution.to_dict()) == distribution
    for value in (None, {"temperature": 1.0, "top_p": 0.95}, {**FULL, "min_p": 0.0}):
        with pytest.raises(ValueError, match="exactly"):
            SamplingDistribution.from_dict(value)


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
    named = sample_training(
        prompt_ids=[1],
        output_ids=[3],
        rollout_log_probs=[-0.5],
        runtime_load_id="e:1",
        finish_reason="stop",
        distribution=SamplingDistribution(0.6, 1.0, -1),
    )
    assert named["distribution"] == FULL
    with pytest.raises(ValueError, match="one log-prob per sampled token"):
        sample_training(
            prompt_ids=[1],
            output_ids=[3, 4],
            rollout_log_probs=[-0.5],
            runtime_load_id="e:1",
            finish_reason="stop",
            distribution=SamplingDistribution(0.6, 1.0, -1),
        )
