"""The token-native chat handler's serving fields (RFC #1097): a scenario's sampling values are defaults."""

from __future__ import annotations

import pytest

from reef.inference.sglang.chat import SGLangInferenceHandler


def handler() -> SGLangInferenceHandler:
    return SGLangInferenceHandler(
        "http://unused", model_path="model", sampling_defaults={"temperature": 0.7, "max_new_tokens": 64}
    )


@pytest.mark.unit
def test_the_fields_take_the_scenario_value_else_the_sampling_default():
    assert handler().serving_config({}) == {"temperature": 0.7}
    assert handler().serving_config({"top_p": 0.9, "top_k": -1}) == {"temperature": 0.7, "top_p": 0.9, "top_k": -1}
    assert handler().serving_config({"temperature": 0}) == {"temperature": 0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "override",
    [
        {"max_new_tokens": 128},
        {"min_p": 0.1},
        {"temperature": -0.1},
        {"temperature": True},
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"top_k": 2.0},
    ],
)
def test_another_field_or_value_is_refused(override):
    with pytest.raises(ValueError):
        handler().serving_config(override)


@pytest.mark.unit
def test_a_scenario_value_applies_only_where_the_client_sets_none():
    body = {"model": "m", "messages": [], "temperature": 1.0, "top_p": None}
    sent = handler().request_payload(body, serving_config={"temperature": 0.3, "top_p": 0.9, "min_replicas": 1})
    assert sent == {"model": "m", "messages": [], "temperature": 1.0, "top_p": 0.9}
    assert body["top_p"] is None
