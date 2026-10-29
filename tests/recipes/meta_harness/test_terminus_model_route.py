"""A terminus tree never chooses where a model call goes: Reef's model binding is its only route.

Render refuses every Terminus 2 argument that chooses the endpoint, the
provider, the credential or the model, in the tree alone (as validation
renders it) and with the binding's overlays (as an episode renders it).
Arguments that tune a run stay admitted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, RenderError, render_tree, validate_tree

OTHER_URL = "http://127.0.0.1:9/v1"


def render(adapter: str, files: Mapping[str, str], *, bound: bool) -> dict[str, str]:
    descriptor = get_adapter(adapter)
    binding = ModelBinding(base_url="http://127.0.0.1:8900", model="served", api_key="k-bound")
    overlays = binding.config_overlays(descriptor) if bound else ()
    return render_tree(HarnessTree(dict(files)), descriptor, overlays)


def config(data: dict[str, Any]) -> dict[str, str]:
    return {"config/primary.json": json.dumps(data)}


TERMINUS_REFUSED = [
    pytest.param({"model_name": "openai/m"}, False, id="model-name-in-tree"),
    pytest.param({"api_base": OTHER_URL}, False, id="endpoint-in-tree"),
    pytest.param({"llm_kwargs": {"base_url": OTHER_URL}}, True, id="constructor-endpoint"),
    pytest.param({"llm_call_kwargs": {"base_url": OTHER_URL}}, True, id="call-base-url"),
    pytest.param({"llm_call_kwargs": {"api_base": OTHER_URL}}, True, id="call-api-base"),
    pytest.param({"llm_call_kwargs": {"custom_llm_provider": "openai"}}, True, id="call-provider"),
    pytest.param({"llm_call_kwargs": {"model": "openai/m"}}, True, id="call-model"),
    pytest.param(
        {"llm_call_kwargs": {"fallbacks": [{"model": "m", "api_base": OTHER_URL}]}}, True, id="call-fallback"
    ),
    pytest.param({"llm_call_kwargs": {"aws_bedrock_runtime_endpoint": OTHER_URL}}, True, id="call-cloud-endpoint"),
    pytest.param(
        {"llm_call_kwargs": {"success_callback": ["langfuse"], "langfuse_host": OTHER_URL}}, True, id="logging"
    ),
    pytest.param({"llm_call_kwargs": {"models": ["m"]}}, True, id="call-fallback-models"),
    pytest.param({"llm_call_kwargs": {"extra_body": {"model": "m"}}}, True, id="body-model"),
    pytest.param({"llm_call_kwargs": {"extra_body": {"models": ["m"]}}}, True, id="body-fallback-models"),
]


@pytest.mark.parametrize(("data", "bound"), TERMINUS_REFUSED)
def test_terminus_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("terminus", config(data), bound=bound)


def test_terminus_refuses_a_request_body_that_is_not_an_object() -> None:
    with pytest.raises(
        RenderError, match=r"on llm_call_kwargs\.extra_body in terminus/config\.json: litellm reads it as an object"
    ):
        render("terminus", config({"llm_call_kwargs": {"extra_body": '{"model": "m"}'}}), bound=True)


def test_terminus_keeps_the_arguments_a_tree_tunes() -> None:
    tuned = {"max_turns": 12, "llm_call_kwargs": {"top_k": 20, "extra_body": {"provider": {"sort": "price"}}}}
    rendered = json.loads(render("terminus", config(tuned), bound=True)["terminus/config.json"])
    assert rendered["llm_call_kwargs"] == tuned["llm_call_kwargs"]
    assert rendered["llm_kwargs"] == {"api_key": "k-bound", "custom_llm_provider": "litellm_proxy"}
    assert rendered["model_name"] == "served"
    render("terminus", config(tuned), bound=False)


@pytest.mark.parametrize(
    "data", [{"llm_call_kwargs": {"base_url": OTHER_URL}}, {"llm_call_kwargs": {"extra_body": {"model": "m"}}}]
)
def test_validation_refuses_the_tree_and_says_why(data: dict[str, Any]) -> None:
    (problem,) = validate_tree(HarnessTree(config(data)), get_adapter("terminus")).problems
    assert "Reef's model binding" in problem
