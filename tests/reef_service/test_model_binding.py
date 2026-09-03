"""The model bindings reef hands to methods and episodes: one value per
endpoint, translated per API dialect, with the named set a method may call."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.harness.render import render_composition
from reef.recipe import RecipeConfigError
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.train.cordis_backend import CordisRecipe


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch: pytest.MonkeyPatch, reply: Any) -> list[dict[str, Any]]:
    """Route urlopen to a recorder; ``reply`` is the JSON body or SSE bytes."""
    seen: list[dict[str, Any]] = []

    def urlopen(request, timeout=None):
        seen.append(
            {
                "url": request.full_url,
                "headers": {k.lower(): v for k, v in request.header_items()},
                "body": json.loads(request.data),
                "timeout": timeout,
            }
        )
        return _Response(reply if isinstance(reply, bytes) else json.dumps(reply).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return seen


# -- dialects ----------------------------------------------------------------


def test_openai_chat_posts_chat_completions_with_a_bearer(monkeypatch) -> None:
    seen = _capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    binding = ModelBinding("http://up/", "m", api_key="k")
    assert (
        binding.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], temperature=0.2) == "hi"
    )
    call = seen[0]
    assert call["url"] == "http://up/v1/chat/completions"
    assert call["headers"]["authorization"] == "Bearer k"
    assert call["body"]["model"] == "m" and call["body"]["temperature"] == 0.2
    assert [m["role"] for m in call["body"]["messages"]] == ["system", "user"]


def test_responses_chat_posts_input_with_a_bearer(monkeypatch) -> None:
    reply = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ]
    }
    seen = _capture(monkeypatch, reply)
    binding = ModelBinding("http://up/", "m", api_key="k", api="responses")
    assert binding.chat([{"role": "user", "content": "u"}], temperature=0.2, max_tokens=16) == "hi"
    call = seen[0]
    assert call["url"] == "http://up/v1/responses"
    assert call["headers"]["authorization"] == "Bearer k"
    assert call["body"]["input"] == [{"role": "user", "content": "u"}]
    assert call["body"]["max_output_tokens"] == 16 and "max_tokens" not in call["body"]
    with pytest.raises(ValueError, match="both max_tokens and max_output_tokens"):
        binding.chat([{"role": "user", "content": "u"}], max_tokens=1, max_output_tokens=2)


def test_anthropic_chat_posts_messages_with_x_api_key_and_a_system_field(monkeypatch) -> None:
    seen = _capture(monkeypatch, {"content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "!"}]})
    binding = ModelBinding("http://up", "claude", api_key="k", api="anthropic")
    assert binding.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], max_tokens=16) == "hi!"
    call = seen[0]
    assert call["url"] == "http://up/v1/messages"
    assert call["headers"]["x-api-key"] == "k" and call["headers"]["anthropic-version"]
    assert "authorization" not in call["headers"]
    assert call["body"]["system"] == "s"
    assert call["body"]["messages"] == [{"role": "user", "content": "u"}]  # system never rides as a message
    assert call["body"]["max_tokens"] == 16


def test_anthropic_chat_supplies_the_required_max_tokens(monkeypatch) -> None:
    seen = _capture(monkeypatch, {"content": [{"type": "text", "text": "x"}]})
    ModelBinding("http://up", "claude", api="anthropic").chat([{"role": "user", "content": "u"}])
    assert seen[0]["body"]["max_tokens"] > 0


def test_streams_fold_to_one_reply_in_all_dialects(monkeypatch) -> None:
    openai = b'data: {"choices":[{"delta":{"role":"assistant","content":"a"}}]}\ndata: {"choices":[{"delta":{"content":"b"}}]}\ndata: [DONE]\n'
    _capture(monkeypatch, openai)
    assert ModelBinding("http://up", "m").chat([{"role": "user", "content": "u"}], stream=True) == "ab"
    anthropic = (
        b'data: {"type":"message_start","message":{"model":"claude"}}\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"b"}}\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n'
    )
    _capture(monkeypatch, anthropic)
    assert (
        ModelBinding("http://up", "claude", api="anthropic").chat([{"role": "user", "content": "u"}], stream=True)
        == "ab"
    )
    responses = (
        b'data: {"type":"response.output_text.delta","delta":"a"}\n'
        b'data: {"type":"response.output_text.delta","delta":"b"}\n'
        b"data: [DONE]\n"
    )
    _capture(monkeypatch, responses)
    assert (
        ModelBinding("http://up", "m", api="responses").chat([{"role": "user", "content": "u"}], stream=True) == "ab"
    )


def test_unknown_api_is_refused() -> None:
    with pytest.raises(ValueError, match="api must be one of"):
        ModelBinding("http://up", "m", api="cohere")


def test_episode_templates_follow_the_dialect() -> None:
    pi = get_adapter("pi")
    openai = render_composition(ModelBinding("http://up", "m", api_key="k").compose_nodes(pi), pi)
    anthropic = render_composition(ModelBinding("http://up", "m", api_key="k", api="anthropic").compose_nodes(pi), pi)
    assert json.loads(openai["pi-agent/models.json"])["providers"]["reef"]["api"] == "openai-completions"
    assert json.loads(openai["pi-agent/models.json"])["providers"]["reef"]["baseUrl"] == "http://up/v1"
    assert json.loads(anthropic["pi-agent/models.json"])["providers"]["reef"]["api"] == "anthropic-messages"
    assert json.loads(anthropic["pi-agent/models.json"])["providers"]["reef"]["baseUrl"] == "http://up"
    for files in (openai, anthropic):
        assert json.loads(files["pi-agent/settings.json"])["defaultModel"] == "reef/m"


def test_the_dialect_rides_the_proxy_runtime_into_the_binding() -> None:
    runtime = InferenceProxyRuntime(model_path="claude", base_url="http://up", api_key="k", api="anthropic")
    binding = ModelBinding.from_runtime(runtime)
    assert (binding.api, binding.model, binding.api_key) == ("anthropic", "claude", "k")
    responses = ModelBinding.from_runtime(
        InferenceProxyRuntime(model_path="gpt", base_url="http://up", api="responses")
    )
    assert responses.api == "responses"
    with pytest.raises(ValueError, match="api must be one of"):
        InferenceProxyRuntime(base_url="http://up", api="grpc")


# -- the named set -----------------------------------------------------------


def test_model_bindings_expose_served_and_named_models() -> None:
    served = ModelBinding("http://up", "small")
    teacher = ModelBinding("http://big", "large", api_key="k")
    models = ModelBindings(served=served, named={"teacher": teacher})
    assert models.served is served and models["served"] is served and models["teacher"] is teacher
    assert list(models) == ["served", "teacher"] and len(models) == 2
    with pytest.raises(KeyError, match=r"no model named 'judge'; declared under evolution.models: teacher"):
        models["judge"]
    with pytest.raises(ValueError, match="'served' is reserved"):
        ModelBindings(served=served, named={"served": teacher})


def test_from_config_reads_the_key_from_the_named_environment_variable() -> None:
    binding = ModelBinding.from_config(
        {"url": "https://api.example/", "model": "gpt", "api_key_env": "TEACHER_KEY", "timeout_s": 30},
        {"TEACHER_KEY": "sk-teacher"},
        where="evolution.models.teacher",
    )
    assert (binding.base_url, binding.model, binding.api_key, binding.api, binding.timeout_s) == (
        "https://api.example",
        "gpt",
        "sk-teacher",
        "openai",
        30.0,
    )
    assert ModelBinding.from_config({"url": "http://u", "model": "m"}, {}).api_key is None
    with pytest.raises(ValueError, match=r"evolution\.models\.t\.url must be a non-empty string"):
        ModelBinding.from_config({"model": "m"}, {}, where="evolution.models.t")


def test_recipe_declares_named_models_under_evolution_models(tmp_path) -> None:
    module = tmp_path / "demo_models.py"
    module.write_text(
        "def propose(nodes, samples, models):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        config = {
            "model": {"path": "small"},
            "evolution": {
                "propose": "demo_models:propose",
                "evaluate": "demo_models:evaluate",
                "tasks": ["t"],
                "models": {"teacher": {"url": "http://big", "model": "large", "api_key_env": "TEACHER_KEY"}},
            },
        }
        runtime = InferenceProxyRuntime(model_path="small", base_url="http://up")
        built = CordisRecipe.from_environment({"TEACHER_KEY": "sk-t"}, config=config, runtime=runtime)
        models = built.model_bindings()
        assert models.served.model == "small" and models["teacher"].api_key == "sk-t"

        bad = {**config, "evolution": {**config["evolution"], "models": {"served": {"url": "http://x", "model": "m"}}}}
        with pytest.raises(RecipeConfigError, match="may not name a model 'served'"):
            CordisRecipe.from_environment({}, config=bad, runtime=runtime)
        bad = {**config, "evolution": {**config["evolution"], "models": {"t": {"model": "m"}}}}
        with pytest.raises(RecipeConfigError, match=r"evolution\.models\.t\.url"):
            CordisRecipe.from_environment({}, config=bad, runtime=runtime)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("demo_models", None)
