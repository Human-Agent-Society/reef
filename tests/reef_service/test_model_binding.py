"""The model bindings reef hands to methods and episodes: one value per
endpoint, translated per API dialect, with the named set a method may call."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import NO_KEY_PLACEHOLDER, ModelBinding, ModelBindings
from reef.harness.tree.render import render_composition
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


def test_complete_reports_the_tokens_the_endpoint_counted_in_every_dialect(monkeypatch) -> None:
    """``usage`` rides the response object and ``last_usage`` keeps it for a
    wrapper of ``chat``, normalised to input/output tokens; a stream folds its
    usage in from the chunk or event that carried it."""
    from reef.harness.episodes.model_binding import usage_of

    binding = ModelBinding("http://up", "m")
    assert binding.last_usage() is None
    _capture(
        monkeypatch,
        {
            "choices": [{"message": {"role": "assistant", "content": "a"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )
    assert binding.chat([{"role": "user", "content": "u"}]) == "a"
    assert binding.last_usage() == {"input_tokens": 12, "output_tokens": 3}
    _capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "b"}}]})
    binding.chat([{"role": "user", "content": "u"}])
    assert binding.last_usage() is None
    openai = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"a"}}]}\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":1}}\ndata: [DONE]\n'
    )
    _capture(monkeypatch, openai)
    response = binding.complete({"messages": [], "stream": True})
    assert response["choices"][0]["message"]["content"] == "a" and usage_of(response) == {
        "input_tokens": 7,
        "output_tokens": 1,
    }
    anthropic = (
        b'data: {"type":"message_start","message":{"model":"claude","usage":{"input_tokens":20,"output_tokens":1}}}\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n'
    )
    _capture(monkeypatch, anthropic)
    claude = ModelBinding("http://up", "claude", api="anthropic")
    assert claude.chat([{"role": "user", "content": "u"}], stream=True) == "a"
    assert claude.last_usage() == {"input_tokens": 20, "output_tokens": 5}
    assert usage_of({"usage": {"input_tokens": True}}) is None and usage_of({"usage": {"output_tokens": 4}}) == {
        "input_tokens": 0,
        "output_tokens": 4,
    }


def test_the_budgeted_binding_records_each_calls_usage_for_the_step(monkeypatch) -> None:
    from reef.train.cordis_backend.backend import _BudgetedBinding

    record: list[dict[str, Any]] = []
    budgeted = _BudgetedBinding(ModelBinding("http://up", "m"), [0], 0, record)
    _capture(
        monkeypatch,
        {
            "choices": [{"message": {"role": "assistant", "content": "a"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2},
        },
    )
    assert budgeted.chat([{"role": "user", "content": "u"}]) == "a"
    _capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "b"}}]})
    budgeted.chat([{"role": "user", "content": "u"}])
    assert record[0]["usage"] == {"input_tokens": 9, "output_tokens": 2} and "usage" not in record[1]


def test_chat_record_keeps_provider_reasoning_separate_from_reply(monkeypatch) -> None:
    from reef.train.cordis_backend.backend import RECORD_TEXT_CAP, _BudgetedBinding

    response = {
        "id": "response-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "the answer",
                    "reasoning": "a" * (RECORD_TEXT_CAP + 3),
                    "reasoning_details": [{"type": "reasoning.text", "text": "sk-test-0123456789abcdefghijklmnop"}],
                }
            }
        ],
    }
    record: list[dict[str, Any]] = []
    inner = ModelBinding("http://up", "m")
    budgeted = _BudgetedBinding(inner, [0], 0, record)
    _capture(monkeypatch, response)
    assert budgeted.chat([]) == "the answer"
    assert record[0]["reply"] == "the answer"
    captured = record[0]["response"]["choices"][0]["message"]
    assert captured["reasoning"].endswith("[clipped 3 chars]")
    assert captured["reasoning_details"][0]["text"] == "[redacted credential]"
    # Capturing is a copy: clipping and redaction must not alter the live provider response.
    assert inner.last_response() == response
    _capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "plain"}}]})
    assert budgeted.chat([]) == "plain"
    assert "reasoning" not in record[1]["response"]["choices"][0]["message"]


def test_stream_keeps_reasoning_fields_and_merges_detail_fragments(monkeypatch) -> None:
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning": "first ",
                        "reasoning_content": "one ",
                        "reasoning_details": [{"index": 0, "type": "reasoning.text", "text": "step "}],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "reasoning": "second",
                        "reasoning_content": "two",
                        "reasoning_details": [
                            {"index": 0, "text": "by step"},
                            {"index": 1, "type": "reasoning.encrypted", "data": "opaque"},
                        ],
                    }
                }
            ]
        },
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
    ]
    _capture(monkeypatch, "".join(f"data: {json.dumps(chunk)}\n" for chunk in chunks).encode())
    binding = ModelBinding("http://up", "m")
    assert binding.chat([], stream=True) == "answer"
    message = binding.last_response()["choices"][0]["message"]
    assert message["reasoning"] == "first second"
    assert message["reasoning_content"] == "one two"
    assert message["reasoning_details"] == [
        {"index": 0, "type": "reasoning.text", "text": "step by step"},
        {"index": 1, "type": "reasoning.encrypted", "data": "opaque"},
    ]


def test_anthropic_stream_keeps_thinking_and_signature_blocks(monkeypatch) -> None:
    events = [
        {"type": "message_start", "message": {"model": "claude"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "consider "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "this"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "signed"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "opaque"}},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "answer"}},
    ]
    _capture(monkeypatch, "".join(f"data: {json.dumps(event)}\n" for event in events).encode())
    binding = ModelBinding("http://up", "claude", api="anthropic")
    assert binding.chat([], stream=True) == "answer"
    assert binding.last_response()["content"] == [
        {"type": "thinking", "thinking": "consider this", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "answer"},
    ]


def test_responses_stream_keeps_reasoning_output_items(monkeypatch) -> None:
    output = [
        {"id": "reason-1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Consider this."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
    ]
    events = [
        {"type": "response.output_item.done", "output_index": index, "item": item} for index, item in enumerate(output)
    ]
    _capture(monkeypatch, "".join(f"data: {json.dumps(event)}\n" for event in events).encode())
    binding = ModelBinding("http://up", "m", api="responses")
    assert binding.chat([], stream=True) == "answer"
    assert binding.last_response()["output"] == output
    # A stream ending after text deltas must retain that text alongside the completed reasoning item.
    events[-1] = {"type": "response.output_text.delta", "delta": "answer"}
    _capture(monkeypatch, "".join(f"data: {json.dumps(event)}\n" for event in events).encode())
    assert binding.chat([], stream=True) == "answer"
    assert binding.last_response()["output"] == output


def test_record_does_not_reuse_a_previous_response_when_custom_chat_returns_text(monkeypatch) -> None:
    from reef.train.cordis_backend.backend import _BudgetedBinding

    class CustomChat(ModelBinding):
        def chat(self, messages, **params):
            return "custom text"

    inner = CustomChat("http://up", "m")
    _capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "previous", "reasoning": "old"}}]})
    inner.complete({"messages": []})
    assert inner.last_response() is not None
    record: list[dict[str, Any]] = []
    assert _BudgetedBinding(inner, [0], 0, record).chat([]) == "custom text"
    assert "response" not in record[0]


def test_failed_request_clears_the_previous_response(monkeypatch) -> None:
    from reef.harness.episodes.model_binding import ModelBindingError

    binding = ModelBinding("http://up", "m")
    _capture(monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "answer", "reasoning": "old"}}]})
    binding.chat([])
    _capture(monkeypatch, b"invalid json")
    with pytest.raises(ModelBindingError):
        binding.chat([])
    assert binding.last_response() is None


def test_proposer_error_retains_reasoning_when_provider_returns_no_final_text(monkeypatch) -> None:
    from reef.harness.episodes.model_binding import ModelBindingError
    from reef.train.cordis_backend.backend import _BudgetedBinding

    record: list[dict[str, Any]] = []
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": None, "reasoning": "Still considering."},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 20},
    }
    _capture(monkeypatch, response)
    with pytest.raises(ModelBindingError, match="non-text"):
        _BudgetedBinding(ModelBinding("http://up", "m"), [0], 0, record).chat([])
    assert "error" in record[0] and "reply" not in record[0]
    assert record[0]["response"] == response
    assert record[0]["usage"] == {"input_tokens": 3, "output_tokens": 20}


def test_reasoning_fragments_do_not_merge_unidentified_blocks_or_stringify_null_signatures() -> None:
    from reef.harness.episodes.model_binding import _merge_reasoning_details

    details: list[Any] = []
    _merge_reasoning_details(details, [{"type": "reasoning.text", "index": 0, "text": "step", "signature": None}])
    _merge_reasoning_details(details, [{"index": 0, "signature": "signed"}])
    _merge_reasoning_details(details, [{"index": 0, "signature": None, "text": None}])
    _merge_reasoning_details(details, [{"type": "reasoning.text", "id": None, "text": "one"}])
    _merge_reasoning_details(details, [{"type": "reasoning.text", "id": None, "text": "two"}])
    assert details == [
        {"type": "reasoning.text", "index": 0, "text": "step", "signature": "signed"},
        {"type": "reasoning.text", "id": None, "text": "one"},
        {"type": "reasoning.text", "id": None, "text": "two"},
    ]


@pytest.mark.parametrize("partial_json", ['{"path":"app.py"}', '{"path":'])
def test_anthropic_stream_preserves_tool_input_or_its_incomplete_fragments(monkeypatch, partial_json) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "read-1", "name": "read_file", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial_json[:4]},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial_json[4:]},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    _capture(monkeypatch, "".join(f"data: {json.dumps(event)}\n" for event in events).encode())
    response = ModelBinding("http://up", "claude", api="anthropic").complete({"messages": [], "stream": True})
    block = response["content"][0]
    assert block["id"] == "read-1" and block["name"] == "read_file"
    if partial_json.endswith("}"):
        assert block["input"] == {"path": "app.py"}
        assert "partial_json" not in block
    else:
        assert "input" not in block
        assert block["partial_json"] == partial_json


def test_compose_nodes_repeats_the_model_entries_for_every_client_model() -> None:
    """With client models, the provider block lists them all, the served one first
    and still the default; a template with no such list is unchanged."""
    binding = ModelBinding("http://up", "served", api_key="k")
    pi = binding.compose_nodes(get_adapter("pi"), models=("other/big", "served", "other/small"))
    providers = next(data for _, data in pi if "providers" in data["data"])["data"]["providers"]["reef"]
    assert [m["id"] for m in providers["models"]] == ["served", "other/big", "other/small"]
    primary = next(data for _, data in pi if "defaultModel" in data["data"])["data"]
    assert primary["defaultModel"] == "reef/served"
    opencode = binding.compose_nodes(get_adapter("opencode"), models=("other/big",))
    data = opencode[0][1]["data"]
    assert list(data["provider"]["reef"]["models"]) == ["served", "other/big"] and data["model"] == "reef/served"
    assert binding.compose_nodes(get_adapter("opencode")) == binding.compose_nodes(
        get_adapter("opencode"), models=("served",)
    )


def test_recipe_reads_client_models_into_the_harness_surface(tmp_path) -> None:
    module = tmp_path / "demo_client_models.py"
    module.write_text(
        "def propose(nodes, samples, models):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        config = {
            "model": {"path": "small"},
            "evolution": {
                "propose": "demo_client_models:propose",
                "evaluate": "demo_client_models:evaluate",
                "tasks": ["t"],
                "client_models": ["big/one", "big/two"],
            },
        }
        runtime = InferenceProxyRuntime(model_path="small", base_url="http://up")
        built = CordisRecipe.from_environment({}, config=config, runtime=runtime)
        assert built.build_surface("s").harness.client_models == ("big/one", "big/two")
        bad = {**config, "evolution": {**config["evolution"], "client_models": "big/one"}}
        with pytest.raises(RecipeConfigError, match=r"evolution\.client_models"):
            CordisRecipe.from_environment({}, config=bad, runtime=runtime)
    finally:
        sys.path.remove(str(tmp_path))


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


def test_an_endpoint_without_a_key_still_renders_a_key_the_agent_accepts() -> None:
    """A local endpoint needs no key, but pi refuses to start without one.

    Rendering the empty string made every evaluation episode exit 1 with
    "No API key found for the selected model" before reaching the endpoint,
    so both sides of the gate scored 0 and no proposal could ever publish.
    """
    pi = get_adapter("pi")
    files = render_composition(ModelBinding("http://127.0.0.1:11434", "m").compose_nodes(pi), pi)
    assert json.loads(files["pi-agent/models.json"])["providers"]["reef"]["apiKey"] == NO_KEY_PLACEHOLDER


def test_a_real_key_is_rendered_verbatim_over_the_placeholder() -> None:
    pi = get_adapter("pi")
    files = render_composition(ModelBinding("http://up", "m", api_key="sk-real").compose_nodes(pi), pi)
    assert json.loads(files["pi-agent/models.json"])["providers"]["reef"]["apiKey"] == "sk-real"


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
