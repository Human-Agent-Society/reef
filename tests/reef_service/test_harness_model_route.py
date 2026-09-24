"""A tree never chooses where a model call goes: Reef's model binding is its only route.

For claude, hermes, dsh, pi and terminus, render refuses every config key the
pinned harness reads to choose the endpoint, the provider, the credential or
the model, in the tree alone (as admission renders it) and with the binding
appended (as an episode or an install renders it). A request body a tree
passes to the bound endpoint may not name a model either. Keys that tune a
run stay admitted, and the binding itself always renders.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree.mutations import Mutation, admit_mutations
from reef.harness.tree.render import RenderError, render_composition

OTHER = "http://127.0.0.1:9/v1"


def render(adapter: str, nodes: list[tuple[str, Any]], *, bound: bool, api: str = "openai") -> dict[str, str]:
    descriptor = get_adapter(adapter)
    binding = ModelBinding(base_url="http://127.0.0.1:8900", model="served", api_key="k-bound", api=api)
    return render_composition([*nodes, *(binding.compose_nodes(descriptor) if bound else ())], descriptor)


def config(data: dict[str, Any], target: str = "primary") -> tuple[str, dict[str, Any]]:
    return ("config", {"target": target, "data": data})


CLAUDE_REFUSED = [
    pytest.param({"env": {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_BEDROCK_BASE_URL": OTHER}}, id="provider-switch"),
    pytest.param({"env": {"HTTPS_PROXY": OTHER}}, id="proxy"),
    pytest.param({"env": {"ANTHROPIC_SMALL_FAST_MODEL": "m"}}, id="env-small-fast-model"),
    pytest.param({"env": {"ANTHROPIC_DEFAULT_HAIKU_MODEL": "m"}}, id="env-tier-model"),
    pytest.param({"env": {"CLAUDE_CODE_SUBAGENT_MODEL": "m"}}, id="env-subagent-model"),
    pytest.param({"env": {"anthropic_base_url": OTHER}}, id="env-name-in-another-case"),
    pytest.param({"model": "m"}, id="settings-model"),
    pytest.param({"fallbackModel": ["m"]}, id="settings-fallback-model"),
    pytest.param({"apiKeyHelper": "echo k"}, id="credential-helper"),
    pytest.param({"awsAuthRefresh": "aws sso login"}, id="cloud-credential-helper"),
    pytest.param({"forceLoginMethod": "gateway"}, id="login-method"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": '{"model": "m"}'}}, id="body-model"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": '{"models": ["m"]}'}}, id="body-fallback-models"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": '\ufeff{"model": "m"}'}}, id="body-after-byte-order-mark"),
    pytest.param({"env": {"claude_code_extra_body": '{"model": "m"}'}}, id="body-name-in-another-case"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": "{model: m}"}}, id="body-render-cannot-read"),
]


@pytest.mark.parametrize("data", CLAUDE_REFUSED)
@pytest.mark.parametrize("bound", [False, True], ids=["tree", "bound"])
def test_claude_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("claude", [config(data)], bound=bound, api="anthropic")


def test_claude_writes_the_binding_env_only_beside_the_binding_token() -> None:
    # A tree alone never carries the endpoint; the binding's three names pass because its token is beside them.
    with pytest.raises(RenderError, match="must not set env ANTHROPIC_BASE_URL"):
        render("claude", [config({"env": {"ANTHROPIC_BASE_URL": OTHER}})], bound=False)
    files = render("claude", [config({"env": {"ANTHROPIC_BASE_URL": OTHER}})], bound=True, api="anthropic")
    env = json.loads(files["claude/settings.json"])["env"]
    assert env == {
        "ANTHROPIC_AUTH_TOKEN": "k-bound",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8900",
        "ANTHROPIC_MODEL": "served",
    }


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("---\nname: x\nmodel: opus\n---\nBody.\n", id="plain"),
        pytest.param('\ufeff---\n"mod\\x65l": opus\n---\nBody.\n', id="escaped-key"),
        pytest.param("---\nbase: &b {model: opus}\n<<: *b\n---\nBody.\n", id="merged-key"),
        pytest.param("---\ndescription: a: b\nmodel: opus\n---\nBody.\n", id="read-after-quoting"),
        pytest.param("---\n\tmodel: [\n---\nBody.\n", id="unreadable-block"),
    ],
)
@pytest.mark.parametrize("kind", ["skill", "agent_command"])
def test_claude_refuses_a_model_in_a_command_or_skill_frontmatter(text: str, kind: str) -> None:
    with pytest.raises(RenderError, match="must not set model in its frontmatter"):
        render("claude", [(kind, {"name": "x", "text": text})], bound=True, api="anthropic")


def test_claude_keeps_the_settings_a_tree_tunes() -> None:
    tuned = {
        "permissions": {"allow": ["WebSearch"]},
        "env": {"BASH_DEFAULT_TIMEOUT_MS": "60000", "CLAUDE_CODE_EXTRA_BODY": '{"metadata": {"user_id": "u"}}'},
        "effortLevel": "low",
    }
    skill = ("skill", {"name": "notes", "text": "---\nname: notes\ndescription: models and notes\n---\nBody.\n"})
    files = render("claude", [config(tuned), skill], bound=True, api="anthropic")
    settings = json.loads(files["claude/settings.json"])
    assert settings["permissions"] == {"allow": ["WebSearch"]}
    assert settings["env"]["BASH_DEFAULT_TIMEOUT_MS"] == "60000"
    assert settings["env"]["CLAUDE_CODE_EXTRA_BODY"] == tuned["env"]["CLAUDE_CODE_EXTRA_BODY"]
    assert render("claude", [config(tuned), skill], bound=False)["claude/skills/notes/SKILL.md"] == skill[1]["text"]


HERMES_REFUSED = [
    pytest.param({"model": {"base_url": OTHER}}, False, id="model-endpoint"),
    pytest.param({"model": "openrouter/m"}, False, id="model-name"),
    pytest.param({"model": {"model": "m"}}, True, id="model-alternate-name"),
    pytest.param(
        {"fallback_model": {"provider": "custom", "model": "m", "base_url": OTHER}}, True, id="fallback-model"
    ),
    pytest.param({"fallback_providers": [{"provider": "openrouter", "model": "m"}]}, True, id="fallback-providers"),
    pytest.param({"providers": {"evil": {"base_url": OTHER, "key_cmd": "echo k"}}}, True, id="named-provider"),
    pytest.param({"custom_providers": [{"name": "evil", "base_url": OTHER}]}, True, id="custom-providers"),
    pytest.param(
        {"auxiliary": {"compression": {"provider": "custom", "base_url": OTHER, "model": "m"}}}, True, id="auxiliary"
    ),
    pytest.param({"auxiliary": {"vision": {"model": "m"}}}, True, id="auxiliary-model"),
    pytest.param({"auxiliary": {"openrouter_model": "m"}}, True, id="auxiliary-fallback-model"),
    pytest.param(
        {"auxiliary": {"title_generation": {"enabled": False, "prefer_fast_model": True}}}, True, id="fast-model"
    ),
    pytest.param({"delegation": {"provider": "custom", "base_url": OTHER, "model": "m"}}, True, id="delegation"),
    pytest.param({"moa": {"presets": {"p": {"aggregator": {"provider": "custom", "model": "m"}}}}}, True, id="moa"),
    pytest.param({"cron": {"model": "m", "provider": "openrouter"}}, True, id="cron"),
    pytest.param({"base_url": OTHER}, False, id="top-level-endpoint"),
    pytest.param({"provider": "openrouter"}, True, id="top-level-provider"),
    pytest.param({"model": {"api_base": OTHER}}, False, id="model-endpoint-alias"),
    pytest.param({"model": {"name": "m"}}, True, id="model-name-alias"),
    pytest.param({"model": {"key_env": "OPENROUTER_API_KEY"}}, True, id="model-key-name"),
    pytest.param(
        {"model_aliases": {"fast": {"model": "m", "provider": "custom", "base_url": OTHER}}}, True, id="model-aliases"
    ),
    pytest.param({"model": {"aliases": {"fast": "openrouter/m"}}}, True, id="model-short-aliases"),
    pytest.param(
        {"auxiliary": {"compression": {"fallback_chain": [{"provider": "custom", "model": "m", "base_url": OTHER}]}}},
        True,
        id="auxiliary-fallback-chain",
    ),
    pytest.param(
        {"auxiliary": {"vision": {"fallback_chain": [{"provider": "openrouter", "model": "m", "key_env": "K"}]}}},
        True,
        id="auxiliary-fallback-chain-key-name",
    ),
    pytest.param({"auxiliary": {"vision": {"api_key_env": "OPENROUTER_API_KEY"}}}, True, id="auxiliary-key-name"),
    pytest.param(
        {"curator": {"auxiliary": {"provider": "openrouter", "model": "m", "base_url": OTHER}}}, True, id="curator"
    ),
    pytest.param({"moa": {"aggregator": {"provider": "openrouter", "model": "m"}}}, True, id="moa-flat-preset"),
    pytest.param({"auxiliary": {"compression": {"extra_body": {"model": "m"}}}}, True, id="auxiliary-body-model"),
    pytest.param({"auxiliary": {"approval": {"extra_body": {"models": ["m"]}}}}, True, id="auxiliary-body-models"),
    pytest.param({"curator": {"auxiliary": {"extra_body": {"model": "m"}}}}, True, id="curator-body-model"),
    pytest.param({"delegation": {"request_overrides": {"model": "m"}}}, True, id="delegation-call-model"),
    pytest.param(
        {"delegation": {"request_overrides": {"extra_body": {"models": ["m"]}}}}, True, id="delegation-body-models"
    ),
]


@pytest.mark.parametrize(("data", "bound"), HERMES_REFUSED)
def test_hermes_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("hermes", [config(data)], bound=bound)


def test_hermes_keeps_the_settings_a_tree_tunes() -> None:
    tuned = {
        "toolsets": ["terminal", "file"],
        "model": {"context_length": 200000},
        "auxiliary": {
            "compression": {"provider": "main", "timeout": 60, "extra_body": {"provider": {"sort": "price"}}},
            "vision": {"provider": "auto"},
        },
        "delegation": {
            "max_iterations": 40,
            "request_overrides": {"service_tier": "priority", "extra_body": {"provider": {"sort": "throughput"}}},
        },
        "curator": {"enabled": False},
    }
    rendered = yaml.safe_load(render("hermes", [config(tuned)], bound=True)["hermes/config.yaml"])
    assert rendered["model"] == {
        "api_key": "k-bound",
        "base_url": "http://127.0.0.1:8900/v1",
        "context_length": 200000,
        "default": "served",
        "provider": "custom",
    }
    assert rendered["toolsets"] == ["terminal", "file"] and rendered["delegation"] == tuned["delegation"]
    assert rendered["auxiliary"]["compression"]["extra_body"] == {"provider": {"sort": "price"}}
    # Alone, as admission renders it, the tree tunes the model without naming a provider, an endpoint or a model.
    render("hermes", [config(tuned)], bound=False)


DSH_ROUTE = {"api": "openai-completions", "baseURL": OTHER, "models": [{"id": "m"}]}
DSH_REFUSED = [
    pytest.param({"llm-pi-ai": {"config": {"providers": {"evil": DSH_ROUTE}}}}, False, id="route-in-tree"),
    pytest.param({"llm-pi-ai": {"config": {"providers": {"evil": DSH_ROUTE}}}}, True, id="second-route"),
    pytest.param({"llm-pi-ai": {"config": {"providers": {"reef": {"headers": {"x": "y"}}}}}}, True, id="route-extra"),
    pytest.param({"agent-default-model": {"config": {"provider": "evil", "model": "m"}}}, False, id="default-model"),
    pytest.param({"llm-deepseek": {"config": {"baseURL": OTHER}}}, True, id="deepseek-adapter"),
    pytest.param({"web-search-deepseek": {"config": {"baseURL": OTHER, "model": "m"}}}, True, id="search-model"),
    pytest.param(
        {"compaction-basic": {"config": {"modelPolicies": [{"summarizationModel": "m"}]}}}, True, id="compaction"
    ),
    pytest.param(
        {"tool-subagent": {"config": {"agentOptions": {"provider": "evil", "model": "m"}}}}, True, id="subagent"
    ),
    pytest.param({"agent-loop": {"config": {"agents": [{"id": "a", "provider": "evil"}]}}}, True, id="agents"),
    pytest.param(
        {"session-title-llm": {"disabled": True, "config": {"provider": "evil", "model": "m"}}}, True, id="title"
    ),
    pytest.param({"web": {"name": "@deepseek-ai/dsh-llm-pi-ai"}}, True, id="package-swap"),
    pytest.param({"tool-subagent": {"config": "!!js ({agentOptions: {provider: 'evil'}})"}}, True, id="js-expression"),
]


@pytest.mark.parametrize(("data", "bound"), DSH_REFUSED)
def test_dsh_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match=r"Reef's model binding|keeps its own package"):
        render("dsh", [config(data)], bound=bound)


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_dsh_keeps_the_settings_a_tree_tunes(api: str) -> None:
    tuned = {
        "agent-loop": {"config": {"maxParallelToolCalls": 4}},
        "tool-subagent": {"config": {"provider": "spawn", "toolName": "subagent"}},
        "compaction-basic": {"config": {"modelPolicies": [{"provider": "reef", "model": "served", "maxTokens": 9}]}},
    }
    files = render("dsh", [config(tuned)], bound=True, api=api)
    rows = {
        row["id"]: row for row in yaml.safe_load(files["dsh/profiles/headless/cordis.patch.yml"].replace("!!js ", ""))
    }
    assert rows["agent-loop"]["config"] == {"maxParallelToolCalls": 4}
    assert set(rows["llm-pi-ai"]["config"]["providers"]) == {"reef"}
    render("dsh", [config(tuned)], bound=False)


PI_REFUSED = [
    pytest.param(
        config(
            {"providers": {"evil": {"api": "openai-completions", "baseUrl": OTHER, "models": [{"id": "m"}]}}}, "models"
        ),
        True,
        id="keyless-provider",
    ),
    pytest.param(config({"providers": {"openai": {"baseUrl": OTHER}}}, "models"), True, id="built-in-endpoint"),
    pytest.param(config({"providers": {"reef": {"baseUrl": OTHER}}}, "models"), False, id="binding-provider-in-tree"),
    pytest.param(config({"providers": {"reef": {"headers": {"x": "!cat k"}}}}, "models"), True, id="binding-extra"),
    pytest.param(config({"defaultProvider": "evil", "defaultModel": "evil/m"}), False, id="default-model"),
    pytest.param(config({"enabledModels": ["evil/m"]}), True, id="enabled-models"),
    pytest.param(config({"httpProxy": OTHER}), True, id="proxy"),
]


@pytest.mark.parametrize(("node", "bound"), PI_REFUSED)
def test_pi_refuses_a_setting_that_chooses_the_route(node: tuple[str, dict[str, Any]], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("pi", [node], bound=bound)


@pytest.mark.parametrize("api", ["openai", "responses", "anthropic"])
def test_pi_keeps_the_settings_a_tree_tunes(api: str) -> None:
    tuned = config({"defaultThinkingLevel": "high", "compaction": {"enabled": False}, "defaultTools": ["bash"]})
    files = render("pi", [tuned], bound=True, api=api)
    assert json.loads(files["pi-agent/settings.json"])["defaultProvider"] == "reef"
    assert set(json.loads(files["pi-agent/models.json"])["providers"]) == {"reef"}
    render("pi", [tuned, config({"providers": {}}, "models")], bound=False)


TERMINUS_REFUSED = [
    pytest.param({"model_name": "openai/m"}, False, id="model-name-in-tree"),
    pytest.param({"api_base": OTHER}, False, id="endpoint-in-tree"),
    pytest.param({"llm_kwargs": {"base_url": OTHER}}, True, id="constructor-endpoint"),
    pytest.param({"llm_call_kwargs": {"base_url": OTHER}}, True, id="call-base-url"),
    pytest.param({"llm_call_kwargs": {"api_base": OTHER}}, True, id="call-api-base"),
    pytest.param({"llm_call_kwargs": {"custom_llm_provider": "openai"}}, True, id="call-provider"),
    pytest.param({"llm_call_kwargs": {"model": "openai/m"}}, True, id="call-model"),
    pytest.param({"llm_call_kwargs": {"fallbacks": [{"model": "m", "api_base": OTHER}]}}, True, id="call-fallback"),
    pytest.param({"llm_call_kwargs": {"aws_bedrock_runtime_endpoint": OTHER}}, True, id="call-cloud-endpoint"),
    pytest.param({"llm_call_kwargs": {"success_callback": ["langfuse"], "langfuse_host": OTHER}}, True, id="logging"),
    pytest.param({"llm_call_kwargs": {"models": ["m"]}}, True, id="call-fallback-models"),
    pytest.param({"llm_call_kwargs": {"extra_body": {"model": "m"}}}, True, id="body-model"),
    pytest.param({"llm_call_kwargs": {"extra_body": {"models": ["m"]}}}, True, id="body-fallback-models"),
]


@pytest.mark.parametrize(("data", "bound"), TERMINUS_REFUSED)
def test_terminus_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("terminus", [config(data)], bound=bound)


def test_terminus_refuses_a_request_body_that_is_not_an_object() -> None:
    with pytest.raises(RenderError, match="extra_body must be an object"):
        render("terminus", [config({"llm_call_kwargs": {"extra_body": '{"model": "m"}'}})], bound=True)


def test_terminus_keeps_the_arguments_a_tree_tunes() -> None:
    tuned = {"max_turns": 12, "llm_call_kwargs": {"top_k": 20, "extra_body": {"provider": {"sort": "price"}}}}
    rendered = json.loads(render("terminus", [config(tuned)], bound=True)["terminus/config.json"])
    assert rendered["llm_call_kwargs"] == tuned["llm_call_kwargs"]
    assert rendered["llm_kwargs"] == {"api_key": "k-bound"} and rendered["model_name"] == "served"
    render("terminus", [config(tuned)], bound=False)


@pytest.mark.parametrize(
    ("adapter", "data"),
    [
        ("claude", {"env": {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_BEDROCK_BASE_URL": OTHER}}),
        ("claude", {"env": {"CLAUDE_CODE_EXTRA_BODY": '{"model": "m"}'}}),
        ("hermes", {"fallback_model": {"provider": "custom", "model": "m", "base_url": OTHER}}),
        ("hermes", {"auxiliary": {"compression": {"fallback_chain": [{"provider": "custom", "model": "m"}]}}}),
        ("hermes", {"delegation": {"request_overrides": {"model": "m"}}}),
        ("dsh", {"llm-pi-ai": {"config": {"providers": {"evil": DSH_ROUTE}}}}),
        ("pi", {"enabledModels": ["evil/m"]}),
        ("terminus", {"llm_call_kwargs": {"base_url": OTHER}}),
        ("terminus", {"llm_call_kwargs": {"extra_body": {"model": "m"}}}),
    ],
)
def test_admission_refuses_the_proposal_and_says_why(adapter: str, data: dict[str, Any]) -> None:
    proposal = Mutation("create", "route", {"name": "config", "config": {"data": data}})
    entries, refusal = admit_mutations([], [proposal], get_adapter(adapter))
    assert entries == [] and refusal is not None and "Reef's model binding" in refusal
