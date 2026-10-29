"""A tree never chooses where a model call goes: Reef's model binding is its only route.

For claude, hermes, dsh and pi, render refuses every config key the
pinned harness reads to choose the endpoint, the provider, the credential or
the model, in the tree alone (as validation renders it) and with the binding's
overlays (as an episode renders it). A request body a tree passes to the
bound endpoint may not name a model either. Keys that tune a run stay
admitted, and the binding itself always renders.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest
import yaml

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, RenderError, render_tree, validate_tree

OTHER_URL = "http://127.0.0.1:9/v1"


def render(adapter: str, files: Mapping[str, str], *, bound: bool, api: str = "openai") -> dict[str, str]:
    descriptor = get_adapter(adapter)
    binding = ModelBinding(base_url="http://127.0.0.1:8900", model="served", api_key="k-bound", api=api)
    overlays = binding.config_overlays(descriptor) if bound else ()
    return render_tree(HarnessTree(dict(files)), descriptor, overlays)


def config(data: dict[str, Any], target: str = "primary") -> dict[str, str]:
    return {f"config/{target}.json": json.dumps(data)}


CLAUDE_REFUSED = [
    pytest.param(
        {"env": {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_BEDROCK_BASE_URL": OTHER_URL}}, id="provider-switch"
    ),
    pytest.param({"env": {"HTTPS_PROXY": OTHER_URL}}, id="proxy"),
    pytest.param({"env": {"ANTHROPIC_SMALL_FAST_MODEL": "m"}}, id="env-small-fast-model"),
    pytest.param({"env": {"ANTHROPIC_DEFAULT_HAIKU_MODEL": "m"}}, id="env-tier-model"),
    pytest.param({"env": {"CLAUDE_CODE_SUBAGENT_MODEL": "m"}}, id="env-subagent-model"),
    pytest.param({"env": {"anthropic_base_url": OTHER_URL}}, id="env-name-in-another-case"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": '{"model": "m"}'}}, id="body-model"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": '{"models": ["m"]}'}}, id="body-fallback-models"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": '\ufeff{"model": "m"}'}}, id="body-after-byte-order-mark"),
    pytest.param({"env": {"claude_code_extra_body": '{"model": "m"}'}}, id="body-name-in-another-case"),
    pytest.param({"env": {"CLAUDE_CODE_EXTRA_BODY": "{model: m}"}}, id="body-render-cannot-read"),
]

#: settings.json keys whose descriptor check is IsAbsent(): fixed names a tree must not set.
CLAUDE_LOCKED = [
    pytest.param({"model": "m"}, id="settings-model"),
    pytest.param({"fallbackModel": ["m"]}, id="settings-fallback-model"),
    pytest.param({"apiKeyHelper": "echo k"}, id="credential-helper"),
    pytest.param({"awsAuthRefresh": "aws sso login"}, id="cloud-credential-helper"),
    pytest.param({"forceLoginMethod": "gateway"}, id="login-method"),
]


@pytest.mark.parametrize("data", CLAUDE_REFUSED)
@pytest.mark.parametrize("bound", [False, True], ids=["tree", "bound"])
def test_claude_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("claude", config(data), bound=bound, api="anthropic")


@pytest.mark.parametrize("data", CLAUDE_LOCKED)
@pytest.mark.parametrize("bound", [False, True], ids=["tree", "bound"])
def test_claude_refuses_a_key_the_descriptor_locks(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match=r"on \w+ in claude/settings\.json: Reef's model binding"):
        render("claude", config(data), bound=bound, api="anthropic")


def test_claude_writes_the_binding_env_only_beside_the_binding_token() -> None:
    # A tree alone never carries the endpoint; the binding's three names pass because its token is beside them.
    with pytest.raises(RenderError, match=re.escape("must not set env.ANTHROPIC_BASE_URL in claude/settings.json")):
        render("claude", config({"env": {"ANTHROPIC_BASE_URL": OTHER_URL}}), bound=False)
    files = render("claude", config({"env": {"ANTHROPIC_BASE_URL": OTHER_URL}}), bound=True, api="anthropic")
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
@pytest.mark.parametrize("path", ["skills/x/SKILL.md", "commands/x.md"])
def test_claude_refuses_a_model_in_a_command_or_skill_frontmatter(text: str, path: str) -> None:
    with pytest.raises(RenderError, match="must not set model in its frontmatter"):
        render("claude", {path: text}, bound=True, api="anthropic")


def test_claude_keeps_the_settings_a_tree_tunes() -> None:
    tuned = {
        "permissions": {"allow": ["WebSearch"]},
        "env": {"BASH_DEFAULT_TIMEOUT_MS": "60000", "CLAUDE_CODE_EXTRA_BODY": '{"metadata": {"user_id": "u"}}'},
        "effortLevel": "low",
    }
    skill = "---\nname: notes\ndescription: models and notes\n---\nBody.\n"
    files = render("claude", {**config(tuned), "skills/notes/SKILL.md": skill}, bound=True, api="anthropic")
    settings = json.loads(files["claude/settings.json"])
    assert settings["permissions"] == {"allow": ["WebSearch"]}
    assert settings["env"]["BASH_DEFAULT_TIMEOUT_MS"] == "60000"
    assert settings["env"]["CLAUDE_CODE_EXTRA_BODY"] == tuned["env"]["CLAUDE_CODE_EXTRA_BODY"]
    alone = render("claude", {**config(tuned), "skills/notes/SKILL.md": skill}, bound=False)
    assert alone["claude/skills/notes/SKILL.md"] == skill


HERMES_REFUSED = [
    pytest.param({"model": {"base_url": OTHER_URL}}, False, id="model-endpoint"),
    pytest.param({"model": "openrouter/m"}, False, id="model-name"),
    pytest.param({"model": {"model": "m"}}, True, id="model-alternate-name"),
    pytest.param(
        {"auxiliary": {"compression": {"provider": "custom", "base_url": OTHER_URL, "model": "m"}}},
        True,
        id="auxiliary",
    ),
    pytest.param({"auxiliary": {"vision": {"model": "m"}}}, True, id="auxiliary-model"),
    pytest.param({"auxiliary": {"openrouter_model": "m"}}, True, id="auxiliary-fallback-model"),
    pytest.param(
        {"auxiliary": {"title_generation": {"enabled": False, "prefer_fast_model": True}}}, True, id="fast-model"
    ),
    pytest.param({"delegation": {"provider": "custom", "base_url": OTHER_URL, "model": "m"}}, True, id="delegation"),
    pytest.param({"moa": {"presets": {"p": {"aggregator": {"provider": "custom", "model": "m"}}}}}, True, id="moa"),
    pytest.param({"cron": {"model": "m", "provider": "openrouter"}}, True, id="cron"),
    pytest.param({"model": {"api_base": OTHER_URL}}, False, id="model-endpoint-alias"),
    pytest.param({"model": {"name": "m"}}, True, id="model-name-alias"),
    pytest.param({"model": {"key_env": "OPENROUTER_API_KEY"}}, True, id="model-key-name"),
    pytest.param({"model": {"aliases": {"fast": "openrouter/m"}}}, True, id="model-short-aliases"),
    pytest.param(
        {
            "auxiliary": {
                "compression": {"fallback_chain": [{"provider": "custom", "model": "m", "base_url": OTHER_URL}]}
            }
        },
        True,
        id="auxiliary-fallback-chain",
    ),
    pytest.param(
        {"auxiliary": {"vision": {"fallback_chain": [{"provider": "openrouter", "model": "m", "key_env": "K"}]}}},
        True,
        id="auxiliary-fallback-chain-key-name",
    ),
    pytest.param({"auxiliary": {"vision": {"api_key_env": "OPENROUTER_API_KEY"}}}, True, id="auxiliary-key-name"),
    # The older compression keys, which hermes moves into auxiliary.compression.
    pytest.param({"compression": {"summary_base_url": OTHER_URL}}, True, id="compression-summary-base-url"),
    pytest.param({"compression": {"summary_model": "m"}}, True, id="compression-summary-model"),
    pytest.param({"compression": {"summary_provider": "openrouter"}}, True, id="compression-summary-provider"),
    pytest.param(
        {"curator": {"auxiliary": {"provider": "openrouter", "model": "m", "base_url": OTHER_URL}}}, True, id="curator"
    ),
    pytest.param({"moa": {"aggregator": {"provider": "openrouter", "model": "m"}}}, True, id="moa-flat-preset"),
    pytest.param({"auxiliary": {"compression": {"extra_body": {"model": "m"}}}}, True, id="auxiliary-body-model"),
    pytest.param({"auxiliary": {"approval": {"extra_body": {"models": ["m"]}}}}, True, id="auxiliary-body-models"),
    pytest.param({"curator": {"auxiliary": {"extra_body": {"model": "m"}}}}, True, id="curator-body-model"),
    pytest.param({"delegation": {"request_overrides": {"model": "m"}}}, True, id="delegation-call-model"),
    pytest.param(
        {"delegation": {"request_overrides": {"extra_body": {"models": ["m"]}}}}, True, id="delegation-body-models"
    ),
    pytest.param({"model": {"openai_runtime": "codex_app_server"}}, True, id="model-codex-runtime"),
    pytest.param({"auxiliary": {"compression": {"api_mode": "anthropic_messages"}}}, True, id="auxiliary-api-mode"),
    pytest.param({"delegation": {"api_mode": "codex_responses"}}, True, id="delegation-api-mode"),
]


@pytest.mark.parametrize(("data", "bound"), HERMES_REFUSED)
def test_hermes_refuses_a_setting_that_chooses_the_route(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("hermes", config(data), bound=bound)


#: Top-level keys whose descriptor check is IsAbsent(): hermes moves them into ``model``, and a tree must not set
#: them at all.
HERMES_LOCKED = [
    pytest.param({"fallback_model": {"provider": "custom", "model": "m", "base_url": OTHER_URL}}, id="fallback-model"),
    pytest.param({"fallback_providers": [{"provider": "openrouter", "model": "m"}]}, id="fallback-providers"),
    pytest.param({"providers": {"evil": {"base_url": OTHER_URL, "key_cmd": "echo k"}}}, id="named-provider"),
    pytest.param({"custom_providers": [{"name": "evil", "base_url": OTHER_URL}]}, id="custom-providers"),
    pytest.param({"base_url": OTHER_URL}, id="top-level-endpoint"),
    pytest.param({"provider": "openrouter"}, id="top-level-provider"),
    pytest.param(
        {"model_aliases": {"fast": {"model": "m", "provider": "custom", "base_url": OTHER_URL}}}, id="model-aliases"
    ),
]


@pytest.mark.parametrize("data", HERMES_LOCKED)
@pytest.mark.parametrize("bound", [False, True], ids=["tree", "bound"])
def test_hermes_refuses_a_key_the_descriptor_locks(data: dict[str, Any], bound: bool) -> None:
    with pytest.raises(RenderError, match=r"on \w+ in hermes/config\.yaml: Reef's model binding"):
        render("hermes", config(data), bound=bound)


#: Each api mode hermes 0.21.0 takes for model.api_mode, and spellings its alias map reads as one. For the binding's
#: custom provider bedrock_converse calls AWS Bedrock and codex_app_server runs a codex subprocess, off the bound
#: endpoint; the binding sets no api mode, so every one is refused.
HERMES_API_MODES = [
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
    "codex_app_server",
    "bedrock",
    "Bedrock",
    " bedrock-converse ",
    "CODEX_APP_SERVER",
    "openai",
]


@pytest.mark.parametrize("api_mode", HERMES_API_MODES)
@pytest.mark.parametrize("bound", [False, True], ids=["tree", "bound"])
def test_hermes_refuses_an_api_mode_the_binding_does_not_set(api_mode: str, bound: bool) -> None:
    with pytest.raises(RenderError, match=r"on model\.api_mode in hermes/config\.yaml: Reef's model binding"):
        render("hermes", config({"model": {"api_mode": api_mode}}), bound=bound)


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
        "compression": {"summary_provider": "auto", "threshold": 0.5},
    }
    rendered = yaml.safe_load(render("hermes", config(tuned), bound=True)["hermes/config.yaml"])
    assert rendered["model"] == {
        "api_key": "k-bound",
        "base_url": "http://127.0.0.1:8900/v1",
        "context_length": 200000,
        "default": "served",
        "provider": "custom",
    }
    assert rendered["toolsets"] == ["terminal", "file"] and rendered["delegation"] == tuned["delegation"]
    assert rendered["auxiliary"]["compression"]["extra_body"] == {"provider": {"sort": "price"}}
    # Alone, as validation renders it, the tree tunes the model without naming a provider, an endpoint or a model.
    render("hermes", config(tuned), bound=False)


DSH_ROUTE = {"api": "openai-completions", "baseURL": OTHER_URL, "models": [{"id": "m"}]}
DSH_REFUSED = [
    pytest.param({"llm-pi-ai": {"config": {"providers": {"evil": DSH_ROUTE}}}}, False, id="route-in-tree"),
    pytest.param({"llm-pi-ai": {"config": {"providers": {"evil": DSH_ROUTE}}}}, True, id="second-route"),
    pytest.param({"llm-pi-ai": {"config": {"providers": {"reef": {"headers": {"x": "y"}}}}}}, True, id="route-extra"),
    pytest.param({"agent-default-model": {"config": {"provider": "evil", "model": "m"}}}, False, id="default-model"),
    pytest.param({"llm-deepseek": {"config": {"baseURL": OTHER_URL}}}, True, id="deepseek-adapter"),
    pytest.param({"web-search-deepseek": {"config": {"baseURL": OTHER_URL, "model": "m"}}}, True, id="search-model"),
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


@pytest.mark.parametrize("target", ["primary", "web"])
@pytest.mark.parametrize(("data", "bound"), DSH_REFUSED)
def test_dsh_refuses_a_setting_that_chooses_the_route_in_either_profile(
    data: dict[str, Any], bound: bool, target: str
) -> None:
    with pytest.raises(RenderError, match=r"Reef's model binding|keeps its own package"):
        render("dsh", config(data, target), bound=bound)


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_dsh_keeps_the_settings_a_tree_tunes(api: str) -> None:
    tuned = {
        "agent-loop": {"config": {"maxParallelToolCalls": 4}},
        "tool-subagent": {"config": {"provider": "spawn", "toolName": "subagent"}},
        "compaction-basic": {"config": {"modelPolicies": [{"provider": "reef", "model": "served", "maxTokens": 9}]}},
    }
    files = render("dsh", config(tuned), bound=True, api=api)
    rows = {
        row["id"]: row for row in yaml.safe_load(files["dsh/profiles/headless/cordis.patch.yml"].replace("!!js ", ""))
    }
    assert rows["agent-loop"]["config"] == {"maxParallelToolCalls": 4}
    assert set(rows["llm-pi-ai"]["config"]["providers"]) == {"reef"}
    render("dsh", config(tuned), bound=False)


PI_REFUSED = [
    pytest.param(
        config(
            {"providers": {"evil": {"api": "openai-completions", "baseUrl": OTHER_URL, "models": [{"id": "m"}]}}},
            "models",
        ),
        True,
        id="keyless-provider",
    ),
    pytest.param(config({"providers": {"openai": {"baseUrl": OTHER_URL}}}, "models"), True, id="built-in-endpoint"),
    pytest.param(
        config({"providers": {"reef": {"baseUrl": OTHER_URL}}}, "models"), False, id="binding-provider-in-tree"
    ),
    pytest.param(config({"providers": {"reef": {"headers": {"x": "!cat k"}}}}, "models"), True, id="binding-extra"),
    pytest.param(config({"defaultProvider": "evil", "defaultModel": "evil/m"}), False, id="default-model"),
]

#: settings.json keys whose descriptor check is IsAbsent(): fixed names a tree must not set.
PI_LOCKED = [
    pytest.param(config({"enabledModels": ["evil/m"]}), id="enabled-models"),
    pytest.param(config({"httpProxy": OTHER_URL}), id="proxy"),
]


@pytest.mark.parametrize(("files", "bound"), PI_REFUSED)
def test_pi_refuses_a_setting_that_chooses_the_route(files: dict[str, str], bound: bool) -> None:
    with pytest.raises(RenderError, match="Reef's model binding"):
        render("pi", files, bound=bound)


@pytest.mark.parametrize("files", PI_LOCKED)
@pytest.mark.parametrize("bound", [False, True], ids=["tree", "bound"])
def test_pi_refuses_a_key_the_descriptor_locks(files: dict[str, str], bound: bool) -> None:
    with pytest.raises(RenderError, match=r"on \w+ in pi-agent/settings\.json: Reef's model binding"):
        render("pi", files, bound=bound)


@pytest.mark.parametrize("api", ["openai", "responses", "anthropic"])
def test_pi_keeps_the_settings_a_tree_tunes(api: str) -> None:
    tuned = config({"defaultThinkingLevel": "high"})
    files = render("pi", tuned, bound=True, api=api)
    assert json.loads(files["pi-agent/settings.json"])["defaultProvider"] == "reef"
    assert set(json.loads(files["pi-agent/models.json"])["providers"]) == {"reef"}
    render("pi", {**tuned, **config({"providers": {}}, "models")}, bound=False)


@pytest.mark.parametrize(
    ("adapter", "data", "refusal"),
    [
        (
            "claude",
            {"env": {"CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_BEDROCK_BASE_URL": OTHER_URL}},
            "Reef's model binding",
        ),
        ("claude", {"env": {"CLAUDE_CODE_EXTRA_BODY": '{"model": "m"}'}}, "Reef's model binding"),
        ("claude", {"fallbackModel": ["m"]}, "Reef's model binding"),
        (
            "hermes",
            {"fallback_model": {"provider": "custom", "model": "m", "base_url": OTHER_URL}},
            "Reef's model binding",
        ),
        (
            "hermes",
            {"auxiliary": {"compression": {"fallback_chain": [{"provider": "custom", "model": "m"}]}}},
            "Reef's model binding",
        ),
        ("hermes", {"delegation": {"request_overrides": {"model": "m"}}}, "Reef's model binding"),
        ("hermes", {"model": {"api_mode": "bedrock_converse"}}, "Reef's model binding"),
        ("dsh", {"llm-pi-ai": {"config": {"providers": {"evil": DSH_ROUTE}}}}, "Reef's model binding"),
        ("pi", {"enabledModels": ["evil/m"]}, "Reef's model binding"),
    ],
)
def test_validation_refuses_the_tree_and_says_why(adapter: str, data: dict[str, Any], refusal: str) -> None:
    (problem,) = validate_tree(HarnessTree(config(data)), get_adapter(adapter)).problems
    assert refusal in problem
