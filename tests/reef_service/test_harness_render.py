"""Guarantees of reef.harness.tree.render and the bundled adapter descriptors."""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import reef.harness.adapters
from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.adapters.descriptor import AdapterRenderer, ClientState, DescriptorError, load_descriptor
from reef.harness.adapters.hermes.quirks import DEFAULT_IDENTITY
from reef.harness.adapters.opencode.quirks import read_frontmatter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError
from reef.harness.tree.mutations import Mutation, admit_mutations
from reef.harness.tree.render import RenderError, render_composition

GOLDENS = Path(__file__).parent / "data" / "harness_goldens"

# One node of every kind; the models config target exists only on pi.
NODES = [
    ("config", {"data": {"defaultModel": "qwen/qwen3-8b", "defaultProvider": "qwen"}}),
    (
        "config",
        {
            "target": "models",
            "data": {
                "providers": {
                    "qwen": {
                        "api": "openai-completions",
                        "apiKey": "dummy",
                        "baseUrl": "http://localhost:8000/v1",
                        "models": [{"id": "qwen3-8b"}],
                    }
                }
            },
        },
    ),
    ("rules", {"text": "Answer briefly."}),
    ("rules", {"text": "Prefer the standard library."}),
    ("skill", {"name": "notes", "text": "# Notes skill\n\nKeep short notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize $1."}),
    ("code_extension", {"name": "tracer", "code": "export default function tracer() {}"}),
]


def golden_tree(adapter: str) -> dict[str, str]:
    root = GOLDENS / adapter
    return {path.relative_to(root).as_posix(): path.read_text() for path in sorted(root.rglob("*")) if path.is_file()}


def test_pi_render_matches_the_golden_tree() -> None:
    # pi keeps every model call on the binding, so the provider and model choice in NODES are swapped for a setting.
    nodes = [("config", {"data": {"defaultThinkingLevel": "off"}}), *(node for node in NODES if node[0] != "config")]
    assert render_composition(nodes, get_adapter("pi")) == golden_tree("pi")


def test_opencode_render_matches_the_golden_tree() -> None:
    nodes = [node for node in NODES if node[1].get("target") != "models"]
    assert render_composition(nodes, get_adapter("opencode")) == golden_tree("opencode")


def test_claude_render_matches_the_golden_tree() -> None:
    nodes = [node for node in NODES if node[1].get("target") != "models"]
    assert render_composition(nodes, get_adapter("claude")) == golden_tree("claude")


def test_codex_render_matches_the_golden_tree() -> None:
    nodes = [node for node in NODES if node[0] != "config" and node[0] != "code_extension"]
    assert render_composition(nodes, get_adapter("codex")) == golden_tree("codex")


def test_codex_binding_uses_the_responses_api() -> None:
    descriptor = get_adapter("codex")
    base = render_composition([], descriptor)
    assert "model_provider" not in tomllib.loads(base["codex/config.toml"])

    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1", api="responses")
    files = render_composition([*binding.compose_nodes(descriptor)], descriptor)
    config = tomllib.loads(files["codex/config.toml"])
    assert config["model"] == "m1" and config["model_provider"] == "reef"
    assert config["model_providers"]["reef"] == {
        "name": "Reef",
        "base_url": "http://127.0.0.1:9/v1",
        "wire_api": "responses",
        "supports_websockets": False,
        "experimental_bearer_token": "k-1",
    }


def test_codex_rejects_code_extensions_until_hooks_have_separate_isolation() -> None:
    with pytest.raises(RenderError, match="native hooks run outside the command sandbox"):
        render_composition(
            [("code_extension", {"name": "tracer", "code": "export default () => null"})], get_adapter("codex")
        )


def test_codex_quirk_rejects_reopened_hermetic_switches() -> None:
    descriptor = get_adapter("codex")
    for approval_policy in ("on-request", "never"):
        with pytest.raises(RenderError, match="may not set approval_policy"):
            render_composition([("config", {"data": {"approval_policy": approval_policy}})], descriptor)
    with pytest.raises(RenderError, match="web_search must be one of disabled, cached, indexed, live"):
        render_composition([("config", {"data": {"web_search": "always"}})], descriptor)
    for data, message in (
        ({"features": {"apps": True}}, r"features\.apps disabled"),
        ({"features": {"hooks": True}}, r"features\.hooks disabled"),
        ({"features": {"plugins": True}}, r"features\.plugins disabled"),
        ({"features": {"skill_mcp_dependency_install": True}}, r"skill_mcp_dependency_install disabled"),
        ({"sandbox_workspace_write": {"network_access": True}}, r"network_access false"),
        ({"sandbox_workspace_write": {"writable_roots": ["/tmp/outside"]}}, "may not add"),
        ({"otel": {"exporter": "otlp-http"}}, "OpenTelemetry exporter disabled"),
        ({"mcp_servers": {"untrusted": {"command": "tool"}}}, "config keys are not admitted"),
        ({"hooks": {"SessionStart": []}}, "config keys are not admitted"),
        ({"notify": ["untrusted-command"]}, "config keys are not admitted"),
        ({"model_instructions_file": "/etc/passwd"}, "config keys are not admitted"),
        ({"experimental_compact_prompt_file": "/etc/passwd"}, "config keys are not admitted"),
        ({"model_catalog_json": "/etc/passwd"}, "config keys are not admitted"),
        ({"agents": {"reviewer": {"config_file": "/etc/passwd"}}}, "config keys are not admitted"),
    ):
        with pytest.raises(RenderError, match=message):
            render_composition([("config", {"data": data})], descriptor)


def test_codex_episode_argv_pins_approvals_and_web_search_off_over_the_session_config() -> None:
    """A reef-codex session reads config.toml: no approval_policy, so Codex asks the person on request, and the
    tree's web_search. The episode argv overrides both before the prompt."""
    descriptor = get_adapter("codex")
    files = render_composition([("config", {"data": {"web_search": "live"}})], descriptor)
    config = tomllib.loads(files["codex/config.toml"])
    assert config["web_search"] == "live" and "approval_policy" not in config
    argv = list(descriptor.argv)
    overrides = [argv[index + 1] for index, token in enumerate(argv) if token == "--config"]
    assert overrides == ['approval_policy="never"', 'web_search="disabled"'] and argv[-1] == "{prompt}"


def test_codex_renders_a_command_as_a_skill_and_refuses_one_that_shares_a_skills_name() -> None:
    """Codex 0.153.4 loads no custom prompts, so a command is a skill typed as $name, in the one skill root."""
    descriptor = get_adapter("codex")
    command = ("agent_command", {"name": "reefine", "text": "File the text after $reefine as a request."})
    files = render_composition([command], descriptor)
    assert files["codex/skills/reefine/SKILL.md"] == (
        "---\nname: reefine\ndescription: File the text after $reefine as a request.\n---\n"
        "File the text after $reefine as a request.\n"
    )
    assert not any(path.startswith(("codex/prompts/", ".agents/")) for path in files)
    skill = ("skill", {"name": "reefine", "text": "# reefine"})
    collision = "'codex/skills/reefine/SKILL.md': skill 'reefine' and agent_command 'reefine'; rename one"
    with pytest.raises(RenderError, match=re.escape(collision)):
        render_composition([skill, command], descriptor)


def test_codex_accepts_admitted_model_tuning() -> None:
    files = render_composition(
        [("config", {"data": {"model_reasoning_effort": "high", "model_verbosity": "low"}})],
        get_adapter("codex"),
    )
    config = tomllib.loads(files["codex/config.toml"])
    assert config["model_reasoning_effort"] == "high"
    assert config["model_verbosity"] == "low"


def test_codex_requires_the_responses_dialect() -> None:
    # The error names the serve flag that picks the dialect the adapter declares.
    with pytest.raises(
        ModelBindingError,
        match=r"declares no model_binding for the .openai. api .*serve with --inference.upstream-api responses",
    ):
        ModelBinding("http://up", "m").compose_nodes(get_adapter("codex"))


def _terminus_nodes():
    # terminus rejects code_extension, and its config target holds Terminus 2
    # constructor arguments rather than a pi-shaped settings object.
    nodes = [node for node in NODES if node[0] not in ("config", "code_extension")]
    return [*nodes, ("config", {"data": {"max_turns": 40}})]


def test_terminus_render_matches_the_golden_tree() -> None:
    assert render_composition(_terminus_nodes(), get_adapter("terminus")) == golden_tree("terminus")


def test_terminus_folds_agent_commands_into_a_second_skill_root() -> None:
    files = render_composition(_terminus_nodes(), get_adapter("terminus"))
    # Both roots carry a SKILL.md, and the quirk synthesized frontmatter on each.
    assert files["terminus/skills/notes/SKILL.md"].startswith("---\nname: notes\n")
    assert files["terminus-commands/summarize/SKILL.md"].startswith("---\nname: summarize\n")


def test_terminus_rejects_config_keys_that_are_not_constructor_arguments() -> None:
    with pytest.raises(RenderError, match="not Terminus 2 arguments"):
        render_composition([("config", {"data": {"tmux_pane_width": 200}})], get_adapter("terminus"))


@pytest.mark.parametrize("turns", [0, -1, True, "many"])
def test_terminus_rejects_a_max_turns_that_is_not_a_positive_integer(turns) -> None:
    with pytest.raises(RenderError, match="max_turns must be a positive integer"):
        render_composition([("config", {"data": {"max_turns": turns}})], get_adapter("terminus"))


def test_terminus_extension_requires_an_agent_class() -> None:
    with pytest.raises(RenderError, match="must define class Agent"):
        render_composition(
            [("code_extension", {"name": "assemble", "code": "def assemble(s, r, f): return None\n"})],
            get_adapter("terminus"),
        )


def test_terminus_renders_one_extension_without_executing_it() -> None:
    node = ("code_extension", {"name": "agent", "code": "raise RuntimeError('must not run')\nclass Agent: pass\n"})
    files = render_composition([node], get_adapter("terminus"))
    assert files["terminus/context/agent.py"] == node[1]["code"]
    with pytest.raises(RenderError, match="exactly one code_extension"):
        render_composition([node, ("code_extension", {**node[1], "name": "second"})], get_adapter("terminus"))


@pytest.mark.parametrize("model", ["m1", "qwen/qwen3-coder"])
def test_terminus_binding_renders_the_litellm_provider(model: str) -> None:
    descriptor = get_adapter("terminus")
    binding = ModelBinding(base_url="http://127.0.0.1:9", model=model, api_key="k-1")
    files = render_composition([*binding.compose_nodes(descriptor)], descriptor)
    config = json.loads(files["terminus/config.json"])
    # The served name stays the model name Harbor looks the context limit up under; litellm_proxy routes litellm
    # to api_base whatever vendor prefix that name carries, with the tree's call arguments in the request body.
    assert config["model_name"] == model
    assert config["api_base"] == "http://127.0.0.1:9/v1"
    assert config["llm_kwargs"] == {"api_key": "k-1", "custom_llm_provider": "litellm_proxy"}


DSH_PATCH = "dsh/profiles/headless/cordis.patch.yml"
DSH_WEB_PATCH = "dsh/profiles/web/cordis.patch.yml"


def _dsh_nodes():
    # dsh's config target is keyed by plugin id, so the pi shaped config nodes are swapped for one of its own.
    nodes = [node for node in NODES if node[0] != "config"]
    return [("config", {"data": {"agent-loop": {"config": {"maxSteps": 40}}}}), *nodes]


def test_dsh_render_matches_the_golden_tree() -> None:
    assert render_composition(_dsh_nodes(), get_adapter("dsh")) == golden_tree("dsh")


def test_dsh_quirks_emit_the_patch_layer_the_env_file_and_skill_frontmatter() -> None:
    descriptor = get_adapter("dsh")
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1")
    files = render_composition([*_dsh_nodes(), *binding.compose_nodes(descriptor)], descriptor)
    patch = yaml.safe_load(files[DSH_PATCH].replace("!!js ", ""))
    by_id = {row["id"]: row for row in patch if "id" in row}
    # Every entry in id order, the defaults that keep an episode readable and quiet, the binding, then the inserts.
    assert [row.get("id", "insert") for row in patch] == [
        "agent-default-model",
        "agent-loop",
        "llm-pi-ai",
        "session-persistence-jsonl",
        "session-telemetry-otel",
        "session-title-llm",
        "insert",
    ]
    assert by_id["session-persistence-jsonl"]["config"] == {"compression": "none", "root": "dshHomePath('sessions')"}
    assert "root: !!js 'dshHomePath(''sessions'')'" in files[DSH_PATCH]
    assert by_id["session-telemetry-otel"] == {"id": "session-telemetry-otel", "disabled": True}
    route = by_id["llm-pi-ai"]["config"]["providers"]["reef"]
    assert route["baseURL"] == "http://127.0.0.1:9/v1" and route["apiKeyEnv"] == "REEF_API_KEY"
    assert by_id["agent-default-model"]["config"] == {"provider": "reef", "model": "m1"}
    assert patch[-1] == {"insert": [{"id": "extension-tracer", "name": "./extensions/tracer.mjs"}]}
    assert files["dsh/.env"] == "REEF_API_KEY=k-1\n"
    # A skill without frontmatter gets name and description; a command is a user invocable skill.
    assert (
        files["dsh/skills/notes/SKILL.md"]
        == "---\nname: notes\ndescription: Notes skill\n---\n# Notes skill\n\nKeep short notes.\n"
    )
    assert files["dsh-agents/skills/summarize/SKILL.md"] == (
        "---\nname: summarize\ndescription: Summarize $1.\ndisable-model-invocation: true\n---\nSummarize $1.\n"
    )
    own = ("skill", {"name": "own", "text": "---\nname: own\ndescription: mine\n---\nBody.\n"})
    assert render_composition([own], descriptor)["dsh/skills/own/SKILL.md"] == own[1]["text"]
    own = ("skill", {"name": "own", "text": "---\r\nname: own\r\ndescription: mine\r\n---\r\nBody.\r\n"})
    assert render_composition([own], descriptor)["dsh/skills/own/SKILL.md"] == own[1]["text"]
    # The web profile reef-dsh web boots: the same defaults and binding (the config node targets the headless
    # patch alone), the headless profile's extension by relative path, and a manifest that reads the patch once.
    web = yaml.safe_load(files[DSH_WEB_PATCH].replace("!!js ", ""))
    assert web[:-1] == [row for row in patch[:-1] if row["id"] != "agent-loop"]
    assert web[-1] == {"insert": [{"id": "extension-tracer", "name": "../headless/extensions/tracer.mjs"}]}
    assert json.loads(files["dsh/profiles/web/package.json"]) == {
        "name": "dsh-profile-web",
        "private": True,
        "dependencies": {},
        "dsh": {
            "profile": {"bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"], "patchReload": "startup"}
        },
    }


def test_dsh_command_with_its_own_frontmatter_stays_user_only() -> None:
    """A command's frontmatter, read the way dsh reads it, keeps the node's keys, but only the person can run it,
    and a name or description dsh would not accept is filled in; frontmatter that does not parse is refused."""
    descriptor = get_adapter("dsh")
    path = "dsh-agents/skills/chat/SKILL.md"
    user_only = "---\nname: chat\ndescription: Chat\ndisable-model-invocation: true\n---\n"

    def command(text: str) -> str:
        return render_composition([("agent_command", {"name": "chat", "text": text})], descriptor)[path]

    own = "---\nname: chat\ndescription: Enter chat mode\nwhenToUse: on request\n---\n# Chat\n\nSearch only.\n"
    assert command(own) == (
        "---\nname: chat\ndescription: Enter chat mode\nwhenToUse: on request\ndisable-model-invocation: true\n"
        "---\n# Chat\n\nSearch only.\n"
    )
    # The node cannot make its command model invocable or hide it from the person, and dsh ignores a skill that
    # holds a camelCase invocation key.
    flipped = "---\nname: chat\ndescription: Chat\ndisable-model-invocation: false\nuser-invocable: false\n---\nBody\n"
    assert command(flipped) == user_only + "Body\n"
    legacy = "---\nname: chat\ndescription: Chat\nuserInvocable: true\ndisableModelInvocation: false\nmodelInvocable: true\n"
    assert command(legacy + "---\nBody\n") == user_only + "Body\n"
    # dsh ignores a skill whose name is not a skill name or whose description is empty or not a string, so a header
    # that lacks either, or holds another value, gets the one a header without frontmatter gets.
    for header in (
        "description: Chat",
        "name:\ndescription: ''",
        "name: 123\ndescription: [a]",
        "name: Chat Mode",
        "description: 09",
    ):
        assert command(f"---\n{header}\n---\n# Chat\n") == user_only + "# Chat\n"
    for empty in ("---\n---\n# Chat\n", "---\n~\n---\n# Chat\n"):
        assert command(empty) == user_only + "# Chat\n"
    # dsh reads YAML 1.2, where Yes and 1:30 are strings, so the header does too, and every header is written so
    # that YAML 1.2 reads a string back where YAML 1.1 or 1.2 would read another type.
    for written in ("Yes", "off", "1:30", "=", "'09'", "'0o17'"):
        value = written.strip("'")
        assert command(f"---\nname: chat\ndescription: {written}\n---\nBody\n") == (
            f"---\nname: chat\ndescription: '{value}'\ndisable-model-invocation: true\n---\nBody\n"
        )
    assert command("1e3\n") == "---\nname: chat\ndescription: '1e3'\ndisable-model-invocation: true\n---\n1e3\n"
    # A scalar tagged ! is a string to dsh, so the name true and the description 123 are kept, a list tagged ! is
    # a list, and a quoted true with a newline is written quoted again.
    assert command("---\nname: ! true\ndescription: ! 123\nx: ! [a]\n---\nBody\n") == (
        "---\nname: 'true'\ndescription: '123'\nx:\n- a\ndisable-model-invocation: true\n---\nBody\n"
    )
    assert command('---\nname: chat\ndescription: ! "true\\n"\n---\nBody\n') == (
        "---\nname: chat\ndescription: 'true\n\n  '\ndisable-model-invocation: true\n---\nBody\n"
    )
    # dsh takes a fence line less one trailing carriage return, and a close at the end of the file.
    assert command("---\r\nname: chat\r\ndescription: Chat\r\n---\r\nBody\r\n") == user_only + "Body\r\n"
    assert command("---\nname: chat\ndescription: Chat\n---") == user_only
    for broken, reason in (
        ("---\nname: [chat\n---\nBody\n", "not valid YAML"),
        ("---\nname: chat\ndescription: !!binary aGk=\n---\nBody\n", "not valid YAML: a value has the tag"),
        ("---\n- chat\n---\nBody\n", "not a YAML mapping"),
        ("---\nname: chat\nx: " + "[" * 3000 + "]" * 3000 + "\n---\nBody\n", "nested too deeply"),
        # An integer past Python's digit limit fails its conversion; the proposal is refused, the step goes on.
        ("---\nname: chat\nx: " + "9" * 4301 + "\n---\nBody\n", "cannot read"),
        ("---\nname: chat\ndescription: Chat\nBody\n", "never closes"),
        ("---\r\nname: chat\r\ndescription: Chat\r\nBody\r\n", "never closes"),
    ):
        with pytest.raises(RenderError, match=f"{path} .*{reason}"):
            command(broken)


def test_dsh_command_frontmatter_that_cannot_be_written_again_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A header that loaded can still fail to be written again (nesting the writer recurses on); that is a refusal
    of the proposal, never an exception out of render."""
    import reef.harness.adapters.dsh.quirks as dsh_quirks

    def deep(*args: object, **kwargs: object) -> str:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(dsh_quirks.yaml, "dump", deep)
    text = "---\nname: chat\ndescription: Chat\n---\nBody\n"
    with pytest.raises(RenderError, match="cannot write again"):
        dsh_quirks._with_frontmatter("dsh-agents/skills/chat/SKILL.md", text, True)


@pytest.mark.parametrize(
    ("api", "route_api", "base_url"),
    [
        ("openai", "openai-completions", "http://127.0.0.1:9/v1"),
        ("anthropic", "anthropic-messages", "http://127.0.0.1:9"),
    ],
)
def test_dsh_binds_both_profiles_in_every_dialect(api, route_api, base_url) -> None:
    """The headless and the web patch each carry the same Reef route and default model, in the dialect bound."""
    descriptor = get_adapter("dsh")
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1", api=api)
    files = render_composition([*_dsh_nodes(), *binding.compose_nodes(descriptor)], descriptor)
    route = {
        "displayName": "Reef",
        "apiKeyEnv": "REEF_API_KEY",
        "api": route_api,
        "baseURL": base_url,
        "models": [{"id": "m1"}],
    }
    for patch_path in (DSH_PATCH, DSH_WEB_PATCH):
        by_id = {row["id"]: row for row in yaml.safe_load(files[patch_path].replace("!!js ", "")) if "id" in row}
        assert by_id["llm-pi-ai"]["config"] == {"providers": {"reef": route}}, patch_path
        assert by_id["agent-default-model"]["config"] == {"provider": "reef", "model": "m1"}, patch_path
    assert files["dsh/.env"] == "REEF_API_KEY=k-1\n"


def test_dsh_quirks_refuse_a_patch_that_breaks_the_episode() -> None:
    descriptor = get_adapter("dsh")
    with pytest.raises(RenderError, match="uncompressed"):
        render_composition(
            [("config", {"data": {"session-persistence-jsonl": {"config": {"compression": "zstd"}}}})], descriptor
        )
    with pytest.raises(RenderError, match="session-telemetry-otel disabled"):
        render_composition([("config", {"data": {"session-telemetry-otel": {"disabled": False}}})], descriptor)
    with pytest.raises(RenderError, match="must be an object"):
        render_composition([("config", {"data": {"agent-loop": "nope"}})], descriptor)
    with pytest.raises(RenderError, match="must be an object"):
        render_composition([("config", {"data": {"session-persistence-jsonl": "zstd"}})], descriptor)
    # The web profile's patch is held to the same checks: a compressed web profile refuses the shared sessions root.
    with pytest.raises(RenderError, match=f"uncompressed .* in {DSH_WEB_PATCH}"):
        render_composition(
            [
                (
                    "config",
                    {"target": "web", "data": {"session-persistence-jsonl": {"config": {"compression": "zstd"}}}},
                )
            ],
            descriptor,
        )
    with pytest.raises(RenderError, match=f"session-title-llm disabled in {DSH_WEB_PATCH}"):
        render_composition(
            [("config", {"target": "web", "data": {"session-title-llm": {"disabled": False}}})], descriptor
        )
    # And its manifest keeps the patch read once at start: with live reload dsh web exits at start.
    live = {"dsh": {"profile": {"patchReload": "live"}}}
    with pytest.raises(RenderError, match="patchReload startup"):
        render_composition([("config", {"target": "web_manifest", "data": live})], descriptor)


HERMES_CONFIG = "hermes/config.yaml"


def _hermes_nodes():
    # hermes's config target is its config.yaml, so the pi shaped config nodes are swapped for one of its own.
    nodes = [node for node in NODES if node[0] != "config"]
    extension = ("code_extension", {"name": "tracer", "code": "def register(ctx):\n    pass\n"})
    return [
        ("config", {"data": {"agent": {"max_turns": 40}}}),
        *[n for n in nodes if n[0] != "code_extension"],
        extension,
    ]


def test_hermes_render_matches_the_golden_tree() -> None:
    assert render_composition(_hermes_nodes(), get_adapter("hermes")) == golden_tree("hermes")


def test_hermes_quirks_emit_the_config_the_plugin_grants_and_skill_frontmatter() -> None:
    descriptor = get_adapter("hermes")
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1")
    files = render_composition([*_hermes_nodes(), *binding.compose_nodes(descriptor)], descriptor)
    config = yaml.safe_load(files[HERMES_CONFIG])
    assert config["model"] == {
        "provider": "custom",
        "default": "m1",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "k-1",
    }
    # An OpenRouter host is hermes's own provider, whose key it reads from the environment or its home's .env.
    assert files["hermes/.env"] == "OPENAI_API_KEY=k-1\n"
    assert config["agent"] == {"max_turns": 40}
    # The defaults that keep an episode hermetic and single request, and the second skill root, found beside
    # the episode home and, in a reef-hermes session whose home is a temp copy, at the install root.
    assert config["security"] == {"tirith_enabled": False} and "approval" not in config
    assert config["auxiliary"] == {"title_generation": {"enabled": False}}
    assert config["memory"] == {"nudge_interval": 0} and config["sessions"] == {"write_json_snapshots": True}
    # No background review or curator writes skills into the tree: in a reef-hermes session it is the release.
    assert config["curator"] == {"enabled": False}
    assert config["skills"] == {
        "creation_nudge_interval": 0,
        "external_dirs": ["${HERMES_HOME}/../hermes-commands", "${REEF_HARNESS_DEST}/hermes-commands"],
    }
    # A rendered plugin is enabled and granted, and gets its manifest.
    assert config["plugins"] == {
        "enabled": ["tracer"],
        "entries": {"tracer": {"granted_capabilities": ["tools.override"]}},
    }
    assert files["hermes/plugins/tracer/plugin.yaml"] == "name: tracer\nversion: '0.1'\ndescription: tracer\n"
    assert files["hermes/plugins/tracer/__init__.py"] == "def register(ctx):\n    pass\n"
    assert files["hermes/.no-bundled-skills"] == ""
    # A skill without frontmatter gets name and description under both roots; the author's frontmatter is left alone.
    assert (
        files["hermes/skills/notes/SKILL.md"]
        == "---\nname: notes\ndescription: Notes skill\n---\n# Notes skill\n\nKeep short notes.\n"
    )
    assert (
        files["hermes-commands/summarize/SKILL.md"]
        == "---\nname: summarize\ndescription: Summarize $1.\n---\nSummarize $1.\n"
    )
    own = ("skill", {"name": "own", "text": "---\nname: own\ndescription: mine\n---\nBody.\n"})
    assert render_composition([own], descriptor)["hermes/skills/own/SKILL.md"] == own[1]["text"]
    # The rules follow hermes's own identity, which hermes writes only to a SOUL.md that does not exist yet; a tree
    # that already starts with it is left as it is.
    assert files["hermes/SOUL.md"] == f"{DEFAULT_IDENTITY}\n\nAnswer briefly.\n\nPrefer the standard library.\n"
    kept = render_composition([("rules", {"text": f"{DEFAULT_IDENTITY}\n\nMine."})], descriptor)["hermes/SOUL.md"]
    assert kept.count(DEFAULT_IDENTITY) == 1


def test_hermes_quirks_refuse_a_config_that_breaks_the_episode() -> None:
    descriptor = get_adapter("hermes")
    with pytest.raises(RenderError, match="tirith_enabled false"):
        render_composition([("config", {"data": {"security": {"tirith_enabled": True}}})], descriptor)
    with pytest.raises(RenderError, match=r"title_generation\.enabled false"):
        render_composition([("config", {"data": {"auxiliary": {"title_generation": {"enabled": True}}}})], descriptor)
    with pytest.raises(RenderError, match="write_json_snapshots true"):
        render_composition([("config", {"data": {"sessions": {"write_json_snapshots": False}}})], descriptor)
    for review in ({"memory": {"nudge_interval": 10}}, {"skills": {"creation_nudge_interval": 10}}):
        with pytest.raises(RenderError, match=r"skills\.creation_nudge_interval 0"):
            render_composition([("config", {"data": review})], descriptor)
    with pytest.raises(RenderError, match=r"curator\.enabled false"):
        render_composition([("config", {"data": {"curator": {"enabled": True}}})], descriptor)


def test_hermes_admission_refuses_a_config_section_that_is_not_an_object() -> None:
    """A config mutation that turns a section the render checks read into a string or a list is a refused
    proposal, not an error raised out of the admission."""
    descriptor = get_adapter("hermes")
    sections = (
        {"security": "off"},
        {"auxiliary": {"title_generation": "off"}},
        {"memory": "on"},
        {"skills": "notes"},
        {"curator": "on"},
        {"sessions": [True]},
    )
    for data in sections:
        entries, refusal = admit_mutations(
            [], [Mutation("create", "c1", {"name": "config", "config": {"data": data}})], descriptor
        )
        assert entries == [] and refusal is not None and refusal.startswith("hermes composition must keep"), data
    tracer = Mutation(
        "create",
        "e1",
        {"name": "code_extension", "config": {"name": "tracer", "code": "def register(ctx):\n    pass\n"}},
    )
    for plugins in ("tracer", {"entries": ["tracer"]}, {"entries": {"tracer": "on"}}):
        config = Mutation("create", "c1", {"name": "config", "config": {"data": {"plugins": plugins}}})
        entries, refusal = admit_mutations([], [config, tracer], descriptor)
        assert entries == [] and refusal is not None and "each rendered plugin's entry objects" in refusal, plugins


def test_hermes_admission_refuses_plugin_names_that_are_not_a_list_of_strings() -> None:
    """The grant adds the rendered plugin to plugins.enabled and tools.override to its granted_capabilities. A value
    there that is not a list of strings is a refused proposal: not an error raised out of the admission, and not a
    string or an object read one character or key at a time. The tree's own names stay ahead of the grant's."""
    descriptor = get_adapter("hermes")
    extension = ("code_extension", {"name": "tracer", "code": "def register(ctx):\n    pass\n"})
    tracer = Mutation("create", "e1", {"name": "code_extension", "config": extension[1]})
    for value in (1, True, 1.5, "", "tracer", {"tracer": True}, ["tracer", 2]):
        for key, plugins in (
            ("plugins.enabled", {"enabled": value}),
            ("plugins.entries.tracer.granted_capabilities", {"entries": {"tracer": {"granted_capabilities": value}}}),
        ):
            config = Mutation("create", "c1", {"name": "config", "config": {"data": {"plugins": plugins}}})
            entries, refusal = admit_mutations([], [config, tracer], descriptor)
            assert entries == [] and refusal == f"hermes composition must keep {key} a list of strings", (key, value)
    own = {"enabled": ["other"], "entries": {"tracer": {"granted_capabilities": ["llm.model_override"]}}}
    files = render_composition([("config", {"data": {"plugins": own}}), extension], descriptor)
    assert yaml.safe_load(files[HERMES_CONFIG])["plugins"] == {
        "enabled": ["other", "tracer"],
        "entries": {"tracer": {"granted_capabilities": ["llm.model_override", "tools.override"]}},
    }


def test_hermes_quirks_add_both_commands_roots_after_the_external_dirs_a_tree_sets() -> None:
    """A config node's list replaces the one below it, so a tree that sets skills.external_dirs would drop the
    commands roots and every agent command would be unknown to hermes; the roots follow the tree's own entries, and
    a string is one entry, as hermes reads it. A value hermes cannot read as entries is a refused proposal."""
    descriptor = get_adapter("hermes")
    roots = ["${HERMES_HOME}/../hermes-commands", "${REEF_HARNESS_DEST}/hermes-commands"]
    for listed, expected in (
        ([], roots),
        (None, roots),
        ("extra", ["extra", *roots]),
        (["extra"], ["extra", *roots]),
        ([roots[1], "extra"], [roots[1], "extra", roots[0]]),
    ):
        files = render_composition([("config", {"data": {"skills": {"external_dirs": listed}}})], descriptor)
        assert yaml.safe_load(files[HERMES_CONFIG])["skills"]["external_dirs"] == expected, listed
    refused = "hermes composition must keep skills.external_dirs a list of strings"
    for listed in (1, True, 1.5, {"extra": True}, ["extra", 2]):
        config = Mutation(
            "create", "c1", {"name": "config", "config": {"data": {"skills": {"external_dirs": listed}}}}
        )
        entries, refusal = admit_mutations([], [config], descriptor)
        assert entries == [] and refusal == refused, listed


NATIVE_TOOL = (
    "native_tool",
    {
        "name": "shout",
        "description": "Upper-case a string.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "code": "def run(args, workdir):\n    return str(args.get('text', '')).upper()\n",
    },
)


NATIVE_HOOK = (
    "native_hook",
    {"name": "guard", "event": "post_execute", "code": "def listen(payload, next):\n    return next()\n"},
)


def test_native_render_matches_the_golden_tree() -> None:
    # The loop never reads a command or an extension, so native declares no path for either kind.
    nodes = [node for node in NODES if node[0] not in ("agent_command", "code_extension")]
    rendered = render_composition([*nodes, NATIVE_TOOL, NATIVE_HOOK], get_adapter("native"))
    assert rendered == golden_tree("native")
    descriptor = get_adapter("native")
    assert descriptor.tree_path == "native/tree.json" and get_adapter("pi").tree_path is None
    assert get_adapter("opencode").tree_path is None
    for kind, node in (("agent_command", "summarize"), ("code_extension", "tracer")):
        with pytest.raises(RenderError, match=f"does not render {kind} nodes"):
            render_composition([node_ for node_ in NODES if node_[1].get("name") == node], descriptor)


def test_config_nodes_deep_merge_in_tree_order() -> None:
    files = render_composition(
        [
            ("config", {"data": {"compaction": {"enabled": True, "keep": 4}, "defaultThinkingLevel": "off"}}),
            ("config", {"data": {"compaction": {"keep": 8}}}),  # later node wins per key
        ],
        get_adapter("pi"),
    )
    assert '"defaultThinkingLevel": "off"' in files["pi-agent/settings.json"]  # sibling keys survive the merge
    assert '"enabled": true' in files["pi-agent/settings.json"]
    assert '"keep": 8' in files["pi-agent/settings.json"]


def test_unknown_config_target_is_rejected() -> None:
    with pytest.raises(RenderError, match="no config target 'models'"):
        render_composition([("config", {"target": "models", "data": {}})], get_adapter("opencode"))


def test_two_nodes_cannot_render_to_the_same_path() -> None:
    skill = ("skill", {"name": "notes", "text": "# notes"})
    collision = "same path 'pi-agent/skills/notes/SKILL.md': skill 'notes' and skill 'notes'"
    with pytest.raises(RenderError, match=re.escape(collision)):
        render_composition([skill, skill], get_adapter("pi"))


def test_opencode_quirk_rejects_reopened_autoupdate() -> None:
    with pytest.raises(RenderError, match="autoupdate false"):
        render_composition([("config", {"data": {"autoupdate": True}})], get_adapter("opencode"))


EVIL_PROVIDER = {"evil": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://127.0.0.1:9/v1"}}}
#: The model binding's shape with an endpoint of the tree's own; the only key a tree can hold is an empty one.
BINDING_COPY = {
    "provider": {
        "reef": {
            "npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": "http://127.0.0.1:9/v1", "apiKey": ""},
            "models": {"x": {}},
        }
    },
    "model": "reef/x",
}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"provider": EVIL_PROVIDER}, "must not set provider"),
        ({"provider": {"reef": {"options": {"baseURL": "http://127.0.0.1:9/v1"}}}}, "must not set provider"),
        (BINDING_COPY, "must not set provider"),
        ({"model": "reef/other"}, "must not set model"),
        ({"small_model": "reef/served"}, "must not set small_model"),
        ({"disabled_providers": ["reef"]}, "must not set disabled_providers"),
        ({"enabled_providers": ["reef", "opencode"]}, r"must keep enabled_providers \['reef'\]"),
        ({"enabled_providers": []}, r"must keep enabled_providers \['reef'\]"),
        ({"agent": {"build": {"model": "evil/m"}}}, "agent 'build' must not choose a model"),
        ({"mode": {"chat": {"model": "evil/m"}}}, "mode 'chat' must not choose a model"),
        ({"command": {"hi": {"template": "Hi.", "model": "evil/m"}}}, r"command 'hi' in opencode\.json must not"),
    ],
)
def test_opencode_quirk_refuses_a_composition_that_chooses_the_model(data: dict, message: str) -> None:
    """Only the binding reef appends writes provider and model, so admission, which renders the tree alone, refuses
    a tree that sets either or picks a model elsewhere. With the binding appended, the binding wins the keys it
    writes, and a provider or a model choice beyond them is still refused."""
    descriptor = get_adapter("opencode")
    with pytest.raises(RenderError, match=message):
        render_composition([("config", {"data": data})], descriptor)
    bound = [("config", {"data": data}), *ModelBinding("http://127.0.0.1:8900", "served").compose_nodes(descriptor)]
    if set(data) <= {"provider", "model"} and set(data.get("provider", {})) <= {"reef"}:
        config = json.loads(render_composition(bound, descriptor)["opencode/opencode.json"])
        assert config["provider"]["reef"]["options"]["baseURL"] == "http://127.0.0.1:8900/v1"
        assert config["model"] == "reef/served"
    else:
        with pytest.raises(RenderError, match=message):
            render_composition(bound, descriptor)


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_opencode_quirk_admits_the_model_binding(api: str) -> None:
    """The binding's own provider and model pass the quirk, with the client models listed beside the served one."""
    descriptor = get_adapter("opencode")
    binding = ModelBinding(base_url="http://127.0.0.1:8900", model="served", api_key="k", api=api)
    files = render_composition(binding.compose_nodes(descriptor, models=("other/big",)), descriptor)
    config = json.loads(files["opencode/opencode.json"])
    assert config["model"] == "reef/served" and set(config["provider"]["reef"]["models"]) == {"served", "other/big"}


def test_opencode_admission_refuses_a_tree_that_copies_the_binding() -> None:
    """The binding always writes a non-empty apiKey, and admission refuses a tree holding one, so a tree cannot
    publish the binding's shape with an endpoint of its own."""
    descriptor = get_adapter("opencode")

    def admit(api_key: str) -> str | None:
        data = {**BINDING_COPY, "provider": {"reef": {**BINDING_COPY["provider"]["reef"]}}}
        data["provider"]["reef"]["options"] = {"baseURL": "http://127.0.0.1:9/v1", "apiKey": api_key}
        options = {"name": "config", "config": {"data": data}}
        return admit_mutations([], [Mutation("create", "copy", options)], descriptor)[1]

    assert "must not set provider" in (admit("") or "")
    assert "carries an inline credential" in (admit("sk-tree") or "")


def test_opencode_runs_offer_only_the_binding_provider() -> None:
    """Without a provider list opencode also offers its own zen provider, so a model choice the check missed would
    leave Reef; the descriptor lists only the binding's provider."""
    config = json.loads(render_composition([], get_adapter("opencode"))["opencode/opencode.json"])
    assert config["enabled_providers"] == ["reef"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("\ufeff---\nagent: ghost\n---\nSay hi.", "starts with a byte order mark"),
        ('---json\n{"agent": "ghost"}\n---\nSay hi.', "names the engine 'json' after ---"),
        ('---json\n{"model": "opencode/big-pickle"}\n---\nSay hi.', "names the engine 'json' after ---"),
        ("---js\n{agent: 'ghost'}\n---\nSay hi.", "names the engine 'js' after ---"),
        ("---\nagent: ghost\n", "has no closing --- line"),
        ("---\nagent: ghost\nallowed-tools: a: b\n---\nSay hi.", "mapping values are not allowed here at line 3"),
        (
            "---\nagent: ghost\ndescription: x\n  y: a: b\n---\nSay hi.",
            "mapping values are not allowed here at line 4",
        ),
        ("---\nagent: ghost\nagent: build\n---\nSay hi.", "the key 'agent' appears twice at line 3"),
        (
            "---\ndescription: a: b\nagent: ghost\nagent: build\n---\nSay hi.",
            "cannot be read even with its values that hold ': ' rewritten as opencode rewrites them: the key 'agent'",
        ),
        ("---\ndescription: " + "[" * 5000 + "]" * 5000 + "\n---\nSay hi.", "cannot be read: maximum recursion depth"),
        ("---\n- build\n---\nSay hi.", "frontmatter is a list"),
        ("---\ndescription: 5\n---\nSay hi.", "field 'description' must be a str, got 5"),
        ("---\nsubtask: yes\n---\nSay hi.", "field 'subtask' must be a bool, got 'yes'"),
        ("---\ndescription: !!int 1e5\n---\nSay hi.", "a value has the tag 'tag:yaml.org,2002:int' at line 2"),
        ("---\ndescription: ! 5\n---\nSay hi.", "a value has the tag '!' at line 2"),
        ("---\ndescription: 2001-13-45\n---\nSay hi.", "the date '2001-13-45' does not exist at line 2"),
    ],
)
def test_opencode_refuses_frontmatter_it_cannot_read_as_opencode_does(text: str, message: str) -> None:
    """opencode's gray-matter strips a byte order mark, takes text after --- as another engine, reads to the end of
    the file with no closing line, and fails on a repeated key and on YAML its rewrite of values that hold ': '
    cannot repair (a key with a dash, an indented line), where opencode reads the file with no keys or skips it; for
    each form a check that read the file its own way would miss the agent or the model opencode sees. A field of
    the wrong type, with js-yaml's booleans (true and false only), makes opencode refuse its whole config. js-yaml
    checks a tag against its value and moves a date that does not exist forward to a real one, where PyYAML's
    constructors fail, so a tagged value and such a date are refused rather than read another way, and so is a
    value nested too deep for PyYAML."""
    descriptor = get_adapter("opencode")
    with pytest.raises(RenderError, match=f"opencode command 'hi' .*{message}"):
        render_composition([("agent_command", {"name": "hi", "text": text})], descriptor)
    if "field" not in message:
        with pytest.raises(RenderError, match=f"opencode skill 'notes' .*{message}"):
            render_composition([("skill", {"name": "notes", "text": text})], descriptor)


def test_opencode_admits_the_frontmatter_forms_it_reads_as_opencode_does() -> None:
    """No frontmatter, a line of four dashes (which gray-matter reads as none), comments only, CRLF line ends, and a
    plain block with fields of the right types."""
    descriptor = get_adapter("opencode")
    for text in (
        "Say hi to $ARGUMENTS.",
        "----\nagent: ghost\n----\nSay hi.",
        "---\n# a note\n---\nSay hi.",
        "---\r\nagent: plan\r\n---\r\nSay hi.",
        "---\ndescription: Say hi\nagent: build\nsubtask: true\nvariant: high\n---\nSay hi.",
    ):
        render_composition([("agent_command", {"name": "hi", "text": text})], descriptor)
    render_composition(
        [("skill", {"name": "notes", "text": "---\nname: notes\ndescription: Notes.\n---\n"})], descriptor
    )


def test_opencode_gives_a_skill_without_frontmatter_the_name_and_description_it_lists_skills_by() -> None:
    """opencode lists no skill whose SKILL.md lacks name and description, so a bare one gets them: the directory and
    the first line, as strings to js-yaml too; a file with a block of its own is left as written."""
    descriptor = get_adapter("opencode")
    for text, description in (("# Answer style\n\nBe brief.", "Answer style"), ("yes\n", "yes"), ("1e5", "1e5")):
        files = render_composition([("skill", {"name": "answer-style", "text": text})], descriptor)
        rendered = files["opencode/skill/answer-style/SKILL.md"]
        assert read_frontmatter("skill", rendered) == {"name": "answer-style", "description": description}
        assert rendered.rstrip("\n").endswith(text.rstrip("\n"))
    own = "---\nname: notes\ndescription: Notes.\n---\n"
    assert (
        render_composition([("skill", {"name": "notes", "text": own})], descriptor)["opencode/skill/notes/SKILL.md"]
        == own
    )
    # A block of its own without both keys would never be listed: render refuses it rather than ship it unseen.
    for text in ("---\ndescription: Notes.\n---\n", "---\nname: notes\n---\n", "---\n---\nBe brief.\n"):
        with pytest.raises(RenderError, match="must set name and description"):
            render_composition([("skill", {"name": "notes", "text": text})], descriptor)


def test_opencode_reads_frontmatter_again_as_opencode_rewrites_it() -> None:
    """When js-yaml cannot read a block, opencode rewrites each top level value that holds ': ' and is not quoted as
    a block scalar and reads the file again; render reads the same keys, so it admits a file opencode loads and
    still sees an agent, a model or a name next to such a value. PyYAML also fails on a tab after a colon, which
    js-yaml reads (``description:<tab>{a: b}`` is a mapping to opencode), so render reads a file again only after a
    failure both readers share."""
    descriptor = get_adapter("opencode")
    chat = "Enter chat mode: conversation with web search only, no other tools"
    for text, description in (
        (f"---\ndescription: {chat}\nagent: build\n---\nSay hi.", chat),
        ("---\r\ndescription: a: b\r\nagent: plan\r\n---\r\nSay hi.", "a: b"),
        ("---\ndescription: a: 'b'\n---\nSay hi.", "a: 'b'"),
        ("---\ndescription :  a: b  \n# a: note\n---\nSay hi.", "a: b"),
        ("---\ndescription: `/chat`: web search only\n---\nSay hi.", "`/chat`: web search only"),
    ):
        assert read_frontmatter("command 'hi'", text)["description"] == description
        render_composition([("agent_command", {"name": "hi", "text": text})], descriptor)
    for text, message in (
        ("---\nagent: ghost\ndescription: note: more\n---\nSay hi.", "names agent 'ghost'"),
        ("---\nmodel: opencode/big-pickle\ndescription: note: more\n---\nSay hi.", "must not choose a model"),
        ("---\nname: reefine\ndescription: note: more\n---\nSay hi.", "must not set name 'reefine'"),
        ("---\ndescription: !!str a: b\n---\nSay hi.", "a value has the tag"),
        ("---\ndescription:\t{a: b}\n---\nSay hi.", "cannot start any token at line 2"),
    ):
        with pytest.raises(RenderError, match=f"opencode command 'hi' .*{message}"):
            render_composition([("agent_command", {"name": "hi", "text": text})], descriptor)


@pytest.mark.parametrize(
    ("value", "number"),
    [
        ("1e5", 100000.0),
        ("1.5e3", 1500.0),
        ("1E5", 100000.0),
        ("1.", 1.0),
        ("017", 15),
        ("0x1F", 31),
        ("1_000", 1000),
        ("1:30", 90),
        ("09", None),
        ("01.5", None),
        ("1_", None),
        ("0b_", None),
        ("=", None),
        ("-.nan", None),
        ("2001-1-1", None),
        ("._e0", math.nan),
        (" ._e-5 ", math.nan),
        (".__e9", math.nan),
    ],
)
def test_opencode_reads_frontmatter_values_as_js_yaml_does(value: str, number: float | None) -> None:
    """js-yaml 3, opencode's frontmatter reader, reads 1e5 and 1.5e3 as numbers, which PyYAML reads as strings, and
    09, 01.5, 1_, 0b_ and = as strings, which PyYAML reads as numbers or cannot read. It reads ._e0 as NaN, where
    PyYAML's float constructor fails. A number in a string field makes opencode refuse its whole config, so render
    refuses exactly the values opencode reads as numbers."""
    descriptor = get_adapter("opencode")
    for field in ("description", "variant"):
        nodes = [("agent_command", {"name": "hi", "text": f"---\n{field}: {value}\n---\nSay hi."})]
        if number is None:
            render_composition(nodes, descriptor)
        else:
            with pytest.raises(RenderError, match=re.escape(f"field {field!r} must be a str, got {number!r}")):
                render_composition(nodes, descriptor)


def test_opencode_command_file_keeps_its_own_name() -> None:
    """opencode files a command file under the name in its frontmatter, so a file with another name could take the
    place of /reefine; render refuses a name that is not the file's."""
    descriptor = get_adapter("opencode")

    def command(name: str, frontmatter_name: str) -> tuple[str, dict]:
        return ("agent_command", {"name": name, "text": f"---\nname: {frontmatter_name}\n---\nSay hi."})

    render_composition([command("chat", "chat")], descriptor)
    with pytest.raises(RenderError, match="command 'chat' frontmatter must not set name 'reefine'"):
        render_composition([command("reefine", "reefine"), command("chat", "reefine")], descriptor)
    with pytest.raises(RenderError, match="command 'chat' frontmatter must not set name 5"):
        render_composition([command("chat", "5")], descriptor)


def test_opencode_tree_keeps_an_agent_that_starts_a_run() -> None:
    """With no default_agent opencode starts a run with its first agent that is neither a subagent nor hidden, and
    fails every run when none is left; its title, summary and compaction agents stay hidden unless the tree says
    otherwise. An agent given another name is looked up by that name and fails the run that uses it."""
    descriptor = get_adapter("opencode")

    def render(data: dict) -> None:
        render_composition([("config", {"data": data})], descriptor)

    off = {"build": {"disable": True}, "plan": {"disable": True}}
    for data in (
        {"agent": {**off, "chat": {"prompt": "You chat."}}},
        {"agent": {**off, "title": {"hidden": False}}},
        {"agent": off, "mode": {"chat": {"prompt": "You chat."}}},
        {"agent": {"build": {"name": "build"}}},
    ):
        render(data)
    for data in (
        {"agent": off},
        {"agent": {"build": {"mode": "subagent"}, "plan": {"disable": True}}},
        {"agent": {"build": {"hidden": True}, "plan": {"disable": True}}},
        {"agent": {**off, "title": {"prompt": "Name it."}}},
        {"agent": {**off, "chat": {"prompt": "You chat.", "mode": "subagent"}}},
    ):
        with pytest.raises(RenderError, match="leaves no agent that can start a run"):
            render(data)
    with pytest.raises(RenderError, match="opencode agent 'build' must not set name 'ghost'"):
        render({"agent": {"build": {"name": "ghost"}}})
    with pytest.raises(RenderError, match="opencode mode 'chat' must not set name 'talk'"):
        render({"mode": {"chat": {"prompt": "You chat.", "name": "talk"}}})


def test_opencode_reads_agent_flags_with_the_types_its_schema_allows() -> None:
    """opencode's schema allows only a bool for an agent's disable and hidden and only subagent, primary or all for
    its mode, in the agent and the mode sections, and refuses its whole config otherwise, so render refuses these
    rather than read null as a primary agent. opencode merges a mode entry over the agent entry of the same name,
    so a mode entry can turn a disabled agent back on."""
    descriptor = get_adapter("opencode")

    def render(data: dict) -> None:
        render_composition([("config", {"data": data})], descriptor)

    off = {"build": {"disable": True}, "plan": {"disable": True}}
    render({"agent": off, "mode": {"build": {"disable": False}}})
    for data, message in (
        ({"agent": {"build": {"disable": 1}, "plan": {"disable": 1}}}, "agent 'build' field 'disable' must be a bool"),
        ({"agent": {**off, "general": {"mode": None}}}, "agent 'general' field 'mode' must be one of"),
        ({"agent": {**off, "chat": {"mode": "secondary"}}}, "agent 'chat' field 'mode' must be one of"),
        ({"default_agent": "general", "agent": {"general": {"mode": None}}}, "agent 'general' field 'mode' must be"),
        ({"agent": {"chat": {"hidden": "yes"}}}, "agent 'chat' field 'hidden' must be a bool"),
        ({"mode": {"chat": {"disable": 1}}}, "mode 'chat' field 'disable' must be a bool"),
        ({"agent": {"build": None}}, "agent 'build' must be an object"),
        ({"agent": []}, "agent must be an object of agents"),
    ):
        with pytest.raises(RenderError, match=f"opencode {message}"):
            render(data)


def test_opencode_default_agent_must_start_a_run() -> None:
    """opencode fails every run whose default_agent is missing, a subagent or hidden, so render refuses it."""
    descriptor = get_adapter("opencode")

    def render(data: dict) -> None:
        render_composition([("config", {"data": data})], descriptor)

    chat = {"prompt": "You chat."}
    for data in (
        {"default_agent": "build"},
        {"default_agent": "plan"},
        {"default_agent": "chat", "agent": {"chat": chat}},
        {"default_agent": "chat", "agent": {"chat": {**chat, "mode": "primary"}}},
        {"default_agent": "chat", "mode": {"chat": chat}},
    ):
        render(data)
    for data, message in (
        ({"default_agent": "ghost"}, "names agent 'ghost', which the tree does not define"),
        ({"default_agent": "general"}, "names agent 'general', a subagent or a hidden agent"),
        ({"default_agent": "chat", "agent": {"chat": {**chat, "mode": "subagent"}}}, "names agent 'chat', a subagent"),
        ({"default_agent": "chat", "agent": {"chat": {**chat, "hidden": True}}}, "names agent 'chat', a subagent"),
        ({"default_agent": "build", "agent": {"build": {"disable": True}}}, "names agent 'build', which the tree"),
        ({"default_agent": 5}, "must name an agent as a string, got 5"),
    ):
        with pytest.raises(RenderError, match=f"opencode default_agent {message}"):
            render(data)


def test_opencode_command_agent_must_be_defined_or_built_in() -> None:
    """opencode fails a command whose agent does not exist at run time, so render refuses it."""
    descriptor = get_adapter("opencode")

    def command(agent: str) -> tuple[str, dict]:
        return ("agent_command", {"name": "hi", "text": f"---\nagent: {agent}\n---\nSay hi to $ARGUMENTS."})

    chat = {"mode": "primary", "permission": {"*": "deny", "websearch": "allow"}}
    for agent in ("build", "plan", "general", "explore", "title"):
        render_composition([command(agent)], descriptor)
    render_composition([("config", {"data": {"agent": {"chat": chat}}}), command("chat")], descriptor)
    render_composition([("config", {"data": {"mode": {"chat": chat}}}), command("chat")], descriptor)
    with pytest.raises(RenderError, match="command 'hi' names agent 'ghost', which the tree does not define"):
        render_composition([command("ghost")], descriptor)
    with pytest.raises(RenderError, match="names agent 'plan'"):
        render_composition([("config", {"data": {"agent": {"plan": {"disable": True}}}}), command("plan")], descriptor)
    inline = {"command": {"hi": {"template": "Hi.", "agent": "ghost"}}}
    with pytest.raises(RenderError, match=r"command 'hi' in opencode\.json names agent 'ghost'"):
        render_composition([("config", {"data": inline})], descriptor)
    model = ("agent_command", {"name": "hi", "text": "---\nmodel: evil/m\n---\nSay hi."})
    with pytest.raises(RenderError, match="command 'hi' must not choose a model"):
        render_composition([model], descriptor)


def test_opencode_descriptor_turns_on_websearch_for_interactive_runs_only() -> None:
    """opencode registers websearch for provider reef only with OPENCODE_ENABLE_EXA; episodes do not search the web."""
    descriptor = get_adapter("opencode")
    assert descriptor.client_env == {"OPENCODE_ENABLE_EXA": "1"}
    assert "OPENCODE_ENABLE_EXA" not in descriptor.env


def test_claude_quirk_rejects_reopened_hermetic_switches() -> None:
    with pytest.raises(RenderError, match="includeCoAuthoredBy"):
        render_composition([("config", {"data": {"includeCoAuthoredBy": True}})], get_adapter("claude"))
    with pytest.raises(RenderError, match="DISABLE_AUTOUPDATER"):
        render_composition([("config", {"data": {"env": {"DISABLE_AUTOUPDATER": "0"}}})], get_adapter("claude"))


@pytest.mark.parametrize("key", ["DISABLE_UPDATES", "DISABLE_AUTOUPDATER", "disable_updates"])
@pytest.mark.parametrize("value", ["x", "maybe", "2", "1"])
def test_claude_quirk_refuses_a_tree_that_sets_the_updater_switches(key: str, value: str) -> None:
    """Claude Code copies settings.env over its environment and reads the updater switches as on only for 1, true,
    yes or on, so a tree must not set them at all: the episode env and reef-claude's client_env own them. Windows
    matches env names in any case."""
    with pytest.raises(RenderError, match=f"must not set {key} in settings.env"):
        render_composition([("config", {"data": {"env": {key: value}}})], get_adapter("claude"))


def test_claude_quirk_renders_a_tree_env_without_the_updater_switches() -> None:
    assert "env" not in json.loads(render_composition([], get_adapter("claude"))["claude/settings.json"])
    node = ("config", {"data": {"env": {"BASH_MAX_TIMEOUT_MS": "600000"}}})
    rendered = json.loads(render_composition([node], get_adapter("claude"))["claude/settings.json"])
    assert rendered["env"] == {"BASH_MAX_TIMEOUT_MS": "600000"}


@pytest.mark.parametrize("value", [None, "enable", False])
def test_claude_quirk_keeps_deep_link_registration_off(value: object) -> None:
    """An interactive reef-claude run must not register the pinned binary as the person's claude-cli:// handler."""
    rendered = json.loads(render_composition([], get_adapter("claude"))["claude/settings.json"])
    assert rendered["disableDeepLinkRegistration"] == "disable"
    with pytest.raises(RenderError, match="disableDeepLinkRegistration"):
        render_composition([("config", {"data": {"disableDeepLinkRegistration": value}})], get_adapter("claude"))


def test_bundled_adapters_are_discoverable() -> None:
    assert set(available_adapters()) >= {"claude", "codex", "dsh", "opencode", "pi"}


def test_pi_descriptor_declares_what_an_interactive_run_needs() -> None:
    """The wrapper adds ``client_env`` to a person's run; the install script names ``client_tools`` missing from PATH."""
    descriptor = get_adapter("pi")
    assert descriptor.client_env == {"PI_SKIP_VERSION_CHECK": "1"}
    assert "PI_OFFLINE" not in descriptor.client_env  # an interactive run talks to reef
    assert descriptor.client_tools == (("rg", "ripgrep"), ("fd", "fd"))


def test_claude_descriptor_turns_deep_link_registration_off_on_the_command_line() -> None:
    """Claude Code skips a whole settings.json that fails its schema, so reef-claude passes the setting as flag
    settings, which no tree can change."""
    assert get_adapter("claude").client_args == ("--settings", '{"disableDeepLinkRegistration":"disable"}')


def test_claude_descriptor_names_its_version_flags_and_turns_its_own_updater_off() -> None:
    """A version flag gets nothing ahead of it. Claude Code's own update and install commands, and its background
    updater, would install the latest release over the person's claude, so a reef-claude run turns them off."""
    descriptor = get_adapter("claude")
    assert descriptor.client_version_args == ("--version", "-v", "-V")
    assert descriptor.client_env == {"DISABLE_AUTOUPDATER": "1", "DISABLE_UPDATES": "1"}


@pytest.mark.parametrize("key", ["client_args", "client_version_args"])
@pytest.mark.parametrize("value", ["--settings", [1], [""]])
def test_descriptor_client_argument_lists_are_lists_of_strings(tmp_path, key: str, value: object) -> None:
    data = yaml.safe_load((Path(reef.harness.adapters.__file__).parent / "claude" / "descriptor.yaml").read_text())
    data[key] = value
    target = tmp_path / "descriptor.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match=f"'{key}'"):
        load_descriptor(target)


def test_adapter_renderer_steps_run_in_order_on_their_own_files() -> None:
    calls: list[tuple[str, str]] = []

    class RecordingRenderer(AdapterRenderer):
        @staticmethod
        def process_config(path, config):
            calls.append(("process_config", path))
            return config

        @staticmethod
        def process_skill(path, text):
            calls.append(("process_skill", path))
            return text.upper()

        @staticmethod
        def process_command(path, text):
            calls.append(("process_command", path))
            return text

        @staticmethod
        def finalize_render(files):
            calls.append(("finalize_render", ""))
            return files

    nodes = [("config", {"data": {"defaultThinkingLevel": "off"}})]
    nodes += [node for node in NODES if node[0] in ("skill", "agent_command")]
    files = render_composition(nodes, replace(get_adapter("pi"), renderer=RecordingRenderer))
    assert calls == [
        ("process_config", "pi-agent/settings.json"),
        ("process_config", "pi-agent/models.json"),
        ("process_skill", "pi-agent/skills/notes/SKILL.md"),
        ("process_command", "pi-agent/prompts/summarize.md"),
        ("finalize_render", ""),
    ]
    assert files["pi-agent/skills/notes/SKILL.md"] == "# NOTES SKILL\n\nKEEP SHORT NOTES.\n"
    assert json.loads(files["pi-agent/settings.json"])["defaultThinkingLevel"] == "off"

    # The render refuses a key Reef's model binding writes whatever the adapter's renderer does.
    with pytest.raises(RenderError, match=re.escape("must not set defaultModel in pi-agent/settings.json")):
        render_composition(
            [("config", {"data": {"defaultModel": "other/m"}})], replace(get_adapter("pi"), renderer=RecordingRenderer)
        )


@pytest.mark.parametrize(
    "data", [{"model": "gpt-5"}, {"model_provider": "reef"}, {"model_providers": {"reef": {"name": "Reef"}}}]
)
def test_codex_refuses_a_tree_that_sets_what_the_binding_writes(data: dict[str, object]) -> None:
    with pytest.raises(RenderError, match="Reef's model binding writes it"):
        render_composition([("config", {"data": data})], get_adapter("codex"))


def test_an_adapter_renderer_step_must_be_a_static_method() -> None:
    with pytest.raises(TypeError, match=r"StatefulRenderer\.process_skill must be a static method"):

        class StatefulRenderer(AdapterRenderer):
            def process_skill(self, path, text):
                return text


def test_a_quirks_module_with_only_finalize_render_still_gets_the_last_word(tmp_path, monkeypatch) -> None:
    """An external quirks module written before AdapterRenderer keeps working, with a deprecation warning."""
    (tmp_path / "legacy_quirks.py").write_text(
        "def finalize_render(files):\n    return {**files, 'native/LEGACY': 'seen ' + str(len(files)) + '\\n'}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    data = yaml.safe_load((Path(reef.harness.adapters.__file__).parent / "native" / "descriptor.yaml").read_text())
    data["quirks"] = "legacy_quirks"
    target = tmp_path / "descriptor.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="finalize_render, which is deprecated"):
        descriptor = load_descriptor(target)
    files = render_composition([], descriptor)
    assert files["native/LEGACY"] == f"seen {len(files) - 1}\n"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "from reef.harness.adapters.descriptor import AdapterRenderer\n"
            "class FirstRenderer(AdapterRenderer):\n    pass\n"
            "class SecondRenderer(AdapterRenderer):\n    pass\n",
            "more than one AdapterRenderer subclass: FirstRenderer, SecondRenderer",
        ),
        (
            "from reef.harness.adapters.descriptor import AdapterRenderer\n"
            "class Renderer(AdapterRenderer):\n    pass\n"
            "def finalize_render(files):\n    return files\n",
            "an AdapterRenderer subclass or finalize_render, not both",
        ),
    ],
    ids=["two-renderers", "renderer-and-finalize-render"],
)
def test_a_quirks_module_must_define_one_adapter_renderer(tmp_path, monkeypatch, source: str, message: str) -> None:
    # One module name per case, since Python caches an imported module by its name.
    module_name = f"invalid_quirks_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    data = yaml.safe_load((Path(reef.harness.adapters.__file__).parent / "native" / "descriptor.yaml").read_text())
    data["quirks"] = module_name
    target = tmp_path / "descriptor.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match=message):
        load_descriptor(target)


def test_bundled_descriptors_keep_the_state_their_resume_and_setup_read() -> None:
    """A reef-<adapter> run keeps what the binary's resume and first-run setup read in the installed tree."""
    kept = {
        name: tuple(replace(state, preference_keys=()) for state in get_adapter(name).client_state)
        for name in ("pi", "claude", "codex", "hermes", "dsh")
    }
    assert kept == {
        # pi writes settings.json in place at its first interactive start (lastChangelogVersion).
        "pi": (ClientState("pi-agent/sessions", "directory"), ClientState("pi-agent/settings.json", "file")),
        "claude": (ClientState("claude/projects", "directory"), ClientState("claude/.claude.json", "file")),
        "codex": (ClientState("codex/sessions", "directory"),),
        "hermes": (
            ClientState("hermes/state.db", "sqlite"),
            ClientState("hermes/sessions", "directory"),
            ClientState("hermes/logs", "directory"),
        ),
        "dsh": (
            ClientState("dsh/.credentials.yaml", "file"),
            ClientState("dsh/settings.yaml", "file"),
            ClientState("dsh/.agent-presets", "directory"),
            ClientState("dsh/sessions", "directory"),
            ClientState("dsh/storages", "directory"),
        ),
    }


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"path": "pi-agent/sessions", "kind": "link"}, "'kind'"),
        ({"kind": "directory"}, "'path'"),
        ({"path": "sessions", "kind": "directory"}, "not below 'pi-agent'"),
        ({"path": "pi-agent", "kind": "directory"}, "not below 'pi-agent'"),
        (
            {"path": "pi-agent/sessions", "kind": "directory", "preference_keys": ["theme"]},
            "'preference_keys' is for a 'file' entry",
        ),
        (
            {"path": "pi-agent/settings.json", "kind": "file", "preference_keys": "theme"},
            "'preference_keys' must be a list",
        ),
    ],
)
def test_descriptor_client_state_is_a_known_kind_below_the_composition(tmp_path, entry, message: str) -> None:
    """The wrapper links only the composition into its temp copy, so state elsewhere would never reach the binary."""
    data = yaml.safe_load((Path(reef.harness.adapters.__file__).parent / "pi" / "descriptor.yaml").read_text())
    data["client_state"] = [entry]
    target = tmp_path / "descriptor.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match=message):
        load_descriptor(target)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host_env", ["DOCKER_HOST"], "'host_env' must map variable names"),
        ("host_env", {"NOT A NAME": ""}, "'host_env' must map variable names"),
        ("host_env", {"PI_CODING_AGENT_DIR": ""}, "'env' already sets: PI_CODING_AGENT_DIR"),
        ("is_root_bind_mounted", "yes", "'is_root_bind_mounted' must be a boolean"),
    ],
)
def test_descriptor_host_env_and_root_placement_are_validated(tmp_path, field: str, value, message: str) -> None:
    data = yaml.safe_load((Path(reef.harness.adapters.__file__).parent / "pi" / "descriptor.yaml").read_text())
    data[field] = value
    target = tmp_path / "descriptor.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match=message):
        load_descriptor(target)


def test_only_terminus_keeps_host_environment_or_has_a_bind_mounted_root() -> None:
    """Every other bundled adapter's episode stays hermetic: no service variable beyond PATH and TMPDIR."""
    terminus = get_adapter("terminus")
    assert terminus.host_env == {"DOCKER_HOST": "", "DOCKER_CONTEXT": "", "DOCKER_CONFIG": "{home}/.docker"}
    assert terminus.is_root_bind_mounted
    for name in sorted(set(available_adapters()) - {"terminus"}):
        descriptor = get_adapter(name)
        assert descriptor.host_env == {} and not descriptor.is_root_bind_mounted, name


def test_a_client_state_file_that_is_a_config_target_names_the_keys_the_binary_may_change() -> None:
    """The install writes a config target and the check skips client state, so a client state file that is also a
    config target (pi's settings.json) names the keys its binary may change; the check covers every other key."""
    for name in available_adapters():
        descriptor = get_adapter(name)
        targets = {target.path for target in descriptor.config_targets.values()}
        for state in descriptor.client_state:
            assert state.kind != "file" or state.path not in targets or state.preference_keys, (name, state.path)


def test_pi_settings_keys_that_load_code_are_not_preference_keys() -> None:
    """From these settings keys pi 0.84.2 loads code, skills, prompts or themes, runs a program, or decides whose code
    loads, so none of them may be a key a session changes unchecked."""
    (settings,) = [state for state in get_adapter("pi").client_state if state.path == "pi-agent/settings.json"]
    checked_keys = {
        "packages",
        "extensions",
        "skills",
        "prompts",
        "themes",
        "shellPath",
        "shellCommandPrefix",
        "npmCommand",
        "externalEditor",
        "defaultProjectTrust",
    }
    assert checked_keys.isdisjoint(settings.preference_keys)
    # What pi writes by itself, and what /model saves.
    assert {"lastChangelogVersion", "theme", "defaultModel", "defaultProvider"} <= set(settings.preference_keys)


def test_pi_settings_keys_pi_rewrites_on_load_are_rendered_in_their_new_form() -> None:
    """pi 0.84.2 rewrites queueMode, a boolean websockets and a skills object when it loads settings.json and writes
    the new form back on its next save; the render writes that form, so pi's save changes no key the check covers."""
    files = render_composition(
        [
            (
                "config",
                {
                    "data": {
                        "queueMode": "all",
                        "websockets": False,
                        "skills": {"customDirectories": ["extra"], "enableSkillCommands": True},
                    }
                },
            )
        ],
        get_adapter("pi"),
    )
    assert json.loads(files["pi-agent/settings.json"]) == {
        "steeringMode": "all",
        "transport": "sse",
        "skills": ["extra"],
        "enableSkillCommands": True,
    }


def test_pi_skill_without_frontmatter_gets_name_and_description() -> None:
    files = render_composition(
        [("skill", {"name": "notes", "text": "# Notes skill\n\nKeep short notes.\n"})], get_adapter("pi")
    )
    assert (
        files["pi-agent/skills/notes/SKILL.md"]
        == "---\nname: notes\ndescription: Notes skill\n---\n# Notes skill\n\nKeep short notes.\n"
    )
    own = ("skill", {"name": "own", "text": "---\nname: own\ndescription: mine\n---\nBody.\n"})
    assert render_composition([own], get_adapter("pi"))["pi-agent/skills/own/SKILL.md"] == own[1]["text"]


@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_catalog_uses_bound_capabilities(reasoning: bool) -> None:
    from reef.core.model_metadata import ModelMetadata

    descriptor = get_adapter("codex")
    binding = ModelBinding("http://up", "custom/model", api="responses", metadata=ModelMetadata(640_000, reasoning))
    files = render_composition(binding.compose_nodes(descriptor), descriptor)
    config = tomllib.loads(files["codex/config.toml"])
    assert config["model_catalog_json"] == "models.json"
    model = next(model for model in json.loads(files["codex/models.json"])["models"] if model["slug"] == binding.model)
    assert model["slug"] == binding.model
    assert model["context_window"] == model["max_context_window"] == 640_000
    assert bool(model["supported_reasoning_levels"]) is reasoning
    assert model["supports_reasoning_summary_parameter"] is reasoning


@pytest.mark.parametrize(
    "metadata",
    [
        {"context_window": True, "reasoning": True},
        {"context_window": 0, "reasoning": True},
        {"context_window": 100, "reasoning": "yes"},
        {"context_window": 100, "reasoning": True, "base_instructions": "override"},
    ],
)
def test_codex_rejects_model_metadata_from_a_tree(metadata: dict[str, object]) -> None:
    # Only Reef's model binding writes the model metadata, whatever it holds.
    with pytest.raises(
        RenderError, match=re.escape("must not set models in codex/models.json: Reef's model binding writes it")
    ):
        render_composition(
            [("config", {"target": "models", "data": {"models": {"m": metadata}}})], get_adapter("codex")
        )


@pytest.mark.parametrize(
    ("model", "native_model"),
    [
        ("gpt-5.4", "gpt-5.4"),
        ("openai/gpt-5.4", "gpt-5.4"),
        ("gpt-5.4-2026-03-05", "gpt-5.4"),
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("gpt-6-astra", "gpt-6-astra"),
        ("openai/gpt-6-astra", "gpt-6-astra"),
    ],
)
@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_overrides_native_capabilities_and_keeps_instructions(
    model: str, native_model: str, reasoning: bool
) -> None:
    from reef.core.model_metadata import ModelMetadata
    from reef.harness.adapters.codex.quirks import bundled_model_catalog

    descriptor = get_adapter("codex")
    binding = ModelBinding("http://up", model, api="responses", metadata=ModelMetadata(640_000, reasoning))
    files = render_composition(binding.compose_nodes(descriptor), descriptor)
    assert tomllib.loads(files["codex/config.toml"])["model_catalog_json"] == "models.json"
    bundled = bundled_model_catalog()
    catalog = {entry["slug"]: entry for entry in json.loads(files["codex/models.json"])["models"]}
    native = bundled[native_model]
    expected = {
        **native,
        "slug": model,
        "context_window": 640_000,
        "max_context_window": 640_000,
        "supports_reasoning_summary_parameter": reasoning,
        "supported_reasoning_levels": native["supported_reasoning_levels"] if reasoning else [],
        "default_reasoning_level": native["default_reasoning_level"] if reasoning else None,
    }
    assert catalog == {**bundled, model: expected}
