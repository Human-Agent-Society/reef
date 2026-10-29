"""Guarantees of reef.harness.tree.render and the bundled adapter descriptors."""

from __future__ import annotations

import json
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tests.support.harness_render import descriptor_data, write_descriptor

from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters import BUILTIN_ADAPTERS, available_adapters, get_adapter
from reef.harness.adapters.codex.render import bundled_model_catalog
from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterDescriptor, AdapterRenderer, DescriptorError, load_descriptor
from reef.harness.adapters.hermes.render import DEFAULT_IDENTITY
from reef.harness.adapters.opencode.render import read_frontmatter
from reef.harness.tasks.model_binding import ModelBinding, ModelBindingError
from reef.harness.tree import HarnessTree, RenderError, render_tree, validate_tree

GOLDENS = Path(__file__).parent / "data" / "harness_goldens"

PI = get_adapter("pi")
OPENCODE = get_adapter("opencode")
CLAUDE = get_adapter("claude")
CODEX = get_adapter("codex")
DSH = get_adapter("dsh")
HERMES = get_adapter("hermes")
NATIVE = get_adapter("native")


def config(data: Mapping[str, object], target: str = "primary") -> dict[str, str]:
    """The tree file of one config target."""
    return {f"config/{target}.json": json.dumps(data)}


def render(
    files: Mapping[str, str], descriptor: AdapterDescriptor, binding: ModelBinding | None = None
) -> dict[str, str]:
    """``files`` as a tree rendered for ``descriptor``, pointed at ``binding`` when one is given."""
    overlays = () if binding is None else binding.config_overlays(descriptor)
    return render_tree(HarnessTree(dict(files)), descriptor, overlays)


def problems(files: Mapping[str, str], descriptor: AdapterDescriptor) -> tuple[str, ...]:
    return validate_tree(HarnessTree(dict(files)), descriptor).problems


def golden_tree(adapter: str) -> dict[str, str]:
    root = GOLDENS / adapter
    return {path.relative_to(root).as_posix(): path.read_text() for path in sorted(root.rglob("*")) if path.is_file()}


# One file of every kind an adapter shares; each source below adds the config its harness reads.
COMMON = {
    "rules.md": "Answer briefly.\n\nPrefer the standard library.\n",
    "skills/notes/SKILL.md": "# Notes skill\n\nKeep short notes.",
    "commands/summarize.md": "Summarize $1.",
}
TRACER = "export default function tracer() {}"
CLAUDE_HOOKS = {
    "hooks": {
        "SessionStart": [
            {"hooks": [{"type": "command", "command": 'node "${CLAUDE_PLUGIN_ROOT}/extensions/tracer.mjs"'}]}
        ]
    }
}
PYTHON_TRACER = "def register(ctx):\n    pass\n"
SETTINGS = {"defaultModel": "qwen/qwen3-8b", "defaultProvider": "qwen"}
QWEN_MODELS = {
    "providers": {
        "qwen": {
            "api": "openai-completions",
            "apiKey": "dummy",
            "baseUrl": "http://localhost:8000/v1",
            "models": [{"id": "qwen3-8b"}],
        }
    }
}
SHOUT_TOOL = (
    "def run(args, workdir):\n    return str(args.get('text', '')).upper()\n\n"
    "NAME = 'shout'\nDESCRIPTION = 'Upper-case a string.'\n"
    "PARAMETERS = {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}\n"
    "CAPABILITIES = []\n"
)
GUARD_HOOK = "def listen(payload, next):\n    return next()\n\nNAME = 'guard'\nEVENT = 'post_execute'\n"

#: The source tree each adapter's golden tree renders from.
SOURCES = {
    # pi keeps every model call on the binding, so its tree holds the one setting its package carries.
    "pi": {**COMMON, **config({"defaultTools": []}), "extensions/tracer.ts": TRACER},
    "opencode": {**COMMON, **config({"theme": "light"}), "extensions/tracer.ts": TRACER},
    # claude's extensions run as the plugin's hooks.
    "claude": {**COMMON, **config(CLAUDE_HOOKS, "hooks"), "extensions/tracer.mjs": TRACER},
    # codex's plugin carries skills alone, a command among them.
    "codex": {path: text for path, text in COMMON.items() if path != "rules.md"},
    # dsh's config is keyed by plugin id.
    "dsh": {**COMMON, **config({"agent-loop": {"config": {"maxSteps": 40}}}), "extensions/tracer.mjs": TRACER},
    # hermes's config is its config.yaml, and its extensions are Python plugins.
    "hermes": {**COMMON, **config({"agent": {"max_turns": 40}}), "extensions/tracer.py": PYTHON_TRACER},
    # The native loop reads no command or extension, but tools and hooks. Validation refuses the inline apiKey;
    # the render does not check it.
    "native": {
        "rules.md": COMMON["rules.md"],
        "skills/notes/SKILL.md": COMMON["skills/notes/SKILL.md"],
        **config(SETTINGS),
        **config(QWEN_MODELS, "models"),
        "native/tools/shout.py": SHOUT_TOOL,
        "native/hooks/guard.py": GUARD_HOOK,
    },
}


@pytest.mark.parametrize("adapter", sorted(SOURCES))
def test_render_matches_the_golden_tree(adapter: str) -> None:
    assert render(SOURCES[adapter], get_adapter(adapter)) == golden_tree(adapter)


def test_render_refuses_a_tree_it_cannot_lay_out() -> None:
    for files, message in (
        ({"RULES.md": "x"}, "RULES.md is not a path of the opencode harness tree"),
        ({"config/primary.json": "{"}, "config/primary.json is not JSON"),
        ({"config/primary.json": "[]"}, "config/primary.json must hold a JSON object"),
        (config({}, "models"), "adapter 'opencode' declares no config target 'models'"),
        ({"native/tools/shout.py": SHOUT_TOOL}, "native/tools/shout.py is not a path of the opencode harness tree"),
    ):
        with pytest.raises(RenderError, match=re.escape(message)):
            render(files, OPENCODE)
    with pytest.raises(RenderError, match="declares no config target 'models'"):
        render_tree(HarnessTree({}), OPENCODE, [("models", {})])


def test_config_merges_the_tree_over_the_defaults_and_each_overlay_over_the_tree() -> None:
    tree = HarnessTree(config({"compaction": {"auto": True, "prune": False}, "theme": "light"}))
    files = render_tree(tree, OPENCODE, [("primary", {"compaction": {"prune": True}})])
    # Sibling keys survive the merge, and the overlay wins the key it writes.
    assert json.loads(files["plugin/config.json"]) == {"compaction": {"auto": True, "prune": True}, "theme": "light"}
    # hermes's descriptor defaults sit below the tree's own keys.
    settings = yaml.safe_load(render(config({"agent": {"max_turns": 40}}), HERMES)["hermes/config.yaml"])
    assert settings["agent"]["max_turns"] == 40 and settings["security"]["tirith_enabled"] is False


def test_native_and_codex_take_no_part_their_harness_never_reads() -> None:
    # The loop never reads a command or an extension, so native declares neither part; Codex's hooks would run
    # outside its command sandbox, so it declares no extension part.
    for path, descriptor in (
        ("commands/summarize.md", NATIVE),
        ("extensions/tracer.py", NATIVE),
        ("extensions/tracer.mjs", CODEX),
    ):
        with pytest.raises(
            RenderError, match=f"{re.escape(path)} is not a path of the {descriptor.name} harness tree"
        ):
            render({path: "x"}, descriptor)


def test_codex_binding_uses_the_responses_api() -> None:
    assert "model_provider" not in tomllib.loads(render({}, CODEX)["codex/config.toml"])

    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1", api="responses")
    config_toml = tomllib.loads(render({}, CODEX, binding)["codex/config.toml"])
    assert config_toml["model"] == "m1" and config_toml["model_provider"] == "reef"
    assert config_toml["model_providers"]["reef"] == {
        "name": "Reef",
        "base_url": "http://127.0.0.1:9/v1",
        "wire_api": "responses",
        "supports_websockets": False,
        "experimental_bearer_token": "k-1",
    }


@pytest.mark.parametrize(
    "data",
    [
        {"approval_policy": "never"},
        {"web_search": "live"},
        {"model_reasoning_effort": "high"},
        {"features": {"plugins": False}},
        {"mcp_servers": {"untrusted": {"command": "tool"}}},
        {"model_catalog_json": "/etc/passwd"},
    ],
)
def test_codex_takes_no_config_key_from_a_tree(data: dict) -> None:
    """A Codex plugin carries no config key, so config.toml holds the run's defaults and the binding alone."""
    with pytest.raises(RenderError, match=r"in codex/config\.toml"):
        render(config(data), CODEX)


def test_codex_run_argv_pins_approvals_and_web_search_off_over_the_config() -> None:
    """The rendered config.toml holds no approval_policy; the run argv pins both before the prompt."""
    config_toml = tomllib.loads(render({}, CODEX)["codex/config.toml"])
    assert config_toml["web_search"] == "disabled" and "approval_policy" not in config_toml
    argv = list(CODEX.argv)
    overrides = [argv[index + 1] for index, token in enumerate(argv) if token == "--config"]
    assert overrides == ['approval_policy="never"', 'web_search="disabled"'] and argv[-1] == "{prompt}"


def test_codex_renders_a_command_as_a_skill_and_refuses_one_that_shares_a_skills_name() -> None:
    """Codex loads no custom prompts, so a command is a skill of the plugin, typed as $reef:name."""
    command = {"commands/reefine.md": "File the text after $reefine as a request."}
    files = render(command, CODEX)
    assert files["plugin/skills/reefine/SKILL.md"] == (
        "---\nname: reefine\ndescription: File the text after $reefine as a request.\n---\n"
        "File the text after $reefine as a request.\n"
    )
    assert not any(path.startswith("codex/prompts/") for path in files)
    collision = "skills/reefine/SKILL.md and commands/reefine.md both render to 'plugin/skills/reefine/SKILL.md'"
    with pytest.raises(RenderError, match=re.escape(collision + "; rename one")):
        render({**command, "skills/reefine/SKILL.md": "# reefine"}, CODEX)


def test_codex_run_installs_the_plugin_as_codex_plugin_add_does() -> None:
    """A run's CODEX_HOME holds the plugin's copy and the two config entries that codex plugin add writes."""
    files = render({"skills/fix/SKILL.md": "# Fix"}, CODEX)
    assert files["codex/plugins/cache/reef/reef/0.0.0/skills/fix/SKILL.md"] == files["plugin/skills/fix/SKILL.md"]
    assert json.loads(files["plugin/.codex-plugin/plugin.json"])["name"] == "reef"
    config_toml = tomllib.loads(files["codex/config.toml"])
    assert config_toml["marketplaces"]["reef"] == {"source_type": "local", "source": "{root}"}
    assert config_toml["plugins"]["reef@reef"] == {"enabled": True}


def test_codex_requires_the_responses_dialect() -> None:
    # The error names the serve flag that picks the dialect the adapter declares.
    with pytest.raises(
        ModelBindingError,
        match=r"declares no model_binding for the .openai. api .*serve with --inference.upstream-api responses",
    ):
        ModelBinding("http://up", "m").config_overlays(CODEX)


@pytest.mark.parametrize(
    "data", [{"model": "gpt-5"}, {"model_provider": "reef"}, {"model_providers": {"reef": {"name": "Reef"}}}]
)
def test_codex_refuses_a_tree_that_sets_what_the_binding_writes(data: dict[str, object]) -> None:
    with pytest.raises(RenderError, match="Reef's model binding writes it"):
        render(config(data), CODEX)


@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_catalog_uses_bound_capabilities(reasoning: bool) -> None:
    binding = ModelBinding("http://up", "custom/model", api="responses", metadata=ModelMetadata(640_000, reasoning))
    files = render({}, CODEX, binding)
    assert tomllib.loads(files["codex/config.toml"])["model_catalog_json"] == "models.json"
    model = next(model for model in json.loads(files["codex/models.json"])["models"] if model["slug"] == binding.model)
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
        render(config({"models": {"m": metadata}}, "models"), CODEX)


@pytest.mark.parametrize(
    ("model", "native_model"),
    [
        ("gpt-5.5", "gpt-5.5"),
        ("openai/gpt-5.5", "gpt-5.5"),
        ("gpt-5.5-2026-03-05", "gpt-5.5"),
        ("gpt-5.6-luna", "gpt-5.6-luna"),
        ("gpt-6-astra", "gpt-6-astra"),
        ("openai/gpt-6-astra", "gpt-6-astra"),
    ],
)
@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_overrides_native_capabilities_and_keeps_instructions(
    model: str, native_model: str, reasoning: bool
) -> None:
    binding = ModelBinding("http://up", model, api="responses", metadata=ModelMetadata(640_000, reasoning))
    files = render({}, CODEX, binding)
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


DSH_PATCH = "dsh/profiles/headless/cordis.patch.yml"
DSH_WEB_PATCH = "dsh/profiles/web/cordis.patch.yml"


def test_dsh_render_steps_emit_the_patch_layer_the_env_file_and_skill_frontmatter() -> None:
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1")
    files = render(SOURCES["dsh"], DSH, binding)
    patch = yaml.safe_load(files[DSH_PATCH].replace("!!js ", ""))
    by_id = {row["id"]: row for row in patch if "id" in row}
    # Every entry in id order, the defaults that keep a run readable and quiet, the binding, then the inserts.
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
    for own in (
        "---\nname: own\ndescription: mine\n---\nBody.\n",
        "---\r\nname: own\r\ndescription: mine\r\n---\r\nBody.\r\n",
    ):
        assert render({"skills/own/SKILL.md": own}, DSH)["dsh/skills/own/SKILL.md"] == own
    # The web profile reef-dsh web boots: the same defaults and binding (the tree's primary config targets the
    # headless patch alone), the headless profile's extension by relative path, and a manifest that reads the patch
    # once.
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
    """A command's frontmatter, read the way dsh reads it, keeps the file's keys, but only the person can run it,
    and a name or description dsh would not accept is filled in; frontmatter that does not parse is refused."""
    path = "dsh-agents/skills/chat/SKILL.md"
    user_only = "---\nname: chat\ndescription: Chat\ndisable-model-invocation: true\n---\n"

    def command(text: str) -> str:
        return render({"commands/chat.md": text}, DSH)[path]

    own = "---\nname: chat\ndescription: Enter chat mode\nwhenToUse: on request\n---\n# Chat\n\nSearch only.\n"
    assert command(own) == (
        "---\nname: chat\ndescription: Enter chat mode\nwhenToUse: on request\ndisable-model-invocation: true\n"
        "---\n# Chat\n\nSearch only.\n"
    )
    # The file cannot make its command model invocable or hide it from the person, and dsh ignores a skill that
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
        # An integer past Python's digit limit fails its conversion; the tree is refused, the step goes on.
        ("---\nname: chat\nx: " + "9" * 4301 + "\n---\nBody\n", "cannot read"),
        ("---\nname: chat\ndescription: Chat\nBody\n", "never closes"),
        ("---\r\nname: chat\r\ndescription: Chat\r\nBody\r\n", "never closes"),
    ):
        with pytest.raises(RenderError, match=f"{path} .*{reason}"):
            command(broken)


def test_dsh_command_frontmatter_that_cannot_be_written_again_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A header that loaded can still fail to be written again (nesting the writer recurses on); that is a refusal
    of the tree, never an exception out of render."""
    import reef.harness.adapters.dsh.render as dsh_render

    def deep(*args: object, **kwargs: object) -> str:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(dsh_render.yaml, "dump", deep)
    text = "---\nname: chat\ndescription: Chat\n---\nBody\n"
    with pytest.raises(RenderError, match="cannot write again"):
        dsh_render._with_frontmatter("dsh-agents/skills/chat/SKILL.md", text, True)


@pytest.mark.parametrize(
    ("api", "route_api", "base_url"),
    [
        ("openai", "openai-completions", "http://127.0.0.1:9/v1"),
        ("anthropic", "anthropic-messages", "http://127.0.0.1:9"),
    ],
)
def test_dsh_binds_both_profiles_in_every_dialect(api, route_api, base_url) -> None:
    """The headless and the web patch each carry the same Reef route and default model, in the dialect bound."""
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1", api=api)
    files = render(SOURCES["dsh"], DSH, binding)
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


def test_dsh_locks_refuse_a_patch_that_breaks_the_run() -> None:
    with pytest.raises(
        RenderError,
        match=r"on session-persistence-jsonl\.config\.compression in dsh/profiles/headless/cordis\.patch\.yml",
    ):
        render(config({"session-persistence-jsonl": {"config": {"compression": "zstd"}}}), DSH)
    with pytest.raises(RenderError, match=r"on session-telemetry-otel\.disabled in"):
        render(config({"session-telemetry-otel": {"disabled": False}}), DSH)
    with pytest.raises(RenderError, match=r"key 'agent-loop' in .*: a patch entry is an object"):
        render(config({"agent-loop": "nope"}), DSH)
    # A patch entry that is not an object takes the checked keys below it away; the check refuses it as such,
    # where the chained reads a hand written check needs would crash on the shape.
    with pytest.raises(RenderError, match=r"on session-persistence-jsonl\.config\.compression in"):
        render(config({"session-persistence-jsonl": "zstd"}), DSH)
    # The web profile's patch is held to the same checks: a compressed web profile refuses the shared sessions root.
    with pytest.raises(RenderError, match=rf"on session-persistence-jsonl\.config\.compression in {DSH_WEB_PATCH}"):
        render(config({"session-persistence-jsonl": {"config": {"compression": "zstd"}}}, "web"), DSH)
    with pytest.raises(RenderError, match=f"on session-title-llm.disabled in {DSH_WEB_PATCH}"):
        render(config({"session-title-llm": {"disabled": False}}, "web"), DSH)
    # And its manifest keeps the patch read once at start: with live reload dsh web exits at start.
    live = {"dsh": {"profile": {"patchReload": "live"}}}
    with pytest.raises(RenderError, match=r"on dsh\.profile\.patchReload in"):
        render(config(live, "web_manifest"), DSH)


HERMES_CONFIG = "hermes/config.yaml"


def test_hermes_render_steps_emit_the_config_the_plugin_grants_and_skill_frontmatter() -> None:
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="m1", api_key="k-1")
    files = render(SOURCES["hermes"], HERMES, binding)
    config_yaml = yaml.safe_load(files[HERMES_CONFIG])
    assert config_yaml["model"] == {
        "provider": "custom",
        "default": "m1",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "k-1",
    }
    # An OpenRouter host is hermes's own provider, whose key it reads from the environment or its home's .env.
    assert files["hermes/.env"] == "OPENAI_API_KEY=k-1\n"
    assert config_yaml["agent"] == {"max_turns": 40}
    # The defaults that keep a run hermetic and single request, and the commands root beside the run home.
    assert config_yaml["security"] == {"tirith_enabled": False} and "approval" not in config_yaml
    assert config_yaml["auxiliary"] == {"title_generation": {"enabled": False}}
    assert config_yaml["memory"] == {"nudge_interval": 0} and config_yaml["sessions"] == {"write_json_snapshots": True}
    # No background review or curator writes skills into the rendered tree.
    assert config_yaml["curator"] == {"enabled": False}
    assert config_yaml["skills"]["creation_nudge_interval"] == 0
    assert "${HERMES_HOME}/../hermes-commands" in config_yaml["skills"]["external_dirs"]
    # A rendered plugin is enabled and granted, and gets its manifest.
    assert config_yaml["plugins"] == {
        "enabled": ["tracer"],
        "entries": {"tracer": {"granted_capabilities": ["tools.override"]}},
    }
    assert files["hermes/plugins/tracer/plugin.yaml"] == "name: tracer\nversion: '0.1'\ndescription: tracer\n"
    assert files["hermes/plugins/tracer/__init__.py"] == PYTHON_TRACER
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
    own = "---\nname: own\ndescription: mine\n---\nBody.\n"
    assert render({"skills/own/SKILL.md": own}, HERMES)["hermes/skills/own/SKILL.md"] == own
    # The rules follow hermes's own identity, which hermes writes only to a SOUL.md that does not exist yet; a tree
    # that already starts with it is left as it is.
    assert files["hermes/SOUL.md"] == f"{DEFAULT_IDENTITY}\n\nAnswer briefly.\n\nPrefer the standard library.\n"
    kept = render({"rules.md": f"{DEFAULT_IDENTITY}\n\nMine."}, HERMES)["hermes/SOUL.md"]
    assert kept.count(DEFAULT_IDENTITY) == 1


def test_hermes_locks_refuse_a_config_that_breaks_the_run() -> None:
    with pytest.raises(RenderError, match=r"on security\.tirith_enabled in"):
        render(config({"security": {"tirith_enabled": True}}), HERMES)
    with pytest.raises(RenderError, match=r"on auxiliary\.title_generation\.enabled in"):
        render(config({"auxiliary": {"title_generation": {"enabled": True}}}), HERMES)
    with pytest.raises(RenderError, match=r"on sessions\.write_json_snapshots in"):
        render(config({"sessions": {"write_json_snapshots": False}}), HERMES)
    with pytest.raises(RenderError, match=r"on memory\.nudge_interval in"):
        render(config({"memory": {"nudge_interval": 10}}), HERMES)
    with pytest.raises(RenderError, match=r"on skills\.creation_nudge_interval in"):
        render(config({"skills": {"creation_nudge_interval": 10}}), HERMES)
    with pytest.raises(RenderError, match=r"on curator\.enabled in"):
        render(config({"curator": {"enabled": True}}), HERMES)


def test_hermes_validation_refuses_a_config_section_that_is_not_an_object() -> None:
    """A config that turns a section the render checks read into a string or a list is a validation problem, not
    an error raised out of validation."""
    sections = (
        {"security": "off"},
        {"auxiliary": {"title_generation": "off"}},
        {"memory": "on"},
        {"skills": "notes"},
        {"curator": "on"},
        {"sessions": [True]},
    )
    for data in sections:
        (problem,) = problems(config(data), HERMES)
        assert problem.startswith("hermes harness tree fails the check on"), data
    for plugins in ("tracer", {"entries": ["tracer"]}, {"entries": {"tracer": "on"}}):
        (problem,) = problems({**config({"plugins": plugins}), "extensions/tracer.py": PYTHON_TRACER}, HERMES)
        assert "each rendered plugin's entry objects" in problem, plugins


def test_hermes_validation_refuses_plugin_names_that_are_not_a_list_of_strings() -> None:
    """The grant adds the rendered plugin to plugins.enabled and tools.override to its granted_capabilities. A value
    there that is not a list of strings is a validation problem: not an error raised out of validation, and not a
    string or an object read one character or key at a time. The tree's own names stay ahead of the grant's."""
    for value in (1, True, 1.5, "", "tracer", {"tracer": True}, ["tracer", 2]):
        for key, plugins in (
            ("plugins.enabled", {"enabled": value}),
            ("plugins.entries.tracer.granted_capabilities", {"entries": {"tracer": {"granted_capabilities": value}}}),
        ):
            files = {**config({"plugins": plugins}), "extensions/tracer.py": PYTHON_TRACER}
            assert problems(files, HERMES) == (f"hermes harness tree must keep {key} a list of strings",), (key, value)
    own = {"enabled": ["other"], "entries": {"tracer": {"granted_capabilities": ["llm.model_override"]}}}
    files = render({**config({"plugins": own}), "extensions/tracer.py": PYTHON_TRACER}, HERMES)
    assert yaml.safe_load(files[HERMES_CONFIG])["plugins"] == {
        "enabled": ["other", "tracer"],
        "entries": {"tracer": {"granted_capabilities": ["llm.model_override", "tools.override"]}},
    }


def test_hermes_render_steps_add_the_commands_roots_after_the_external_dirs_a_tree_sets() -> None:
    """A tree's list replaces the default one, so a tree that sets skills.external_dirs would drop the commands
    roots and every agent command would be unknown to hermes; the roots follow the tree's own entries, a root the
    tree lists already is not added twice, and a string is one entry, as hermes reads it. A value hermes cannot
    read as entries is a validation problem."""
    roots = yaml.safe_load(render({}, HERMES)[HERMES_CONFIG])["skills"]["external_dirs"]
    assert "${HERMES_HOME}/../hermes-commands" in roots
    for listed, expected in (
        ([], roots),
        (None, roots),
        ("extra", ["extra", *roots]),
        (["extra"], ["extra", *roots]),
        ([roots[-1], "extra"], [roots[-1], "extra", *roots[:-1]]),
    ):
        files = render(config({"skills": {"external_dirs": listed}}), HERMES)
        assert yaml.safe_load(files[HERMES_CONFIG])["skills"]["external_dirs"] == expected, listed
    refused = ("hermes harness tree must keep skills.external_dirs a list of strings",)
    for listed in (1, True, 1.5, {"extra": True}, ["extra", 2]):
        assert problems(config({"skills": {"external_dirs": listed}}), HERMES) == refused, listed


def test_opencode_descriptor_lock_rejects_reopened_autoupdate() -> None:
    with pytest.raises(RenderError, match=r"on autoupdate in plugin/config\.json: runs keep autoupdate"):
        render(config({"autoupdate": True}), OPENCODE)


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
        ({"provider": EVIL_PROVIDER}, "on provider in"),
        ({"provider": {"reef": {"options": {"baseURL": "http://127.0.0.1:9/v1"}}}}, "on provider in"),
        (BINDING_COPY, "on provider in"),
        ({"model": "reef/other"}, "on model in"),
        ({"small_model": "reef/served"}, r"on small_model in plugin/config\.json: Reef's model binding"),
        ({"disabled_providers": ["reef"]}, "on disabled_providers in"),
        ({"enabled_providers": ["reef", "opencode"]}, "on enabled_providers in"),
        ({"enabled_providers": []}, "on enabled_providers in"),
        ({"agent": {"build": {"model": "evil/m"}}}, r"on agent\.build\.model in"),
        ({"mode": {"chat": {"model": "evil/m"}}}, r"on mode\.chat\.model in"),
        ({"command": {"hi": {"template": "Hi.", "model": "evil/m"}}}, r"on command\.hi\.model in"),
    ],
)
def test_opencode_render_refuses_a_tree_that_chooses_the_model(data: dict, message: str) -> None:
    """Reef's plugin merges the tree's config over opencode's own, after the binding, so the tree's config names no
    provider and chooses no model, with the binding or without it."""
    for binding in (None, ModelBinding("http://127.0.0.1:8900", "served")):
        with pytest.raises(RenderError, match=message):
            render(config(data), OPENCODE, binding)


@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_opencode_render_admits_the_model_binding(api: str) -> None:
    """The binding's own provider and model pass the renderer, with the further models listed beside the served one."""
    binding = ModelBinding(base_url="http://127.0.0.1:8900", model="served", api_key="k", api=api)
    files = render_tree(HarnessTree({}), OPENCODE, binding.config_overlays(OPENCODE, models=("other/big",)))
    config_json = json.loads(files["opencode/opencode.json"])
    assert config_json["model"] == "reef/served"
    assert set(config_json["provider"]["reef"]["models"]) == {"served", "other/big"}


def test_opencode_validation_refuses_a_tree_that_copies_the_binding() -> None:
    """The binding always writes a non-empty apiKey, and validation refuses a tree holding one, so a tree cannot
    publish the binding's shape with an endpoint of its own."""

    def copy_problems(api_key: str) -> tuple[str, ...]:
        reef = {**BINDING_COPY["provider"]["reef"], "options": {"baseURL": "http://127.0.0.1:9/v1", "apiKey": api_key}}
        return problems(config({**BINDING_COPY, "provider": {"reef": reef}}), OPENCODE)

    (empty,) = copy_problems("")
    assert "on provider in" in empty
    (inline,) = copy_problems("sk-tree")
    assert "carries an inline credential" in inline


def test_opencode_runs_offer_only_the_binding_provider() -> None:
    """Without a provider list opencode also offers its own zen provider, so a model choice the check missed would
    leave Reef; the descriptor lists only the binding's provider. A run does not search the web, so the env
    does not register opencode's websearch tool."""
    assert json.loads(render({}, OPENCODE)["opencode/opencode.json"])["enabled_providers"] == ["reef"]
    assert "OPENCODE_ENABLE_EXA" not in OPENCODE.env


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("﻿---\nagent: ghost\n---\nSay hi.", "starts with a byte order mark"),
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
    with pytest.raises(RenderError, match=f"opencode command 'hi' .*{message}"):
        render({"commands/hi.md": text}, OPENCODE)
    if "field" not in message:
        with pytest.raises(RenderError, match=f"opencode skill 'notes' .*{message}"):
            render({"skills/notes/SKILL.md": text}, OPENCODE)


def test_opencode_admits_the_frontmatter_forms_it_reads_as_opencode_does() -> None:
    """No frontmatter, a line of four dashes (which gray-matter reads as none), comments only, CRLF line ends, and a
    plain block with fields of the right types."""
    for text in (
        "Say hi to $ARGUMENTS.",
        "----\nagent: ghost\n----\nSay hi.",
        "---\n# a note\n---\nSay hi.",
        "---\r\nagent: plan\r\n---\r\nSay hi.",
        "---\ndescription: Say hi\nagent: build\nsubtask: true\nvariant: high\n---\nSay hi.",
    ):
        render({"commands/hi.md": text}, OPENCODE)
    render({"skills/notes/SKILL.md": "---\nname: notes\ndescription: Notes.\n---\n"}, OPENCODE)


def test_opencode_gives_a_skill_without_frontmatter_the_name_and_description_it_lists_skills_by() -> None:
    """opencode lists no skill whose SKILL.md lacks name and description, so a bare one gets them: the directory and
    the first line, as strings to js-yaml too; a file with a block of its own is left as written."""
    for text, description in (("# Answer style\n\nBe brief.", "Answer style"), ("yes\n", "yes"), ("1e5", "1e5")):
        rendered = render({"skills/answer-style/SKILL.md": text}, OPENCODE)["plugin/skill/answer-style/SKILL.md"]
        assert read_frontmatter("skill", rendered) == {"name": "answer-style", "description": description}
        assert rendered.rstrip("\n").endswith(text.rstrip("\n"))
    own = "---\nname: notes\ndescription: Notes.\n---\n"
    assert render({"skills/notes/SKILL.md": own}, OPENCODE)["plugin/skill/notes/SKILL.md"] == own
    # A block of its own without both keys would never be listed: render refuses it rather than ship it unseen.
    for text in ("---\ndescription: Notes.\n---\n", "---\nname: notes\n---\n", "---\n---\nBe brief.\n"):
        with pytest.raises(RenderError, match="must set name and description"):
            render({"skills/notes/SKILL.md": text}, OPENCODE)


def test_opencode_reads_frontmatter_again_as_opencode_rewrites_it() -> None:
    """When js-yaml cannot read a block, opencode rewrites each top level value that holds ': ' and is not quoted as
    a block scalar and reads the file again; render reads the same keys, so it admits a file opencode loads and
    still sees an agent, a model or a name next to such a value. PyYAML also fails on a tab after a colon, which
    js-yaml reads (``description:<tab>{a: b}`` is a mapping to opencode), so render reads a file again only after a
    failure both readers share."""
    chat = "Enter chat mode: conversation with web search only, no other tools"
    for text, description in (
        (f"---\ndescription: {chat}\nagent: build\n---\nSay hi.", chat),
        ("---\r\ndescription: a: b\r\nagent: plan\r\n---\r\nSay hi.", "a: b"),
        ("---\ndescription: a: 'b'\n---\nSay hi.", "a: 'b'"),
        ("---\ndescription :  a: b  \n# a: note\n---\nSay hi.", "a: b"),
        ("---\ndescription: `/chat`: web search only\n---\nSay hi.", "`/chat`: web search only"),
    ):
        assert read_frontmatter("command 'hi'", text)["description"] == description
        render({"commands/hi.md": text}, OPENCODE)
    for text, message in (
        ("---\nagent: ghost\ndescription: note: more\n---\nSay hi.", "names agent 'ghost'"),
        ("---\nmodel: opencode/big-pickle\ndescription: note: more\n---\nSay hi.", "must not choose a model"),
        ("---\nname: reefine\ndescription: note: more\n---\nSay hi.", "must not set name 'reefine'"),
        ("---\ndescription: !!str a: b\n---\nSay hi.", "a value has the tag"),
        ("---\ndescription:\t{a: b}\n---\nSay hi.", "cannot start any token at line 2"),
    ):
        with pytest.raises(RenderError, match=f"opencode command 'hi' .*{message}"):
            render({"commands/hi.md": text}, OPENCODE)


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
    for field in ("description", "variant"):
        files = {"commands/hi.md": f"---\n{field}: {value}\n---\nSay hi."}
        if number is None:
            render(files, OPENCODE)
        else:
            with pytest.raises(RenderError, match=re.escape(f"field {field!r} must be a str, got {number!r}")):
                render(files, OPENCODE)


def test_opencode_command_file_keeps_its_own_name() -> None:
    """opencode files a command file under the name in its frontmatter, so a file with another name could take the
    place of /reefine; render refuses a name that is not the file's."""

    def command(name: str, frontmatter_name: str) -> dict[str, str]:
        return {f"commands/{name}.md": f"---\nname: {frontmatter_name}\n---\nSay hi."}

    render(command("chat", "chat"), OPENCODE)
    with pytest.raises(RenderError, match="command 'chat' frontmatter must not set name 'reefine'"):
        render({**command("reefine", "reefine"), **command("chat", "reefine")}, OPENCODE)
    with pytest.raises(RenderError, match="command 'chat' frontmatter must not set name 5"):
        render(command("chat", "5"), OPENCODE)


def test_opencode_tree_keeps_an_agent_that_starts_a_run() -> None:
    """With no default_agent opencode starts a run with its first agent that is neither a subagent nor hidden, and
    fails every run when none is left; its title, summary and compaction agents stay hidden unless the tree says
    otherwise. An agent given another name is looked up by that name and fails the run that uses it."""
    off = {"build": {"disable": True}, "plan": {"disable": True}}
    for data in (
        {"agent": {**off, "chat": {"prompt": "You chat."}}},
        {"agent": {**off, "title": {"hidden": False}}},
        {"agent": off, "mode": {"chat": {"prompt": "You chat."}}},
        {"agent": {"build": {"name": "build"}}},
    ):
        render(config(data), OPENCODE)
    for data in (
        {"agent": off},
        {"agent": {"build": {"mode": "subagent"}, "plan": {"disable": True}}},
        {"agent": {"build": {"hidden": True}, "plan": {"disable": True}}},
        {"agent": {**off, "title": {"prompt": "Name it."}}},
        {"agent": {**off, "chat": {"prompt": "You chat.", "mode": "subagent"}}},
    ):
        with pytest.raises(RenderError, match="leaves no agent that can start a run"):
            render(config(data), OPENCODE)
    with pytest.raises(RenderError, match="opencode agent 'build' must not set name 'ghost'"):
        render(config({"agent": {"build": {"name": "ghost"}}}), OPENCODE)
    with pytest.raises(RenderError, match="opencode mode 'chat' must not set name 'talk'"):
        render(config({"mode": {"chat": {"prompt": "You chat.", "name": "talk"}}}), OPENCODE)


def test_opencode_reads_agent_flags_with_the_types_its_schema_allows() -> None:
    """opencode's schema allows only a bool for an agent's disable and hidden and only subagent, primary or all for
    its mode, in the agent and the mode sections, and refuses its whole config otherwise, so render refuses these
    rather than read null as a primary agent. opencode merges a mode entry over the agent entry of the same name,
    so a mode entry can turn a disabled agent back on."""
    off = {"build": {"disable": True}, "plan": {"disable": True}}
    render(config({"agent": off, "mode": {"build": {"disable": False}}}), OPENCODE)
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
            render(config(data), OPENCODE)


def test_opencode_default_agent_must_start_a_run() -> None:
    """opencode fails every run whose default_agent is missing, a subagent or hidden, so render refuses it."""
    chat = {"prompt": "You chat."}
    for data in (
        {"default_agent": "build"},
        {"default_agent": "plan"},
        {"default_agent": "chat", "agent": {"chat": chat}},
        {"default_agent": "chat", "agent": {"chat": {**chat, "mode": "primary"}}},
        {"default_agent": "chat", "mode": {"chat": chat}},
    ):
        render(config(data), OPENCODE)
    for data, message in (
        ({"default_agent": "ghost"}, "names agent 'ghost', which the tree does not define"),
        ({"default_agent": "general"}, "names agent 'general', a subagent or a hidden agent"),
        ({"default_agent": "chat", "agent": {"chat": {**chat, "mode": "subagent"}}}, "names agent 'chat', a subagent"),
        ({"default_agent": "chat", "agent": {"chat": {**chat, "hidden": True}}}, "names agent 'chat', a subagent"),
        ({"default_agent": "build", "agent": {"build": {"disable": True}}}, "names agent 'build', which the tree"),
        ({"default_agent": 5}, "must name an agent as a string, got 5"),
    ):
        with pytest.raises(RenderError, match=f"opencode default_agent {message}"):
            render(config(data), OPENCODE)


def test_opencode_command_agent_must_be_defined_or_built_in() -> None:
    """opencode fails a command whose agent does not exist at run time, so render refuses it."""

    def command(agent: str) -> dict[str, str]:
        return {"commands/hi.md": f"---\nagent: {agent}\n---\nSay hi to $ARGUMENTS."}

    chat = {"mode": "primary", "permission": {"*": "deny", "websearch": "allow"}}
    for agent in ("build", "plan", "general", "explore", "title"):
        render(command(agent), OPENCODE)
    render({**config({"agent": {"chat": chat}}), **command("chat")}, OPENCODE)
    render({**config({"mode": {"chat": chat}}), **command("chat")}, OPENCODE)
    with pytest.raises(RenderError, match="command 'hi' names agent 'ghost', which the tree does not define"):
        render(command("ghost"), OPENCODE)
    with pytest.raises(RenderError, match="names agent 'plan'"):
        render({**config({"agent": {"plan": {"disable": True}}}), **command("plan")}, OPENCODE)
    inline = {"command": {"hi": {"template": "Hi.", "agent": "ghost"}}}
    with pytest.raises(RenderError, match=r"command 'hi' in config\.json names agent 'ghost'"):
        render(config(inline), OPENCODE)
    with pytest.raises(RenderError, match="command 'hi' must not choose a model"):
        render({"commands/hi.md": "---\nmodel: evil/m\n---\nSay hi."}, OPENCODE)


@pytest.mark.parametrize(
    "data",
    [
        {"includeCoAuthoredBy": True},
        {"env": {"DISABLE_AUTOUPDATER": "0"}},
        {"env": {"BASH_MAX_TIMEOUT_MS": "600000"}},
        {"cleanupPeriodDays": 7},
        {"hooks": {"SessionStart": []}},
    ],
)
def test_claude_takes_no_setting_from_a_tree(data: dict) -> None:
    """A Claude Code plugin carries no settings, so settings.json holds the run's defaults and the binding."""
    with pytest.raises(RenderError, match=r"in claude/settings\.json"):
        render(config(data), CLAUDE)


def test_claude_renders_the_plugin_with_its_rules_as_an_output_style_and_its_hooks() -> None:
    hooks = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "node x.mjs"}]}]}}
    files = render({"rules.md": "Run the tests first.\n", **config(hooks, "hooks")}, CLAUDE)
    assert json.loads(files["plugin/.claude-plugin/plugin.json"]) == {"name": "reef"}
    style = files["plugin/output-styles/reef.md"]
    assert "force-for-plugin: true" in style and style.endswith("Run the tests first.\n")
    assert json.loads(files["plugin/hooks/hooks.json"]) == hooks
    # No hooks, no hooks file; no rules, no output style.
    assert not {"plugin/hooks/hooks.json", "plugin/output-styles/reef.md"} & set(render({}, CLAUDE))


@pytest.mark.parametrize("value", [None, "enable", False])
def test_claude_render_keeps_deep_link_registration_off(value: object) -> None:
    """A run must not register the pinned binary as the person's claude-cli:// handler."""
    assert json.loads(render({}, CLAUDE)["claude/settings.json"])["disableDeepLinkRegistration"] == "disable"
    with pytest.raises(RenderError, match="disableDeepLinkRegistration"):
        render(config({"disableDeepLinkRegistration": value}), CLAUDE)


def test_pi_settings_admit_only_what_reefs_extension_applies() -> None:
    """A pi package carries no settings.json pi reads, so Reef's extension applies the one setting a tree may hold."""
    files = render(config({"defaultTools": []}), PI)
    assert json.loads(files["plugin/settings.json"]) == {"defaultTools": []}
    assert "setActiveTools" in files["plugin/extensions/reef/index.ts"]
    for data in ({"compaction": {"enabled": False}}, {"defaultThinkingLevel": "high"}, {"defaultTools": ["bash"]}):
        with pytest.raises(RenderError, match=re.escape("plugin/settings.json")):
            render(config(data), PI)


def test_pi_skill_without_frontmatter_gets_name_and_description() -> None:
    files = render({"skills/notes/SKILL.md": "# Notes skill\n\nKeep short notes.\n"}, PI)
    assert (
        files["plugin/skills/notes/SKILL.md"]
        == "---\nname: notes\ndescription: Notes skill\n---\n# Notes skill\n\nKeep short notes.\n"
    )
    own = "---\nname: own\ndescription: mine\n---\nBody.\n"
    assert render({"skills/own/SKILL.md": own}, PI)["plugin/skills/own/SKILL.md"] == own


def test_bundled_adapters_are_discoverable() -> None:
    assert set(available_adapters()) >= set(SOURCES)


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
        def check_model_route(configs, skills, commands):
            calls.append(("check_model_route", " ".join([*skills.values(), *commands])))

        @staticmethod
        def finalize_render(files):
            calls.append(("finalize_render", ""))
            return files

    descriptor = replace(PI, renderer=RecordingRenderer)
    files = render(
        {
            **config({"defaultTools": []}),
            "skills/notes/SKILL.md": COMMON["skills/notes/SKILL.md"],
            "commands/summarize.md": COMMON["commands/summarize.md"],
        },
        descriptor,
    )
    # Every config first, in target order; then each skill and command file; the model route sees them processed.
    assert calls[:3] == [
        ("process_config", "plugin/settings.json"),
        ("process_config", "pi-agent/settings.json"),
        ("process_config", "pi-agent/models.json"),
    ]
    assert sorted(calls[3:5]) == [
        ("process_command", "plugin/prompts/summarize.md"),
        ("process_skill", "plugin/skills/notes/SKILL.md"),
    ]
    assert calls[5:] == [
        ("check_model_route", "# NOTES SKILL\n\nKEEP SHORT NOTES.\n plugin/prompts/summarize.md"),
        ("finalize_render", ""),
    ]
    assert files["plugin/skills/notes/SKILL.md"] == "# NOTES SKILL\n\nKEEP SHORT NOTES.\n"
    assert json.loads(files["plugin/settings.json"])["defaultTools"] == []

    # The render refuses a key Reef's model binding writes whatever the adapter's renderer does.
    with pytest.raises(RenderError, match=re.escape("must not set defaultModel in pi-agent/settings.json")):
        render(config({"defaultModel": "other/m"}, "agent"), descriptor)


def test_with_skill_frontmatter_adds_the_header_only_to_bare_text() -> None:
    path = "harness/skills/notes/SKILL.md"
    bare = "# Notes skill\n\nKeep short notes.\n"
    assert with_skill_frontmatter(path, bare) == "---\nname: notes\ndescription: Notes skill\n---\n" + bare
    assert with_skill_frontmatter(path, "") == "---\nname: notes\ndescription: notes\n---\n"
    headed = "---\nname: notes\ndescription: Short notes\n---\nKeep short notes.\n"
    assert with_skill_frontmatter(path, headed) == headed


def test_default_renderer_leaves_skill_and_command_files_unchanged() -> None:
    bare = "Summarize the thread.\n"
    assert AdapterRenderer.process_skill("harness/skills/summarize/SKILL.md", bare) == bare
    assert AdapterRenderer.process_command("harness-commands/summarize/SKILL.md", bare) == bare


def test_an_adapter_renderer_step_must_be_a_static_method() -> None:
    with pytest.raises(TypeError, match=r"StatefulRenderer\.process_skill must be a static method"):

        class StatefulRenderer(AdapterRenderer):
            def process_skill(self, path, text):
                return text


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("reef.harness.adapters.pi.render", "must be 'package.module:ClassName'"),
        ("reef.harness.adapters.pi.render:Missing", "cannot import"),
        ("reef.harness.adapters.descriptor:ExecutionValidator", "must name a class that inherits"),
    ],
)
def test_a_renderer_must_name_an_adapter_renderer_class(tmp_path, reference: str, message: str) -> None:
    target = write_descriptor(tmp_path, {**descriptor_data("native"), "renderer": reference})
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
    target = write_descriptor(tmp_path, {**descriptor_data("pi"), field: value})
    with pytest.raises(DescriptorError, match=message):
        load_descriptor(target)


def test_no_bundled_adapter_keeps_host_environment_or_has_a_bind_mounted_root() -> None:
    """Every bundled adapter's run stays hermetic: no service variable beyond PATH and TMPDIR."""
    for name in BUILTIN_ADAPTERS:
        descriptor = get_adapter(name)
        assert descriptor.host_env == {} and not descriptor.is_root_bind_mounted, name
