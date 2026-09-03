"""Guarantees of reef.harness.render and the bundled adapter descriptors."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.model_binding import ModelBinding
from reef.harness.render import RenderError, render_composition

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
    assert render_composition(NODES, get_adapter("pi")) == golden_tree("pi")


def test_opencode_render_matches_the_golden_tree() -> None:
    nodes = [node for node in NODES if node[1].get("target") != "models"]
    assert render_composition(nodes, get_adapter("opencode")) == golden_tree("opencode")


def test_claude_render_matches_the_golden_tree() -> None:
    nodes = [node for node in NODES if node[1].get("target") != "models"]
    assert render_composition(nodes, get_adapter("claude")) == golden_tree("claude")


DSH_PATCH = "dsh/profiles/headless/cordis.patch.yml"


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
    assert config["agent"] == {"max_turns": 40}
    # The defaults that keep an episode hermetic and single request, and the second skill root.
    assert config["approval"] == {"tirith_enabled": False}
    assert config["auxiliary"] == {"title_generation": {"enabled": False}}
    assert config["memory"] == {"nudge_interval": 0} and config["sessions"] == {"write_json_snapshots": True}
    assert config["skills"] == {"external_dirs": ["${HERMES_HOME}/../hermes-commands"]}
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
    assert files["hermes/SOUL.md"] == "Answer briefly.\n\nPrefer the standard library.\n"


def test_hermes_quirks_refuse_a_config_that_breaks_the_episode() -> None:
    descriptor = get_adapter("hermes")
    with pytest.raises(RenderError, match="tirith_enabled false"):
        render_composition([("config", {"data": {"approval": {"tirith_enabled": True}}})], descriptor)
    with pytest.raises(RenderError, match=r"title_generation\.enabled false"):
        render_composition([("config", {"data": {"auxiliary": {"title_generation": {"enabled": True}}}})], descriptor)
    with pytest.raises(RenderError, match="write_json_snapshots true"):
        render_composition([("config", {"data": {"sessions": {"write_json_snapshots": False}}})], descriptor)


def test_config_nodes_deep_merge_in_tree_order() -> None:
    files = render_composition(
        [
            ("config", {"data": {"compaction": {"enabled": True, "keep": 4}, "defaultProvider": "a"}}),
            ("config", {"data": {"compaction": {"keep": 8}}}),  # later node wins per key
        ],
        get_adapter("pi"),
    )
    assert '"defaultProvider": "a"' in files["pi-agent/settings.json"]  # sibling keys survive the merge
    assert '"enabled": true' in files["pi-agent/settings.json"]
    assert '"keep": 8' in files["pi-agent/settings.json"]


def test_unknown_config_target_is_rejected() -> None:
    with pytest.raises(RenderError, match="no config target 'models'"):
        render_composition([("config", {"target": "models", "data": {}})], get_adapter("opencode"))


def test_two_nodes_cannot_render_to_the_same_path() -> None:
    skill = ("skill", {"name": "notes", "text": "# notes"})
    with pytest.raises(RenderError, match="same path"):
        render_composition([skill, skill], get_adapter("pi"))


def test_opencode_quirk_rejects_reopened_autoupdate() -> None:
    with pytest.raises(RenderError, match="autoupdate false"):
        render_composition([("config", {"data": {"autoupdate": True}})], get_adapter("opencode"))


def test_claude_quirk_rejects_reopened_hermetic_switches() -> None:
    with pytest.raises(RenderError, match="includeCoAuthoredBy"):
        render_composition([("config", {"data": {"includeCoAuthoredBy": True}})], get_adapter("claude"))
    with pytest.raises(RenderError, match="DISABLE_AUTOUPDATER"):
        render_composition([("config", {"data": {"env": {"DISABLE_AUTOUPDATER": "0"}}})], get_adapter("claude"))


def test_bundled_adapters_are_discoverable() -> None:
    assert set(available_adapters()) >= {"claude", "opencode", "pi"}
