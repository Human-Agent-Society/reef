"""The adapter mapping in both directions: a tree rendered for an adapter reads back as the same tree."""

from __future__ import annotations

import json

import pytest

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.sync import TOKEN_PLACEHOLDER, TreeMerge, import_files, layer_install, merge_lines, merge_trees
from reef.harness.tree import HarnessTree, render_tree

SKILL = "---\nname: fix\ndescription: Fix the failing test first.\n---\n\nRun the tests, then fix.\n"
COMMAND = "---\ndescription: Ship the change.\n---\n\nRun the tests and commit.\n"
SYNCED = [name for name in available_adapters() if get_adapter(name).layer is not None]


def test_the_adapters_that_sync_are_the_ones_whose_roots_are_known() -> None:
    assert SYNCED == ["claude", "codex", "opencode", "pi"]


@pytest.mark.parametrize("adapter", SYNCED)
def test_a_rendered_tree_reads_back_as_the_tree(adapter: str) -> None:
    descriptor = get_adapter(adapter)
    files = {"skills/fix/SKILL.md": SKILL, "config/primary.json": "{}\n"}
    if "rules" in descriptor.layout.parts:
        files["rules.md"] = "Answer briefly.\n"
    command = descriptor.layout.parts["command"]
    # Codex renders a command where it renders a skill, so its command reads back as the skill below.
    if command.renders != descriptor.layout.parts["skill"].renders:
        files[command.tree_path("ship")] = COMMAND
    tree = HarnessTree(files)
    # The config files, the plugin's manifests and Reef's own code read back as nothing.
    assert import_files(render_tree(tree, descriptor), tree, descriptor).tree == tree


def test_an_import_replaces_the_heads_files_and_ignores_what_no_part_renders_to() -> None:
    pi = get_adapter("pi")
    config = json.dumps({"defaultTools": []}) + "\n"
    head = HarnessTree({"config/primary.json": config, "rules.md": "Old.\n", "skills/fix/SKILL.md": SKILL})
    files = {
        "plugin/rules.md": "Mine.\n",
        "plugin/prompts/ship.md": COMMAND,
        "pi-agent/auth.json": '{"anthropic": "a credential"}\n',
        "pi-agent/sessions/one.jsonl": "{}\n",
    }
    imported = import_files(files, head, pi)
    # The person removed the skill, so the tree has none; the head's config stays.
    assert imported.tree == HarnessTree(
        {"config/primary.json": config, "rules.md": "Mine.\n", "commands/ship.md": COMMAND}
    )
    assert imported.ignored == ("pi-agent/auth.json", "pi-agent/sessions/one.jsonl")


def test_codex_reads_a_file_where_a_skill_and_a_command_both_render_as_the_skill() -> None:
    imported = import_files({"plugin/skills/ship/SKILL.md": COMMAND}, HarnessTree({}), get_adapter("codex"))
    assert imported.tree == HarnessTree({"skills/ship/SKILL.md": COMMAND})


def test_the_merge_keeps_each_sides_own_changes_and_merges_text_parts_by_line() -> None:
    layout = get_adapter("claude").layout
    base = HarnessTree(
        {"rules.md": "Test first.\nBe brief.\n", "extensions/a.js": "a\n", "skills/fix/SKILL.md": SKILL}
    )
    head = HarnessTree(
        {"rules.md": "Test first.\nBe very brief.\n", "extensions/a.js": "b\n", "skills/fix/SKILL.md": SKILL}
    )
    # The client added a memory note, removed the skill, and changed a line of the rules the head left alone.
    client = HarnessTree(
        {"rules.md": "Test first, always.\nBe brief.\n", "extensions/a.js": "a\n", "memory/MEMORY.md": "- note\n"}
    )
    merged = HarnessTree(
        {"rules.md": "Test first, always.\nBe very brief.\n", "extensions/a.js": "b\n", "memory/MEMORY.md": "- note\n"}
    )
    assert merge_trees(base, head, client, layout) == TreeMerge(merged, ())
    # A code file merges whole, so two changes to it are a conflict; so are two changes to one line.
    clash = HarnessTree({**client.files, "extensions/a.js": "c\n", "rules.md": "Test first.\nBe terse.\n"})
    assert merge_trees(base, head, clash, layout) == TreeMerge(None, ("extensions/a.js", "rules.md"))
    assert merge_trees(base, head, head, layout).tree == head


def test_two_notes_added_at_one_place_keep_both_the_heads_first() -> None:
    assert merge_lines("- a\n", "- a\n- from Reef\n", "- a\n- from you\n") == "- a\n- from Reef\n- from you\n"
    assert merge_lines("x\ny\n", "X\ny\n", "X\ny\n") == "X\ny\n"
    assert merge_lines("x\ny\n", "X\ny\n", "x2\ny\n") is None


BINDINGS = {"claude": "anthropic", "codex": "responses", "opencode": "openai", "pi": "openai"}


@pytest.mark.parametrize("adapter", sorted(BINDINGS))
def test_a_layer_installs_the_release_as_the_agents_plugin_with_the_token_off_the_command_line(adapter) -> None:
    descriptor = get_adapter(adapter)
    served = ModelBinding(base_url="http://upstream.invalid", model="m1", api=BINDINGS[adapter])
    tree = HarnessTree({"skills/fix/SKILL.md": SKILL})
    layer = layer_install(tree, descriptor, served, "http://reef.invalid/reef/scenarios/s", "/home/a/install")
    assert all(TOKEN_PLACEHOLDER not in arg for command in (layer.args, *layer.register) for arg in command)
    assert TOKEN_PLACEHOLDER in "".join(layer.files.values()) + "".join(layer.env.values())
    assert any("{name}" in template for template in layer.synced)
    # The agent's own commands install the plugin, and the session that Reef starts binds the model; each was
    # checked with the agent's binary.
    expected = {
        "claude": (
            ["--settings", "/home/a/install/claude/settings.json"],
            [("plugin", "marketplace", "add", "/home/a/install"), ("plugin", "install", "reef@reef")],
        ),
        "codex": (
            ["-c", 'model_provider="reef"', "-c", 'model="m1"'],
            [("plugin", "marketplace", "add", "/home/a/install"), ("plugin", "add", "reef@reef")],
        ),
        "opencode": ([], [("plugin", "-g", "/home/a/install/plugin")]),
        "pi": (["--model", "reef/m1"], [("install", "/home/a/install/plugin")]),
    }[adapter]
    assert list(layer.args[: len(expected[0])]) == expected[0]
    assert list(layer.register) == expected[1]
    assert "plugin/skills/fix/SKILL.md" in layer.files or "plugin/skill/fix/SKILL.md" in layer.files
    if adapter == "claude":
        marketplace = json.loads(layer.files[".claude-plugin/marketplace.json"])
        assert marketplace["plugins"] == [{"name": "reef", "source": "./plugin"}]
    if adapter == "codex":
        # Codex copies the whole plugin into its cache, so the install keeps no config file with the credential.
        assert not any(path.startswith("codex/") for path in layer.files)
        assert (
            json.loads(layer.files[".agents/plugins/marketplace.json"])["plugins"][0]["source"]["path"] == "./plugin"
        )
    if adapter == "opencode":
        assert layer.env["OPENCODE_CONFIG"] == "/home/a/install/opencode/opencode.json"
    if adapter == "pi":
        assert "plugin/extensions/reef-provider/index.ts" in layer.files
    # The episode defaults stay in episodes: a layer never opens every permission of the person's agent.
    assert "permission" not in "".join(layer.files.values())
