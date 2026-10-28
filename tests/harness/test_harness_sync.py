"""The adapter mapping in both directions: a tree rendered for an adapter reads back as the same tree."""

from __future__ import annotations

import json

import pytest

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.sync import import_files
from reef.harness.tree import HarnessTree, render_tree

SKILL = "---\nname: fix\ndescription: Fix the failing test first.\n---\n\nRun the tests, then fix.\n"
COMMAND = "---\ndescription: Ship the change.\n---\n\nRun the tests and commit.\n"
SYNCED = [name for name in available_adapters() if get_adapter(name).roots]


def test_the_adapters_that_sync_are_the_ones_whose_roots_are_known() -> None:
    assert SYNCED == ["claude", "codex", "opencode", "pi"]


@pytest.mark.parametrize("adapter", SYNCED)
def test_a_rendered_tree_reads_back_as_the_tree(adapter: str) -> None:
    descriptor = get_adapter(adapter)
    files = {"rules.md": "Answer briefly.\n", "skills/fix/SKILL.md": SKILL, "config/primary.json": "{}\n"}
    command = descriptor.layout.parts["command"]
    # Codex renders a command where it renders a skill, so its command reads back as the skill below.
    if command.renders != descriptor.layout.parts["skill"].renders:
        files[command.tree_path("ship")] = COMMAND
    tree = HarnessTree(files)
    rendered = render_tree(tree, descriptor)
    imported = import_files(rendered, tree, descriptor)
    assert imported.tree == tree
    # Config files are the binding's: the import keeps the head's and ignores the rendered ones.
    assert imported.ignored and set(imported.ignored) <= {target.path for target in descriptor.config_targets.values()}


def test_an_import_replaces_the_heads_files_and_ignores_what_no_part_renders_to() -> None:
    pi = get_adapter("pi")
    config = json.dumps({"defaultThinkingLevel": "off"}) + "\n"
    head = HarnessTree({"config/primary.json": config, "rules.md": "Old.\n", "skills/fix/SKILL.md": SKILL})
    files = {
        "pi-agent/AGENTS.md": "Mine.\n",
        "pi-agent/prompts/ship.md": COMMAND,
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
    imported = import_files({"codex/skills/ship/SKILL.md": COMMAND}, HarnessTree({}), get_adapter("codex"))
    assert imported.tree == HarnessTree({"skills/ship/SKILL.md": COMMAND})
