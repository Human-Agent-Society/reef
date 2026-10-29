"""Claude Code adapter render steps: the plugin's own files, the model a frontmatter names, and the layer's files.

A release is the plugin ``reef`` in ``plugin/``. ``finalize_render`` writes
its manifest, and its output style from ``plugin/rules.md``, which Claude Code
adds to its system prompt while the plugin is on (``force-for-plugin``). It
drops ``plugin/hooks/hooks.json`` when the tree has no hooks. ``layer_files``
adds the local marketplace at the install root that a person's machine
installs the plugin from.

Every model call stays on Reef's model binding, and a ``model`` in the
frontmatter of a command or a skill is the tree choosing where calls go.
``check_model_route`` refuses it, reading the frontmatter as Claude Code
reads it, which no config check can.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import yaml

from reef.harness.adapters.descriptor import AdapterRenderer
from reef.harness.tree.render import RenderError

RULES = "plugin/rules.md"
HOOKS = "plugin/hooks/hooks.json"
#: The output style that holds the rules: Claude Code adds it to its own system prompt, which it keeps.
OUTPUT_STYLE = "plugin/output-styles/reef.md"
OUTPUT_STYLE_FRONTMATTER = (
    "---\nname: Reef\ndescription: The rules of the Reef harness release\n"
    "force-for-plugin: true\nkeep-coding-instructions: true\n---\n\n"
)

#: Claude Code's frontmatter block: after a byte order mark, an opening --- line, then the text up to the next ---.
FRONTMATTER_BLOCK = re.compile(r"---\s*\n([\s\S]*?)---\s*\n?")
#: A plain ``key: value`` line and the value characters Claude Code quotes when its YAML reader refuses the block.
PLAIN_LINE = re.compile(r"^([a-zA-Z_-]+):\s+(.+)$")
YAML_SYNTAX = re.compile(r"[{}[\]*&#!|>%@`]|: ")


def quoted_values(block: str) -> str:
    """``block`` as Claude Code reads it again when its YAML reader refuses it: plain values holding YAML quoted."""
    lines = []
    for line in block.split("\n"):
        match = PLAIN_LINE.match(line)
        if match is None:
            lines.append(line)
            continue
        value = match.group(2)
        quoted = value[0] in "\"'" and value[-1] == value[0]
        if not quoted and YAML_SYNTAX.search(value):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{match.group(1)}: "{escaped}"')
        else:
            lines.append(line)
    return re.sub(r"(?m)^\t+", lambda tabs: "  " * len(tabs.group(0)), "\n".join(lines))


def top_level_keys(block: str) -> set[str] | None:
    """The top-level keys of a YAML block, merged keys included; None when PyYAML cannot read it."""
    loader = yaml.SafeLoader(block)
    try:
        node = loader.get_single_node()
        if not isinstance(node, yaml.MappingNode):
            return set()
        loader.flatten_mapping(node)
    except (yaml.YAMLError, RecursionError):
        return None
    finally:
        loader.dispose()
    return {key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)}


def frontmatter_chooses_model(text: str) -> bool:
    """Whether Claude Code can read a ``model`` key in the frontmatter of a command or a skill.

    A block PyYAML cannot read as Claude Code does counts when it holds the word ``model`` or an escape that could
    spell it, since Bun's YAML reader may still read it.
    """
    match = FRONTMATTER_BLOCK.match(text.removeprefix("\ufeff"))
    if match is None:
        return False
    block = match.group(1)
    for candidate in (block, quoted_values(block)):
        keys = top_level_keys(candidate)
        if keys is not None:
            return "model" in keys
    return "model" in block or "\\" in block


class ClaudeAdapterRenderer(AdapterRenderer):
    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        plugin = {**files, "plugin/.claude-plugin/plugin.json": '{"name": "reef"}\n'}
        if not json.loads(plugin[HOOKS]):
            plugin.pop(HOOKS)
        if RULES in plugin:
            plugin[OUTPUT_STYLE] = OUTPUT_STYLE_FRONTMATTER + plugin[RULES]
        return plugin

    @staticmethod
    def layer_files(files: dict[str, str]) -> dict[str, str]:
        marketplace = {"name": "reef", "owner": {"name": "Reef"}, "plugins": [{"name": "reef", "source": "./plugin"}]}
        return {**files, ".claude-plugin/marketplace.json": json.dumps(marketplace, indent=2) + "\n"}

    @staticmethod
    def check_model_route(
        configs: Mapping[str, Mapping[str, Any]], skills: Mapping[str, str], commands: Mapping[str, str]
    ) -> None:
        for path, text in {**skills, **commands}.items():
            if frontmatter_chooses_model(text):
                raise RenderError(
                    f"claude {path} must not set model in its frontmatter: Reef's model binding chooses the model"
                )
