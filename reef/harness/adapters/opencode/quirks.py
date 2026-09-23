"""opencode adapter quirks: boot mutations and enforced config invariants.

opencode's boot mutates its own composition directories: it injects a
``$schema`` key into config files that lack one (an in-place edit of a
rendered file, invisible to the residue scan), writes a ``.gitignore``, and
npm-installs ``@opencode-ai/plugin`` with a ``node_modules`` tree and
lockfiles. The whitelist below names exactly those artifacts so the episode
inverse tolerates them and nothing else.

``finalize_render`` enforces the traps a mutated config node could reopen: a
benchmark episode must never autoupdate the binary mid-campaign or upload a
share link, so a composition that overrides either is rejected at render -
the same gate that rejects an invalid node.

It also keeps the model Reef's. The descriptor sets ``enabled_providers`` to
``["reef"]``, so opencode offers no provider but the binding's, its own zen
provider included, and a composition must keep exactly that list. The model
binding Reef appends at render is the only writer of ``provider`` and
``model``: it renders after the tree, wins every key it writes, and always
writes a non-empty ``apiKey``, which a tree cannot hold because admission
refuses an inline credential. So the two keys pass only in the binding's
shape with that key, which a tree rendered alone, as admission renders it,
never has. A provider, a model choice (``small_model``, an agent's or a
command's ``model``) or ``disabled_providers`` anywhere else is a composition
pointing opencode at another endpoint.

opencode reads the frontmatter of a command or a skill with gray-matter,
which strips a byte order mark, takes the text after the opening ``---`` as
the name of another engine (JSON, JavaScript), reads to the end of the file
when no line closes the block, and fails on YAML js-yaml cannot read, which
opencode then reads again with its values that hold a colon rewritten. The
check reads only the plain form, a ``---`` line, a YAML mapping with no tags
and a closing ``---`` line, with js-yaml's types for plain values (``1e5`` is
a number and ``yes`` a string), where both readers agree, and refuses every
other form, so no command reaches opencode with an agent, a model or a name
the check did not see. A command file's frontmatter ``name`` must be the
file's own, since opencode files the command under that name, in place of
the command of that name. A command's ``agent`` and ``default_agent`` must
name an agent the run has, an agent must keep its own name, and a tree with
no ``default_agent`` must keep an agent that is neither a subagent nor
hidden, since opencode otherwise fails the command, or every run, with an
opaque server error; a command field of the wrong type makes opencode refuse
its whole configuration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Hashable, Mapping
from typing import ClassVar

import yaml

from reef.harness.tree.render import RenderError

_CONFIG_PATH = "opencode/opencode.json"
COMMAND_DIR = "opencode/command/"
SKILL_DIR = "opencode/skill/"

#: What the model binding writes: one ``reef`` provider holding these keys, its options, and empty model entries.
BINDING_PROVIDER = "reef"
BINDING_PROVIDER_KEYS = frozenset({"models", "npm", "options"})
BINDING_OPTION_KEYS = frozenset({"apiKey", "baseURL"})
#: Top-level keys the binding never writes that choose which model or which providers a run uses.
MODEL_CHOICE_KEYS = ("small_model", "disabled_providers")
#: The agents opencode builds in, with their modes. Its title, summary and compaction agents are hidden: they hold
#: the prompts of opencode's own calls, and a command may name one, but none starts a run.
BUILTIN_AGENT_MODES = {
    "build": "primary",
    "plan": "primary",
    "general": "subagent",
    "explore": "subagent",
    "compaction": "primary",
    "summary": "primary",
    "title": "primary",
}
BUILTIN_HIDDEN_AGENTS = frozenset({"compaction", "summary", "title"})
#: The command fields opencode reads beside its model and the type each must have; any other type fails its whole
#: configuration.
COMMAND_FIELD_TYPES: dict[str, type] = {"agent": str, "description": str, "subtask": bool, "variant": str}
#: How js-yaml 3, the YAML reader gray-matter gives opencode, types a plain value: each tag with its pattern and the
#: characters the value can start with, tried in this order. PyYAML follows YAML 1.1 instead, where yes and no are
#: booleans, 1e5 and 1.5e3 are strings (its floats need a dot and a signed exponent), and 01.5 and 1_ are numbers.
JS_YAML_RESOLVERS = (
    ("tag:yaml.org,2002:null", r"^(?:~|null|Null|NULL|)$", ("~", "n", "N", "")),
    ("tag:yaml.org,2002:bool", r"^(?:true|True|TRUE|false|False|FALSE)$", tuple("tTfF")),
    (
        "tag:yaml.org,2002:int",
        r"^[-+]?(?:0|0b[01_]*[01]|0x[0-9a-fA-F_]*[0-9a-fA-F]|0[0-7_]*[0-7]"
        r"|[1-9](?:[0-9_]*[0-9])?|[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
        tuple("-+0123456789"),
    ),
    (
        "tag:yaml.org,2002:float",
        r"^(?:[-+]?(?:0|[1-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))(?<!_)$",
        tuple("-+0123456789."),
    ),
    (
        "tag:yaml.org,2002:timestamp",
        r"^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$",
        tuple("0123456789"),
    ),
    ("tag:yaml.org,2002:merge", r"^(?:<<)$", ("<",)),
)
FRONTMATTER_FORM = "write the frontmatter as a --- line, a YAML mapping with no tags, and a closing --- line"

cleanup_whitelist = (
    "opencode/.gitignore",
    "opencode/package.json",
    "opencode/package-lock.json",
    "opencode/bun.lock",
    "opencode/node_modules/**",
)


class FrontmatterLoader(yaml.SafeLoader):
    """``yaml.safe_load`` as js-yaml reads frontmatter: its types for plain values, and no value it reads otherwise.

    PyYAML reads a repeated key as its last value, where js-yaml fails and opencode reads the file another way. A
    tagged value and a date that does not exist (2001-13-45) are refused too: js-yaml checks a tag against its value
    where PyYAML's constructors can fail on it, and moves such a date forward to a real one where PyYAML fails.
    """

    yaml_implicit_resolvers: ClassVar[dict[str, list[tuple[str, re.Pattern[str]]]]] = {}

    def compose_node(self, parent: yaml.Node | None, index: object) -> yaml.Node | None:
        event = self.peek_event()
        if isinstance(event, (yaml.ScalarEvent, yaml.CollectionStartEvent)) and event.tag is not None:
            raise yaml.MarkedYAMLError(problem=f"a value has the tag {event.tag!r}", problem_mark=event.start_mark)
        return super().compose_node(parent, index)

    def construct_yaml_timestamp(self, node: yaml.ScalarNode) -> object:
        try:
            return super().construct_yaml_timestamp(node)
        except ValueError as error:
            raise yaml.MarkedYAMLError(
                problem=f"the date {node.value!r} does not exist", problem_mark=node.start_mark
            ) from error

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, object]:
        seen: set[str] = set()
        for key, _ in node.value:
            if isinstance(key, yaml.ScalarNode):
                if key.value in seen:
                    raise yaml.MarkedYAMLError(
                        problem=f"the key {key.value!r} appears twice", problem_mark=key.start_mark
                    )
                seen.add(key.value)
        return super().construct_mapping(node, deep=deep)


for tag, pattern, first in JS_YAML_RESOLVERS:
    FrontmatterLoader.add_implicit_resolver(tag, re.compile(pattern), first)
FrontmatterLoader.add_constructor("tag:yaml.org,2002:timestamp", FrontmatterLoader.construct_yaml_timestamp)


def check_binding_shape(config: Mapping[str, object]) -> None:
    """``provider`` and ``model`` appear together in the binding's shape, with its key, or not at all."""
    provider = config.get("provider")
    model = config.get("model")
    if provider is None and model is None:
        return
    if provider is None or model is None:
        key = "model" if provider is None else "provider"
        raise RenderError(f"opencode composition must not set {key}: Reef's model binding writes provider and model")
    refusal = (
        f"opencode composition must not set provider: Reef's model binding writes the only one, {BINDING_PROVIDER!r}"
    )
    if not isinstance(provider, dict) or set(provider) != {BINDING_PROVIDER}:
        raise RenderError(refusal)
    reef = provider[BINDING_PROVIDER]
    if not isinstance(reef, dict) or set(reef) != BINDING_PROVIDER_KEYS:
        raise RenderError(refusal)
    options, models = reef["options"], reef["models"]
    if not isinstance(options, dict) or set(options) != BINDING_OPTION_KEYS or not isinstance(models, dict):
        raise RenderError(refusal)
    # The binding's key is never empty and a tree cannot hold one, so a tree that copies this shape fails here.
    api_key = options["apiKey"]
    if not isinstance(api_key, str) or not api_key.strip():
        raise RenderError(refusal)
    if any(entry != {} for entry in models.values()):
        raise RenderError(refusal)
    if not isinstance(model, str) or model.removeprefix(f"{BINDING_PROVIDER}/") not in models:
        raise RenderError("opencode composition must not set model: Reef's model binding chooses it")


def check_agent_name(where: str, agent: object, agents: Collection[str]) -> None:
    """``agent`` names an agent the run has."""
    if not isinstance(agent, str):
        raise RenderError(f"opencode {where} must name an agent as a string, got {agent!r}")
    if agent not in agents:
        known = ", ".join(sorted(agents))
        raise RenderError(f"opencode {where} names agent {agent!r}, which the tree does not define (agents: {known})")


def check_command(where: str, command: Mapping[object, object], agents: Collection[str]) -> None:
    """A command chooses no model, its fields have the types opencode reads, and its agent is one the run has."""
    if "model" in command:
        raise RenderError(f"opencode {where} must not choose a model: Reef's model binding chooses it")
    for field, expected in COMMAND_FIELD_TYPES.items():
        if field in command and not isinstance(command[field], expected):
            raise RenderError(
                f"opencode {where} field {field!r} must be a {expected.__name__}, got {command[field]!r}"
            )
    if "agent" in command:
        check_agent_name(where, command["agent"], agents)


def read_frontmatter(where: str, text: str) -> Mapping[object, object]:
    """The keys of a markdown file's frontmatter as opencode reads them, empty when it has none.

    gray-matter finds frontmatter only when the text starts with ``---`` and a fourth character other than ``-``,
    and closes it at the first later line starting with ``---``. Every other form where it and this reader could
    see different keys is refused rather than guessed at.
    """
    if text.startswith("\ufeff"):
        raise RenderError(f"opencode {where} starts with a byte order mark; {FRONTMATTER_FORM}")
    if not text.startswith("---") or text.startswith("----"):
        return {}
    block = text[3:]
    first_line_end = block.find("\n")
    closing = block.find("\n---")
    if first_line_end == -1 or closing == -1:
        raise RenderError(f"opencode {where} frontmatter has no closing --- line; {FRONTMATTER_FORM}")
    engine = block[:first_line_end].strip()
    if engine:
        raise RenderError(f"opencode {where} frontmatter names the engine {engine!r} after ---; {FRONTMATTER_FORM}")
    try:
        data = yaml.load(block[first_line_end + 1 : closing], Loader=FrontmatterLoader)
    except yaml.YAMLError as error:
        # A marked error counts lines from the block, which starts on the file's second line.
        if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
            detail = f"{error.problem or error.context} at line {error.problem_mark.line + 2}"
        else:
            detail = " ".join(str(error).split())
        raise RenderError(
            f"opencode {where} frontmatter cannot be read: {detail}; {FRONTMATTER_FORM}, "
            "quoting a value that holds ': '"
        ) from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RenderError(f"opencode {where} frontmatter is a {type(data).__name__}; {FRONTMATTER_FORM}")
    return data


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG_PATH])
    if config.get("autoupdate") is not False:
        raise RenderError("opencode composition must keep autoupdate false for benchmark episodes")
    if config.get("share") != "disabled":
        raise RenderError("opencode composition must keep share disabled for benchmark episodes")
    if config.get("enabled_providers") != [BINDING_PROVIDER]:
        raise RenderError(
            f"opencode composition must keep enabled_providers [{BINDING_PROVIDER!r}]: "
            "a run uses only the provider Reef's model binding writes"
        )
    check_binding_shape(config)
    for key in MODEL_CHOICE_KEYS:
        if key in config:
            raise RenderError(f"opencode composition must not set {key}: Reef's model binding chooses the model")
    modes = dict(BUILTIN_AGENT_MODES)
    disabled: set[str] = set()
    hidden = set(BUILTIN_HIDDEN_AGENTS)
    # ``mode`` is opencode's deprecated name for ``agent``; opencode folds its entries in as primary agents.
    for section in ("agent", "mode"):
        entries = config.get(section)
        for name, agent in entries.items() if isinstance(entries, dict) else ():
            if not isinstance(agent, dict):
                continue
            if "model" in agent:
                raise RenderError(
                    f"opencode {section} {name!r} must not choose a model: Reef's model binding chooses it"
                )
            # opencode files an agent under its key but runs it by this name, so a run that uses it finds no agent.
            if "name" in agent and agent["name"] != name:
                raise RenderError(
                    f"opencode {section} {name!r} must not set name {agent['name']!r}: opencode then finds no agent "
                    "by that name and fails a run that uses it"
                )
            # An agent opencode does not build in takes either role unless its mode says which.
            modes[name] = "primary" if section == "mode" else str(agent.get("mode", modes.get(name, "all")))
            if agent.get("disable") is True:
                disabled.add(name)
            if agent.get("hidden") is True:
                hidden.add(name)
            elif agent.get("hidden") is False:
                hidden.discard(name)
    agents = {name: mode for name, mode in modes.items() if name not in disabled}
    if "default_agent" in config:
        default = config["default_agent"]
        check_agent_name("default_agent", default, agents)
        if agents[default] == "subagent" or default in hidden:
            raise RenderError(
                f"opencode default_agent names agent {default!r}, a subagent or a hidden agent, which cannot start a run"
            )
    elif all(mode == "subagent" or name in hidden for name, mode in agents.items()):
        # With no default_agent opencode starts a run with the first agent that is neither, and fails with none.
        raise RenderError(
            "opencode composition leaves no agent that can start a run: every agent it keeps is a subagent or hidden"
        )
    commands = config.get("command")
    for name, command in commands.items() if isinstance(commands, dict) else ():
        if isinstance(command, dict):
            check_command(f"command {name!r} in opencode.json", command, agents)
    for path, text in files.items():
        if path.startswith(COMMAND_DIR) and path.endswith(".md"):
            command_name = path[len(COMMAND_DIR) : -len(".md")]
            where = f"command {command_name!r}"
            frontmatter = read_frontmatter(where, text)
            # opencode files a command file under the name in its frontmatter, so another name would put this
            # file in the place of the command of that name, /reefine included.
            if "name" in frontmatter and frontmatter["name"] != command_name:
                raise RenderError(
                    f"opencode {where} frontmatter must not set name {frontmatter['name']!r}: opencode files the "
                    "command under that name, in place of any command already named so"
                )
            check_command(where, frontmatter, agents)
        elif path.startswith(SKILL_DIR) and path.endswith("/SKILL.md"):
            read_frontmatter(f"skill {path[len(SKILL_DIR) : -len('/SKILL.md')]!r}", text)
    return files
