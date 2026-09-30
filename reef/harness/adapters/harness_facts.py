"""What a text proposer is told about each harness it writes entries for, beside pi.

pi's surface is its extension API, which a proposer reads from the tree's own
reference skill. Every other adapter takes rules, skills, commands and, where
its config can enforce a behavior, a few config keys. Its directory's
``harness_facts.yaml`` says how a person types a command there, what the
command file holds, which tools the harness has of its own (web search among
them) and how a mode is built, so a request is answered with that harness's
own means and its review judges the answer by them. An adapter without the
file (pi, native, an external one) has no facts.

A harness that keeps no mode state of its own writes ``conversation_mode``
(its ``name``, how a typed ``skill`` reaches the message, the ``escape`` a
person has around a mode, and ``toggle``, what turns the mode off, a command
by default) in place of ``mode``; ``CONVERSATION_MODE`` holds the words.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from reef.harness.adapters import BUILTIN_ADAPTERS
from reef.harness.adapters.descriptor import DescriptorError


@dataclass(frozen=True)
class HarnessFacts:
    """One harness's surface as a request prompt and its review describe it.

    ``command`` says how a person invokes an agent_command and what its text
    holds; ``tools`` names the harness's own tools; ``mode`` says how a mode
    is turned on, shown and turned off there. ``config_keys`` are the top
    level keys a request's config entry may set in the primary config file,
    and ``config_example`` shows one. ``machine``, when set, says where the
    change runs, for a harness that runs somewhere other than the person's
    machine; the prompt then gives it in place of the client's report."""

    title: str
    command: str
    tools: str
    mode: str
    config_keys: tuple[str, ...] = ()
    config_example: str = ""
    machine: str = ""


#: How a mode is built on a harness that keeps no mode state: guidance the model follows, filled per harness.
CONVERSATION_MODE = (
    "{title} keeps no mode state of its own and a command cannot take a tool away, so a mode here is guidance the "
    "model follows while every tool stays in its list: the command that turns it on says so in its reply and "
    "names how to leave it, the rules say how the agent behaves while it is on and that every reply shows it is "
    "on, and the same command with the word off turns it off. The mode's state lives in the conversation, in the "
    "command's reply and the header each reply starts with, never in a file or a marker a tool writes or reads, and "
    "the rules make no tool call on any turn: a rule applies to every turn of every session, the mode's turns or "
    "not. While the mode is on, the model declines a skill that the person's message loads, other than the mode's "
    "own command ({skill}), and tries the tool the mode allows before it refuses a question that tool can answer. "
    "A reply that declines names the mode's off command as the way out and never suggests a route around the mode, "
    "such as a shell command the person runs themselves ({escape}). "
    "The design, the How to use paragraph and the command's reply say the model follows the mode and never "
    "claim the other tools are unavailable. A request for a hard restriction (no other tool or skill may run at "
    "all) is only partly met this way, and no answer here can do more: the review lists that point under limits."
)


def conversation_mode(name: str, skill: str, escape: str, toggle: str = "command") -> str:
    """The mode words for a harness that keeps no mode state; ``toggle`` names what the person types to turn it off."""
    mode = CONVERSATION_MODE.format(title=name, skill=skill, escape=escape)
    return mode if toggle == "command" else mode.replace("the same command", f"the same {toggle}")


def load_harness_facts(path: Path) -> HarnessFacts:
    """Read one ``harness_facts.yaml``; a missing field or a wrong type raises :class:`DescriptorError`."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DescriptorError(f"cannot read harness facts {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DescriptorError(f"harness facts {path} must be a mapping")
    conversation = data.get("conversation_mode")
    mode: object
    if isinstance(conversation, dict):
        mode = conversation_mode(
            str(conversation.get("name", "")),
            str(conversation.get("skill", "")),
            str(conversation.get("escape", "")),
            str(conversation.get("toggle", "command")),
        )
    else:
        mode = data.get("mode")
    config_keys = data.get("config_keys", [])
    fields = {"title": data.get("title"), "command": data.get("command"), "tools": data.get("tools"), "mode": mode}
    missing = sorted(key for key, value in fields.items() if not isinstance(value, str) or not value)
    if missing or not isinstance(config_keys, list):
        raise DescriptorError(f"harness facts {path} need text for {', '.join(missing) or 'config_keys as a list'}")
    return HarnessFacts(
        title=str(fields["title"]),
        command=str(fields["command"]),
        tools=str(fields["tools"]),
        mode=str(mode),
        config_keys=tuple(str(key) for key in config_keys),
        config_example=str(data.get("config_example", "")),
        machine=str(data.get("machine", "")),
    )


@cache
def harness_facts(adapter: str) -> HarnessFacts | None:
    """The facts for a bundled ``adapter`` from its directory; ``None`` where it has none."""
    path = Path(__file__).parent / adapter / "harness_facts.yaml"
    if adapter not in BUILTIN_ADAPTERS or not path.is_file():
        return None
    return load_harness_facts(path)


__all__ = ["CONVERSATION_MODE", "HarnessFacts", "conversation_mode", "harness_facts", "load_harness_facts"]
