"""What Reefine's prompts say about each harness they write files for, beside pi.

pi's surface is its extension API, which a prompt carries from the reference
beside the method (``pi_extension_api.md``). Every other adapter takes
rules, skills, commands and, where its config can enforce a behavior, a few
config keys. Its ``<adapter>.yaml`` in this directory says how a person types a command there, what the
command file holds, which tools the harness has of its own (web search among
them) and how a mode is built, so a request is answered with that harness's
own means and its review judges the answer by them. An adapter without the
file (pi, native, an external one) has no facts.

A harness that keeps no mode state of its own writes ``conversation_mode``
(its ``name``, how a typed ``skill`` reaches the message, the ``escape`` a
person has around a mode, and ``toggle``, what turns the mode off, a command
by default) in place of ``mode``; ``conversation_mode.md`` holds the words.
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

    ``command`` says how a person invokes a command file and what its text
    holds; ``tools`` names the harness's own tools; ``mode`` says how a mode
    is turned on, shown and turned off there. ``config_keys`` are the top
    level keys a request may change in the primary config file, and
    ``config_example`` shows one. ``machine``, when set, says where the
    change runs, for a harness that runs somewhere other than the person's
    machine; the prompt then gives it in place of the client's report."""

    title: str
    command: str
    tools: str
    mode: str
    config_keys: tuple[str, ...] = ()
    config_example: str = ""
    machine: str = ""


#: How a mode is built on a harness that keeps no mode state: guidance the model follows, filled per harness with
#: ``title``, ``skill`` and ``escape``.
CONVERSATION_MODE = Path(__file__).with_name("conversation_mode.md").read_text(encoding="utf-8").removesuffix("\n")


def conversation_mode(name: str, skill: str, escape: str, toggle: str = "command") -> str:
    """The mode words for a harness that keeps no mode state; ``toggle`` names what the person types to turn it off."""
    mode = CONVERSATION_MODE.format(title=name, skill=skill, escape=escape)
    return mode if toggle == "command" else mode.replace("the same command", f"the same {toggle}")


def load_harness_facts(path: Path) -> HarnessFacts:
    """Read one adapter's facts file; a missing field or a wrong type raises :class:`DescriptorError`."""
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
    """The facts for a bundled ``adapter`` from ``<adapter>.yaml`` beside this module; ``None`` where it has none."""
    path = Path(__file__).with_name(f"{adapter}.yaml")
    if adapter not in BUILTIN_ADAPTERS or not path.is_file():
        return None
    return load_harness_facts(path)


__all__ = ["CONVERSATION_MODE", "HarnessFacts", "conversation_mode", "harness_facts", "load_harness_facts"]
