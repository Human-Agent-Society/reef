"""The shipped harness requests entries: the ``/reefine`` command, per adapter.

Like the update notice, the requests entries are composition, not runtime
code: ``evolution.requests: true`` appends them to the seed. On pi they are a
``code_extension`` carrying ``requests.ts`` (the ``/reefine`` command, which
files the person's request through native manual training, and its watch)
and a ``skill`` carrying the pi extension API reference the service proposer
reads before it writes an extension. Every other adapter with a command
surface gets one ``agent_command`` named ``reefine`` instead: its text tells
the session's model to file the request with ``reef-<adapter> evolve``, wait
for the step with ``reef-<adapter> wait`` and offer ``reef-<adapter>
update``, through the wrapper the session was started by
(``REEF_HARNESS_WRAPPER``). The ids are reef's own (``RESERVED_ENTRY_IDS``),
so a proposal cannot rewrite or remove them; the seed and a recovered state
carry them as they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.harness.adapters.descriptor import DescriptorError

REQUESTS_ENTRY_ID = "reef-requests"
REQUESTS_SKILL_ID = "reef-pi-extension-api"
#: The command a person types in the harness session.
REQUESTS_COMMAND = "reefine"

_ADAPTERS = Path(__file__).parents[1] / "adapters"
#: Per adapter: the extension file and the skill body it ships.
_ASSETS = {
    "pi": (_ADAPTERS / "pi" / "requests.ts", _ADAPTERS / "pi" / "pi_extension_api.md"),
}
_COMMAND_TEXT = Path(__file__).with_name("reefine_command.md")


@dataclass(frozen=True)
class _Command:
    """How an adapter's command file receives what the person typed after ``/reefine``.

    ``request`` names it in the command text: the binary's own placeholder
    where it substitutes one, else where the model finds it. ``frontmatter``
    is written before the text when the binary reads one; empty, the
    adapter's quirks module synthesizes it."""

    request: str
    frontmatter: str = ""


#: The adapters whose interactive session reaches a rendered agent_command, and how each passes the request on.
_COMMANDS = {
    "claude": _Command(
        request='"$ARGUMENTS"',
        frontmatter="---\ndescription: Ask Reef to change this harness\nargument-hint: <what it should do>\n---\n",
    ),
    "codex": _Command(request='"$ARGUMENTS"', frontmatter="---\ndescription: Ask Reef to change this harness\n---\n"),
    "opencode": _Command(
        request='"$ARGUMENTS"', frontmatter="---\ndescription: Ask Reef to change this harness\n---\n"
    ),
    "hermes": _Command(request="the text after /reefine in the person's message"),
    "dsh": _Command(request="the text after /reefine in the person's message"),
}


def _read(asset: Path, adapter: str) -> str:
    try:
        return asset.read_text(encoding="utf-8")
    except OSError as exc:
        raise DescriptorError(f"adapter {adapter!r} requests asset {asset.name} cannot be read: {exc}") from exc


def ships_requests(adapter: str) -> bool:
    """Whether the adapter ships a ``/reefine`` command: pi's extension, or the command file."""
    return adapter in _ASSETS or adapter in _COMMANDS


def command_text(adapter: str) -> str:
    """The ``/reefine`` command file's text for an adapter without its own extension."""
    command = _COMMANDS.get(adapter)
    if command is None:
        raise DescriptorError(f"adapter {adapter!r} ships no requests command")
    body = _read(_COMMAND_TEXT, adapter).format(request=command.request, adapter=adapter)
    return command.frontmatter + body


def request_entries(adapter: str) -> tuple[dict[str, Any], ...]:
    """The seed entry options for the adapter's shipped ``/reefine``, in seed order: pi's extension and its skill,
    or the command file."""
    assets = _ASSETS.get(adapter)
    if assets is not None:
        extension, skill = assets
        return (
            {
                "id": REQUESTS_ENTRY_ID,
                "name": "code_extension",
                "config": {"name": REQUESTS_ENTRY_ID, "code": _read(extension, adapter)},
            },
            {
                "id": REQUESTS_SKILL_ID,
                "name": "skill",
                "config": {"name": REQUESTS_SKILL_ID, "text": _read(skill, adapter)},
            },
        )
    if adapter not in _COMMANDS:
        raise DescriptorError(f"adapter {adapter!r} ships no requests extension")
    return (
        {
            "id": REQUESTS_ENTRY_ID,
            "name": "agent_command",
            "config": {"name": REQUESTS_COMMAND, "text": command_text(adapter)},
        },
    )


__all__ = [
    "REQUESTS_COMMAND",
    "REQUESTS_ENTRY_ID",
    "REQUESTS_SKILL_ID",
    "command_text",
    "request_entries",
    "ships_requests",
]
