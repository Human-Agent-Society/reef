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
(``REEF_HARNESS_WRAPPER``), in the form each harness's permission check
lets through and with the timeout its shell tool takes. The ids are reef's own (``RESERVED_ENTRY_IDS``),
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
    """How one adapter's ``/reefine`` command file reaches the request and the wrapper.

    ``command`` is what the person types; ``request`` names the typed request
    in the command text: the binary's own placeholder where it substitutes
    one, else where the model finds it. ``wrapper`` is how the command runs
    the wrapper, the form the harness's own permission check lets through;
    ``shell_timeout`` says how its shell tool counts a timeout.
    ``frontmatter`` is written before the text when the binary reads one;
    empty, the adapter's quirks module synthesizes it."""

    request: str
    shell_timeout: str
    command: str = "/reefine"
    wrapper: str = '"$REEF_HARNESS_WRAPPER"'
    wrapper_note: str = ""
    frontmatter: str = ""


_TYPED = "the text after /reefine in the person's message"

#: The adapters whose interactive session reaches a rendered agent_command, and how each passes the request on.
_COMMANDS = {
    # Claude Code refuses an allowed-tools rule on a variable, so the command runs the wrapper by name: the
    # session's PATH starts with the install root.
    "claude": _Command(
        request='"$ARGUMENTS"',
        wrapper="reef-claude",
        shell_timeout="The Bash tool's own timeout counts milliseconds: give it 150000.",
        frontmatter=(
            "---\ndescription: Ask Reef to change this harness\nargument-hint: <what it should do>\n"
            "allowed-tools: Bash(reef-claude evolve:*), Bash(reef-claude wait:*)\n---\n"
        ),
    ),
    # Codex refuses a custom /command, so the command is a skill typed $reefine; its sandbox lets only the
    # wrapper's absolute path, which the session's rules name, reach Reef.
    "codex": _Command(
        request="the text after $reefine in the person's message",
        command="$reefine",
        wrapper="<wrapper>",
        wrapper_note=(
            " Here <wrapper> is the absolute path REEF_HARNESS_WRAPPER holds: run printenv REEF_HARNESS_WRAPPER "
            "once and write that path itself in each command, never the variable, since only that path may reach "
            "Reef from the sandbox."
        ),
        shell_timeout="If the shell tool returns while it still runs, wait for it to finish.",
    ),
    # opencode runs a command with the session's current agent; a mode agent without bash could not file it.
    "opencode": _Command(
        request='"$ARGUMENTS"',
        shell_timeout="The bash tool's own timeout counts milliseconds: give it 150000.",
        frontmatter="---\ndescription: Ask Reef to change this harness\nagent: build\n---\n",
    ),
    "hermes": _Command(
        request=(
            "the instruction at the end of the person's message, after 'The user has provided the following "
            "instruction alongside the skill invocation:'"
        ),
        shell_timeout="The terminal tool's own timeout counts seconds too: give it 150.",
    ),
    "dsh": _Command(request=_TYPED, shell_timeout="Give the bash tool timeoutMs 150000."),
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
    body = _read(_COMMAND_TEXT, adapter).format(
        command=command.command,
        request=command.request,
        adapter=adapter,
        wrapper=command.wrapper,
        wrapper_note=command.wrapper_note,
        shell_timeout=command.shell_timeout,
    )
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
