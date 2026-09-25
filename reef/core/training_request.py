"""A user's training instruction and the session and release it came from."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.requirements import parse_requires

#: The commands a client reports as on its PATH or not, so a proposer builds for the machine the change runs on,
#: not the one it is tried in: players, notifiers, openers and clipboards per platform, and common tools.
CLIENT_COMMANDS = (
    "afplay",
    "say",
    "osascript",
    "open",
    "pbcopy",
    "terminal-notifier",
    "xdg-open",
    "notify-send",
    "paplay",
    "pw-play",
    "aplay",
    "wl-copy",
    "xclip",
    "powershell.exe",
    "wslview",
    "ffplay",
    "ffmpeg",
    "mpv",
    "mpg123",
    "sox",
    "espeak",
    "curl",
    "git",
    "gh",
    "python3",
    "node",
    "brew",
    "apt-get",
)
#: At most this many commands in one report.
MAX_CLIENT_COMMANDS = 64
_CLIENT_WORD = re.compile(r"[A-Za-z0-9._+\- ()]{1,64}")
_COMMAND_NAME = re.compile(r"[A-Za-z0-9._+\-]{1,40}")


def parse_client(value: Any) -> dict[str, Any]:
    """A client's report of its machine: ``platform``, ``arch`` and ``release`` as short words, and ``commands``
    mapping a command name to whether it is on the PATH, at most ``MAX_CLIENT_COMMANDS``. The report only
    informs a proposer, so what does not fit that shape is dropped rather than refusing the request; it is the
    client's word, data a proposer reads, never an instruction."""
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, Any] = {}
    for key in ("platform", "arch", "release"):
        word = value.get(key)
        if isinstance(word, str) and _CLIENT_WORD.fullmatch(word.strip()):
            parsed[key] = word.strip()
    commands = value.get("commands")
    if isinstance(commands, Mapping):
        kept = {
            name: present
            for name, present in commands.items()
            if isinstance(name, str) and _COMMAND_NAME.fullmatch(name) and isinstance(present, bool)
        }
        if kept:
            parsed["commands"] = dict(list(kept.items())[:MAX_CLIENT_COMMANDS])
    return parsed


@dataclass(frozen=True)
class TrainingRequest:
    """A training instruction, independent of inference batches and feedback.

    ``id`` is filled from the enclosing AgentRecord when it becomes a batch.
    Session and release identify the request's source; they do not select an inference batch.
    ``requires`` is what the change needs from the person's machine, at most
    ``MAX_REQUIRES`` ``{name, kind, check}`` items of the shape
    ``reef.core.requirements.parse_requires`` admits; default none.
    ``client`` is the requesting client's report of its machine (see
    :func:`parse_client`); empty when it sent none.
    """

    text: str
    session: str
    release_id: str
    id: str = ""
    # Out of the hash: the items are dicts, and the frozen contract is what the other fields carry.
    requires: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)
    client: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if len(self.text) > 4000:
            raise ValueError("text must not exceed 4000 characters")
        if not isinstance(self.session, str) or not isinstance(self.release_id, str):
            raise ValueError("session and release_id must be strings")
        object.__setattr__(self, "requires", tuple(parse_requires(self.requires)))
        object.__setattr__(self, "client", parse_client(self.client))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingRequest:
        fields: dict[str, str] = {}
        for key in ("text", "session", "release_id"):
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            fields[key] = value
        requires = payload.get("requires")
        return cls(**fields, requires=() if requires is None else requires, client=parse_client(payload.get("client")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "session": self.session,
            "release_id": self.release_id,
            "requires": [dict(item) for item in self.requires],
            # Only when the client reported one, so a request without it keeps its earlier shape.
            **({"client": dict(self.client)} if self.client else {}),
        }
