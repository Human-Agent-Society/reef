"""Training-request validation and the queued instruction value."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

from reef.core.requirements import parse_requires

ClientReport = dict[str, str | dict[str, bool]]


class TrainingRequestPayload(TypedDict):
    """Normalized fields stored in a TRAIN record; identity comes from its envelope."""

    text: str
    session: str
    release_id: str
    requires: list[dict[str, str]]
    client: NotRequired[ClientReport]


#: At most this many commands in one report.
MAX_CLIENT_COMMANDS = 64
CLIENT_WORD = re.compile(r"[A-Za-z0-9._+\- ()]{1,64}")
COMMAND_NAME = re.compile(r"[A-Za-z0-9._+\-]{1,40}")


def parse_client(value: object) -> ClientReport:
    """A client's report of its machine: ``platform``, ``arch`` and ``release`` as short words, and ``commands``
    mapping a command name to whether it is on the PATH, at most ``MAX_CLIENT_COMMANDS``. The report only
    informs a proposer, so what does not fit that shape is dropped rather than refusing the request; it is the
    client's word, data a proposer reads, never an instruction."""
    if not isinstance(value, Mapping):
        return {}
    parsed: ClientReport = {}
    for key in ("platform", "arch", "release"):
        word = value.get(key)
        if isinstance(word, str) and CLIENT_WORD.fullmatch(word.strip()):
            parsed[key] = word.strip()
    commands = value.get("commands")
    if isinstance(commands, Mapping):
        kept = {
            name: present
            for name, present in commands.items()
            if isinstance(name, str) and COMMAND_NAME.fullmatch(name) and isinstance(present, bool)
        }
        if kept:
            parsed["commands"] = dict(list(kept.items())[:MAX_CLIENT_COMMANDS])
    return parsed


def parse_training_request_fields(payload: Mapping[str, object]) -> tuple[str, str, str]:
    """Read the required text, session and release strings for normalization or construction."""
    fields: dict[str, str] = {}
    for key in ("text", "session", "release_id"):
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        fields[key] = value
    return fields["text"], fields["session"], fields["release_id"]


def normalize_training_request_payload(payload: Mapping[str, object]) -> TrainingRequestPayload:
    """Validate a request and return its wire fields, dropping unknown keys.

    Invalid instruction fields or requirements raise ValueError. Client machine
    information is advisory: malformed entries are dropped. Text is preserved,
    absent requirements become an empty list, and an empty client is omitted.
    """
    text, session, release_id = parse_training_request_fields(payload)
    if not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > 4000:
        raise ValueError("text must not exceed 4000 characters")
    normalized: TrainingRequestPayload = {
        "text": text,
        "session": session,
        "release_id": release_id,
        "requires": parse_requires(payload.get("requires")),
    }
    client = parse_client(payload.get("client"))
    if client:
        normalized["client"] = client
    return normalized


@dataclass(frozen=True)
class TrainingRequest:
    """A training instruction, independent of inference batches and feedback.

    ``id`` comes from the enclosing AgentRecord when the processor queues the request.
    Session and release identify the request's source; they do not select an inference batch.
    ``requires`` is what the change needs from the person's machine, at most
    ``MAX_REQUIRES`` ``{name, kind, check}`` items of the shape
    ``reef.core.requirements.parse_requires`` admits; default none.
    ``client`` is the requesting client's report of its machine (see
    :func:`reef.core.training_request.parse_client`); empty when it sent none.

    Construction delegates to the shared payload normalizer. Admission code
    calls that function directly; processors create this value for their queue.
    """

    text: str
    session: str
    release_id: str
    id: str = ""
    # Out of the hash: the items are dicts, and the frozen contract is what the other fields carry.
    requires: tuple[Mapping[str, str], ...] = field(default=(), hash=False)
    client: Mapping[str, str | dict[str, bool]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        normalized = normalize_training_request_payload(
            {
                "text": self.text,
                "session": self.session,
                "release_id": self.release_id,
                "requires": self.requires,
                "client": self.client,
            }
        )
        object.__setattr__(self, "requires", tuple(normalized["requires"]))
        object.__setattr__(self, "client", normalized.get("client", {}))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, request_id: str = "") -> TrainingRequest:
        """Build a queued request, using its enclosing record's ID when supplied."""
        text, session, release_id = parse_training_request_fields(payload)
        return cls(
            text=text,
            session=session,
            release_id=release_id,
            id=request_id,
            requires=payload.get("requires", ()),
            client=payload.get("client", {}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "session": self.session,
            "release_id": self.release_id,
            "requires": [dict(item) for item in self.requires],
            # Only when the client reported one, so a request without it keeps its earlier shape.
            **({"client": dict(self.client)} if self.client else {}),
        }
