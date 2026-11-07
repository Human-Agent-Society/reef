"""Training-request validation and the queued request value."""

from __future__ import annotations

import json
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
    artifact_id: str
    requires: list[dict[str, str]]
    client: NotRequired[ClientReport]
    parameters: NotRequired[dict[str, Any]]
    samples: NotRequired[list[str]]


#: At most this many commands in one report.
MAX_CLIENT_COMMANDS = 64
#: At most this many bytes of JSON in a request's job parameters.
MAX_PARAMETERS_BYTES = 16384
CLIENT_WORD = re.compile(r"[A-Za-z0-9._+\- ()]{1,64}")
COMMAND_NAME = re.compile(r"[A-Za-z0-9._+\-]{1,40}")


def parse_client(value: object) -> ClientReport:
    """A client's report of its machine: ``platform``, ``arch`` and ``release`` as short words, and ``commands``
    mapping a command name to whether it is on the PATH, at most ``MAX_CLIENT_COMMANDS``. The report only
    informs a proposer, so what does not fit that shape is dropped rather than refusing the request; it is the
    client's word, data a proposer reads, never a request."""
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


def normalize_training_request_payload(payload: Mapping[str, object]) -> TrainingRequestPayload:
    """Validate a request and return its wire fields, dropping unknown keys.

    Invalid request fields, requirements or parameters raise ValueError. Client machine
    information is advisory: malformed entries are dropped. Text is preserved,
    absent requirements become an empty list, and an empty client or empty parameters are omitted.
    ``parameters`` is a JSON object with string keys, at most ``MAX_PARAMETERS_BYTES`` bytes of JSON: the
    parameters of the request's training job, merged into the recipe's (``Sample2Job.job_parameters``).
    ``samples`` is a non-empty list of distinct non-empty sample IDs: the samples that the request's job trains. The
    service's request size limit bounds its length.
    """
    fields: dict[str, str] = {}
    for key in ("text", "session", "artifact_id"):
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        fields[key] = value
    text = fields["text"]
    if not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > 4000:
        raise ValueError("text must not exceed 4000 characters")
    normalized: TrainingRequestPayload = {
        "text": text,
        "session": fields["session"],
        "artifact_id": fields["artifact_id"],
        "requires": parse_requires(payload.get("requires")),
    }
    client = parse_client(payload.get("client"))
    if client:
        normalized["client"] = client
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, Mapping) or not all(isinstance(key, str) for key in parameters):
        raise ValueError("parameters must be a JSON object")
    try:
        encoded = json.dumps(parameters, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"parameters must be JSON: {exc}") from exc
    if len(encoded.encode()) > MAX_PARAMETERS_BYTES:
        raise ValueError(f"parameters must not exceed {MAX_PARAMETERS_BYTES} bytes of JSON")
    if parameters:
        normalized["parameters"] = json.loads(encoded)
    if "samples" in payload:
        samples = payload["samples"]
        if (
            not isinstance(samples, list)
            or not samples
            or any(not isinstance(sample_id, str) or not sample_id for sample_id in samples)
            or len(set(samples)) != len(samples)
        ):
            raise ValueError("samples must be a non-empty list of distinct non-empty sample IDs")
        normalized["samples"] = list(samples)
    return normalized


def request_parameter(request: TrainingRequest) -> dict[str, Any]:
    """``request`` as the ``request`` job parameter of a stage that reads training requests: its ID and its fields
    without ``samples``, which the job data directory holds."""
    fields = {key: value for key, value in request.to_dict().items() if key != "samples"}
    return {"id": request.id, **fields}


def request_from_parameters(parameters: Mapping[str, Any]) -> TrainingRequest | None:
    """The training request of the ``request`` parameter that ``request_parameter`` wrote, or None; ``ValueError``
    for another value."""
    value = parameters.get("request")
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("parameters.request must be a training request with its id, or null")
    return TrainingRequest.from_dict({key: item for key, item in value.items() if key != "id"}, request_id=value["id"])


@dataclass(frozen=True)
class TrainingRequest:
    """A training request, independent of inference batches and feedback.

    ``id`` comes from the enclosing AgentRecord when the processor queues the request.
    Session and artifact identify the request's source; they do not select an inference batch.
    ``requires`` is what the change needs from the person's machine, at most
    ``MAX_REQUIRES`` ``{name, kind, check}`` items of the shape
    ``reef.core.requirements.parse_requires`` admits; default none.
    ``client`` is the requesting client's report of its machine (see
    :func:`reef.core.training_request.parse_client`); empty when it sent none.
    ``parameters`` are the parameters of the request's training job, merged into the recipe's; empty when it sent none.
    ``samples`` are the IDs of the samples that the request's job trains; empty when it named none.

    :meth:`from_dict` validates a payload with
    :func:`normalize_training_request_payload`; the constructor takes values
    that are already valid and does not check them again.
    """

    text: str
    session: str
    artifact_id: str
    id: str = ""
    # Out of the hash: the items are dicts, and the frozen contract is what the other fields carry.
    requires: tuple[Mapping[str, str], ...] = field(default=(), hash=False)
    client: Mapping[str, str | dict[str, bool]] = field(default_factory=dict, hash=False)
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)
    samples: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, request_id: str = "") -> TrainingRequest:
        """Validate a payload and build a queued request, using its enclosing record's ID when supplied."""
        normalized = normalize_training_request_payload(payload)
        return cls(
            text=normalized["text"],
            session=normalized["session"],
            artifact_id=normalized["artifact_id"],
            id=request_id,
            requires=tuple(normalized["requires"]),
            client=normalized.get("client", {}),
            parameters=normalized.get("parameters", {}),
            samples=tuple(normalized.get("samples", ())),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "session": self.session,
            "artifact_id": self.artifact_id,
            "requires": [dict(item) for item in self.requires],
            # Only when the client reported one, so a request without it keeps its earlier shape.
            **({"client": dict(self.client)} if self.client else {}),
            **({"parameters": dict(self.parameters)} if self.parameters else {}),
            **({"samples": list(self.samples)} if self.samples else {}),
        }
