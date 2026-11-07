"""The HTTP wire shapes: ``x-reef-*`` request headers and the report body.

Only the service reads these. ``AgentRecord`` — what a request becomes once
stored — lives in ``reef.core``; this module is the boundary between an
aiohttp request and that record.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.errors import ReefError
from reef.core.records_types import RequestType, parse_references

SCENARIO_HEADER = "x-reef-scenario"
#: Harness-stamped side-channel context. The
#: service carries each pair through to the INFERENCE record so a processor
#: can correlate on it, and reads one value itself: ``release``, the snapshot
#: a client runs, which names the record's artifact where the client pulls
#: the snapshot (``recorded_ref``). Header-free correlation stays the
#: fallback; a tag is what a harness offers when its agent does not resend
#: the transcript it is continuing.
TAG_HEADER_PREFIX = "x-reef-tag-"


class HeaderError(ReefError):
    """Raised when x-reef-* request headers are missing or invalid."""


def strict_json_object(data: bytes) -> dict[str, object]:
    """Decode one JSON object, refusing repeated keys and non-finite numbers.

    Raises:
        ValueError: The data is not such an object, is not valid text, or nests too deeply to decode.
    """

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("JSON object repeats a key")
        return result

    def finite_number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON number is not finite")
        return value

    def no_constant(name: str) -> float:
        raise ValueError(f"JSON constant {name} is not allowed")

    try:
        value = json.loads(
            data, object_pairs_hook=unique_object, parse_float=finite_number, parse_constant=no_constant
        )
    except RecursionError as exc:
        raise ValueError("JSON nests too deeply") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


@dataclass(frozen=True)
class RequestHeaders:
    scenario: str
    request_type: RequestType
    #: ``x-reef-tag-<name>`` pairs, lowercased names, opaque values.
    tags: Mapping[str, str] = field(default_factory=dict)


def scenario_header(headers: Mapping[str, str]) -> str:
    """The scenario that a request's headers name; ``HeaderError`` without a scenario."""
    normalized = {key.lower(): value for key, value in headers.items()}
    scenario = normalized.get(SCENARIO_HEADER, "").strip()
    if not scenario:
        raise HeaderError(f"missing or empty {SCENARIO_HEADER}")
    return scenario


def parse_request_headers(headers: Mapping[str, str], request_type: RequestType) -> RequestHeaders:
    normalized = {key.lower(): value for key, value in headers.items()}
    scenario = scenario_header(headers)

    tags = {
        key[len(TAG_HEADER_PREFIX) :]: value.strip()
        for key, value in normalized.items()
        if key.startswith(TAG_HEADER_PREFIX) and len(key) > len(TAG_HEADER_PREFIX) and value.strip()
    }

    return RequestHeaders(
        scenario=scenario,
        request_type=request_type,
        tags=tags,
    )


@dataclass(frozen=True)
class ReportPayload:
    """A report carries feedback about an inference, of which a numeric score is one
    optional, common form.

    ``feedback`` is intentionally opaque to reef's core: it can be a plain string for
    simple text feedback, or a structured object (a rubric breakdown, judge output,
    multi-dimensional scores, or whatever a given recipe cares about) when a recipe
    wants richer content. Reef's core only validates that it is a string or a JSON
    object — never a specific internal shape. Interpretation is left to whichever
    recipe's processor reads ``report.payload["feedback"]``, the same way ``metadata``,
    ``TrainingSignal.metrics`` are opaque payloads
    interpreted by consumers rather than validated centrally.
    """

    score: float | None = None
    feedback: str | Mapping[str, Any] | None = None
    references: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReportPayload:
        score = payload.get("score")
        if score is not None:
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError("score must be a number")
            if not math.isfinite(score):
                raise ValueError("score must be finite")
            score = float(score)
        feedback = payload.get("feedback")
        if feedback is not None and not isinstance(feedback, (str, Mapping)):
            raise ValueError("feedback must be a string or an object")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            score=score,
            feedback=dict(feedback) if isinstance(feedback, Mapping) else feedback,
            references=parse_references(payload),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.score is not None:
            result["score"] = self.score
        if self.feedback is not None:
            result["feedback"] = dict(self.feedback) if isinstance(self.feedback, Mapping) else self.feedback
        if self.references:
            result["references"] = list(self.references)
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


def with_request_tags(payload: Mapping[str, Any], parsed: RequestHeaders) -> Mapping[str, Any]:
    """Carry ``x-reef-tag-*`` through to the INFERENCE record's metadata.

    Only inference: a tag is context about a served exchange, and the
    processors that read one correlate on the inference side. This stores
    every pair as sent; only :func:`recorded_ref` reads a value.
    """
    if parsed.request_type is not RequestType.INFERENCE or not parsed.tags:
        return payload
    metadata = dict(payload.get("metadata") or {})
    metadata["tags"] = {**(metadata.get("tags") or {}), **parsed.tags}
    return {**payload, "metadata": metadata}
