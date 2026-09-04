"""The HTTP wire shapes: ``x-reef-*`` request headers and the report body.

Only the service reads these. ``AgentRecord`` — what a request becomes once
stored — lives in ``reef.core``; this module is the boundary between an
aiohttp request and that record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.errors import ReefError
from reef.core.records_types import RequestType, parse_references

SCENARIO_HEADER = "x-reef-scenario"
RELEASE_ID_HEADER = "x-reef-release-id"
AGENT_RECORD_ID_HEADER = "x-reef-agent-record-id"
#: Harness-stamped side-channel context (method-integration RFC §3.2). The
#: service never reads a tag's meaning — it carries the pair through to
#: the INFERENCE record so a processor can correlate on it. Header-free
#: correlation stays the fallback; a tag is what a harness offers when its
#: agent does not resend the transcript it is continuing.
TAG_HEADER_PREFIX = "x-reef-tag-"


class HeaderError(ReefError):
    """Raised when x-reef-* request headers are missing or invalid."""


@dataclass(frozen=True)
class RequestHeaders:
    scenario: str
    request_type: RequestType
    release_id: str | None = None
    #: Optional client-selected inference receipt for retry-safe requests.
    agent_record_id: str | None = None
    #: ``x-reef-tag-<name>`` pairs, lowercased names, opaque values.
    tags: Mapping[str, str] = field(default_factory=dict)


def parse_request_headers(headers: Mapping[str, str], request_type: RequestType) -> RequestHeaders:
    normalized = {key.lower(): value for key, value in headers.items()}
    scenario = normalized.get(SCENARIO_HEADER, "").strip()
    if not scenario:
        raise HeaderError(f"missing or empty {SCENARIO_HEADER}")
    release_id = normalized.get(RELEASE_ID_HEADER, "").strip() or None
    agent_record_id = normalized.get(AGENT_RECORD_ID_HEADER, "").strip() or None

    tags = {
        key[len(TAG_HEADER_PREFIX) :]: value.strip()
        for key, value in normalized.items()
        if key.startswith(TAG_HEADER_PREFIX) and len(key) > len(TAG_HEADER_PREFIX) and value.strip()
    }

    return RequestHeaders(
        scenario=scenario,
        request_type=request_type,
        release_id=release_id,
        agent_record_id=agent_record_id,
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
    ``SlimeStepSignal.metrics`` are opaque payloads
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
