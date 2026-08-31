from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from reef.core.errors import ReefError
from reef.core.records_types import RequestType

SCENARIO_HEADER = "x-reef-scenario"
ARTIFACT_VERSION_HEADER = "x-reef-artifact-version"
CAPTURE_HEADER = "x-reef-capture"
#: Harness-stamped side-channel context (method-integration RFC §3.2). The
#: request plane never reads a tag's meaning — it carries the pair through to
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
    artifact_version: str | None = None
    #: ``x-reef-tag-<name>`` pairs, lowercased names, opaque values.
    tags: Mapping[str, str] = field(default_factory=dict)
    #: Whether an inference exchange becomes a durable record. Reports do not
    #: use this transport control and always keep the default.
    capture: bool = True


def parse_request_headers(headers: Mapping[str, str], request_type: RequestType) -> RequestHeaders:
    normalized = {key.lower(): value for key, value in headers.items()}
    scenario = normalized.get(SCENARIO_HEADER, "").strip()
    if not scenario:
        raise HeaderError(f"missing or empty {SCENARIO_HEADER}")
    artifact_version = normalized.get(ARTIFACT_VERSION_HEADER, "").strip() or None
    capture = True
    if request_type is RequestType.INFERENCE and CAPTURE_HEADER in normalized:
        capture_value = normalized[CAPTURE_HEADER].strip().lower()
        if capture_value not in ("true", "false"):
            raise HeaderError(f"{CAPTURE_HEADER} must be 'true' or 'false'")
        capture = capture_value == "true"

    tags = {
        key[len(TAG_HEADER_PREFIX) :]: value.strip()
        for key, value in normalized.items()
        if key.startswith(TAG_HEADER_PREFIX) and len(key) > len(TAG_HEADER_PREFIX) and value.strip()
    }

    return RequestHeaders(
        scenario=scenario,
        request_type=request_type,
        artifact_version=artifact_version,
        capture=capture,
        tags=tags,
    )
