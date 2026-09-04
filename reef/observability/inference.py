"""Provider-neutral observation contracts for recorded model interactions.

The record store remains authoritative.  These immutable snapshots are emitted
only after (or, for failures, instead of) record acceptance and observers must
never participate in request correctness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceTrace:
    """One inference attempt sequence and its final Reef/delivery state."""

    agent_record_id: str
    scenario: str
    recipe: str | None
    path: str
    started_at: float
    ended_at: float
    retry_count: int
    completion_state: str
    delivery_state: str
    record_accepted: bool
    streaming: bool
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    content_id: str | None = None
    release_id: str | None = None
    serving_runtime_load_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReportFeedback:
    """A durably accepted report to attach to each referenced inference."""

    report_record_id: str
    scenario: str
    references: tuple[str, ...]
    score: float | None = None
    feedback: str | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InferenceObserver:
    """Service-edge observer. Implementations must isolate every failure."""

    def record_inference(self, trace: InferenceTrace) -> None:
        return None

    def record_feedback(self, feedback: ReportFeedback) -> None:
        return None

    def close(self) -> None:
        pass


class NullInferenceObserver(InferenceObserver):
    """No-op observer used by every default Reef installation."""


__all__ = [
    "InferenceObserver",
    "InferenceTrace",
    "NullInferenceObserver",
    "ReportFeedback",
]
