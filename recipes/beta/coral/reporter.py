"""Report finalized CORAL attempts to Reef as training signal.

One ``POST /reef/report`` per attempt: the grader score, the attempt's
captured inference references, the CORAL coordinates in metadata, and a
deterministic client-supplied id so grader re-runs dedup server-side.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from recipes.beta.coral.journal import deterministic_report_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttemptReport:
    """The reef-facing projection of one finalized CORAL attempt."""

    scenario: str
    agent_id: str
    commit_hash: str
    score: float | None
    status: str
    parent_hash: str | None = None
    run_id: str | None = None
    feedback: str | Mapping[str, Any] | None = None
    references: tuple[str, ...] = ()
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "coral": {
                "agent_id": self.agent_id,
                "commit_hash": self.commit_hash,
                "status": self.status,
                "parent_hash": self.parent_hash,
                "run_id": self.run_id,
            },
            **dict(self.extra_metadata),
        }
        body: dict[str, Any] = {
            "agent_record_id": deterministic_report_id(self.scenario, self.agent_id, self.commit_hash),
            "metadata": metadata,
        }
        if self.score is not None:
            body["score"] = float(self.score)
        if self.feedback is not None:
            body["feedback"] = self.feedback if isinstance(self.feedback, str) else dict(self.feedback)
        if self.references:
            body["references"] = list(self.references)
        return body


def report_attempt(
    reef_url: str,
    report: AttemptReport,
    *,
    token: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST one report; returns reef's acknowledgement.

    Raises ``urllib.error.HTTPError`` on rejection — a conflicting resend
    (same client id, different payload) is a bug worth failing loudly on.
    """
    body = report.payload()
    headers = {
        "Content-Type": "application/json",
        "x-reef-scenario": report.scenario,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        reef_url.rstrip("/") + "/reef/report",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
