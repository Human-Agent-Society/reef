"""Submission-log ingestion: the judge's history becomes trace rows.

Every submission is scored by the judge, so the submission log is a
trusted score-over-time record. The verifier writes it to
``/logs/verifier/submissions.jsonl`` (one ``{"t": <sec>, "score": <x>}``
line per submission); after the trial this file sits inside the trial
directory, and :func:`load_trace` turns it into :class:`TracePoint`s.
"""

from __future__ import annotations

import json
from pathlib import Path

from reef_eval.types import TracePoint

SUBMISSIONS_LOG = "submissions.jsonl"

# Directories the agent's own output lands in. A file it wrote is not
# evidence about itself, so these are never a source for the trace.
AGENT_WRITABLE = ("agent", "artifacts")


def parse_submissions(text: str) -> list[TracePoint]:
    """Parse submission-log lines, skipping malformed ones."""
    points: list[TracePoint] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            t = float(obj.pop("t"))
            score = float(obj.pop("score"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        points.append(TracePoint(t=t, score=score, data=obj))
    points.sort(key=lambda p: p.t)
    return points


def find_submissions_log(root: Path) -> Path | None:
    """Locate the verifier's submission log under a trial directory.

    The verifier's own directory is searched first. The fallback, for
    layouts that do not use a ``verifier/`` directory, is anywhere else
    under the trial except what the agent wrote: the anytime curve is only
    trustworthy because every score in it came from the judge, so a log
    found under :data:`AGENT_WRITABLE` is passed over rather than believed.
    """
    if not root.is_dir():
        return None
    verifier = root / "verifier"
    if verifier.is_dir():
        trusted = sorted(verifier.rglob(SUBMISSIONS_LOG))
        if trusted:
            return trusted[0]
    hits = [
        hit
        for hit in sorted(root.rglob(SUBMISSIONS_LOG))
        if not set(hit.relative_to(root).parts) & set(AGENT_WRITABLE)
    ]
    return hits[0] if hits else None


def load_trace(root: Path) -> list[TracePoint]:
    log = find_submissions_log(root)
    if log is None:
        return []
    try:
        return parse_submissions(log.read_text())
    except OSError:
        return []
