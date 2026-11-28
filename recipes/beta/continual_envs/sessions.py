"""The traffic sessions of a generation round, and the record folders that a generation run reads from them.

A round of the generated-task stage carries the records of its source sessions: each session's inference records,
then its report (``reef.recipe.generated_tasks.processor.session_records``). The report scores the session and
references its last call, whose request holds the whole conversation. One session becomes one record folder::

    <report record id>/session.json   {record_id, score, feedback, calls, request, response}

``request`` and ``response`` are the last call as the provider saw it. So the folder holds what the agent was asked,
each tool result that the environment gave, and the agent's last reply. Of the report it keeps only ``score`` and
``feedback``, never its metadata, so no verifier output reaches the generation run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from reef.core.records_types import AgentRecord, RequestType
from reef.core.values import JsonValue
from reef.samples.reports import report_score

#: The file of a session in its record folder.
SESSION_FILE = "session.json"
#: The keys that Reef adds to the provider's request body when it records an inference.
RECORDED_KEYS = frozenset({"metadata", "artifact_id", "response"})


@dataclass(frozen=True)
class TrafficSession:
    """One reported session of live traffic.

    Values: ``report`` is a report record. ``calls`` is a non-empty tuple of inference records in arrival order, and
    its last call is the one that ``report`` references last. ``score`` is the report's score, or None when the report
    has none.
    Purpose: ``traffic_sessions`` makes it from a round's records; the generator sends the failed ones to a
    generation run (``record_files``) and gives each task the record IDs of its session (``record_ids``).
    """

    report: AgentRecord
    calls: tuple[AgentRecord, ...]

    def __post_init__(self) -> None:
        if self.report.request_type is not RequestType.REPORT:
            raise ValueError("a traffic session's report must be a report record")
        if not self.calls or any(call.request_type is not RequestType.INFERENCE for call in self.calls):
            raise ValueError("a traffic session must have at least one inference record")
        if not self.report.references or self.report.references[-1] != self.calls[-1].agent_record_id:
            raise ValueError("a traffic session's report must reference its last call")

    @property
    def score(self) -> float | None:
        return report_score(self.report)

    @property
    def record_ids(self) -> tuple[str, ...]:
        """The IDs of every record of the session: its calls, then its report."""
        return (*(call.agent_record_id for call in self.calls), self.report.agent_record_id)

    def document(self) -> dict[str, JsonValue]:
        """The content of the session's ``session.json``."""
        last = self.calls[-1].payload
        feedback = self.report.payload.get("feedback")
        return {
            "record_id": self.report.agent_record_id,
            "score": self.score,
            "feedback": feedback if isinstance(feedback, str) else None,
            "calls": len(self.calls),
            "request": {key: value for key, value in last.items() if key not in RECORDED_KEYS},
            "response": last.get("response"),
        }


def traffic_sessions(records: Sequence[AgentRecord]) -> tuple[TrafficSession, ...]:
    """The sessions of a round's records, in order.

    Each report forms a session with the inference records after the previous report, up to the call that it
    references last. A report that references no call of that group forms no session.
    """
    sessions, calls = [], []
    for record in records:
        if record.request_type is RequestType.INFERENCE:
            calls.append(record)
            continue
        ids = [call.agent_record_id for call in calls]
        if record.references and record.references[-1] in ids:
            sessions.append(TrafficSession(record, tuple(calls[: ids.index(record.references[-1]) + 1])))
        calls = []
    return tuple(sessions)


def record_files(sessions: Sequence[TrafficSession], max_bytes: int) -> tuple[dict[str, str], tuple[str, ...]]:
    """The files of a ``records`` run source, and the report IDs of the sessions that do not fit.

    The files are ``{"<report record id>/session.json": text}``. The sessions fit in order while their files
    together hold at most ``max_bytes`` bytes; a session that does not fit is left out, and a later one can fit.
    """
    files: dict[str, str] = {}
    left_out: list[str] = []
    size_bytes = 0
    for session in sessions:
        text = json.dumps(session.document(), ensure_ascii=False)
        length_bytes = len(text.encode())
        if size_bytes + length_bytes > max_bytes:
            left_out.append(session.report.agent_record_id)
            continue
        files[f"{session.report.agent_record_id}/{SESSION_FILE}"] = text
        size_bytes += length_bytes
    return files, tuple(left_out)
