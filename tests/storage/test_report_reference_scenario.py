"""A report whose reference lives in another scenario is refused with a hint to send it with the inference's scenario."""

from types import SimpleNamespace

import pytest

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.dispatcher.ingest import validate_record


class _EmptyRecords:
    def existing_receipt(self, item: AgentRecord) -> None:
        return None

    def get_for_audit(self, scenario: str, agent_record_id: str) -> None:
        return None


class _Operations:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def increment(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1


def test_report_with_a_reference_from_another_scenario_names_the_scenario_header() -> None:
    current = SimpleNamespace(records=_EmptyRecords(), report_type=None, operations=_Operations())
    report = AgentRecord.create(
        scenario="x-reply",
        request_type=RequestType.REPORT,
        payload={"score": 12.0, "feedback": "views after 24 h", "references": ["inference-recorded-under-lab-reply"]},
    )

    with pytest.raises(ReportValidationError, match="same x-reef-scenario as the inference it references"):
        validate_record(current, report, {})
    assert current.operations.counts == {"ingest/rejected_report_total": 1}
