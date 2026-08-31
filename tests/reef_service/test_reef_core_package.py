from __future__ import annotations

import importlib.util

import pytest

from reef.core import AgentRecord, RequestType
from reef.records import RecordConflict, RecordStore


@pytest.mark.unit
def test_core_package_exports_protocol_and_record_types() -> None:
    assert tuple(RequestType) == (
        RequestType.INFERENCE,
        RequestType.REPORT,
    )
    assert AgentRecord.__module__ == "reef.core.records_types"
    assert RecordStore.__module__ == "reef.records"
    assert all(symbol is not None for symbol in (RecordConflict,))


@pytest.mark.unit
@pytest.mark.parametrize("module", ["reef.headers", "reef.runtime.testing", "reef.wire"])
def test_non_public_modules_are_not_shipped(module: str) -> None:
    assert importlib.util.find_spec(module) is None
