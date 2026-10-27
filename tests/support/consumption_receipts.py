"""Write a consumption receipt as an earlier Reef version did; Reef itself no longer writes them.

Stores that older versions wrote keep these receipts, and scenario recovery
still reads them. Tests use this helper to make such a store.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping

from reef.storage.sql_records import SQLRecordStore


def write_consumption_receipt(
    records: SQLRecordStore,
    scenario: str,
    record_ids: frozenset[str],
    *,
    receipt_id: str,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Insert one receipt row in the layout that ``consumption_receipts`` reads."""
    consumed_ids_json = json.dumps(sorted(record_ids), ensure_ascii=False, separators=(",", ":"))
    metadata_json = json.dumps(dict(metadata or {}), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with records._transaction(write=True) as connection:
        records._insert(
            connection,
            records._tables.consumption,
            {
                "scenario": scenario,
                "receipt_id": receipt_id,
                "consumed_ids_json": consumed_ids_json,
                "metadata_json": metadata_json,
                "recorded_at": time.time(),
                "consumed_ids_sha256": hashlib.sha256(consumed_ids_json.encode()).hexdigest(),
            },
        )
