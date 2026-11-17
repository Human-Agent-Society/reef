"""Agent records and record stores for tests.

``inference``, ``token_inference`` and ``report`` build one record each. ``item_records`` and ``memory_records``
open the record items of one scenario in a data store.
"""

from __future__ import annotations

from reef.core import AgentRecord, RequestType
from reef.storage.data import DataStore, MemoryDataStore
from reef.storage.record_items import ItemRecordStore
from reef.storage.tasks import LocalTaskFileStore


def item_records(data: DataStore, scenario: str = "math", owner: str | None = None) -> ItemRecordStore:
    """The ``ItemRecordStore`` of ``scenario`` in ``data``; ``owner`` defaults to the scenario."""
    owner = owner or scenario
    return ItemRecordStore(data, owner, scenario, data.scenario_generation(owner, scenario))


def memory_records(scenario: str = "math", owner: str | None = None) -> ItemRecordStore:
    """The ``ItemRecordStore`` of ``scenario`` in a new ``MemoryDataStore``."""
    return item_records(MemoryDataStore(LocalTaskFileStore(None)), scenario, owner)


def inference(record_id: str, *, scenario: str = "math", **payload: object) -> AgentRecord:
    """The inference record ``record_id`` of ``scenario``. Its payload is exactly ``payload``."""
    return AgentRecord.create(
        scenario=scenario, request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def token_inference(record_id: str, *, scenario: str = "math", **payload: object) -> AgentRecord:
    """The inference record ``record_id`` of ``scenario`` that holds the training tensors of one two-token call.

    The payload holds ``tokens``, ``loss_mask`` and ``rollout_log_probs``, then the keys of ``payload``.
    """
    tensors: dict[str, object] = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    return inference(record_id, scenario=scenario, **{**tensors, **payload})


def report(
    record_id: str, *references: str, scenario: str = "math", score: object = 1.0, **payload: object
) -> AgentRecord:
    """The report record ``record_id`` of ``scenario`` on the inference records ``references``.

    The payload holds ``score`` and ``references``, then the keys of ``payload``, such as ``feedback`` or
    ``metadata``. ``score`` is a number; a test of invalid reports passes another value. The record's references
    are ``references``.
    """
    body = {"score": score, "references": list(references), **payload}
    return AgentRecord.create(
        scenario=scenario, request_type=RequestType.REPORT, payload=body, agent_record_id=record_id
    )
