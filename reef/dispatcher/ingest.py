"""Validate records and uploaded trajectories and append them to one loaded scenario.

The dispatcher calls these functions under the scenario's lock. They append
records and uploaded trajectories durably and count the ``ingest/*`` metrics of records.
They do not start training: the caller wakes the trainers after new data lands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportValidationError, validate_report_payload
from reef.core.training_request import normalize_training_request_payload
from reef.harness.tree.nodes import directive_shaped, secret_shaped
from reef.scenario.scenario import Scenario
from reef.storage.records import RecordConflict


def training_request_refusal(text: str, requires: Sequence[Mapping[str, Any]] = ()) -> str | None:
    """Why admission refuses an instruction, or ``None``. The reason names the rule, never the text or the item.

    The text becomes proposer input and a catalog row. Thus it must pass the
    same screens as a promoted task prompt. A ``requires`` name, check or
    prompt is shown to the person and recorded in the commit. Thus it must
    pass the same screens too.
    """
    if secret_shaped(text):
        return "the request text carries a credential shaped literal; a request never holds secrets"
    if directive_shaped(text):
        return "the request text carries an instruction override phrasing or a chat template control token"
    for item in requires:
        for value in (str(item.get("name", "")), str(item.get("check") or ""), str(item.get("prompt") or "")):
            if secret_shaped(value):
                return "a requires item carries a credential shaped literal; a request never holds secrets"
            if directive_shaped(value):
                return "a requires item carries an instruction override phrasing or a chat template control token"
    return None


def validate_record(current: Scenario, item: AgentRecord, preceding: Mapping[str, AgentRecord]) -> AgentRecord | None:
    """Check ``item`` before it is appended; return the stored record when ``item`` repeats one.

    ``preceding`` holds the records earlier in the same import batch, which a
    report may reference before they are stored.
    """
    try:
        if item.request_type is RequestType.TRAIN:
            if (existing := current.records.existing_receipt(item)) is not None:
                return existing
            if current.training_mode == "auto":
                raise ValueError("explicit training requests require training_mode='manual' or 'hybrid'")
            if all(component_trainer.trainer.candidate_backend is None for component_trainer in current.trainers):
                raise ValueError("explicit training requests require a training backend")
            request = normalize_training_request_payload(item.payload)
            if item.references:
                raise ValueError("training instructions do not reference inference receipts")
            refusal = training_request_refusal(request["text"], request["requires"])
            if refusal is not None:
                raise ValueError(refusal)
        # Reject a malformed report before it is appended. Then the producer's
        # POST fails with an error that names the broken field. Otherwise the
        # record would fail silently at training time. A recipe without a
        # report schema accepts every report. A scenario of several components
        # accepts a report that any of its components accepts. Each trainer
        # skips a report that has the shape of another component.
        if item.request_type is RequestType.REPORT:
            # An identical retry remains valid after capacity eviction.
            if (existing := current.records.existing_receipt(item)) is not None:
                return existing
            validate_report_payload(item.payload)
            if (report_type := current.report_type) is not None:
                report_type.from_dict(item.payload)
            if len(set(item.references)) != len(item.references):
                raise ReportValidationError("report references must be unique")
            for reference in item.references:
                source = preceding.get(reference)
                if source is None:
                    stored_reference = current.records.get_for_audit(item.scenario, reference)
                    source = stored_reference.item if stored_reference is not None else None
                if source is None or source.request_type is not RequestType.INFERENCE:
                    raise ReportValidationError(
                        f"report reference {reference!r} must identify an existing inference in scenario {item.scenario!r}"
                    )
    except ReportValidationError:
        current.operations.increment("ingest/rejected_report_total")
        raise
    except RecordConflict:
        current.operations.increment("ingest/rejected_conflict_total")
        raise
    except ValueError:
        current.operations.increment("ingest/rejected_request_total")
        raise
    return None


def append_record(current: Scenario, item: AgentRecord) -> tuple[AgentRecord, bool]:
    """Validate and append one record; return the stored record and whether it is new."""
    if (existing := validate_record(current, item, {})) is not None:
        current.operations.increment("ingest/duplicates_total")
        return existing, False
    try:
        with current.operations.measure("ingest/write"):
            appended = current.records.append_result(item)
    except RecordConflict:
        current.operations.increment("ingest/rejected_conflict_total")
        raise
    current.operations.increment("ingest/accepted_total" if appended.inserted else "ingest/duplicates_total")
    return appended.item, appended.inserted


def append_records(current: Scenario, items: Sequence[AgentRecord]) -> tuple[tuple[AgentRecord, ...], bool]:
    """Validate an ordered import batch and append it; return the stored records and whether any is new."""
    preceding: dict[str, AgentRecord] = {}
    for item in items:
        validate_record(current, item, preceding)
        preceding[item.agent_record_id] = item
    try:
        with current.operations.measure("ingest/write"):
            appended = current.records.append_many(items)
    except RecordConflict:
        current.operations.increment("ingest/rejected_conflict_total")
        raise
    for result in appended:
        current.operations.increment("ingest/accepted_total" if result.inserted else "ingest/duplicates_total")
    return tuple(result.item for result in appended), any(result.inserted for result in appended)


def append_trajectories(current: Scenario, trajectories: tuple[tuple[str, TrajectoryItem], ...]) -> bool:
    """Save uploaded ``(trajectory_id, trajectory)`` pairs in one transaction. Return True if one or more is new.

    An uploaded trajectory trains only through a recipe that accepts uploaded
    trajectories. Otherwise every trainer would skip it, so admission refuses it.
    """
    if not any(component_trainer.trainer.pipeline.accepts_trajectories for component_trainer in current.trainers):
        raise ValueError(f"the recipe of scenario {current.name!r} does not accept uploaded trajectories")
    return current.store.trajectories.add(trajectories)
