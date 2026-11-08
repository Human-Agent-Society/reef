"""Validate records, training requests, uploaded trajectories and uploaded tasks of one loaded scenario.

The dispatcher calls these functions under the scenario's lock. They append records, uploaded trajectories and
uploaded Harbor tasks durably, queue training requests (``reef.train.training_requests``), and count the ``ingest/*``
metrics of records and requests. They do not start training: the caller wakes the trainers after new data lands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportValidationError, validate_report_payload
from reef.core.training_request import TrainingRequest, normalize_training_request_payload
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.scenario.scenario import Scenario
from reef.storage.records import RecordConflict
from reef.storage.work_queue import WorkItem
from reef.train.trainer import Trainer


def training_request_refusal(text: str, requires: Sequence[Mapping[str, Any]] = ()) -> str | None:
    """Why admission refuses a request, or ``None``. The reason names the rule, never the text or the item.

    The text becomes proposer input and a catalog row. Thus it must pass the
    same screens as a collected task prompt. A ``requires`` name, check or
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


def file_request(current: Scenario, request_id: str, payload: Mapping[str, Any]) -> WorkItem:
    """Check a training request and queue it in ``current``'s requests; return its queue item.

    A retry of a request that the scenario queued before returns its item, whatever the training mode now.
    Otherwise the scenario must train in ``manual`` mode, the text and ``requires`` must pass the screens
    (``training_request_refusal``), and a trainer that trains must take the request's samples
    (``Trainer.check_request``). Raise ``ValueError`` for a refused request, and
    ``RecordConflict`` for an ID that the scenario queued with another payload.
    """
    if current.training_requests is None:
        raise ValueError(f"scenario {current.name!r} takes no training requests: it has no training request queue")
    try:
        retry = current.training_requests.get(request_id) is not None
        if not retry and current.training_mode == "auto":
            raise ValueError("explicit training requests require training_mode='manual'")
        trainers = [
            component_trainer.trainer for component_trainer in current.trainers if component_trainer.trainer.trains
        ]
        if not retry and not trainers:
            raise ValueError("explicit training requests require a training backend")
        request = normalize_training_request_payload(payload)
        if not retry:
            refusal = training_request_refusal(request["text"], request["requires"])
            if refusal is not None:
                raise ValueError(refusal)
            check_request(trainers, TrainingRequest.from_dict(request))
        return current.training_requests.file(request_id, request)
    except RecordConflict:
        current.operations.increment("ingest/rejected_conflict_total")
        raise
    except ValueError:
        current.operations.increment("ingest/rejected_request_total")
        raise


def validate_record(current: Scenario, item: AgentRecord, preceding: Mapping[str, AgentRecord]) -> AgentRecord | None:
    """Check ``item`` before it is appended; return the stored record when ``item`` repeats one.

    ``preceding`` holds the records earlier in the same import batch, which a
    report may reference before they are stored.
    """
    try:
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
                    source = stored_reference.item if stored_reference is not None else current.used_record(reference)
                if source is None or source.request_type is not RequestType.INFERENCE:
                    raise ReportValidationError(
                        f"report reference {reference!r} must identify an existing inference in scenario {item.scenario!r}"
                    )
            # A report references the last call of each trajectory that it scores (RFC #1054).
            for reference in item.references:
                inner = set(item.references) & set(current.records.trajectory(item.scenario, reference)) - {reference}
                if inner:
                    raise ReportValidationError(
                        f"report references {sorted(inner)!r} inside the trajectory that ends at {reference!r}; "
                        "reference only the last call of each trajectory"
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


def check_request(trainers: Sequence[Trainer], request: TrainingRequest) -> None:
    """Raise ``ValueError`` unless one of ``trainers`` can run ``request``, with the first trainer's reason."""
    reasons: list[str] = []
    for trainer in trainers:
        try:
            trainer.check_request(request)
            return
        except ValueError as exc:
            reasons.append(str(exc))
    raise ValueError(reasons[0])


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


def check_use(current: Scenario, trajectories: bool, train_tasks: bool) -> None:
    """Raise ``ValueError`` unless ``current`` can train on the data items that it is to use.

    ``trajectories`` says whether one or more trajectory item is new to it, and ``train_tasks`` whether one or more
    task item is to be in the ``train`` split. A trajectory needs a trainer that accepts uploaded trajectories
    (``TrainingPipeline.accepts_trajectories``). The check does not read the trajectories: the pipeline renders each
    one when it reads it, and skips one that cannot train (``TrainingPipeline.trainable``). A train task needs a
    trainer that reads imported tasks (``Trainer.accepts_imported_tasks``). The ``eval`` split is always open: an
    artifact evaluation runs its tasks.
    """
    if trajectories and not any(
        component_trainer.trainer.pipeline.accepts_trajectories for component_trainer in current.trainers
    ):
        raise ValueError(f"the recipe of scenario {current.name!r} does not accept uploaded trajectories")
    if train_tasks and not any(
        component_trainer.trainer.accepts_imported_tasks for component_trainer in current.trainers
    ):
        raise ValueError(f"the recipe of scenario {current.name!r} does not train on imported tasks")
