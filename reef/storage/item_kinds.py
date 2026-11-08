"""The kinds of data items and the rules of each kind's content (RFC #1028).

An item's ``metadata.kind`` is one of ``ITEM_KINDS``. The content of an item of each kind follows the rules of that
kind (``ITEM_KIND_RULES``), and every data store checks them when it creates an item (``prepared_items``):

* ``trajectory``: an ATIF document (``TrajectorySample``).
* ``record``: an inference or a report that a scenario made, as ``copied_record`` reads it
  (``reef.storage.record_items``).
* ``artifact``: a reference to an artifact's files (``check_artifact_content``, ``reef.storage.artifact_items``).
* ``task``: a Harbor task directory named after the item, which the store copies into the owner's task file
  partition (``task_partition``).

This module imports no store, so that each store and each kind's own module can import it.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.evaluation import ARTIFACT_KINDS
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord, RequestType, stored_record_artifact_id
from reef.storage.tasks import TaskFileStore

#: Prefixes the task file partition of an owner. A scenario name never contains ``/``, so no scenario's partition
#: starts with it, and archiving a scenario never moves an owner's task files.
TASK_PARTITION_PREFIX = "data/"
#: The keys of a record item's content.
RECORD_CONTENT_KEYS = frozenset(
    {"agent_record_id", "request_type", "payload", "created_at", "references", "artifact_id"}
)
#: The keys of an artifact item's content.
ARTIFACT_CONTENT_KEYS = frozenset({"source_uri", "kind", "component"})


def json_digest(content: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical JSON of ``content``: sorted keys, no spaces, UTF-8."""
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_call_id(evaluation_id: str, task_index: int, repeat: int, attempt: int, index: int) -> str:
    """The record ID of a call of an evaluation rollout: ``<evaluation_id>:<task_index>:<repeat>:<attempt>:<index>``."""
    return f"{evaluation_id}:{task_index}:{repeat}:{attempt}:{index}"


def task_partition(owner: str) -> str:
    """The task file store partition that keeps the task directories of ``owner``."""
    return TASK_PARTITION_PREFIX + owner


def copied_record(item_id: str, content: Mapping[str, Any], origin: str, scenario: str) -> AgentRecord:
    """The copy in ``scenario`` of the record item ``item_id`` with ``content``, made in the scenario ``origin``.

    The copy has the record ID ``item_id``, and the request type, payload, time and release of ``content``. Each
    reference ``x`` becomes ``<origin>/x``, and so does each entry of a ``references`` list in the payload. Raise
    ``ValueError`` unless ``content`` has the keys ``RECORD_CONTENT_KEYS`` with the types that ``record_item``
    gives them, for an inference or a report, and ``item_id`` is ``<origin>/<agent_record_id>``.
    """
    if not isinstance(content, Mapping) or set(content) != RECORD_CONTENT_KEYS:
        raise ValueError(f"a record item's content has exactly the keys {', '.join(sorted(RECORD_CONTENT_KEYS))}")
    record_id, request_type, payload = content["agent_record_id"], content["request_type"], content["payload"]
    created_at, references = content["created_at"], content["references"]
    if not isinstance(record_id, str) or item_id != f"{origin}/{record_id}":
        raise ValueError(f"a record item made in scenario {origin!r} has the ID {origin}/<agent_record_id>")
    if request_type not in (RequestType.INFERENCE.value, RequestType.REPORT.value):
        raise ValueError("a record item's request_type is 'inference' or 'report'")
    if not isinstance(payload, Mapping):
        raise ValueError("a record item's payload is an object")
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
        raise ValueError("a record item's created_at is a number")
    if not isinstance(references, list) or not all(isinstance(reference, str) for reference in references):
        raise ValueError("a record item's references is a list of strings")
    copied = tuple(f"{origin}/{reference}" for reference in references)
    payload = dict(payload)
    if isinstance(payload.get("references"), list):
        payload["references"] = [f"{origin}/{reference}" for reference in payload["references"]]
    return AgentRecord(
        agent_record_id=item_id,
        scenario=scenario,
        request_type=RequestType(request_type),
        payload=payload,
        created_at=float(created_at),
        references=copied,
        artifact_id=stored_record_artifact_id(content, payload),
    )


def check_artifact_content(content: Mapping[str, Any], *, made_by_scenario: bool) -> None:
    """Raise ``ValueError`` unless ``content`` is the content of an artifact item, as ``reef.storage.artifact_items``
    defines it.

    ``made_by_scenario`` is whether the item has an ``origin``. Every value is a non-empty string or null;
    ``kind`` is an ``ArtifactKind``; a client's item has a ``source_uri`` and no ``component``, and an item that a
    scenario made has a ``component``.
    """
    if set(content) != ARTIFACT_CONTENT_KEYS:
        raise ValueError(f"an artifact item's content holds exactly {sorted(ARTIFACT_CONTENT_KEYS)}")
    for key in sorted(content):
        value = content[key]
        if value is not None and not (isinstance(value, str) and value):
            raise ValueError(f"an artifact item's content.{key} is a non-empty string or null")
    if content["kind"] is not None and content["kind"] not in ARTIFACT_KINDS:
        raise ValueError(f"an artifact item's content.kind is one of {ARTIFACT_KINDS} or null")
    if made_by_scenario and content["component"] is None:
        raise ValueError("the artifact item of a scenario's artifact names its component")
    if not made_by_scenario and (content["source_uri"] is None or content["component"] is not None):
        raise ValueError("a client's artifact item names its source_uri, and no component")


class ItemKindRules(ABC):
    """The content rules of one item kind.

    Values: ``kind`` is the kind, one of ``ITEM_KINDS``. ``stored_digest(owner, item_id, content, metadata,
    task_files)`` checks ``content``, the content of the new item ``item_id`` of ``owner`` with the created
    ``metadata``, against the kind's rules, and returns the digest of the content that the store keeps. It raises
    ``ValueError`` for content that breaks the rules. ``current_content(content)`` is ``content`` in the current
    format; an older stored format maps to the same value as its current equivalent. ``matches_stored(content,
    digest)`` is whether stored ``content`` in an older format has the current content with ``digest``.

    Purpose: ``ITEM_KIND_RULES`` holds one per kind; ``prepared_items`` calls it for each item of a create, and a
    create compares a retry with ``matches_stored``.
    """

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def stored_digest(
        self, owner: str, item_id: str, content: object, metadata: Mapping[str, Any], task_files: TaskFileStore
    ) -> str: ...

    def current_content(self, content: Mapping[str, Any] | Path) -> Mapping[str, Any] | Path:
        """``content`` unchanged: this kind has one format."""
        return content

    def matches_stored(self, content: object, digest: str) -> bool:
        """False: without older formats, only an equal digest matches."""
        return False


class JsonItemRules(ItemKindRules):
    """The rules of a kind whose content is a JSON object: its digest is ``json_digest``, after ``check_json``."""

    def stored_digest(
        self, owner: str, item_id: str, content: object, metadata: Mapping[str, Any], task_files: TaskFileStore
    ) -> str:
        if not isinstance(content, Mapping):
            raise ValueError(f"{self.kind} item {item_id!r} needs a JSON object as its content")
        self.check_json(item_id, content, metadata)
        return json_digest(content)

    def matches_stored(self, content: object, digest: str) -> bool:
        if not isinstance(content, Mapping):
            return False
        current = self.current_content(content)
        return current is not content and isinstance(current, Mapping) and json_digest(current) == digest

    @abstractmethod
    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` unless ``content`` follows the kind's rules."""


class TrajectoryRules(JsonItemRules):
    """A trajectory's content is an ATIF document."""

    kind = "trajectory"

    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        TrajectorySample(content)


class RecordRules(JsonItemRules):
    """A record's content and ID are what ``copied_record`` reads, for the scenario of its ``origin``. The record ID of
    an evaluation call is the ``evaluation_call_id`` of its ``origin.evaluation``.

    Its digest is ``json_digest`` of the content without ``created_at``, as the record store's retry fingerprint,
    so a client's retry with a new time matches. An item stored before had ``created_at`` in its digest, and it
    matches the same content too.
    """

    kind = "record"

    def stored_digest(
        self, owner: str, item_id: str, content: object, metadata: Mapping[str, Any], task_files: TaskFileStore
    ) -> str:
        if not isinstance(content, Mapping):
            raise ValueError(f"record item {item_id!r} needs a JSON object as its content")
        self.check_json(item_id, content, metadata)
        return record_digest(content)

    def matches_stored(self, content: object, digest: str) -> bool:
        return isinstance(content, Mapping) and record_digest(content) == digest

    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        origin = metadata["origin"]["scenario"]
        record = copied_record(item_id, content, origin, origin)
        location = metadata["origin"].get("evaluation")
        names = ("evaluation_id", "task_index", "repeat", "attempt", "index")
        if (
            location is not None
            and record.agent_record_id != f"{origin}/{evaluation_call_id(*map(location.get, names))}"
        ):
            raise ValueError(
                f"an evaluation call has the ID {origin}/<evaluation_id>:<task_index>:<repeat>:<attempt>:<index>"
            )


def record_digest(content: Mapping[str, Any]) -> str:
    """The digest of a record item's ``content``: ``json_digest`` without ``created_at``."""
    return json_digest({key: value for key, value in content.items() if key != "created_at"})


class ArtifactRules(JsonItemRules):
    """An artifact's content is what ``check_artifact_content`` takes; a scenario made the item when it has an
    ``origin``."""

    kind = "artifact"

    def current_content(self, content: Mapping[str, Any] | Path) -> Mapping[str, Any] | Path:
        """An older artifact item also held an unused ``content_id``."""
        if not isinstance(content, Mapping) or "content_id" not in content:
            return content
        return {key: value for key, value in content.items() if key != "content_id"}

    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        check_artifact_content(content, made_by_scenario="origin" in metadata)


class TaskRules(ItemKindRules):
    """A task's content is a Harbor task directory named after the item. The store copies it into the owner's task
    file partition, so the digest names stored files."""

    kind = "task"

    def stored_digest(
        self, owner: str, item_id: str, content: object, metadata: Mapping[str, Any], task_files: TaskFileStore
    ) -> str:
        if not isinstance(content, Path) or not content.is_dir():
            raise ValueError(f"task item {item_id!r} needs a task directory as its content")
        if content.name != item_id:
            raise ValueError(f"task item {item_id!r} needs a task directory of the same name")
        return task_files.put(task_partition(owner), content)


#: The rules of each item kind.
ITEM_KIND_RULES: dict[str, ItemKindRules] = {
    rules.kind: rules for rules in (TrajectoryRules(), TaskRules(), RecordRules(), ArtifactRules())
}

__all__ = [
    "ARTIFACT_CONTENT_KEYS",
    "ITEM_KIND_RULES",
    "RECORD_CONTENT_KEYS",
    "TASK_PARTITION_PREFIX",
    "ArtifactRules",
    "ItemKindRules",
    "JsonItemRules",
    "RecordRules",
    "TaskRules",
    "TrajectoryRules",
    "check_artifact_content",
    "copied_record",
    "evaluation_call_id",
    "json_digest",
    "task_partition",
]
