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

from reef.core.artifact_ref import decode_artifact_ref
from reef.core.evaluation import ARTIFACT_KINDS
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord, RequestType
from reef.storage.tasks import TaskFileStore

#: Prefixes the task file partition of an owner. A scenario name never contains ``/``, so no scenario's partition
#: starts with it, and archiving a scenario never moves an owner's task files.
TASK_PARTITION_PREFIX = "data/"
#: The keys of a record item's content.
RECORD_CONTENT_KEYS = frozenset(
    {"agent_record_id", "request_type", "payload", "created_at", "references", "artifact_ref"}
)
#: The keys of an artifact item's content.
ARTIFACT_CONTENT_KEYS = frozenset({"source_uri", "kind", "component", "content_id"})


def json_digest(content: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical JSON of ``content``: sorted keys, no spaces, UTF-8."""
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if not isinstance(content, Mapping) or content.keys() != RECORD_CONTENT_KEYS:
        raise ValueError(f"a record item's content has exactly the keys {', '.join(sorted(RECORD_CONTENT_KEYS))}")
    record_id, request_type, payload = content["agent_record_id"], content["request_type"], content["payload"]
    created_at, references, artifact = content["created_at"], content["references"], content["artifact_ref"]
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
    if artifact is not None and not isinstance(artifact, Mapping):
        raise ValueError("a record item's artifact_ref is an object or null")
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
        artifact_ref=None if artifact is None else decode_artifact_ref(artifact),
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
    for key in sorted(ARTIFACT_CONTENT_KEYS):
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
    ``ValueError`` for content that breaks the rules.

    Purpose: ``ITEM_KIND_RULES`` holds one per kind; ``prepared_items`` calls it for each item of a create.
    """

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def stored_digest(
        self, owner: str, item_id: str, content: object, metadata: Mapping[str, Any], task_files: TaskFileStore
    ) -> str: ...


class JsonItemRules(ItemKindRules):
    """The rules of a kind whose content is a JSON object: its digest is ``json_digest``, after ``check_json``."""

    def stored_digest(
        self, owner: str, item_id: str, content: object, metadata: Mapping[str, Any], task_files: TaskFileStore
    ) -> str:
        if not isinstance(content, Mapping):
            raise ValueError(f"{self.kind} item {item_id!r} needs a JSON object as its content")
        self.check_json(item_id, content, metadata)
        return json_digest(content)

    @abstractmethod
    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` unless ``content`` follows the kind's rules."""


class TrajectoryRules(JsonItemRules):
    """A trajectory's content is an ATIF document."""

    kind = "trajectory"

    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        TrajectorySample(content)


class RecordRules(JsonItemRules):
    """A record's content and ID are what ``copied_record`` reads, for the scenario of its ``origin``."""

    kind = "record"

    def check_json(self, item_id: str, content: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        origin = metadata["origin"]["scenario"]
        copied_record(item_id, content, origin, origin)


class ArtifactRules(JsonItemRules):
    """An artifact's content is what ``check_artifact_content`` takes; a scenario made the item when it has an
    ``origin``."""

    kind = "artifact"

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
    "json_digest",
    "task_partition",
]
