"""Storage contract for the data store of a deployment (RFC #1028).

The data store keeps the data that scenarios train on, once per deployment. An item is content and metadata.
Its content never changes; its metadata says what the item is, which owner it belongs to, and which scenarios
use it. The store has create, read, query, update and delete, and each operation reads or writes the items of
one owner only. Training state, such as train counts and buffers, is not data and stays in the scenario
storage.

This module defines the item values, the metadata rules, the ``DataStore`` plugin base and an in-memory store.
``reef.storage.sql_data`` implements the store in a SQL database.
"""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, get_args

from reef.core.errors import ReefError
from reef.core.plugins import Plugin, PluginConfigError
from reef.storage.tasks import SPLIT_NAMES, TaskFileStore

ItemKind = Literal["trajectory", "task", "record"]
ITEM_KINDS: tuple[ItemKind, ...] = get_args(ItemKind)
#: The metadata keys that Reef reads. ``kind``, ``owner`` and ``origin`` are set at creation and never change.
FIXED_KEYS = ("kind", "owner", "origin")
#: The most items that one query returns.
MAX_QUERY_LIMIT = 1000
#: Prefixes the task file partition of an owner. A scenario name never contains ``/``, so no scenario's partition
#: starts with it, and archiving a scenario never moves an owner's task files.
TASK_PARTITION_PREFIX = "data/"


class ItemConflict(ReefError):
    """An item ID that the owner already stores with other content."""


class ItemNotFound(ReefError):
    """An item ID that the owner does not store."""


def check_owner(owner: object) -> str:
    """Return ``owner`` if it is a valid owner: a non-empty string without ``/``, other than ``.`` and ``..``.

    These are the rules of a scenario name. Raise ``ValueError`` otherwise.
    """
    if not isinstance(owner, str) or not owner or "/" in owner or owner in (".", ".."):
        raise ValueError(f"owner must be a non-empty name without '/', got {owner!r}")
    return owner


def check_item_id(item_id: object) -> str:
    """Return ``item_id`` if it is a non-empty string; raise ``ValueError`` otherwise."""
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"an item ID must be a non-empty string, got {item_id!r}")
    return item_id


def json_digest(content: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical JSON of ``content``: sorted keys, no spaces, UTF-8."""
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_scenarios(value: object, kind: ItemKind) -> dict[str, str]:
    """Return ``value`` as the ``scenarios`` metadata of an item of ``kind``, or raise ``ValueError``.

    ``scenarios`` maps each scenario name (the rules of ``check_owner``) to a split, ``train`` or ``eval``. A
    trajectory and a record are always ``train``.
    """
    if not isinstance(value, Mapping):
        raise ValueError("metadata.scenarios must be an object from scenario name to split")
    scenarios: dict[str, str] = {}
    for scenario, split in value.items():
        check_owner(scenario)
        if split not in SPLIT_NAMES:
            raise ValueError(f"metadata.scenarios.{scenario} must be train or eval, got {split!r}")
        if kind != "task" and split != "train":
            raise ValueError(f"a {kind} is always train, so metadata.scenarios.{scenario} must be train")
        scenarios[scenario] = split
    return scenarios


def check_origin(value: object) -> dict[str, str | None]:
    """Return ``value`` as a record's ``origin``: ``{scenario, release_id, artifact_id}``.

    ``scenario`` is a valid scenario name. ``release_id`` and ``artifact_id`` are non-empty strings, or null for a
    record that names no release or no artifact, such as an imported record. Raise ``ValueError`` otherwise.
    """
    if not isinstance(value, Mapping) or set(value) != {"scenario", "release_id", "artifact_id"}:
        raise ValueError("metadata.origin must be {scenario, release_id, artifact_id}")
    check_owner(value["scenario"])
    for key in ("release_id", "artifact_id"):
        if value[key] is not None and not (isinstance(value[key], str) and value[key]):
            raise ValueError(f"metadata.origin.{key} must be a non-empty string or null")
    return {key: value[key] for key in ("scenario", "release_id", "artifact_id")}


def created_metadata(owner: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """The metadata that an item of ``owner`` is created with, from the client's ``metadata``.

    ``kind`` is required. ``owner``, when given, must be ``owner``; it is set when it is missing. ``scenarios``
    defaults to ``{}``. ``origin`` is required for a record and refused for any other kind. Other keys are kept
    as they are. Raise ``ValueError`` for metadata that breaks these rules.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    kind = metadata.get("kind")
    if kind not in ITEM_KINDS:
        raise ValueError(f"metadata.kind must be one of {', '.join(ITEM_KINDS)}, got {kind!r}")
    if metadata.get("owner", owner) != owner:
        raise ValueError(f"metadata.owner must be {owner!r}, the owner that creates the item")
    created = copy.deepcopy(dict(metadata))
    created["owner"] = owner
    created["scenarios"] = check_scenarios(metadata.get("scenarios", {}), kind)
    if kind == "record":
        created["origin"] = check_origin(metadata.get("origin"))
    elif "origin" in metadata:
        raise ValueError(f"only a record has metadata.origin, not a {kind}")
    return created


def merge_patch(target: Any, patch: Any) -> Any:
    """Apply the JSON merge patch ``patch`` to ``target`` (RFC 7386) and return the result; ``target`` stays."""
    if not isinstance(patch, Mapping):
        return copy.deepcopy(patch)
    merged = copy.deepcopy(dict(target)) if isinstance(target, Mapping) else {}
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = merge_patch(merged.get(key), value)
    return merged


def patched_metadata(metadata: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """``metadata`` with the JSON merge patch ``patch`` applied, or ``ValueError``.

    The patch may not change ``kind``, ``owner`` or ``origin``. The patched ``scenarios`` must be valid for the
    item's kind; removing ``scenarios`` leaves it empty. A record item that its origin scenario uses stays in it:
    that scenario's record store keeps the record.
    """
    if not isinstance(patch, Mapping):
        raise ValueError("a metadata patch must be an object")
    patched = merge_patch(metadata, patch)
    for key in FIXED_KEYS:
        if patched.get(key) != metadata.get(key):
            raise ValueError(f"metadata.{key} is set at creation and never changes")
    patched["scenarios"] = check_scenarios(patched.get("scenarios", {}), metadata["kind"])
    if metadata["kind"] == "record":
        origin = metadata["origin"]["scenario"]
        if origin in metadata["scenarios"] and origin not in patched["scenarios"]:
            raise ValueError(f"a record item stays in {origin!r}, the scenario that made it")
    return patched


@dataclass(frozen=True)
class NewItem:
    """An item that a client creates.

    Values: ``item_id`` passes ``check_item_id``. ``content`` is a JSON object for a trajectory or a record, and
    the path of a task directory for a task; a task's ``item_id`` is the name of that directory. ``metadata`` is the client's metadata, which ``created_metadata``
    checks against the owner.

    Purpose: the input of ``DataStore.create``.
    """

    item_id: str
    content: Mapping[str, Any] | Path
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class StoredItem:
    """An item in a data store.

    Values: ``sequence`` is a positive int, the position of the item in the order the store received its items.
    ``owner`` and ``item_id`` identify the item; no two items of one owner have the same ``item_id``.
    ``metadata`` passes ``created_metadata`` for ``owner``, and its ``kind`` is the item's kind. ``digest`` is the
    SHA-256 of the content: ``json_digest`` of the JSON content, or the task file store's digest of a task
    directory. A trajectory or a record has its JSON ``content``. A task has no ``content``: the task file store
    keeps its directory under ``digest``, in the owner's partition (``task_partition``).

    Purpose: what ``DataStore`` returns. Training reads a scenario's items, and clients read and query them.
    """

    sequence: int
    owner: str
    item_id: str
    metadata: Mapping[str, Any]
    digest: str
    content: Mapping[str, Any] | None = field(default=None, compare=False)

    @property
    def kind(self) -> ItemKind:
        return self.metadata["kind"]

    @property
    def scenarios(self) -> Mapping[str, str]:
        return self.metadata["scenarios"]


def task_partition(owner: str) -> str:
    """The task file store partition that keeps the task directories of ``owner``."""
    return TASK_PARTITION_PREFIX + owner


class DataStore(Plugin, ABC):
    """The data of a deployment: items of content and metadata, each in one owner.

    Values: the state of a store is a finite set of ``StoredItem`` values. Within one owner, no two items have
    the same ``item_id``, and ``sequence`` increases in the order the items arrived. An item's content,
    ``digest`` and fixed metadata (``kind``, ``owner``, ``origin``) never change. Every operation names one owner
    and reads or writes only the items of that owner.

    Operations:

    * ``create(owner, items)`` stores each item, all or none. An item whose ID the owner stores with the same
      digest is unchanged; with another digest it is an ``ItemConflict``, and nothing is stored. Bad metadata or
      content is a ``ValueError``, and nothing is stored. It returns the stored items in the order given.
    * ``read(owner, item_id)`` returns the item, or None.
    * ``query(owner, kind, scenario, after, limit)`` returns the items of ``owner`` with ``sequence > after``,
      in sequence order, at most ``limit``, of ``kind`` and used by ``scenario`` when those are given.
    * ``update(owner, item_ids, patch)`` applies one JSON merge patch to the metadata of each item, all or none
      (``patched_metadata``). An unknown ID is an ``ItemNotFound``. It returns the updated items.
    * ``delete(owner, item_id)`` removes the item and returns True, or returns False for an unknown ID.
    * ``task_path(item)`` returns a local directory with the files of a task item.

    The store checks the form of the metadata. It does not know the scenarios, so the service checks that
    ``scenarios`` names only scenarios of the owner.

    Purpose: a deployment names a subclass in ``storage.data_store``. The service builds it at startup and closes
    it at shutdown.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        """Build the store a deployment names.

        ``config`` is the plugin's own configuration. ``directory`` is the deployment's
        ``storage.agent_record_dir``, for local files. ``task_files`` keeps the directories of task items, in the
        partition of each owner (``task_partition``); the caller owns it and closes it.
        """

    @abstractmethod
    def create(self, owner: str, items: Sequence[NewItem]) -> tuple[StoredItem, ...]: ...

    @abstractmethod
    def read(self, owner: str, item_id: str) -> StoredItem | None: ...

    @abstractmethod
    def query(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
    ) -> tuple[StoredItem, ...]: ...

    @abstractmethod
    def update(self, owner: str, item_ids: Sequence[str], patch: Mapping[str, Any]) -> tuple[StoredItem, ...]: ...

    @abstractmethod
    def delete(self, owner: str, item_id: str) -> bool: ...

    @abstractmethod
    def task_path(self, item: StoredItem) -> Path: ...

    @abstractmethod
    def close(self) -> None:
        """Release the store's resources. A second call does nothing."""


def check_query(owner: str, kind: object, scenario: object, after: object, limit: object) -> None:
    """Raise ``ValueError`` unless the arguments of ``DataStore.query`` are valid."""
    check_owner(owner)
    if kind is not None and kind not in ITEM_KINDS:
        raise ValueError(f"kind must be one of {', '.join(ITEM_KINDS)}, got {kind!r}")
    if scenario is not None:
        check_owner(scenario)
    if isinstance(after, bool) or not isinstance(after, int) or after < 0:
        raise ValueError("after must be a non-negative int")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise ValueError(f"limit must be an int from 1 to {MAX_QUERY_LIMIT}")


def prepared_items(
    owner: str, items: Sequence[NewItem], task_files: TaskFileStore
) -> list[tuple[NewItem, dict[str, Any], str]]:
    """Check a ``create`` call and return each item with its created metadata and digest.

    A task directory is copied into the owner's task file partition, so its digest names stored files. The copy
    happens before the store checks for conflicts; files are kept by digest, so a refused call leaves at most an
    unused copy. Raise ``ValueError`` for an invalid owner, ID, metadata or content, or for an ID given twice.
    """
    check_owner(owner)
    prepared: list[tuple[NewItem, dict[str, Any], str]] = []
    seen: set[str] = set()
    for item in items:
        check_item_id(item.item_id)
        if item.item_id in seen:
            raise ValueError(f"item {item.item_id!r} is given twice")
        seen.add(item.item_id)
        metadata = created_metadata(owner, item.metadata)
        if metadata["kind"] == "task":
            if not isinstance(item.content, Path) or not item.content.is_dir():
                raise ValueError(f"task item {item.item_id!r} needs a task directory as its content")
            if item.content.name != item.item_id:
                raise ValueError(f"task item {item.item_id!r} needs a task directory of the same name")
            digest = task_files.put(task_partition(owner), item.content)
        elif isinstance(item.content, Mapping):
            digest = json_digest(item.content)
        else:
            raise ValueError(f"{metadata['kind']} item {item.item_id!r} needs a JSON object as its content")
        prepared.append((item, metadata, digest))
    return prepared


class MemoryDataStore(DataStore):
    """A data store in memory, for tests and for a deployment that keeps nothing.

    It keeps the items of each owner in a dict from ID to item, and the next sequence. Task directories go to the
    task file store that it is given.
    """

    def __init__(self, task_files: TaskFileStore) -> None:
        self.task_files = task_files
        self.items: dict[str, dict[str, StoredItem]] = {}
        self.next_sequence = 1

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        if config:
            raise PluginConfigError(f"MemoryDataStore takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls(task_files)

    def create(self, owner: str, items: Sequence[NewItem]) -> tuple[StoredItem, ...]:
        prepared = prepared_items(owner, items, self.task_files)
        stored = self.items.setdefault(owner, {})
        for entry in prepared:
            existing = stored.get(entry[0].item_id)
            if existing is not None and existing.digest != entry[2]:
                raise ItemConflict(f"item {entry[0].item_id!r} of {owner!r} is stored with other content")
        result: list[StoredItem] = []
        for item, metadata, digest in prepared:
            existing = stored.get(item.item_id)
            if existing is not None:
                result.append(existing)
                continue
            content = copy.deepcopy(dict(item.content)) if isinstance(item.content, Mapping) else None
            created = StoredItem(self.next_sequence, owner, item.item_id, metadata, digest, content)
            self.next_sequence += 1
            stored[item.item_id] = created
            result.append(created)
        return tuple(result)

    def read(self, owner: str, item_id: str) -> StoredItem | None:
        check_owner(owner)
        return self.items.get(owner, {}).get(item_id)

    def query(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
    ) -> tuple[StoredItem, ...]:
        check_query(owner, kind, scenario, after, limit)
        matched = sorted(
            (
                item
                for item in self.items.get(owner, {}).values()
                if item.sequence > after
                and (kind is None or item.kind == kind)
                and (scenario is None or scenario in item.scenarios)
            ),
            key=lambda item: item.sequence,
        )
        return tuple(matched[:limit])

    def update(self, owner: str, item_ids: Sequence[str], patch: Mapping[str, Any]) -> tuple[StoredItem, ...]:
        check_owner(owner)
        stored = self.items.get(owner, {})
        missing = [item_id for item_id in item_ids if item_id not in stored]
        if missing:
            raise ItemNotFound(f"{owner!r} stores no item {missing[0]!r}")
        updated = [
            StoredItem(
                stored[item_id].sequence,
                owner,
                item_id,
                patched_metadata(stored[item_id].metadata, patch),
                stored[item_id].digest,
                stored[item_id].content,
            )
            for item_id in dict.fromkeys(item_ids)
        ]
        for item in updated:
            stored[item.item_id] = item
        return tuple(updated)

    def delete(self, owner: str, item_id: str) -> bool:
        check_owner(owner)
        return self.items.get(owner, {}).pop(item_id, None) is not None

    def task_path(self, item: StoredItem) -> Path:
        if item.kind != "task":
            raise ValueError(f"item {item.item_id!r} is a {item.kind}, not a task")
        return self.task_files.path(task_partition(item.owner), item.digest)

    def close(self) -> None:
        return None
