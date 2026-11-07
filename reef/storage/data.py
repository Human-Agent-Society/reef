"""Storage contract for the data store of a deployment (RFC #1028).

The data store keeps the data that scenarios train on, once per deployment. An item is content and metadata.
Its content never changes; its metadata says what the item is, which owner it belongs to, and which scenarios
use it. The store has create, read, query, update and delete, and each operation reads or writes the items of
one owner only. Training state, such as train counts, is not data and stays in the scenario
storage.

This module defines the item values, the metadata rules, the ``DataStore`` plugin base and an in-memory store.
``reef.storage.sql_data`` implements the store in a SQL database.
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self, get_args

from reef.core.errors import ReefError
from reef.core.job_data import JobData
from reef.core.plugins import Plugin, PluginConfigError
from reef.core.records_types import AgentRecord, RequestType, stored_record_artifact_id
from reef.core.trajectory_index import TrajectoryIndex
from reef.storage.item_kinds import ITEM_KIND_RULES, task_partition
from reef.storage.job_data import JOB_DATA_PREFIX, JOB_FILE, JobDataRef, job_data_key, job_data_kind, write_job_data
from reef.storage.segments import LocalSegmentStore, SegmentStore
from reef.storage.tasks import SPLIT_NAMES, TaskFileStore

ItemKind = Literal["trajectory", "task", "record", "artifact"]
ITEM_KINDS: tuple[ItemKind, ...] = get_args(ItemKind)
#: The metadata keys that Reef reads. ``kind``, ``owner`` and ``origin`` are set at creation and never change.
FIXED_KEYS = ("kind", "owner", "origin")
#: The most items that one query returns.
MAX_QUERY_LIMIT = 1000


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


def check_scenarios(value: object, kind: ItemKind) -> dict[str, str]:
    """Return ``value`` as the ``scenarios`` metadata of an item of ``kind``, or raise ``ValueError``.

    ``scenarios`` maps each scenario name (the rules of ``check_owner``) to a split, ``train`` or ``eval``. A
    trajectory and a record are always ``train``. An artifact has no ``scenarios``: a scenario uses an artifact
    item by importing it, so its ``scenarios`` is always empty.
    """
    if not isinstance(value, Mapping):
        raise ValueError("metadata.scenarios must be an object from scenario name to split")
    if kind == "artifact" and value:
        raise ValueError("a scenario uses an artifact item by importing it, so metadata.scenarios is empty")
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
    """Return ``{scenario, artifact_id}``.

    ``scenario`` is a valid scenario name. ``artifact_id`` is a non-empty string, or null when a record names no
    artifact. Raise ``ValueError`` for other shapes.
    """
    if not isinstance(value, Mapping) or set(value) != {"scenario", "artifact_id"}:
        raise ValueError("metadata.origin must be {scenario, artifact_id}")
    check_owner(value["scenario"])
    if value["artifact_id"] is not None and not (isinstance(value["artifact_id"], str) and value["artifact_id"]):
        raise ValueError("metadata.origin.artifact_id must be a non-empty string or null")
    return {key: value[key] for key in ("scenario", "artifact_id")}


def created_metadata(owner: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """The metadata that an item of ``owner`` is created with, from the client's ``metadata``.

    ``kind`` is required. ``owner``, when given, must be ``owner``; it is set when it is missing. ``scenarios``
    defaults to ``{}``. ``origin`` is required for a record; an artifact may have one, with a non-null
    ``artifact_id``, when a scenario made it (``reef.storage.artifact_items``); a trajectory or a task may have one
    when a trainer's stage made it (``check_stage_origin``). Other keys are kept as they are. Raise ``ValueError`` for metadata that breaks these rules.
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
    elif kind == "artifact" and "origin" in metadata:
        origin = check_origin(metadata["origin"])
        if origin["artifact_id"] is None:
            raise ValueError("an artifact item's metadata.origin names its artifact_id")
        created["origin"] = origin
    elif "origin" in metadata:
        created["origin"] = check_stage_origin(metadata["origin"])
    return created


def check_stage_origin(value: object) -> dict[str, Any]:
    """Return ``value`` as the ``origin`` of a trajectory or a task that a trainer's stage made, or raise
    ``ValueError``: ``{scenario, component, records}``, the scenario (the rules of ``check_owner``), the trainer's
    component (``""`` for the only trainer of a scenario) and the IDs of the records that the sample came from."""
    if not isinstance(value, Mapping) or set(value) != {"scenario", "component", "records"}:
        raise ValueError("a stage sample's metadata.origin must be {scenario, component, records}")
    check_owner(value["scenario"])
    records = value["records"]
    if (
        not isinstance(value["component"], str)
        or not isinstance(records, list)
        or not all(isinstance(record_id, str) and record_id for record_id in records)
    ):
        raise ValueError("metadata.origin.component is a string and metadata.origin.records a list of record IDs")
    return {"scenario": value["scenario"], "component": value["component"], "records": list(records)}


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

    The patch may not change ``kind``, ``owner`` or ``origin``, or an ``artifact_id`` that the item has already. The patched ``scenarios`` must be valid for the
    item's kind; removing ``scenarios`` leaves it empty. A record item that its origin scenario uses stays in it:
    that scenario's record store keeps the record.
    """
    if not isinstance(patch, Mapping):
        raise ValueError("a metadata patch must be an object")
    patched = merge_patch(metadata, patch)
    for key in FIXED_KEYS:
        if patched.get(key) != metadata.get(key):
            raise ValueError(f"metadata.{key} is set at creation and never changes")
    if metadata.get("artifact_id") not in (None, patched.get("artifact_id")):
        raise ValueError("metadata.artifact_id is set once and never changes")
    patched["scenarios"] = check_scenarios(patched.get("scenarios", {}), metadata["kind"])
    if metadata["kind"] == "record":
        origin = metadata["origin"]["scenario"]
        if origin in metadata["scenarios"] and origin not in patched["scenarios"]:
            raise ValueError(f"a record item stays in {origin!r}, the scenario that made it")
    return patched


@dataclass(frozen=True)
class NewItem:
    """An item that a client creates.

    Values: ``item_id`` passes ``check_item_id``. ``content`` is a JSON object for a trajectory, a record or an artifact, and
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
    SHA-256 of the content: ``json_digest`` of the JSON content, without ``created_at`` for a record
    (``record_digest``), or the task file store's digest of a task directory. A trajectory, a record or an artifact has its JSON ``content``, except in the result of a ``query``
    without content. A task has no ``content``: the task file store
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


#: A metadata key that ``DataStore.groups`` can group by: 1 to 64 letters, digits and underscores.
GROUP_KEY = re.compile(r"[A-Za-z0-9_]{1,64}")


@dataclass(frozen=True)
class ItemGroup:
    """The items of one owner and kind that have one value under one metadata key.

    Values: ``value`` is the string under the key, or None for the items whose metadata has no string there.
    ``item_count`` is a positive int. ``first_sequence`` and ``last_sequence`` are the smallest and the largest
    sequence of the group's items, so ``first_sequence <= last_sequence``. ``metadata`` is the metadata of the item at
    ``first_sequence``.

    Purpose: ``DataStore.groups`` returns it. A client lists the sets of items that an owner uploaded, such as the
    trajectories of one file (``set_id``), without reading each item.
    """

    value: str | None
    item_count: int
    first_sequence: int
    last_sequence: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.value is not None and not isinstance(self.value, str):
            raise ValueError("a group's value is a string or None")
        for name in ("item_count", "first_sequence", "last_sequence"):
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"a group's {name} is a positive int")
        if self.first_sequence > self.last_sequence:
            raise ValueError("a group's first_sequence is at most its last_sequence")


@dataclass(frozen=True)
class RecordLink:
    """The place of one record item in its owner's trajectory tree (RFC #1054).

    Values: ``parent`` is None or a non-empty item ID: the item of the nearest ancestor of the item's trajectory node
    whose item the owner still stores, None when there is none. ``entry_count`` is a non-negative int, the number of
    ATIF entries of the item's request (``CallKey.entry_count``). An ancestor's ``entry_count`` is at most its
    descendant's; it is equal when the descendant retries the ancestor's request.

    Purpose: ``DataStore.record_links`` returns it. A reader of record items forms each trajectory from the links:
    an end is a call that no stored call names as ``parent``, and its calls are
    ``reef.core.trajectory_index.trajectory_calls`` of the path of ``(item ID, entry_count)`` from the root to it.
    """

    parent: str | None
    entry_count: int

    def __post_init__(self) -> None:
        if self.parent is not None and (not isinstance(self.parent, str) or not self.parent):
            raise ValueError("a record link's parent is None or a non-empty item ID")
        if isinstance(self.entry_count, bool) or not isinstance(self.entry_count, int) or self.entry_count < 0:
            raise ValueError("a record link's entry_count is a non-negative int")


#: How long a data store keeps a job data directory that no job used: 7 days.
JOB_DIRECTORY_TTL_SECONDS = 7 * 24 * 3600.0


class DataStore(Plugin, ABC):
    """The data of a deployment: items of content and metadata, each in one owner.

    Values: the state of a store is a finite set of ``StoredItem`` values. Within one owner, no two items have
    the same ``item_id``, and ``sequence`` increases in the order the items arrived. An item's content,
    ``digest`` and fixed metadata (``kind``, ``owner``, ``origin``) never change. Every operation names one owner
    and reads or writes only the items of that owner.

    Operations:

    * ``create(owner, items)`` stores each item, all or none. An item whose ID the owner stores with the same
      digest is unchanged; with another digest it is an ``ItemConflict``, and nothing is stored. Bad metadata or
      content is a ``ValueError``, and nothing is stored. New artifact items omit the obsolete ``content_id``.
      A retry of an older artifact compares all other content fields and preserves the stored content and digest.
      It returns the stored items in the order given, without their content.
    * ``read(owner, item_id)`` returns the item, or None.
    * ``query(owner, kind, scenario, after, limit, content)`` returns the items of ``owner`` with
      ``sequence > after``, in sequence order, at most ``limit``, of ``kind`` and used by ``scenario`` when those
      are given. Without ``content``, each item's ``content`` is None, and the store does not read it.
    * ``scenario_items(owner, scenario, kinds, after, limit)`` returns the items of ``kinds`` of ``owner`` that
      ``scenario`` uses, with content, each with the sequence of its use, in the order of those sequences: the uses
      after ``after``, at most ``limit``. The store reads no item of another kind; without ``content``, each item's
      ``content`` is None, and the store does not read it. Each use of an item by a scenario
      has its own sequence, which increases across the store: a new use, also of an older item, and a use whose
      split changes get a new one. A trainer reads its scenario's items with
      it. With ``excluding_origin``, the record items that this scenario made are left out.
    * ``last_use_sequence(owner, scenario)`` returns the sequence of the newest use by ``scenario``, or 0.
    * ``scenarios_with_records(owner)`` returns each scenario that uses a record item of ``owner``, also a deleted
      one, because a record item keeps its origin. A durable reader of record items keeps one cursor for each: a
      ``sequence`` orders the items of one scenario, and it does not order the items of several (RFC #1219).
    * ``update(owner, item_ids, patch)`` applies one JSON merge patch to the metadata of each item, all or none
      (``patched_metadata``). An unknown ID is an ``ItemNotFound``. It returns the updated items.
    * ``delete(owner, item_id)`` removes the item and returns True, or returns False for an unknown ID.
    * ``task_path(item)`` returns a local directory with the files of a task item.
    * ``record_links(owner, item_ids)`` maps each ID of ``item_ids`` that names a stored record item of ``owner``
      with a message list to its ``RecordLink``. The store links such an item into a trajectory when it creates
      it, by the rule of ``reef.core.trajectory_index`` among the earlier record items of the same origin scenario,
      and the link never changes; a deleted ancestor is skipped. Other IDs are left out.
    * ``groups(owner, kind, key, before, limit)`` groups the items of ``owner`` and ``kind`` by the string under
      ``key`` in their metadata (``ItemGroup``). It returns the groups whose ``first_sequence`` is below ``before``,
      when given, newest first, at most ``limit``.
    * ``maintain()`` does the store's periodic work: it deletes the job data directories that no job used for
      ``job_directory_ttl_seconds``, and a subclass can do more, such as moving content into segments. It changes
      no item. The data service calls it every few minutes.

    The store also keeps the data directory of each training job (``reef.storage.job_data``), one directory for each
    content, in the segment store ``job_directories``:

    * ``write_job_directory(data)`` writes the directory of ``data`` if the store does not hold it, records its use,
      and returns its reference.
    * ``open_job_directory(value)`` gives the reference that ``JobDataRef.to_dict`` wrote, with a local copy of the
      directory, and records its use.
    * ``delete_unused_job_directories(time)`` deletes each directory that no job used since ``time``, and returns
      their number.

    A subclass sets ``job_directories`` and ``job_directory_ttl_seconds``, and keeps the time of the last use of each
    directory, by its key:
    ``record_job_directory_use(key, time)``, ``job_directories_unused_since(time)`` and
    ``delete_job_directory_record(key)``.

    The store checks the form of the metadata. It does not know the scenarios, so the service checks that
    ``scenarios`` names only scenarios of the owner.

    Purpose: a deployment names a subclass in ``storage.data_store``. The service builds it at startup and closes
    it at shutdown.
    """

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        directory: Path,
        task_files: TaskFileStore,
        segments: SegmentStore | None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> Self:
        """Build the store a deployment names.

        ``config`` is the plugin's own configuration. ``directory`` is the deployment's
        ``storage.agent_record_dir``, for local files. ``task_files`` keeps the directories of task items, in the
        partition of each owner (``task_partition``); the caller owns it and closes it. ``segments`` is the
        ``storage.segment_backend`` store, or None when it is unset; a store can keep large content there, in the
        partition of each owner. The data store owns it and closes it with itself. ``job_directory_ttl_seconds`` is
        ``storage.job_directory_ttl_s``.
        """

    @abstractmethod
    def create(self, owner: str, items: Sequence[NewItem]) -> tuple[StoredItem, ...]: ...

    @abstractmethod
    def read(self, owner: str, item_id: str) -> StoredItem | None: ...

    @abstractmethod
    def read_many(self, owner: str, item_ids: Sequence[str]) -> tuple[StoredItem, ...]:
        """The items of ``owner`` that ``item_ids`` name, with their content, each once, in the order of their first
        ID in ``item_ids``. An ID that names no item gives none. A store reads them together, so that a write that
        names many items does not read each one alone."""

    @abstractmethod
    def query(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
        content: bool = True,
    ) -> tuple[StoredItem, ...]: ...

    @abstractmethod
    def scenario_items(
        self,
        owner: str,
        scenario: str,
        *,
        kinds: Sequence[str],
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
        content: bool = True,
        excluding_origin: str | None = None,
    ) -> tuple[tuple[int, StoredItem], ...]: ...

    @abstractmethod
    def last_use_sequence(self, owner: str, scenario: str) -> int: ...

    @abstractmethod
    def scenarios_with_records(self, owner: str) -> tuple[str, ...]: ...

    @abstractmethod
    def update(self, owner: str, item_ids: Sequence[str], patch: Mapping[str, Any]) -> tuple[StoredItem, ...]: ...

    @abstractmethod
    def delete(self, owner: str, item_id: str) -> bool: ...

    @abstractmethod
    def task_path(self, item: StoredItem) -> Path: ...

    @abstractmethod
    def record_links(self, owner: str, item_ids: Sequence[str]) -> dict[str, RecordLink]: ...

    def groups(
        self,
        owner: str,
        *,
        kind: ItemKind,
        key: str,
        scenario: str | None = None,
        before: int | None = None,
        limit: int = MAX_QUERY_LIMIT,
    ) -> tuple[ItemGroup, ...]:
        """The groups of the items of ``owner`` and ``kind`` by ``key``, only of the items that ``scenario`` uses if it
        is given. The default reads every page of ``query`` without content; a store can read the groups at once."""
        check_groups(owner, kind, key, scenario, before, limit)
        found: dict[str | None, ItemGroup] = {}
        after = 0
        while page := self.query(owner, kind=kind, scenario=scenario, after=after, content=False):
            for item in page:
                value = item.metadata.get(key)
                value = value if isinstance(value, str) else None
                group = found.get(value)
                found[value] = (
                    ItemGroup(value, 1, item.sequence, item.sequence, item.metadata)
                    if group is None
                    else replace(group, item_count=group.item_count + 1, last_sequence=item.sequence)
                )
            after = page[-1].sequence
        newest = sorted(found.values(), key=lambda group: group.first_sequence, reverse=True)
        return tuple(group for group in newest if before is None or group.first_sequence < before)[:limit]

    def maintain(self) -> None:
        """Do the store's periodic work: delete the job data directories that no job used for the time to live."""
        self.delete_unused_job_directories(time.time() - self.job_directory_ttl_seconds)

    #: The segment store of the job data directories.
    job_directories: SegmentStore
    #: How long the store keeps a job data directory that no job used, in seconds.
    job_directory_ttl_seconds: float

    @abstractmethod
    def record_job_directory_use(self, key: str, time: float) -> None: ...

    @abstractmethod
    def job_directories_unused_since(self, time: float) -> tuple[str, ...]: ...

    @abstractmethod
    def delete_job_directory_record(self, key: str) -> None: ...

    def write_job_directory(self, data: JobData) -> JobDataRef:
        key = job_data_key(data)
        name = f"{JOB_DATA_PREFIX}/{key}"
        uri = self.job_directories.find_directory(name)
        if uri is None:
            with tempfile.TemporaryDirectory() as staging:
                directory = Path(staging) / key
                write_job_data(data, directory)
                uri = self.job_directories.put_directory(name, directory)
        self.record_job_directory_use(key, time.time())
        return JobDataRef(
            uri, self.job_directories.open_directory(uri), data.data_id, len(data.samples), job_data_kind(data)
        )

    def open_job_directory(self, value: Mapping[str, Any]) -> JobDataRef:
        uri = value.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError("a job data reference has a uri")
        path = self.job_directories.open_directory(uri)
        header = json.loads((path / JOB_FILE).read_text(encoding="utf-8"))
        self.record_job_directory_use(header["key"], time.time())
        return JobDataRef.from_dict(value, path)

    def delete_unused_job_directories(self, time: float) -> int:
        unused = self.job_directories_unused_since(time)
        for key in unused:
            self.job_directories.delete_directory(f"{JOB_DATA_PREFIX}/{key}")
            self.delete_job_directory_record(key)
        return len(unused)

    @abstractmethod
    def close(self) -> None:
        """Release the store's resources. A second call does nothing."""


def check_groups(owner: str, kind: object, key: object, scenario: object, before: object, limit: object) -> None:
    """Raise ``ValueError`` unless the arguments of ``DataStore.groups`` are valid."""
    check_query(owner, kind, scenario, 0, limit)
    if kind is None:
        raise ValueError("groups need a kind")
    if not isinstance(key, str) or not GROUP_KEY.fullmatch(key):
        raise ValueError("key must be 1 to 64 letters, digits and underscores")
    if before is not None and (isinstance(before, bool) or not isinstance(before, int) or before < 1):
        raise ValueError("before must be a positive int")


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


def item_digest_matches(item: StoredItem, digest: str) -> bool:
    """Whether the stored ``item`` has the content of a create with ``digest``.

    A stored item in an older format matches the current content that it maps to
    (``ItemKindRules.matches_stored``), so a retry never rewrites an immutable item.
    """
    return item.digest == digest or ITEM_KIND_RULES[item.metadata["kind"]].matches_stored(item.content, digest)


def prepared_items(
    owner: str, items: Sequence[NewItem], task_files: TaskFileStore
) -> list[tuple[NewItem, dict[str, Any], str]]:
    """Check a ``create`` call and return each item with its created metadata and digest.

    Each item's content follows the rules of its kind (``ITEM_KIND_RULES``), in the current format
    (``ItemKindRules.current_content``). A task directory is copied into the
    owner's task file partition, so its digest names stored files. The copy happens before the store checks for
    conflicts; files are kept by digest, so a refused call leaves at most an unused copy. Raise ``ValueError`` for
    an invalid owner, ID, metadata or content, or for an ID given twice.
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
        rules = ITEM_KIND_RULES[metadata["kind"]]
        item = NewItem(item.item_id, rules.current_content(item.content), item.metadata)
        prepared.append((item, metadata, rules.stored_digest(owner, item.item_id, item.content, metadata, task_files)))
    return prepared


class MemoryDataStore(DataStore):
    """A data store in memory, for tests and for a deployment that keeps nothing.

    It keeps the items of each owner in a dict from ID to item, the use sequence of each item, scenario and split
    (``uses``), and the next sequence, which items and uses share. Task directories go to the task file store that it
    is given. ``trajectories`` links the record items of each owner and origin scenario as
    they are created; a deleted item stays there as a link of later calls.
    """

    def __init__(
        self, task_files: TaskFileStore, job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS
    ) -> None:
        self.task_files = task_files
        self.job_directories = LocalSegmentStore(Path(tempfile.mkdtemp(prefix="reef-job-directories-")))
        self.job_directory_ttl_seconds = job_directory_ttl_seconds
        self.job_directory_uses: dict[str, float] = {}
        self.items: dict[str, dict[str, StoredItem]] = {}
        self.trajectories: dict[tuple[str, str], TrajectoryIndex] = {}
        self.uses: dict[tuple[str, str, str], tuple[str, int]] = {}
        self.next_sequence = 1

    def take_sequence(self) -> int:
        """The next sequence of an item or a use."""
        self.next_sequence += 1
        return self.next_sequence - 1

    def write_uses(self, owner: str, item_id: str, scenarios: Mapping[str, str]) -> None:
        """Keep the use of each unchanged scenario and split, and give each new one its sequence."""
        for key in [key for key in self.uses if key[:2] == (owner, item_id) and key[2] not in scenarios]:
            self.uses.pop(key)
        for scenario, split in scenarios.items():
            used = self.uses.get((owner, item_id, scenario))
            if used is None or used[0] != split:
                self.uses[(owner, item_id, scenario)] = (split, self.take_sequence())

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        directory: Path,
        task_files: TaskFileStore,
        segments: SegmentStore | None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> Self:
        if config:
            raise PluginConfigError(f"MemoryDataStore takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls(task_files, job_directory_ttl_seconds)

    def record_job_directory_use(self, key: str, time: float) -> None:
        self.job_directory_uses[key] = time

    def job_directories_unused_since(self, time: float) -> tuple[str, ...]:
        return tuple(key for key, last_used in self.job_directory_uses.items() if last_used < time)

    def delete_job_directory_record(self, key: str) -> None:
        self.job_directory_uses.pop(key, None)

    def create(self, owner: str, items: Sequence[NewItem]) -> tuple[StoredItem, ...]:
        prepared = prepared_items(owner, items, self.task_files)
        stored = self.items.setdefault(owner, {})
        for entry in prepared:
            existing = stored.get(entry[0].item_id)
            if existing is not None and not item_digest_matches(existing, entry[2]):
                raise ItemConflict(f"item {entry[0].item_id!r} of {owner!r} is stored with other content")
        result: list[StoredItem] = []
        for item, metadata, digest in prepared:
            existing = stored.get(item.item_id)
            if existing is not None:
                result.append(existing)
                continue
            content = copy.deepcopy(dict(item.content)) if isinstance(item.content, Mapping) else None
            created = StoredItem(self.take_sequence(), owner, item.item_id, metadata, digest, content)
            stored[item.item_id] = created
            self.write_uses(owner, item.item_id, metadata["scenarios"])
            if content is not None and metadata["kind"] == "record" and content["request_type"] == "inference":
                scenario = metadata["origin"]["scenario"]
                self.trajectories.setdefault((owner, scenario), TrajectoryIndex()).add(
                    AgentRecord.create(
                        scenario=scenario,
                        request_type=RequestType.INFERENCE,
                        payload=content["payload"],
                        agent_record_id=item.item_id,
                        artifact_id=stored_record_artifact_id(content, content["payload"]),
                    )
                )
            result.append(created)
        return tuple(replace(item, content=None) for item in result)

    def read(self, owner: str, item_id: str) -> StoredItem | None:
        check_owner(owner)
        return self.items.get(owner, {}).get(item_id)

    def read_many(self, owner: str, item_ids: Sequence[str]) -> tuple[StoredItem, ...]:
        check_owner(owner)
        stored = self.items.get(owner, {})
        return tuple(stored[item_id] for item_id in dict.fromkeys(item_ids) if item_id in stored)

    def query(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
        content: bool = True,
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
        return tuple(item if content else replace(item, content=None) for item in matched[:limit])

    def scenario_items(
        self,
        owner: str,
        scenario: str,
        *,
        kinds: Sequence[str],
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
        content: bool = True,
        excluding_origin: str | None = None,
    ) -> tuple[tuple[int, StoredItem], ...]:
        check_query(owner, None, scenario, after, limit)
        stored = self.items.get(owner, {})
        used = sorted(
            (sequence, item_id)
            for (use_owner, item_id, use_scenario), (_, sequence) in self.uses.items()
            if use_owner == owner
            and use_scenario == scenario
            and sequence > after
            and stored[item_id].kind in kinds
            and (excluding_origin is None or not item_id.startswith(f"{excluding_origin}/"))
        )
        return tuple(
            (sequence, stored[item_id] if content else replace(stored[item_id], content=None))
            for sequence, item_id in used[:limit]
        )

    def scenarios_with_records(self, owner: str) -> tuple[str, ...]:
        check_owner(owner)
        stored = self.items.get(owner, {})
        return tuple(
            sorted(
                {
                    use_scenario
                    for (use_owner, item_id, use_scenario) in self.uses
                    if use_owner == owner and stored[item_id].kind == "record"
                }
            )
        )

    def last_use_sequence(self, owner: str, scenario: str) -> int:
        return max(
            (
                sequence
                for (use_owner, _, use_scenario), (_, sequence) in self.uses.items()
                if use_owner == owner and use_scenario == scenario
            ),
            default=0,
        )

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
            self.write_uses(owner, item.item_id, item.scenarios)
        return tuple(updated)

    def delete(self, owner: str, item_id: str) -> bool:
        check_owner(owner)
        self.write_uses(owner, item_id, {})
        return self.items.get(owner, {}).pop(item_id, None) is not None

    def task_path(self, item: StoredItem) -> Path:
        if item.kind != "task":
            raise ValueError(f"item {item.item_id!r} is a {item.kind}, not a task")
        return self.task_files.path(task_partition(item.owner), item.digest)

    def record_links(self, owner: str, item_ids: Sequence[str]) -> dict[str, RecordLink]:
        check_owner(owner)
        stored = self.items.get(owner, {})
        links: dict[str, RecordLink] = {}
        for item_id in item_ids:
            item = stored.get(item_id)
            if item is None or item.kind != "record":
                continue
            index = self.trajectories.get((owner, item.metadata["origin"]["scenario"]))
            call = None if index is None else index.calls.get(item_id)
            if index is None or call is None:
                continue
            parent = call.parent
            while parent is not None and parent not in stored:
                parent = index.calls[parent].parent
            links[item_id] = RecordLink(parent, call.entry_count)
        return links

    def close(self) -> None:
        return None
