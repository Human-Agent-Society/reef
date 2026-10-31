"""Transport-independent handling of the data routes (RFC #1028).

A client creates, reads, queries, updates and deletes the data items of one owner. Each write that adds an item
to a scenario, or removes one, follows the same order:

1. Check the scenarios that the items are to use: each existing one must have the owner, and must be able to train
   on its new items (``Dispatcher.check_use``). A scenario that does not exist yet is checked when it opens.
2. Write the data store.
3. Ask control to sync each scenario that the write touched (``request_data_sync``), and wait up to
   ``SYNC_WAIT_SECONDS`` for the syncs to end. The response says whether they ended (``synced``); a sync that did
   not end runs when control takes it, or when the scenario opens.

No aiohttp types appear here. ``reef.service.routes.data`` adapts it to HTTP.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from harbor.models.task.task import Task as HarborTaskLoader

from reef.core.errors import ReefError
from reef.core.job_data import TrajectorySample
from reef.dispatcher import Dispatcher
from reef.service.tasks import unpack_task_archive
from reef.storage.data import (
    ITEM_KINDS,
    MAX_QUERY_LIMIT,
    DataStore,
    ItemKind,
    ItemNotFound,
    NewItem,
    StoredItem,
    check_item_id,
    check_owner,
    check_scenarios,
    created_metadata,
    patched_metadata,
)
from reef.storage.data_queue import DATA_SYNC_QUEUE
from reef.storage.data_sync import ScenarioUses, request_data_sync
from reef.storage.work_queue import WorkQueue

#: The most items in one create call: the most trajectories of one upload before RFC #1028.
MAX_ITEMS_PER_CREATE = 1000
#: The item kinds by name, so that a query parameter becomes an ``ItemKind``.
KINDS: dict[str, ItemKind] = {kind: kind for kind in ITEM_KINDS}
#: The longest time a write waits for control to sync the scenarios that it touched.
SYNC_WAIT_SECONDS = 30.0
#: The wait between two looks at the sync items.
SYNC_POLL_SECONDS = 0.05


class NoDataStore(ReefError):
    """The deployment has no data store, so it serves no data route."""


@dataclass(frozen=True)
class DataAccess:
    """What the data writes and reads use of a process.

    Values: ``store`` is the deployment's data store, or None for a deployment without one; ``queue`` is the work
    queue that carries the data syncs and transfers; ``scenarios`` checks the scenarios that the items name.

    Purpose: the data routes and the data service make one from their dispatcher (``of``); the functions of this
    module and of ``reef.service.data_transfer`` read it, and no other part of the dispatcher.
    """

    store: DataStore | None
    queue: WorkQueue
    scenarios: ScenarioUses

    @classmethod
    def of(cls, dispatcher: Dispatcher) -> DataAccess:
        return cls(dispatcher.data_store, dispatcher.queue, dispatcher)

    def require_store(self) -> DataStore:
        """The data store; ``NoDataStore`` for a deployment without one."""
        if self.store is None:
            raise NoDataStore("this deployment has no data store")
        return self.store


def item_json(item: StoredItem, *, content: bool) -> dict[str, Any]:
    """An item as JSON: ``{id, sequence, digest, metadata}``, and ``content`` when asked (null for a task)."""
    value: dict[str, Any] = {
        "id": item.item_id,
        "sequence": item.sequence,
        "digest": item.digest,
        "metadata": dict(item.metadata),
    }
    if content:
        value["content"] = None if item.content is None else dict(item.content)
    return value


@dataclass(frozen=True)
class Use:
    """One item that one scenario is to use in a split other than before.

    Values: ``item_id`` is the item's ID. ``trajectory`` is the item's content as a ``TrajectorySample`` for a
    trajectory, else None. ``train_task`` is True exactly for a task in the ``train`` split.

    Purpose: a write collects the new uses of each scenario, and ``check_uses`` checks them before the write.
    """

    item_id: str
    trajectory: TrajectorySample | None
    train_task: bool


def new_use(item_id: str, kind: str, content: Mapping[str, Any] | None, split: str) -> Use:
    trajectory = TrajectorySample(content) if kind == "trajectory" and content is not None else None
    return Use(item_id, trajectory, kind == "task" and split == "train")


def sync_touched(data: DataAccess, owner: str, scenarios: Iterable[str], *, wait: bool = True) -> bool:
    """Ask control to sync each scenario in ``scenarios`` with the items of ``owner``; return whether every sync
    ended, within ``SYNC_WAIT_SECONDS`` when ``wait``, or at once otherwise."""
    queue = data.queue
    pending = {request_data_sync(queue, owner, scenario).id for scenario in sorted(set(scenarios))}
    deadline = time.monotonic() + (SYNC_WAIT_SECONDS if wait else 0.0)
    while pending:
        # A sync item that is gone or final has ended.
        pending = {
            item_id
            for item_id in pending
            if (item := queue.get(DATA_SYNC_QUEUE, item_id)) is not None and not item.final
        }
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(SYNC_POLL_SECONDS)
    return not pending


def check_uses(data: DataAccess, owner: str, uses: Mapping[str, list[Use]]) -> None:
    """Check the new uses of each scenario, step 1 of every write.

    Raise ``ValueError`` for an existing scenario of another owner, or one that cannot train on its new items.
    """
    for scenario, scenario_uses in sorted(uses.items()):
        scenario_owner = data.scenarios.scenario_owner(scenario)
        if scenario_owner is None:
            continue
        if scenario_owner != owner:
            raise ValueError(f"scenario {scenario!r} has the owner {scenario_owner!r}, not {owner!r}")
        trajectories = [(use.item_id, use.trajectory) for use in scenario_uses if use.trajectory is not None]
        data.scenarios.check_use(scenario, trajectories, any(use.train_task for use in scenario_uses))


def create_items(data: DataAccess, owner: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Create record and artifact items from ``{"items": [{"id", "content", "metadata"}]}``; return them.

    1 to ``MAX_ITEMS_PER_CREATE`` items, whose content follows the rules of their kind
    (``reef.storage.item_kinds``). A trajectory comes as JSON Lines (``create_trajectory_items``), and a task needs its files
    (``create_task_items``), so this call refuses both. The response is ``{owner, items, synced}``, without content.
    """
    check_owner(owner)
    entries = body.get("items")
    if body.keys() != {"items"} or not isinstance(entries, list):
        raise ValueError("the body holds only 'items', an array of {id, content, metadata} objects")
    items: list[NewItem] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.keys() != {"id", "content", "metadata"}:
            raise ValueError("each item holds only 'id', 'content' and 'metadata'")
        item_id, content, metadata = check_item_id(entry["id"]), entry["content"], entry["metadata"]
        if not isinstance(content, Mapping):
            raise ValueError(f"item {item_id!r} needs a JSON object as its content")
        kind = created_metadata(owner, metadata)["kind"]
        if kind == "trajectory":
            raise ValueError("trajectories come as JSON Lines: upload them with POST .../data/trajectories")
        if kind == "task":
            raise ValueError("a task item needs its files: create it with POST .../data/tasks")
        items.append(NewItem(item_id, content, metadata))
    return store_items(data, owner, items)


def jsonl_trajectories(lines: Iterable[str], source: str) -> list[tuple[str, Mapping[str, Any]]]:
    """The ATIF documents of the JSON Lines ``lines``, in order, each with its item ID: its ``session_id``.

    A blank line is skipped. Raise ``ValueError``, naming ``source`` and the line, for a line that is not a JSON
    object or not an ATIF document, a document without a ``session_id`` that ``check_item_id`` takes, or a
    ``session_id`` that an earlier line has.
    """
    documents: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
            TrajectorySample(document)
            item_id = check_item_id(document.get("session_id"))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{source} line {number}: {exc}") from exc
        if item_id in seen:
            raise ValueError(f"{source} line {number}: session_id {item_id!r} is on an earlier line")
        seen.add(item_id)
        documents.append((item_id, document))
    return documents


def create_trajectory_items(
    data: DataAccess,
    owner: str,
    documents: Sequence[tuple[str, Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    *,
    wait: bool = True,
) -> dict[str, Any]:
    """Create a trajectory item for each ATIF document of ``documents`` (``jsonl_trajectories``), each with
    ``metadata``, whose ``kind`` is ``trajectory``; return ``{owner, items, synced}`` as ``store_items`` does."""
    check_owner(owner)
    if not isinstance(metadata, Mapping) or metadata.get("kind", "trajectory") != "trajectory":
        raise ValueError("the metadata of trajectory items is an object whose kind is trajectory")
    items = [NewItem(item_id, document, {**metadata, "kind": "trajectory"}) for item_id, document in documents]
    return store_items(data, owner, items, wait=wait)


def store_items(data: DataAccess, owner: str, items: Sequence[NewItem], *, wait: bool = True) -> dict[str, Any]:
    """Check and store 1 to ``MAX_ITEMS_PER_CREATE`` trajectory, record and artifact items of ``owner`` in one
    transaction, then sync the scenarios that they name; return ``{owner, items, synced}``, without content.

    Each item's metadata passes ``created_metadata``, and its content follows the rules of its kind
    (``reef.storage.item_kinds``), which the store checks. ``wait`` is as in ``sync_touched``.
    """
    if not 1 <= len(items) <= MAX_ITEMS_PER_CREATE:
        raise ValueError(f"a create call holds 1 to {MAX_ITEMS_PER_CREATE} items")
    uses: dict[str, list[Use]] = {}
    for item in items:
        checked = created_metadata(owner, item.metadata)
        content = cast(Mapping[str, Any], item.content)
        for scenario, split in checked["scenarios"].items():
            uses.setdefault(scenario, []).append(new_use(item.item_id, checked["kind"], content, split))
    check_uses(data, owner, uses)
    stored = data.require_store().create(owner, list(items))
    synced = sync_touched(data, owner, uses, wait=wait)
    return {"owner": owner, "items": [item_json(item, content=False) for item in stored], "synced": synced}


def create_task_items(data: DataAccess, owner: str, archive: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Create a task item for each Harbor task of the tar.gz ``archive``, each with ``metadata``; return them.

    Raise ``ValueError`` for an archive that ``unpack_task_archive`` refuses, and as ``create_task_directories``.
    """
    with tempfile.TemporaryDirectory(prefix="reef-task-items-") as work:
        return create_task_directories(data, owner, unpack_task_archive(archive, Path(work)), metadata)


def create_task_directories(
    data: DataAccess,
    owner: str,
    directories: Sequence[Path],
    metadata: Mapping[str, Any],
    *,
    wait: bool = True,
) -> dict[str, Any]:
    """Create a task item for each Harbor task directory of ``directories``, named after it, each with ``metadata``;
    return ``{owner, items, synced}`` as ``create_items`` does.

    ``metadata`` is the client's metadata for every task; its ``kind`` is ``task``. Raise ``ValueError`` for a
    directory that Harbor cannot load; nothing is stored then.
    """
    check_owner(owner)
    if not isinstance(metadata, Mapping) or metadata.get("kind", "task") != "task":
        raise ValueError("the metadata of task items is an object whose kind is task")
    scenarios = check_scenarios(metadata.get("scenarios", {}), "task")
    for directory in directories:
        try:
            HarborTaskLoader(directory)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{directory.name} is not a Harbor task: {exc}") from exc
    new_uses = {
        scenario: [new_use(directory.name, "task", None, split) for directory in directories]
        for scenario, split in scenarios.items()
    }
    check_uses(data, owner, new_uses)
    stored = data.require_store().create(
        owner, [NewItem(directory.name, directory, {**metadata, "kind": "task"}) for directory in directories]
    )
    synced = sync_touched(data, owner, scenarios, wait=wait)
    return {"owner": owner, "items": [item_json(item, content=False) for item in stored], "synced": synced}


def query_items(
    data: DataAccess, owner: str, *, kind: str | None, scenario: str | None, after: int, limit: int
) -> dict[str, Any]:
    """The items of ``owner`` after ``after``, at most ``limit``: ``{owner, items, next}``, without content.

    ``next`` is the ``after`` of the next page, or null after the last item.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(ITEM_KINDS)}, got {kind!r}")
    checked_kind = None if kind is None else KINDS[kind]
    found = data.require_store().query(owner, kind=checked_kind, scenario=scenario, after=after, limit=limit)
    return {
        "owner": owner,
        "items": [item_json(item, content=False) for item in found],
        "next": found[-1].sequence if len(found) == limit else None,
    }


def read_item(data: DataAccess, owner: str, item_id: str) -> dict[str, Any]:
    """One item of ``owner`` with its content; ``ItemNotFound`` for an unknown ID."""
    item = data.require_store().read(owner, item_id)
    if item is None:
        raise ItemNotFound(f"{owner!r} stores no item {item_id!r}")
    return {"owner": owner, **item_json(item, content=True)}


def update_items(data: DataAccess, owner: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ``{"ids": [...], "patch": {...}}``, one JSON merge patch on each item's metadata; return the items.

    The scenarios that the patch adds to an item, or moves to another split, are checked first. Raise
    ``ItemNotFound`` for an unknown ID and ``ValueError`` for a patch that ``patched_metadata`` refuses.
    """
    check_owner(owner)
    ids, patch = body.get("ids"), body.get("patch")
    if body.keys() != {"ids", "patch"} or not isinstance(ids, list) or not isinstance(patch, Mapping):
        raise ValueError("the body holds only 'ids', an array of item IDs, and 'patch', a JSON merge patch")
    if not 1 <= len(ids) <= MAX_QUERY_LIMIT:
        raise ValueError(f"an update names 1 to {MAX_QUERY_LIMIT} items")
    store = data.require_store()
    current = touched_items(store, owner, ids)
    uses: dict[str, list[Use]] = {}
    touched: set[str] = set()
    for item in current:
        after = patched_metadata(item.metadata, patch)["scenarios"]
        touched.update(item.scenarios, after)
        for scenario, split in after.items():
            if item.scenarios.get(scenario) != split:
                uses.setdefault(scenario, []).append(new_use(item.item_id, item.kind, item.content, split))
    check_uses(data, owner, uses)
    updated = store.update(owner, ids, patch)
    synced = sync_touched(data, owner, touched)
    return {"owner": owner, "items": [item_json(item, content=False) for item in updated], "synced": synced}


def delete_item(data: DataAccess, owner: str, item_id: str) -> dict[str, Any]:
    """Delete one item of ``owner``; the scenarios that used it lose its upload. ``ItemNotFound`` if unknown.

    Raise ``ValueError`` for the artifact item of an artifact that a scenario of ``owner`` made while the scenario
    exists: the next artifact copy would make the item again (``reef.storage.artifact_items``).
    """
    store = data.require_store()
    (item,) = touched_items(store, owner, [item_id])
    if item.kind == "artifact" and "origin" in item.metadata:
        made_by = item.metadata["origin"]["scenario"]
        if data.scenarios.scenario_owner(made_by) == owner:
            raise ValueError(
                f"artifact item {item_id!r} lists an artifact of scenario {made_by!r}, which still exists"
            )
    store.delete(owner, item_id)
    synced = sync_touched(data, owner, item.scenarios)
    return {"owner": owner, "id": item_id, "deleted": True, "synced": synced}


def touched_items(store: DataStore, owner: str, ids: Sequence[object]) -> list[StoredItem]:
    """The stored items of ``ids``, in the order given; ``ValueError`` for a bad ID, ``ItemNotFound`` if unknown."""
    found: list[StoredItem] = []
    for item_id in dict.fromkeys(ids):
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("an item ID is a non-empty string")
        item = store.read(owner, item_id)
        if item is None:
            raise ItemNotFound(f"{owner!r} stores no item {item_id!r}")
        found.append(item)
    return found


def parsed_metadata(text: str | None) -> Mapping[str, Any]:
    """The JSON object of the ``metadata`` query parameter of a task create call; ``{}`` when it is not given."""
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata is not JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object")
    return value


__all__ = [
    "MAX_ITEMS_PER_CREATE",
    "SYNC_WAIT_SECONDS",
    "DataAccess",
    "NoDataStore",
    "create_items",
    "create_task_directories",
    "create_task_items",
    "create_trajectory_items",
    "delete_item",
    "item_json",
    "jsonl_trajectories",
    "parsed_metadata",
    "query_items",
    "read_item",
    "store_items",
    "update_items",
]
