"""Transport-independent handling of the data routes (RFC #1028).

A client creates, reads, queries, updates and deletes the data items of one owner. Each write that adds an item
to a scenario, or removes one, follows the same order:

1. Check the scenarios that the items are to use: each existing one must have the owner, and must be able to train
   on its new items (``Dispatcher.check_use``). A scenario that does not exist yet is checked when it opens.
2. Write the data store.
3. Sync the uploads of each loaded scenario that the write touched (``Dispatcher.sync_items``).

No aiohttp types appear here. ``reef.service.routes.data`` adapts it to HTTP.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

#: The most items in one create call: the most trajectories of one upload before RFC #1028.
MAX_ITEMS_PER_CREATE = 1000
#: The item kinds by name, so that a query parameter becomes an ``ItemKind``.
KINDS: dict[str, ItemKind] = {kind: kind for kind in ITEM_KINDS}


class NoDataStore(ReefError):
    """The deployment has no data store, so it serves no data route."""


def data_store(dispatcher: Dispatcher) -> DataStore:
    if dispatcher.data_store is None:
        raise NoDataStore("this deployment has no data store")
    return dispatcher.data_store


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


def check_uses(dispatcher: Dispatcher, owner: str, uses: Mapping[str, list[Use]]) -> None:
    """Check the new uses of each scenario, step 1 of every write.

    Raise ``ValueError`` for an existing scenario of another owner, or one that cannot train on its new items.
    """
    for scenario, scenario_uses in sorted(uses.items()):
        scenario_owner = dispatcher.scenario_owner(scenario)
        if scenario_owner is None:
            continue
        if scenario_owner != owner:
            raise ValueError(f"scenario {scenario!r} has the owner {scenario_owner!r}, not {owner!r}")
        trajectories = [(use.item_id, use.trajectory) for use in scenario_uses if use.trajectory is not None]
        dispatcher.check_use(scenario, trajectories, any(use.train_task for use in scenario_uses))


def create_items(dispatcher: Dispatcher, owner: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Create trajectory and record items from ``{"items": [{"id", "content", "metadata"}]}``; return them.

    1 to ``MAX_ITEMS_PER_CREATE`` items. A trajectory's content must be an ATIF document. A task needs its files,
    so ``create_task_items`` creates it. The response is ``{owner, items}``, without content.
    """
    check_owner(owner)
    entries = body.get("items")
    if body.keys() != {"items"} or not isinstance(entries, list):
        raise ValueError("the body holds only 'items', an array of {id, content, metadata} objects")
    if not 1 <= len(entries) <= MAX_ITEMS_PER_CREATE:
        raise ValueError(f"a create call holds 1 to {MAX_ITEMS_PER_CREATE} items")
    items: list[NewItem] = []
    uses: dict[str, list[Use]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.keys() != {"id", "content", "metadata"}:
            raise ValueError("each item holds only 'id', 'content' and 'metadata'")
        item_id, content, metadata = check_item_id(entry["id"]), entry["content"], entry["metadata"]
        if not isinstance(content, Mapping):
            raise ValueError(f"item {item_id!r} needs a JSON object as its content")
        checked = created_metadata(owner, metadata)
        if checked["kind"] == "task":
            raise ValueError("a task item needs its files: create it with POST .../data/tasks")
        for scenario, split in checked["scenarios"].items():
            uses.setdefault(scenario, []).append(new_use(item_id, checked["kind"], content, split))
        if checked["kind"] == "trajectory":
            TrajectorySample(content)
        items.append(NewItem(item_id, content, metadata))
    check_uses(dispatcher, owner, uses)
    stored = data_store(dispatcher).create(owner, items)
    dispatcher.sync_items(owner, uses)
    return {"owner": owner, "items": [item_json(item, content=False) for item in stored]}


def create_task_items(
    dispatcher: Dispatcher, owner: str, archive: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a task item for each Harbor task of the tar.gz ``archive``, each with ``metadata``; return them.

    ``metadata`` is the client's metadata for every task; its ``kind`` is ``task``. Raise ``ValueError`` for an
    archive that ``unpack_task_archive`` refuses or a directory that Harbor cannot load; nothing is stored then.
    """
    check_owner(owner)
    if not isinstance(metadata, Mapping) or metadata.get("kind", "task") != "task":
        raise ValueError("the metadata of task items is an object whose kind is task")
    scenarios = check_scenarios(metadata.get("scenarios", {}), "task")
    with tempfile.TemporaryDirectory(prefix="reef-task-items-") as work:
        directories = unpack_task_archive(archive, Path(work))
        for directory in directories:
            try:
                HarborTaskLoader(directory)
            except (ValueError, OSError) as exc:
                raise ValueError(f"{directory.name} is not a Harbor task: {exc}") from exc
        new_uses = {
            scenario: [new_use(directory.name, "task", None, split) for directory in directories]
            for scenario, split in scenarios.items()
        }
        check_uses(dispatcher, owner, new_uses)
        stored = data_store(dispatcher).create(
            owner, [NewItem(directory.name, directory, {**metadata, "kind": "task"}) for directory in directories]
        )
    dispatcher.sync_items(owner, scenarios)
    return {"owner": owner, "items": [item_json(item, content=False) for item in stored]}


def query_items(
    dispatcher: Dispatcher, owner: str, *, kind: str | None, scenario: str | None, after: int, limit: int
) -> dict[str, Any]:
    """The items of ``owner`` after ``after``, at most ``limit``: ``{owner, items, next}``, without content.

    ``next`` is the ``after`` of the next page, or null after the last item.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(ITEM_KINDS)}, got {kind!r}")
    checked_kind = None if kind is None else KINDS[kind]
    found = data_store(dispatcher).query(owner, kind=checked_kind, scenario=scenario, after=after, limit=limit)
    return {
        "owner": owner,
        "items": [item_json(item, content=False) for item in found],
        "next": found[-1].sequence if len(found) == limit else None,
    }


def read_item(dispatcher: Dispatcher, owner: str, item_id: str) -> dict[str, Any]:
    """One item of ``owner`` with its content; ``ItemNotFound`` for an unknown ID."""
    item = data_store(dispatcher).read(owner, item_id)
    if item is None:
        raise ItemNotFound(f"{owner!r} stores no item {item_id!r}")
    return {"owner": owner, **item_json(item, content=True)}


def update_items(dispatcher: Dispatcher, owner: str, body: Mapping[str, Any]) -> dict[str, Any]:
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
    store = data_store(dispatcher)
    current = touched_items(store, owner, ids)
    uses: dict[str, list[Use]] = {}
    touched: set[str] = set()
    for item in current:
        after = patched_metadata(item.metadata, patch)["scenarios"]
        touched.update(item.scenarios, after)
        for scenario, split in after.items():
            if item.scenarios.get(scenario) != split:
                uses.setdefault(scenario, []).append(new_use(item.item_id, item.kind, item.content, split))
    check_uses(dispatcher, owner, uses)
    updated = store.update(owner, ids, patch)
    dispatcher.sync_items(owner, touched)
    return {"owner": owner, "items": [item_json(item, content=False) for item in updated]}


def delete_item(dispatcher: Dispatcher, owner: str, item_id: str) -> dict[str, Any]:
    """Delete one item of ``owner``; the scenarios that used it lose its upload. ``ItemNotFound`` if unknown."""
    store = data_store(dispatcher)
    (item,) = touched_items(store, owner, [item_id])
    store.delete(owner, item_id)
    dispatcher.sync_items(owner, item.scenarios)
    return {"owner": owner, "id": item_id, "deleted": True}


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
    "NoDataStore",
    "create_items",
    "create_task_items",
    "delete_item",
    "parsed_metadata",
    "query_items",
    "read_item",
    "update_items",
]
