"""Transport-independent handling of the collection routes (RFC #1312).

A collection names benchmarks and other collections of one owner (``reef.storage.task_collections``). A client
stores, lists, reads and deletes the collections of an owner, and adds the tasks of one to a scenario split. No
aiohttp types appear here. ``reef.service.routes.data`` adapts it to HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.service.data import DataAccess, update_items
from reef.storage.data import MAX_QUERY_LIMIT, DataStore, check_owner
from reef.storage.task_collections import CollectionNotFoundError, TaskCollection, collection_benchmarks


def benchmark_tasks(store: DataStore, owner: str, benchmarks: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """The item IDs of the tasks of each benchmark of ``benchmarks`` that ``owner`` stores, in the order stored."""
    found: dict[str, tuple[str, ...]] = {}
    for benchmark in benchmarks:
        item_ids: list[str] = []
        after = 0
        while page := store.query(
            owner, kind="task", benchmark=benchmark, after=after, limit=MAX_QUERY_LIMIT, content=False
        ):
            item_ids.extend(item.item_id for item in page)
            after = page[-1].sequence
        found[benchmark] = tuple(item_ids)
    return found


def collection_tasks(store: DataStore, owner: str, name: str) -> tuple[TaskCollection, dict[str, tuple[str, ...]]]:
    """The collection ``name`` of ``owner``, and the tasks of each of its benchmarks (``collection_benchmarks``,
    ``benchmark_tasks``); ``CollectionNotFoundError`` for an unknown name."""
    stored = {collection.name: collection for collection in store.collections(owner)}
    if name not in stored:
        raise CollectionNotFoundError(f"{owner!r} stores no collection {name!r}")
    return stored[name], benchmark_tasks(store, owner, collection_benchmarks(stored, name))


def save_collection(data: DataAccess, owner: str, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Store ``body``, ``{"benchmarks", "collections", "average"}``, as the collection ``name`` of ``owner``, in
    place of one with the name; return ``{owner, name, benchmarks, collections, average}``.

    Raise ``ValueError`` for a body that ``TaskCollection`` refuses, one that names a collection that ``owner``
    does not store, and one that makes a cycle.
    """
    check_owner(owner)
    if "name" in body:
        raise ValueError("the path gives the name of a collection, not the body")
    collection = TaskCollection.from_json({**body, "name": name})
    store = data.require_store()
    stored = {other.name: other for other in store.collections(owner)}
    # A new cycle goes through this collection, so the check from it finds every one.
    collection_benchmarks({**stored, name: collection}, name)
    store.put_collection(owner, collection)
    return {"owner": owner, **collection.to_json()}


def list_collections(data: DataAccess, owner: str) -> dict[str, Any]:
    """The collections of ``owner``, by name: ``{owner, collections}``."""
    stored = data.require_store().collections(owner)
    return {"owner": owner, "collections": [collection.to_json() for collection in stored]}


def read_collection(data: DataAccess, owner: str, name: str) -> dict[str, Any]:
    """The collection ``name`` of ``owner`` with its task counts; ``CollectionNotFoundError`` for an unknown name.

    ``benchmark_task_counts`` gives ``{benchmark, task_count}`` for each benchmark of the collection and of the
    collections under it. ``task_count`` counts each task once.
    """
    collection, tasks = collection_tasks(data.require_store(), owner, name)
    return {
        "owner": owner,
        **collection.to_json(),
        "benchmark_task_counts": [
            {"benchmark": benchmark, "task_count": len(item_ids)} for benchmark, item_ids in tasks.items()
        ],
        "task_count": len({item_id for item_ids in tasks.values() for item_id in item_ids}),
    }


def delete_collection(data: DataAccess, owner: str, name: str) -> dict[str, Any]:
    """Delete the collection ``name`` of ``owner``; no task item changes. ``CollectionNotFoundError`` if unknown.

    Raise ``ValueError`` while another collection of ``owner`` names it.
    """
    store = data.require_store()
    users = [other.name for other in store.collections(owner) if name in other.collections]
    if users:
        raise ValueError(f"the collections {users} name the collection {name!r}")
    if not store.remove_collection(owner, name):
        raise CollectionNotFoundError(f"{owner!r} stores no collection {name!r}")
    return {"owner": owner, "name": name, "deleted": True}


def add_to_scenario(data: DataAccess, owner: str, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Add each task that the collection ``name`` of ``owner`` has now to one split of one scenario, from
    ``{"scenario", "split"}``; return ``{owner, name, scenario, split, task_count, synced}``.

    The tasks get the metadata patch ``{"scenarios": {scenario: split}}``, at most ``MAX_QUERY_LIMIT`` in one
    update, with the checks of ``update_items``. Raise ``CollectionNotFoundError`` for an unknown name, and
    ``ValueError`` for a collection without tasks.
    """
    scenario, split = body.get("scenario"), body.get("split")
    if body.keys() != {"scenario", "split"} or not isinstance(scenario, str) or not isinstance(split, str):
        raise ValueError("the body holds only 'scenario' and 'split', two strings")
    _, tasks = collection_tasks(data.require_store(), owner, name)
    item_ids = list(dict.fromkeys(item_id for benchmark_ids in tasks.values() for item_id in benchmark_ids))
    if not item_ids:
        raise ValueError(f"the collection {name!r} has no task")
    synced = True
    for start in range(0, len(item_ids), MAX_QUERY_LIMIT):
        update = {"ids": item_ids[start : start + MAX_QUERY_LIMIT], "patch": {"scenarios": {scenario: split}}}
        synced = update_items(data, owner, update)["synced"] and synced
    return {
        "owner": owner,
        "name": name,
        "scenario": scenario,
        "split": split,
        "task_count": len(item_ids),
        "synced": synced,
    }


__all__ = [
    "add_to_scenario",
    "benchmark_tasks",
    "collection_tasks",
    "delete_collection",
    "list_collections",
    "read_collection",
    "save_collection",
]
