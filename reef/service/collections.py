"""Transport-independent handling of the collection routes (RFC #1312) and of a scenario's benchmarks (RFC #1420).

A collection names benchmarks and other collections of one owner (``reef.storage.task_collections``). A client
stores, lists, reads and deletes the collections of an owner. A scenario holds benchmarks, each in one split: it then
uses the tasks of the benchmark, and no task item changes (``DataStore.hold_benchmark``). A client holds and releases
a benchmark, lists a scenario's holds, and holds each benchmark of a collection in one call. No aiohttp types appear
here. ``reef.service.routes.data`` adapts it to HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.service.data import DataAccess, Use, check_uses, sync_touched
from reef.storage.data import MAX_QUERY_LIMIT, DataStore, check_owner
from reef.storage.task_collections import CollectionNotFoundError, TaskCollection, collection_benchmarks
from reef.storage.tasks import SPLIT_NAMES


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
    store.write_collection(owner, collection)
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
    if not store.delete_collection(owner, name):
        raise CollectionNotFoundError(f"{owner!r} stores no collection {name!r}")
    return {"owner": owner, "name": name, "deleted": True}


def checked_hold(data: DataAccess, owner: str, scenario: str, split: object) -> str:
    """``split`` when ``scenario`` of ``owner`` can hold a benchmark in it; ``ValueError`` for another value, for a
    scenario of another owner, and for a ``train`` hold by a scenario that cannot train imported tasks."""
    check_owner(owner)
    if split not in SPLIT_NAMES:
        raise ValueError(f"split is one of {', '.join(SPLIT_NAMES)}")
    check_uses(data, owner, {scenario: [Use(False, split == "train")]})
    return str(split)


def hold_benchmark(
    data: DataAccess, owner: str, scenario: str, benchmark: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    """Make ``scenario`` of ``owner`` hold ``benchmark`` in the split of ``{"split"}``, then sync the scenario;
    return ``{owner, scenario, benchmark, split, task_count, synced}``.

    The scenario uses each task of the benchmark from then on, also a task that joins it later. No task item
    changes. A hold that exists gets the split. Raise ``ValueError`` as ``checked_hold`` does.
    """
    if body.keys() != {"split"}:
        raise ValueError("the body holds only 'split'")
    split = checked_hold(data, owner, scenario, body["split"])
    hold = data.require_store().hold_benchmark(owner, scenario, benchmark, split)
    return {"owner": owner, **hold.to_json(), "synced": sync_touched(data, owner, [scenario])}


def release_benchmark(data: DataAccess, owner: str, scenario: str, benchmark: str) -> dict[str, Any]:
    """End the hold of ``benchmark`` by ``scenario`` of ``owner``, then sync the scenario; return ``{owner, scenario,
    benchmark, released, synced}``. ``released`` is False when the scenario did not hold the benchmark."""
    released = data.require_store().release_benchmark(owner, scenario, benchmark)
    synced = sync_touched(data, owner, [scenario]) if released else True
    return {"owner": owner, "scenario": scenario, "benchmark": benchmark, "released": released, "synced": synced}


def list_held_benchmarks(data: DataAccess, owner: str, scenario: str) -> dict[str, Any]:
    """The benchmarks that ``scenario`` of ``owner`` holds, by name: ``{owner, scenario, benchmarks}``, each
    ``{scenario, benchmark, split, task_count}``."""
    holds = data.require_store().held_benchmarks(owner, scenario)
    return {"owner": owner, "scenario": scenario, "benchmarks": [hold.to_json() for hold in holds]}


def add_to_scenario(data: DataAccess, owner: str, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Make one scenario hold each benchmark that the collection ``name`` of ``owner`` has now, in one split, from
    ``{"scenario", "split"}``; return ``{owner, name, scenario, split, benchmarks, task_count, synced}``.

    ``benchmarks`` is the holds, as ``list_held_benchmarks`` gives them, and ``task_count`` counts each task of
    them once. A benchmark that joins the collection later is not held. Raise ``CollectionNotFoundError`` for an
    unknown name, and ``ValueError`` for a collection without a benchmark and as ``checked_hold`` does.
    """
    scenario = body.get("scenario")
    if body.keys() != {"scenario", "split"} or not isinstance(scenario, str) or not scenario:
        raise ValueError("the body holds only 'scenario' and 'split', two strings")
    split = checked_hold(data, owner, scenario, body["split"])
    store = data.require_store()
    _, tasks = collection_tasks(store, owner, name)
    if not tasks:
        raise ValueError(f"the collection {name!r} has no benchmark")
    holds = [store.hold_benchmark(owner, scenario, benchmark, split) for benchmark in tasks]
    return {
        "owner": owner,
        "name": name,
        "scenario": scenario,
        "split": split,
        "benchmarks": [hold.to_json() for hold in holds],
        "task_count": len({item_id for item_ids in tasks.values() for item_id in item_ids}),
        "synced": sync_touched(data, owner, [scenario]),
    }


__all__ = [
    "add_to_scenario",
    "benchmark_tasks",
    "collection_tasks",
    "delete_collection",
    "hold_benchmark",
    "list_collections",
    "list_held_benchmarks",
    "read_collection",
    "release_benchmark",
    "save_collection",
]
