"""Transport-independent handling of the collection routes (RFC #1312) and of a scenario's benchmarks (RFC #1420).

A collection names benchmarks and other collections of one owner (``reef.storage.task_collections``). A client
stores, lists, reads and deletes the collections of an owner. A scenario holds sets of items: benchmarks, each in one
split, and trajectory sets. It then uses the items of the set, and no item changes (``DataStore.hold``). A client
holds and releases a set, lists a scenario's holds, and holds each benchmark of a collection in one call. No aiohttp
types appear here. ``reef.service.routes.data`` adapts it to HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.service.data import DataAccess, Use, check_uses, sync_touched
from reef.storage.data import MAX_QUERY_LIMIT, DataStore, check_hold, check_owner
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


#: The kinds of sets that a scenario holds, by the name of their routes.
HOLD_ROUTES = {"benchmarks": "benchmark", "trajectory-sets": "trajectory_set"}


def checked_hold(data: DataAccess, owner: str, scenario: str, kind: str, split: object) -> str:
    """``split`` when ``scenario`` of ``owner`` can hold a set of ``kind`` in it; ``ValueError`` for another value,
    for a scenario of another owner, and for a hold that the scenario's recipe cannot train: a ``train`` hold of a
    benchmark needs a recipe that trains imported tasks, and a hold of a trajectory set one that takes uploaded
    trajectories."""
    check_owner(owner)
    if split not in SPLIT_NAMES:
        raise ValueError(f"split is one of {', '.join(SPLIT_NAMES)}")
    check_hold(scenario, kind, "set", split)
    check_uses(data, owner, {scenario: [Use(kind == "trajectory_set", kind == "benchmark" and split == "train")]})
    return str(split)


def hold_set(
    data: DataAccess, owner: str, scenario: str, kind: str, name: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    """Make ``scenario`` of ``owner`` hold the set ``name`` of ``kind``, then sync the scenario; return ``{owner,
    scenario, kind, name, split, item_count, synced}``.

    The body of a benchmark is ``{"split"}``. The body of a trajectory set is ``{}``: a trajectory always trains.
    The scenario uses each item of the set from then on, also an item that joins it later. No item changes. A
    scenario can hold one benchmark in both splits, by two calls. Raise ``ValueError`` as ``checked_hold`` does.
    """
    if kind == "trajectory_set":
        if body:
            raise ValueError("the body of a hold of a trajectory set is empty")
        split = checked_hold(data, owner, scenario, kind, "train")
    else:
        if body.keys() != {"split"}:
            raise ValueError("the body holds only 'split'")
        split = checked_hold(data, owner, scenario, kind, body["split"])
    hold = data.require_store().hold(owner, scenario, kind, name, split)
    return {"owner": owner, **hold.to_json(), "synced": sync_touched(data, owner, [scenario])}


def release_set(
    data: DataAccess, owner: str, scenario: str, kind: str, name: str, split: str | None
) -> dict[str, Any]:
    """End the hold of the set ``name`` of ``kind`` in ``split`` by ``scenario`` of ``owner``, then sync the
    scenario; return ``{owner, scenario, kind, name, split, released, synced}``. ``released`` is False when the
    scenario did not hold the set in that split.

    ``split`` is ``train`` or ``eval`` for a benchmark. For a trajectory set it is None or ``train``. Raise
    ``ValueError`` for another value.
    """
    if kind == "trajectory_set" and split is None:
        split = "train"
    if split not in SPLIT_NAMES:
        raise ValueError(f"split is one of {', '.join(SPLIT_NAMES)}")
    released = data.require_store().release(owner, scenario, kind, name, split)
    synced = sync_touched(data, owner, [scenario]) if released else True
    return {
        "owner": owner,
        "scenario": scenario,
        "kind": kind,
        "name": name,
        "split": split,
        "released": released,
        "synced": synced,
    }


def list_holds(data: DataAccess, owner: str, scenario: str, kind: str) -> dict[str, Any]:
    """The sets of ``kind`` that ``scenario`` of ``owner`` holds, by name: ``{owner, scenario, kind, holds}``, each
    ``{scenario, kind, name, split, item_count}``."""
    holds = [hold for hold in data.require_store().holds(owner, scenario) if hold.kind == kind]
    return {"owner": owner, "scenario": scenario, "kind": kind, "holds": [hold.to_json() for hold in holds]}


def add_to_scenario(data: DataAccess, owner: str, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Make one scenario hold each benchmark that the collection ``name`` of ``owner`` has now, in one split, from
    ``{"scenario", "split"}``; return ``{owner, name, scenario, split, holds, task_count, synced}``.

    ``holds`` is the holds, as ``list_holds`` gives them, and ``task_count`` counts each task of them once. A
    benchmark that joins the collection later is not held. Raise ``CollectionNotFoundError`` for an unknown name,
    and ``ValueError`` for a collection without a benchmark and as ``checked_hold`` does.
    """
    scenario = body.get("scenario")
    if body.keys() != {"scenario", "split"} or not isinstance(scenario, str) or not scenario:
        raise ValueError("the body holds only 'scenario' and 'split', two strings")
    split = checked_hold(data, owner, scenario, "benchmark", body["split"])
    store = data.require_store()
    _, tasks = collection_tasks(store, owner, name)
    if not tasks:
        raise ValueError(f"the collection {name!r} has no benchmark")
    holds = [store.hold(owner, scenario, "benchmark", benchmark, split) for benchmark in tasks]
    return {
        "owner": owner,
        "name": name,
        "scenario": scenario,
        "split": split,
        "holds": [hold.to_json() for hold in holds],
        "task_count": len({item_id for item_ids in tasks.values() for item_id in item_ids}),
        "synced": sync_touched(data, owner, [scenario]),
    }


__all__ = [
    "HOLD_ROUTES",
    "add_to_scenario",
    "benchmark_tasks",
    "collection_tasks",
    "delete_collection",
    "hold_set",
    "list_collections",
    "list_holds",
    "read_collection",
    "release_set",
    "save_collection",
]
