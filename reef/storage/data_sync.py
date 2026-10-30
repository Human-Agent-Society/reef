"""The data sync: a scenario's uploads are copies of the data items that it uses (RFC #1028).

Let ``s`` be a scenario and ``o`` its owner. ``I(s)`` is the set of trajectory and task items of ``o`` whose
``scenarios`` metadata names ``s``. ``U(s)`` is the set of uploads in the sample store of ``s``: the samples of no
component. The sync makes ``U(s)`` match ``I(s)``:

* each item ``i`` of ``I(s)`` has one upload ``u`` with ``u.sample_id = i.item_id``, the split
  ``i.scenarios[s]``, and the content of ``i``;
* ``U(s)`` holds no other upload.

The data store is the source of the content; an upload is a copy, so training reads it as before. Records are
items too, but the sync leaves them alone: the scenario's record store keeps its records.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from reef.core.job_data import TaskSample, TrajectorySample
from reef.storage.data import DataStore, NewItem, StoredItem
from reef.storage.samples import SampleStore, StoredSample
from reef.storage.tasks import SPLIT_NAMES


@dataclass(frozen=True)
class SyncResult:
    """The change that one sync made to the uploads of a scenario.

    Values: ``added`` and ``removed`` are tuples of distinct sample IDs, each in the order the store holds them.
    One ID is in both when its upload was replaced, for example after its split changed.

    Purpose: ``sync_scenario_samples`` returns it. The dispatcher tells each trainer to forget the ``removed``
    samples, and wakes the scenario's workers when ``added`` is not empty.
    """

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()


def scenario_items(data: DataStore, owner: str, scenario: str) -> dict[str, StoredItem]:
    """``I(s)``: the trajectory and task items of ``owner`` that ``scenario`` uses, by ID, in arrival order."""
    items: dict[str, StoredItem] = {}
    after = 0
    while page := data.query(owner, scenario=scenario, after=after):
        items.update((item.item_id, item) for item in page if item.kind != "record")
        after = page[-1].sequence
    return items


def uploads(samples: SampleStore) -> Iterator[StoredSample]:
    """``U(s)``: the samples of no component in ``samples``, in arrival order."""
    after = 0
    while page := samples.page(after_sequence=after):
        yield from (stored for stored in page if stored.component is None)
        after = page[-1].sequence


def is_copy(upload: StoredSample, item: StoredItem, scenario: str) -> bool:
    """Whether ``upload`` is the copy of ``item`` in ``scenario``: the same kind, split and content."""
    split = item.scenarios[scenario]
    if item.kind == "trajectory":
        return isinstance(upload.sample, TrajectorySample) and upload.sample.trajectory == item.content
    return isinstance(upload.sample, TaskSample) and upload.files_digest == item.digest and upload.split == split


def sync_scenario_samples(samples: SampleStore, data: DataStore, owner: str, scenario: str) -> SyncResult:
    """Make the uploads of ``samples``, the sample store of ``scenario``, match the items that it uses.

    The sync first removes each upload that is not the copy of an item of ``I(s)`` (``is_copy``). Then it adds a
    copy of each item of ``I(s)`` that has none, in arrival order: the trajectories in one call, and the tasks of
    each split in one call. It raises as ``SampleStore.add_trajectories`` and ``add_tasks`` do, for example when
    an item's ID is the ID of a sample that a trainer made.
    """
    items = scenario_items(data, owner, scenario)
    present = {upload.sample_id: upload for upload in uploads(samples)}
    stale = tuple(
        sample_id
        for sample_id, upload in present.items()
        if sample_id not in items or not is_copy(upload, items[sample_id], scenario)
    )
    removed = samples.remove_uploads(stale)
    missing = [item for item in items.values() if item.item_id not in present or item.item_id in removed]
    trajectories = tuple(
        (item.item_id, TrajectorySample(item.content))
        for item in missing
        if item.kind == "trajectory" and item.content is not None
    )
    if trajectories:
        samples.add_trajectories(trajectories)
    for split in SPLIT_NAMES:
        directories = tuple(
            data.task_path(item) for item in missing if item.kind == "task" and item.scenarios[scenario] == split
        )
        if directories:
            samples.add_tasks(directories, split)
    return SyncResult(added=tuple(item.item_id for item in missing), removed=removed)


def adopt_uploads(samples: SampleStore, data: DataStore, owner: str, scenario: str) -> None:
    """Make each upload of ``samples`` an item of ``owner`` that ``scenario`` uses, in the upload's split.

    This moves the uploads of a scenario that the data sync opens for the first time into the data store, so
    that the next sync keeps them. An item that ``owner`` stores already with the same content changes nothing
    but its ``scenarios``. It raises ``ItemConflict`` for an item that ``owner`` stores with other content.
    """
    adopted = list(uploads(samples))
    data.create(
        owner,
        [
            (
                NewItem(upload.sample_id, dict(upload.sample.trajectory), {"kind": "trajectory"})
                if isinstance(upload.sample, TrajectorySample)
                else NewItem(upload.sample_id, upload.sample.task_path, {"kind": "task"})
            )
            for upload in adopted
        ],
    )
    for split in SPLIT_NAMES:
        ids = [upload.sample_id for upload in adopted if upload.split == split]
        if ids:
            data.update(owner, ids, {"scenarios": {scenario: split}})


def release_items(data: DataStore, owner: str, scenario: str) -> None:
    """Remove ``scenario`` from the ``scenarios`` of each item of ``owner`` that it uses, page by page.

    The record items that ``scenario`` made keep it: it is their origin. The archive of a scenario calls this, so
    that its items stay and a later scenario with the same name does not use them.
    """
    after = 0
    while page := data.query(owner, scenario=scenario, after=after):
        after = page[-1].sequence
        ids = [
            item.item_id for item in page if item.kind != "record" or item.metadata["origin"]["scenario"] != scenario
        ]
        if ids:
            data.update(owner, ids, {"scenarios": {scenario: None}})


__all__ = [
    "SyncResult",
    "adopt_uploads",
    "release_items",
    "scenario_items",
    "sync_scenario_samples",
    "uploads",
]
