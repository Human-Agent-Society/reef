"""Transport-independent handling of uploaded data items: ATIF trajectories and Harbor tasks.

A data item is not a record. These functions check an upload, make a
``TrainDataItem`` of it, and give it to the dispatcher. No aiohttp types appear
here. ``reef.service.routes.data_items`` adapts these functions to HTTP.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.models.task.task import Task as HarborTaskLoader

from reef.core.batches import TaskItem, TrainDataItem, TrajectoryItem
from reef.dispatcher import Dispatcher
from reef.service.wire import RELEASE_ID_HEADER, SCENARIO_HEADER, HeaderError
from reef.storage.tasks import files_digest

#: The client gives the ID of an uploaded task in this header.
ITEM_ID_HEADER = "x-reef-item-id"
#: The most trajectories in one upload request.
MAX_TRAJECTORIES_PER_UPLOAD = 1000
#: The maximum size in bytes of the unpacked files of one uploaded task.
MAX_TASK_FILES_BYTES = 256 * 1024 * 1024


def upload_headers(headers: Mapping[str, str]) -> tuple[str, str | None, Mapping[str, str]]:
    """Return the scenario and the release ID of an upload, and its headers with lowercase names."""
    normalized = {key.lower(): value.strip() for key, value in headers.items()}
    scenario = normalized.get(SCENARIO_HEADER, "")
    if not scenario:
        raise HeaderError(f"missing or empty {SCENARIO_HEADER}")
    return scenario, normalized.get(RELEASE_ID_HEADER) or None, normalized


def upload_trajectories(dispatcher: Dispatcher, headers: Mapping[str, str], body: Mapping[str, Any]) -> dict[str, Any]:
    """Save the ATIF trajectories of ``body`` as data items in one transaction, and return the receipt.

    ``body`` holds only ``trajectories``: 1 to 1000 objects, each with an
    ``item_id`` and a ``trajectory``.
    """
    scenario, release_id, _ = upload_headers(headers)
    entries = body.get("trajectories")
    if body.keys() != {"trajectories"} or not isinstance(entries, list):
        raise ValueError("body must contain only 'trajectories', an array of trajectory objects")
    if not 1 <= len(entries) <= MAX_TRAJECTORIES_PER_UPLOAD:
        raise ValueError(f"an upload must hold 1 to {MAX_TRAJECTORIES_PER_UPLOAD} trajectories")
    items: list[tuple[str, TrainDataItem]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.keys() != {"item_id", "trajectory"}:
            raise ValueError("each trajectory object must contain only 'item_id' and 'trajectory'")
        item_id, trajectory = entry["item_id"], entry["trajectory"]
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be a non-empty string")
        if not isinstance(trajectory, Mapping):
            raise ValueError("trajectory must be an ATIF document")
        items.append((item_id, TrajectoryItem(trajectory)))
    dispatcher.accept_data_items(scenario, tuple(items), release_id=release_id)
    return {"scenario": scenario, "items": [{"item_id": item_id, "kind": "trajectory"} for item_id, _ in items]}


def upload_task(dispatcher: Dispatcher, headers: Mapping[str, str], archive: bytes) -> dict[str, Any]:
    """Save the Harbor task in ``archive`` as a data item, and return the receipt.

    ``archive`` is a gzip tar of one task directory. The item ID comes from the
    ``x-reef-item-id`` header. Thus a retry is safe.
    """
    scenario, release_id, normalized = upload_headers(headers)
    item_id = normalized.get(ITEM_ID_HEADER, "")
    if not item_id:
        raise HeaderError(f"missing or empty {ITEM_ID_HEADER}")
    with tempfile.TemporaryDirectory(prefix="reef-task-upload-") as staging:
        task_directory = unpack_task_archive(archive, Path(staging))
        digest = files_digest(task_directory)
        dispatcher.accept_data_items(scenario, ((item_id, TaskItem(task_directory)),), release_id=release_id)
    return {
        "scenario": scenario,
        "item_id": item_id,
        "kind": "task",
        "task": {"name": task_directory.name, "digest": digest},
    }


def unpack_task_archive(archive: bytes, root: Path) -> Path:
    """Unpack a gzip tar of one Harbor task directory under ``root``, check it with Harbor, and return it.

    The archive must hold one top-level directory, with only files and
    directories below it. Raises ``ValueError`` for any other archive.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"the task archive entry {member.name!r} is not a file or a directory")
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError(f"the task archive entry {member.name!r} is not inside the task directory")
            top_names = {PurePosixPath(member.name).parts[0] for member in members}
            if len(top_names) != 1:
                raise ValueError("the task archive must hold one task directory")
            if sum(member.size for member in members) > MAX_TASK_FILES_BYTES:
                raise ValueError(f"the task files are larger than {MAX_TASK_FILES_BYTES} bytes")
            tar.extractall(root, members=members, filter="data")
    except tarfile.TarError as exc:
        raise ValueError(f"the task archive is not a valid gzip tar: {exc}") from exc
    task_directory = root / top_names.pop()
    if not task_directory.is_dir():
        raise ValueError("the task archive must hold one task directory")
    try:
        HarborTaskLoader(task_directory)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{task_directory.name} is not a Harbor task: {exc}") from exc
    return task_directory
