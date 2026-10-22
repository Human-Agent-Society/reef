"""Transport-independent handling of uploaded ATIF trajectories.

An uploaded trajectory is not a record. ``upload_trajectories`` checks an
upload, makes a ``TrajectoryItem`` of each trajectory, and gives them to the
dispatcher. No aiohttp types appear here. ``reef.service.routes.trajectories``
adapts it to HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.dispatcher import Dispatcher
from reef.service.wire import RELEASE_ID_HEADER, SCENARIO_HEADER, HeaderError

#: The most trajectories in one upload request.
MAX_TRAJECTORIES_PER_UPLOAD = 1000


def upload_trajectories(dispatcher: Dispatcher, headers: Mapping[str, str], body: Mapping[str, Any]) -> dict[str, Any]:
    """Save the ATIF trajectories of ``body`` in one transaction, and return the receipt.

    ``body`` holds only ``trajectories``: 1 to 1000 objects, each with a
    ``trajectory_id`` and a ``trajectory``.
    """
    normalized = {key.lower(): value.strip() for key, value in headers.items()}
    scenario = normalized.get(SCENARIO_HEADER, "")
    if not scenario:
        raise HeaderError(f"missing or empty {SCENARIO_HEADER}")
    entries = body.get("trajectories")
    if body.keys() != {"trajectories"} or not isinstance(entries, list):
        raise ValueError("body must contain only 'trajectories', an array of trajectory objects")
    if not 1 <= len(entries) <= MAX_TRAJECTORIES_PER_UPLOAD:
        raise ValueError(f"an upload must hold 1 to {MAX_TRAJECTORIES_PER_UPLOAD} trajectories")
    trajectories: list[tuple[str, TrajectoryItem]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.keys() != {"trajectory_id", "trajectory"}:
            raise ValueError("each trajectory object must contain only 'trajectory_id' and 'trajectory'")
        trajectory_id, trajectory = entry["trajectory_id"], entry["trajectory"]
        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            raise ValueError("trajectory_id must be a non-empty string")
        if not isinstance(trajectory, Mapping):
            raise ValueError("trajectory must be an ATIF document")
        trajectories.append((trajectory_id, TrajectoryItem(trajectory)))
    dispatcher.accept_trajectories(scenario, tuple(trajectories), release_id=normalized.get(RELEASE_ID_HEADER) or None)
    return {"scenario": scenario, "trajectory_ids": [trajectory_id for trajectory_id, _ in trajectories]}
