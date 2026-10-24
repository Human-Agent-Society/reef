"""The staleness check: find the held batch units whose training data is too many versions behind."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.core.batches import TrainDataItem, TrajectoryItem
from reef.record2record.sets import RecordSetUnit
from reef.runtime.interfaces import RuntimeLoadId


def producing_runtime_load_ids(tasks: Sequence[TrainDataItem]) -> frozenset[str | None]:
    """Return the runtime load IDs that produced the training data of ``tasks``.

    A sample whose tokens came from several versions gives each version of its
    token spans. A sample without a recorded version gives None. A ``TaskItem``
    gives nothing: the algorithm rolls it out with the weights it trains.
    """
    versions: set[str | None] = set()
    for task in tasks:
        if not isinstance(task, TrajectoryItem):
            continue
        spans = parse_runtime_load_spans(task.training.get("runtime_load_spans", []))
        if spans:
            versions.update(span.runtime_load_id for span in spans)
        else:
            version = task.training.get("runtime_load_id")
            versions.add(version if isinstance(version, str) and version else None)
    return frozenset(versions)


def stale_units(
    units: Sequence[RecordSetUnit],
    producing_runtime_load_ids_by_set: Mapping[Hashable, frozenset[str | None]],
    served_runtime_load_ids: Sequence[str],
    max_staleness: int,
) -> tuple[RecordSetUnit, ...]:
    """Return the units whose training data is more than ``max_staleness`` versions behind.

    ``served_runtime_load_ids`` lists the versions that the scenario served,
    oldest first. The last one is the version that it serves now. The lag of a
    version is the number of versions that the scenario served after it.

    A unit is stale when one of its record sets has a version that is not in the
    list, or a version that lags by more than ``max_staleness``. With
    ``max_staleness`` 0, only data of the current version trains. An empty list
    means that the serving version is unknown, so no unit is stale.
    """
    return tuple(
        unit
        for unit in units
        if is_stale(
            frozenset(
                version
                for set_id in unit.set_ids()
                for version in producing_runtime_load_ids_by_set.get(set_id, frozenset())
            ),
            served_runtime_load_ids,
            max_staleness,
        )
    )


def is_stale(versions: frozenset[str | None], served_runtime_load_ids: Sequence[str], max_staleness: int) -> bool:
    """Whether training data that ``versions`` produced is more than ``max_staleness`` versions behind.

    It is stale when a version is None, is not in ``served_runtime_load_ids``, or lags the last served version
    by more than ``max_staleness``. An empty ``served_runtime_load_ids`` means that the serving version is
    unknown, so no data is stale.
    """
    if not served_runtime_load_ids:
        return False
    newest_position = len(served_runtime_load_ids) - 1
    positions = {version: position for position, version in enumerate(served_runtime_load_ids)}
    for version in versions:
        position = None if version is None else positions.get(version)
        if position is None or newest_position - position > max_staleness:
            return True
    return False


class ServedVersions:
    """The versions that a scenario served, oldest first, since its engine's current incarnation started.

    Values: ``versions`` is a finite list of runtime load IDs with no two neighbors equal; the last one is the
    version that the scenario serves now. ``observe`` appends a version that differs from the last one, and
    clears the list first when the version belongs to another incarnation than the last one, because no
    earlier version can train after the engine restarted.
    Purpose: a stage that runs the staleness check keeps one, fed by the pipeline's
    ``observe_serving_runtime_load_id``, and gives ``versions`` to ``stale_units`` or ``is_stale``.
    """

    def __init__(self) -> None:
        self.versions: list[str] = []

    def observe(self, runtime_load_id: str) -> None:
        """Add the version that the scenario serves now, if the version changed."""
        if self.versions and self.versions[-1] == runtime_load_id:
            return
        previous = canonical_runtime_load_id(self.versions[-1]) if self.versions else None
        current = canonical_runtime_load_id(runtime_load_id)
        if previous is not None and current is not None and previous.incarnation != current.incarnation:
            self.versions.clear()
        self.versions.append(runtime_load_id)


def canonical_runtime_load_id(value: str) -> RuntimeLoadId | None:
    """Parse a canonical runtime load ID. Return None for a non-canonical value."""
    try:
        parsed = RuntimeLoadId.parse(value)
    except (TypeError, ValueError):
        return None
    return parsed if str(parsed) == value else None
