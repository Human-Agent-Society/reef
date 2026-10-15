"""The staleness check: find the held sets whose training data is too many versions behind."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.core.batches import TrainDataItem, TrajectoryItem
from reef.record2record.sets import RecordSet
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


def stale_sets(
    record_sets: Sequence[RecordSet],
    producing_runtime_load_ids_by_set: Mapping[Hashable, frozenset[str | None]],
    served_runtime_load_ids: Sequence[str],
    max_staleness: int,
) -> tuple[RecordSet, ...]:
    """Return the sets whose training data is more than ``max_staleness`` versions behind.

    ``served_runtime_load_ids`` lists the versions that the scenario served,
    oldest first. The last one is the version that it serves now. The lag of a
    version is the number of versions that the scenario served after it.

    A set is stale when one of its formed sets has a version that is not in the
    list, or a version that lags by more than ``max_staleness``. With
    ``max_staleness`` 0, only data of the current version trains. An empty list
    means that the serving version is unknown, so no set is stale.
    """
    if not served_runtime_load_ids:
        return ()
    newest_position = len(served_runtime_load_ids) - 1
    positions = {version: position for position, version in enumerate(served_runtime_load_ids)}
    stale: list[RecordSet] = []
    for record_set in record_sets:
        versions = (
            version
            for formed_set_id in record_set.formed_set_ids()
            for version in producing_runtime_load_ids_by_set.get(formed_set_id, frozenset())
        )
        for version in versions:
            position = None if version is None else positions.get(version)
            if position is None or newest_position - position > max_staleness:
                stale.append(record_set)
                break
    return tuple(stale)


def canonical_runtime_load_id(value: str) -> RuntimeLoadId | None:
    """Parse a canonical runtime load ID. Return None for a non-canonical value."""
    try:
        parsed = RuntimeLoadId.parse(value)
    except (TypeError, ValueError):
        return None
    return parsed if str(parsed) == value else None
