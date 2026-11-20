"""What the samples service judges samples against: the artifacts that a scenario served, and how far data lags."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from reef.core.job_data import Sample, TrajectorySample


@dataclass(frozen=True)
class ServedArtifacts:
    """The artifacts that a scenario served, as its commit history holds them, and the commit that last changed them.

    Values: ``artifact_ids`` is a finite tuple of artifact IDs, oldest first, with no two neighbors equal: the
    artifacts of the served component that the scenario's creation and each activating commit bound. The last one
    is the artifact that the scenario serves now. ``beside_artifact_ids`` is the set of artifacts that the scenario
    serves beside its active selection. ``commit_number`` is a non-negative int: the number of the newest commit that
    changed what the scenario serves (an activation, a restore, a sync, a serve or a stop of serving), or 0 before
    any. A commit number never recurs, so it names one point of the served history. All are empty or 0 for a scenario
    that serves no artifact of its own. The constructor refuses equal neighbors and a negative or non-int
    ``commit_number``.
    Purpose: ``Scenario.served_artifacts`` makes it. The samples service gives it to ``Record2Sample.is_stale`` and
    stores ``commit_number`` when its judgment is done; control waits for that before an automatic job.
    """

    artifact_ids: tuple[str, ...] = ()
    beside_artifact_ids: frozenset[str] = frozenset()
    commit_number: int = 0

    def __post_init__(self) -> None:
        if any(first == second for first, second in pairwise(self.artifact_ids)):
            raise ValueError("served artifact IDs have no two neighbors equal")
        if isinstance(self.commit_number, bool) or not isinstance(self.commit_number, int) or self.commit_number < 0:
            raise ValueError(f"commit_number must be a non-negative int, got {self.commit_number!r}")


def lags_behind(samples: Sequence[Sample], served: ServedArtifacts, max_artifact_lag: int) -> bool:
    """Whether the data of a sample of ``samples`` lags more than ``max_artifact_lag`` artifacts behind the artifact
    that ``served`` serves now.

    A ``TrajectorySample`` lags when the artifact that produced it (``TrajectorySample.artifact_id``) is unknown, is
    not in ``served.artifact_ids``, or has more than ``max_artifact_lag`` artifacts after its last position there. A
    trajectory of an artifact that the scenario serves beside its active selection does not lag. A ``TaskSample``
    never lags: the algorithm rolls it out with the weights that it trains. With an empty ``served.artifact_ids``
    no sample lags.
    """
    if not served.artifact_ids:
        return False
    newest_position = len(served.artifact_ids) - 1
    positions = {artifact_id: position for position, artifact_id in enumerate(served.artifact_ids)}
    for sample in samples:
        if not isinstance(sample, TrajectorySample):
            continue
        produced = sample.artifact_id
        if produced is not None and produced in served.beside_artifact_ids:
            continue
        position = None if produced is None else positions.get(produced)
        if position is None or newest_position - position > max_artifact_lag:
            return True
    return False


__all__ = ["ServedArtifacts", "lags_behind"]
