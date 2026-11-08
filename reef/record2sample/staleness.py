"""The staleness check: whether the training data of samples is too many artifacts behind the served one."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from reef.core.job_data import Sample, TrajectorySample


def producing_artifact_ids(samples: Sequence[Sample]) -> frozenset[str | None]:
    """Return the artifacts that produced the training data of ``samples`` (``TrajectorySample.artifact_id``).

    A sample without a recorded artifact gives None. A ``TaskSample`` gives nothing: the algorithm rolls it out
    with the weights it trains.
    """
    return frozenset(sample.artifact_id for sample in samples if isinstance(sample, TrajectorySample))


def is_stale(artifact_ids: frozenset[str | None], served_artifact_ids: Sequence[str], max_staleness: int) -> bool:
    """Whether training data that ``artifact_ids`` produced is more than ``max_staleness`` artifacts behind.

    It is stale when an artifact is None, is not in ``served_artifact_ids``, or lags the artifact served last by
    more than ``max_staleness``. An empty ``served_artifact_ids`` means that the served artifact is unknown, so no
    data is stale.
    """
    if not served_artifact_ids:
        return False
    newest_position = len(served_artifact_ids) - 1
    positions = {artifact_id: position for position, artifact_id in enumerate(served_artifact_ids)}
    for artifact_id in artifact_ids:
        position = None if artifact_id is None else positions.get(artifact_id)
        if position is None or newest_position - position > max_staleness:
            return True
    return False


class ServedArtifacts:
    """The artifacts that a scenario served, oldest first.

    Values: ``artifact_ids`` is a finite list of artifact IDs with no two neighbors equal; the last one is the
    artifact that the scenario serves now. ``observe`` appends an artifact that differs from the last one, and
    ``restore`` replaces the list. ``beside_artifact_ids`` is the set of artifacts that the scenario serves beside its
    active selection, and ``restore_beside`` replaces it. These artifacts answer requests now, also when they were
    never active.
    Purpose: the pipeline keeps one. The scenario restores it from its commit history before each job's selection, so
    a reload or a restart keeps the artifacts served before it. The pipeline gives ``artifact_ids`` to
    ``Record2Sample.is_stale``, for a sample that ``served_beside`` does not accept.
    """

    def __init__(self) -> None:
        self.artifact_ids: list[str] = []
        self.beside_artifact_ids: frozenset[str] = frozenset()

    def restore_beside(self, artifact_ids: Iterable[str]) -> None:
        self.beside_artifact_ids = frozenset(artifact_ids)

    def served_beside(self, samples: Sequence[Sample]) -> bool:
        """Whether an artifact produced ``samples`` and every such artifact is in ``beside_artifact_ids``."""
        produced = producing_artifact_ids(samples)
        return bool(produced) and produced <= self.beside_artifact_ids

    def observe(self, artifact_id: str) -> None:
        """Add the artifact that the scenario serves now, if it changed."""
        if not self.artifact_ids or self.artifact_ids[-1] != artifact_id:
            self.artifact_ids.append(artifact_id)

    def restore(self, artifact_ids: Sequence[str]) -> None:
        """Replace the list with ``artifact_ids``, a commit history's artifacts, oldest first."""
        self.artifact_ids = list(artifact_ids)


__all__ = ["ServedArtifacts", "is_stale", "producing_artifact_ids"]
