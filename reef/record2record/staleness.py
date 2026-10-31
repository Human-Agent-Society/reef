"""The staleness check: whether the training data of samples is too many artifacts behind the served one."""

from __future__ import annotations

from collections.abc import Sequence

from reef.core.job_data import Sample, TrajectorySample


def producing_artifact_ids(samples: Sequence[Sample]) -> frozenset[str | None]:
    """Return the artifacts that produced the training data of ``samples``, by ``artifact_id``.

    A sample without a recorded artifact gives None. A ``TaskSample`` gives nothing: the algorithm rolls it out
    with the weights it trains.
    """
    artifacts: set[str | None] = set()
    for sample in samples:
        if not isinstance(sample, TrajectorySample):
            continue
        artifact_id = sample.training.get("artifact_id")
        artifacts.add(artifact_id if isinstance(artifact_id, str) and artifact_id else None)
    return frozenset(artifacts)


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
    artifact that the scenario serves now. ``observe`` appends an artifact that differs from the last one. The list
    lives with its pipeline: a pipeline built again (a reload or a restart) starts it empty, so data from the
    artifacts served before it is stale.
    Purpose: a stage that runs the staleness check keeps one, fed by the pipeline's ``observe_serving_artifact_id``,
    and gives ``artifact_ids`` to ``is_stale``.
    """

    def __init__(self) -> None:
        self.artifact_ids: list[str] = []

    def observe(self, artifact_id: str) -> None:
        """Add the artifact that the scenario serves now, if it changed."""
        if not self.artifact_ids or self.artifact_ids[-1] != artifact_id:
            self.artifact_ids.append(artifact_id)


__all__ = ["ServedArtifacts", "is_stale", "producing_artifact_ids"]
