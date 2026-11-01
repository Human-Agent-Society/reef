"""Trajectories as samples: the calls of each trajectory that ended, without a report (RFC #1054).

``TrajectorySamples`` links the inference records that it reads into trajectories (``reef.core.trajectory_index``).
A trajectory ends when its last call has had no continuing call and no retry for ``trajectory_idle_seconds``; the
stage then forms one ``TrajectorySet`` of its calls and makes it one ATIF ``TrajectorySample`` with
``make_trajectory``. It is the stage of a method that needs no report, such as SFT on traffic or on-policy
distillation; such a method overrides ``trajectory_sample``. Every record that is not an inference, such as a
report, ends at once. A method with reports extends ``ReportSamples``, whose default sample is the same trajectory
with the report's score.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

from reef.core.config import config_option
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import make_trajectory
from reef.core.trajectory_index import TrajectoryIndex
from reef.pipeline_context import PipelineContext
from reef.record2sample.base import Record2Sample, Record2SampleSettings, SampleResult
from reef.record2sample.sets import RecordSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class TrajectorySet(RecordSet):
    """The calls of one trajectory, oldest first; the last call ends it.

    Values: a record set whose records are ``calls``, at least one, each an inference record of the stage's
    scenario. ``set_id`` is ``("trajectory", <ID of the last call>)``. The constructor checks both.
    Purpose: ``TrajectorySamples`` forms it and makes its sample (``trajectory_sample``).
    """

    calls: tuple[AgentRecord, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.calls:
            raise ValueError("a trajectory has at least one call")
        if self.set_id != ("trajectory", self.calls[-1].agent_record_id):
            raise ValueError("a trajectory set is named after its last call")

    def records(self) -> tuple[AgentRecord, ...]:
        return self.calls


@dataclass(frozen=True)
class TrajectorySettings(Record2SampleSettings):
    """The config of ``TrajectorySamples``: when a trajectory ends, and the staleness limit.

    Values: ``trajectory_idle_seconds`` is a positive number; ``max_staleness`` is as in ``Record2SampleSettings``.
    The constructor refuses other values.
    Purpose: ``TrajectorySamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    trajectory_idle_seconds: float = config_option(
        300.0, help="The seconds without a continuing call after which a trajectory ends and trains."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        idle = self.trajectory_idle_seconds
        if isinstance(idle, bool) or not isinstance(idle, (int, float)) or idle <= 0:
            raise ValueError(f"trajectory_idle_seconds must be a positive number, got {idle!r}")


class TrajectorySamples(Record2Sample):
    """Make one sample per trajectory whose last call had no continuing call for ``trajectory_idle_seconds``.

    The stage keeps each inference record, with the monotonic time at which it read it, until its release. At each
    ``advance``, a record that ends its trajectory (``TrajectoryIndex.is_end``) and that the stage read at least
    ``trajectory_idle_seconds`` before forms the set of its trajectory's calls; the calls on its path that a retry
    replaced end. A trajectory that holds a call of an earlier set, as a branch of one conversation does, or a call
    that the stage released, ends without a set, with its other calls. The sample of a set is
    ``trajectory_sample``, whose source records are the trajectory's calls.

    Config: ``TrajectorySettings``.
    """

    def __init__(
        self, context: PipelineContext, *, trajectory_idle_seconds: float = 300.0, max_staleness: int = 0
    ) -> None:
        super().__init__(max_staleness=max_staleness)
        self.scenario = context.scenario
        self.trajectory_idle_seconds = trajectory_idle_seconds
        #: The inference records, by ID, until their release.
        self.inferences: dict[str, AgentRecord] = {}
        #: The monotonic time at which the stage read each inference record that no set holds yet.
        self.read_at: dict[str, float] = {}
        self.trajectories = TrajectoryIndex()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        settings = TrajectorySettings.parse(config, plugin_class=cls)
        return cls(
            context, trajectory_idle_seconds=settings.trajectory_idle_seconds, max_staleness=settings.max_staleness
        )

    def trajectory_sample(self, record_set: TrajectorySet) -> TrajectorySample:
        """The sample of a trajectory without a report: its calls as one ATIF trajectory, without a reward."""
        return make_trajectory(record_set.calls)

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        trajectory = trajectory_set(record_set)
        sample = self.trajectory_sample(trajectory)
        return SampleResult.ready((replace(sample, source_agent_record_ids=trajectory.record_ids()),))

    def push(self, record: AgentRecord) -> None:
        if record.scenario != self.scenario:
            raise ValueError("records must belong to the pipeline's scenario")
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        if record.agent_record_id in self.inferences:
            return
        self.inferences[record.agent_record_id] = record
        self.read_at[record.agent_record_id] = time.monotonic()
        self.trajectories.add(record)

    def advance(self, now: float) -> None:
        """Form the set of each trajectory that ended by ``now``."""
        super().advance(now)
        ended = [
            record_id
            for record_id, read_at in self.read_at.items()
            if now - read_at >= self.trajectory_idle_seconds and self.trajectories.is_end(record_id)
        ]
        for record_id in ended:
            path = self.trajectories.path(record_id)
            calls = self.trajectories.trajectory(record_id)
            for call in path:
                self.read_at.pop(call, None)
            # The calls that a retry replaced end with the trajectory, as do all its free calls when it cannot train.
            unavailable = [call for call in calls if call not in self.inferences]
            taken = [call for call in calls if call in self.set_of_record]
            ending = [call for call in path if call not in calls] if not (unavailable or taken) else list(path)
            self.terminal.update(call for call in ending if call in self.inferences and call not in self.set_of_record)
            if unavailable or taken:
                logger.warning(
                    "scenario %r: the trajectory that ends at %s does not train; its calls %r are in another "
                    "set or already released",
                    self.scenario,
                    record_id,
                    taken or unavailable,
                )
                continue
            self.form(
                TrajectorySet(set_id=("trajectory", record_id), calls=tuple(self.inferences[call] for call in calls))
            )

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        for record_id in agent_record_ids:
            self.inferences.pop(record_id, None)
            self.read_at.pop(record_id, None)
        self.trajectories.forget(agent_record_ids)

    def operational_metrics(self) -> Mapping[str, float | int]:
        """The calls that wait for their trajectory to end."""
        return {"waiting_calls": len(self.read_at)}


def trajectory_set(record_set: RecordSet) -> TrajectorySet:
    """Return ``record_set`` as a trajectory set. A trajectory stage forms only trajectory sets."""
    if not isinstance(record_set, TrajectorySet):
        raise TypeError(f"the trajectory stage forms trajectory sets, not {type(record_set).__name__}")
    return record_set


__all__ = ["TrajectorySamples", "TrajectorySet", "TrajectorySettings", "trajectory_set"]
