"""Shared pieces of the generated-task tests: a pipeline that commits each job's submission."""

from __future__ import annotations

from typing import Any

from reef.artifact import SnapshotRef
from reef.pipeline_context import PipelineContext
from reef.storage.commits import CommitRecord
from reef.storage.scenario import ScenarioStore
from reef.train.pipeline import TrainingPipeline


class CommittingPipeline(TrainingPipeline):
    """A pipeline whose ``acknowledge`` also commits what the job data consumed, as the trainer's commit of a
    job's submission does: the commit counts the stored samples trained. Each look also makes the samples first,
    as the samples service does before the trainer reads the sample store."""

    def __init__(self, context: PipelineContext, session: ScenarioStore, **stages: Any) -> None:
        super().__init__(context, **stages)
        self.session = session

    def advance(self) -> None:
        self.produce()
        super().advance()

    def acknowledge(self, data_id: str) -> frozenset[str]:
        consumed = super().acknowledge(data_id)
        step = len(self.session.history())
        commit = CommitRecord(
            scenario="tasks",
            commit_number=step + 1,
            snapshot=SnapshotRef(f"release:{step + 1}", "base"),
            checkpoint=True,
            algorithm_state={},
            high_water_sequence=step + 1,
            high_water_offset=step + 1,
            consumed_ids=consumed,
            recorded_at=1000.0 + step,
        )
        self.session.append_commit(expected_commit_number=step, commit=commit)
        return consumed
