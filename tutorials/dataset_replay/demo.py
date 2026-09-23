"""Run a CPU-only record replay demonstration; artifacts describe batches, not weights."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.recipe import Recipe
from reef.storage.records import RecordStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.trainer import Trainer
from reef.train.types import ProcessorContext, TrainingBatch, TrainStepResult
from .processor import ReplayBatch, ReplayProcessor, ReplayProgress


class DemoBackend(CandidateBackend):
    """Publish a small batch description to exercise real commits and recovery.

    A model backend would train on batch.items, and carry next_progress in its
    returned state alongside its own optimizer/algorithm state in the same way.
    """

    def __init__(self) -> None:
        self.directory = TemporaryDirectory(prefix="reef-replay-batch-")

    def initial_state(self) -> Mapping[str, object]:
        return {}

    def prepare_step(self, batch: TrainingBatch, state: Mapping[str, object], scenario_step: int) -> PreparedStep:
        if not isinstance(batch, ReplayBatch):
            raise TypeError("DemoBackend requires ReplayBatch")
        return PreparedStep.with_candidate(
            UpdateCandidate(batch.batch_id),
            state={**state, "replay": batch.next_progress.to_dict()},
            metrics={
                "dataset_epoch": batch.epoch if batch.phase == "dataset" else None,
                "phase": batch.phase,
                "record_ids": [record_id for item in batch.items for record_id in item.source_agent_record_ids],
            },
        )

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return EvaluationResult("replay-demo", "1", {})

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        path = Path(self.directory.name)
        (path / "batch.json").write_text(json.dumps(prepared.metrics), encoding="utf-8")
        return TrainStepResult(prepared.state, prepared.metrics, artifact=Artifact.local(path))

    def abort_step(self, prepared: PreparedStep) -> None:
        pass

    def close(self) -> None:
        self.directory.cleanup()


@dataclass(frozen=True, kw_only=True)
class ReplayRecipe(Recipe):
    """Wire existing storage and committed state into a recipe-owned processor."""

    dataset_last_sequence: int
    epochs: int = 2
    batch_size: int = 2

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, object] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        if algorithm_state is None:
            progress = ReplayProgress(self.dataset_last_sequence, self.epochs)
            state: Mapping[str, object] = {"replay": progress.to_dict()}
        else:
            progress = ReplayProgress.from_dict(algorithm_state["replay"])
            if (progress.dataset_last_sequence, progress.epochs) != (self.dataset_last_sequence, self.epochs):
                raise ValueError("cannot change the committed dataset range or epoch count on restart")
            state = algorithm_state
        processor = ReplayProcessor(
            ProcessorContext(
                scenario,
                {"batch_size": self.batch_size},
                training_mode=self.training_mode,
                experiment_logger=experiment_logger if experiment_logger is not None else NullExperimentLogger(),
            ),
            records=records,
            progress=progress,
        )
        return Trainer(
            scenario=scenario,
            records=records,
            processor=processor,
            candidate_backend=DemoBackend(),
            candidate_evaluator=None,
            state=state,
        )


def example_record(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="demo",
        agent_record_id=record_id,
        request_type=RequestType.INFERENCE,
        payload={
            "request": {"messages": [{"role": "user", "content": f"Question {record_id}"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "Example answer"}}]},
        },
    )


def main() -> None:
    with TemporaryDirectory(prefix="reef-dataset-replay-") as directory:
        root = Path(directory)
        initial = root / "initial"
        initial.mkdir()
        repository = InMemoryRepositoryBackend.factory(initial, root=root / "artifacts")
        storage = SQLiteScenarioStorage(root / "records")
        with closing(storage.open("demo")) as store:
            # The HTTP batch-import endpoint writes these same AgentRecords.
            store.records.append_many([example_record(record_id) for record_id in ("a", "b", "c")])
            last = store.records.get_for_audit("demo", "c")
            if last is None:
                raise RuntimeError("the imported dataset was evicted")
            recipe = ReplayRecipe(dataset_last_sequence=last.sequence)
        dispatcher = Dispatcher(recipe, repository, scenario_storage=storage)
        with closing(dispatcher):
            scenario = dispatcher.get_or_create_scenario("demo")
            for _ in range(4):
                result = scenario.prepare_training_step()
                if result is None:
                    raise RuntimeError("dataset pass ended early")
                scenario.commit(result)
                print(dict(result.metrics))
            # Chat appends the same record type; it now goes through once.
            scenario.records.append(example_record("live"))
            result = scenario.prepare_training_step()
            if result is None:
                raise RuntimeError("live record did not become ready")
            scenario.commit(result)
            print(dict(result.metrics))


if __name__ == "__main__":
    main()
