"""Scenario aggregate and its durable recovery metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.artifact.repository import Repository
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError
from reef.core.reports import ReportBase
from reef.inference.model_config import ModelConfig
from reef.observability.operations import OperationMetrics
from reef.recipe.checkpoint_strategy import CheckpointStrategy
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, TrainingRuntime
from reef.scenario.binding import ScenarioBinding
from reef.scenario.committer import ScenarioCommitter, StaleTrainingResultError
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord, scenario_metadata_for
from reef.storage.records import RecordStore
from reef.storage.scenario import ScenarioStore
from reef.surface.base import Surface
from reef.train.backend import StepExecution
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import TrainingBatch, TrainStepResult


class Scenario:
    """Scenario aggregate owning one runtime binding, one release chain, and one trainer per component."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        repository: Repository,
        checkpoint_strategy: CheckpointStrategy,
        store: ScenarioStore,
        trainers: tuple[ComponentTrainer, ...],
        model_config: ModelConfig | None = None,
        scenario_step: int = 0,
        process_id: str | None = None,
        recovered_head_record: CommitRecord | None = None,
    ) -> None:
        self.operations = OperationMetrics(
            ("serve/request", "serve/admission", "ingest/write"),
            counters=(
                "serve/retries_total",
                "serve/version_mismatch_total",
                "serve/timeouts_total",
                "ingest/accepted_total",
                "ingest/duplicates_total",
                "ingest/rejected_report_total",
                "ingest/rejected_conflict_total",
                "ingest/rejected_request_total",
            ),
        )
        self._name = name
        self._binding = binding
        self.model_config = model_config or ModelConfig()
        self._surface = binding.surface
        self._store = store
        self._closed = False
        self._trainers = validate_component_trainers(trainers, binding.surface)
        self._artifact_chain = ArtifactReleaseChain(repository, process_id=process_id)
        self._committer = ScenarioCommitter(
            name=name,
            binding=binding,
            artifacts=self._artifact_chain,
            checkpoint_strategy=checkpoint_strategy,
            trainers=self._trainers,
            scenario_step=scenario_step,
            store=store,
            recovered_head_record=recovered_head_record,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def training_runtime(self) -> TrainingRuntime | None:
        return self._binding.training_runtime

    @property
    def runtime(self) -> InferenceRuntime | None:
        """Inference or training runtime bound to this scenario."""
        return self.model_config.runtime or self._binding.runtime

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The recipe's declared report contract, enforced at ingress when set."""
        return self._binding.report_type

    @property
    def inference_handler(self) -> InferenceHandler | None:
        runtime = self.model_config.runtime
        return runtime.inference_handler if runtime is not None else self._binding.inference_handler

    @property
    def repository(self) -> Repository:
        """The release chain's scenario-scoped artifact repository.

        The public read path to artifact heads (base, current, checkpoint);
        mutation goes through Scenario methods so it stays serialized with
        rollback and commit.
        """
        return self._artifact_chain.repository

    @property
    def records(self) -> RecordStore:
        return self._store.records

    @property
    def store(self) -> ScenarioStore:
        """The scenario's record and commit storage session."""
        return self._store

    @property
    def component_trainers(self) -> tuple[ComponentTrainer, ...]:
        """Every trainer of this scenario with the component it evolves, in recipe order."""
        return self._trainers

    @property
    def trainer(self) -> Trainer:
        """Inspection-only view of the first trainer; the only one of a flat scenario.

        Every mutating path goes through a Scenario method so it is serialized
        against rollback and commit by the committer lock. Reading state
        that is not part of a transaction (objective identity, consumption
        watermarks, processor schema) is safe here; do not reserve batches,
        replace results, or compact through this handle.
        """
        return self._trainers[0].trainer

    def trainer_for(self, component: str | None) -> Trainer:
        """The trainer evolving ``component``; ``None`` selects the first trainer, for scenario-wide operations."""
        if component is None:
            return self._trainers[0].trainer
        for bound in self._trainers:
            if bound.component == component:
                return bound.trainer
        raise ReefError(f"scenario {self._name!r} has no trainer for component {component!r}")

    @property
    def dispatched_component(self) -> str | None:
        """The component whose trainer runs a dispatched backend on the training runtime, if any."""
        for bound in self._trainers:
            backend = bound.trainer.candidate_backend
            if backend is not None and backend.dispatched:
                return bound.component
        return None

    @property
    def scenario_step(self) -> int:
        return self._committer.step

    @property
    def surface(self) -> Surface:
        """The serving surface built for this scenario."""
        return self._surface

    def set_training_mode(self, training_mode: str) -> None:
        """Select future batches without waiting for a running backend step."""
        for bound in self._trainers:
            bound.trainer.set_training_mode(training_mode)

    def prepare_training_step(self, component: str | None = None) -> TrainStepResult | None:
        """Prepare one local-backend step while excluding rollback and commit."""
        with self._committer.lock:
            return self.trainer_for(component).run_once(
                self.scenario_step, base_release_id=self.current_artifact_ref().release_id
            )

    def reserve_training_batch(self, component: str | None = None) -> TrainingBatch | None:
        """Reserve one backend-training batch while excluding rollback and commit."""
        with self._committer.lock:
            return self.trainer_for(component).reserve_training_batch(
                base_release_id=self.current_artifact_ref().release_id
            )

    def execute_reserved_training_step(self, component: str | None = None) -> StepExecution:
        """Run the bound dispatched backend for the reserved batch."""
        return self.trainer_for(component).execute_reserved_step(self.scenario_step)

    def reject_pending(self, metrics: Mapping[str, Any] | None = None, *, component: str | None = None) -> None:
        """Drop the reserved batch and durably compact whatever it released."""
        with self._committer.lock:
            self._committer.reject_pending(component, metrics)

    def retry_pending(self, component: str | None = None) -> None:
        """Keep the reserved batch and prepare it again against the release served now."""
        with self._committer.lock:
            self.trainer_for(component).retry_pending()

    def reingest(self, *, up_to_sequence: int, consumed_ids: frozenset[str], component: str | None = None) -> None:
        """Rebuild processor memory from retained rows behind a recovered watermark."""
        with self._committer.lock:
            self.trainer_for(component).reingest(up_to_sequence=up_to_sequence, consumed_ids=consumed_ids)

    def restore_record_progress(self, *, after_sequence: int, offset: int, component: str | None = None) -> None:
        """Resume record consumption at a recovered commit's high-water mark."""
        with self._committer.lock:
            self.trainer_for(component).restore_record_progress(after_sequence=after_sequence, offset=offset)

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """The non-blocking committed step, training outcome, and artifact-head sync status."""
        return self._committer.commit_status

    @property
    def committed_training_job_id(self) -> str | None:
        """Training-job identity proven by the dispatched component's current durable commit."""
        with self._committer.lock:
            record = self._committer.last_record_for(self.dispatched_component)
            if record is None or record.step != self.scenario_step:
                return None
            return record.training_job_id

    @property
    def committed_training_without_job_id(self) -> bool:
        """Whether the dispatched component's current head is a pre-identity training commit."""
        with self._committer.lock:
            record = self._committer.last_record_for(self.dispatched_component)
            if record is None or record.step != self.scenario_step:
                return False
            return record.operation == "training" and record.operation_verified and record.training_job_id is None

    def last_commit_for(self, component: str | None) -> CommitRecord | None:
        """The newest durable commit made by ``component``'s trainer."""
        with self._committer.lock:
            return self._committer.last_record_for(component)

    def metrics_for_version(self, release_id: str) -> Mapping[str, Any] | None:
        """Metrics of the training step that published ``release_id``, if logged."""
        return self._committer.metrics_for_version(release_id)

    def releases(self) -> tuple[dict[str, Any], ...]:
        return self._committer.releases()

    def artifact_for_version(self, release_id: str) -> Artifact:
        """Materialize a scenario release for read-only serving; absence raises ArtifactNotFound."""
        return self._committer.artifact_for_version(release_id)

    def entries_for_version(self, release_id: str) -> tuple[Mapping[str, Any], ...] | None:
        """The composition entries behind a scenario release, if its training commit logged them."""
        return self._committer.entries_for_version(release_id)

    def artifact_with_metrics(
        self,
        release_id: str | None = None,
    ) -> tuple[Artifact, Mapping[str, Any] | None]:
        """Freeze one artifact and its gate metrics without waiting for preparation."""
        return self._committer.artifact_with_metrics(release_id)

    def current_artifact_ref(self) -> ArtifactRef:
        return self._artifact_chain.current

    def rollback(self, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        return self._committer.rollback(release_id, operation=operation)

    def commit(self, result: TrainStepResult, *, component: str | None = None) -> Any:
        """Commit ``component``'s pending result as one atomic version record.

        Raises :class:`StaleTrainingResultError` when the result was prepared
        against a release another component has since replaced; the caller
        then calls :meth:`retry_pending` and prepares the batch again.
        """
        return self._committer.commit(result, component=component)

    def publish_shipped_content(self) -> ArtifactRef | None:
        """Republish the head with the content this Reef ships when it is stale; the new head, or ``None``."""
        return self._committer.publish_shipped_content()

    def close(self) -> None:
        """Tear down what this scenario instance owns: trainers, then its storage session.

        The dispatcher calls this on shutdown and when a durable reload
        replaces the instance — the one guarantee processors with background
        workers rely on (see :meth:`DataProcessor.close`). Safe to call more
        than once; the trainers close first so no processor thread can touch
        the record store after it closes.
        """
        with self._committer.lock:
            if self._closed:
                return
            self._closed = True
            try:
                for bound in self._trainers:
                    bound.trainer.close()
            finally:
                self._store.close()

    def to_metadata(self) -> dict[str, object]:
        return scenario_metadata_for(
            name=self.name,
            base_artifact=self.repository.base_artifact,
            scenario_step=self.scenario_step,
        )


def validate_component_trainers(
    trainers: tuple[ComponentTrainer, ...], surface: Surface
) -> tuple[ComponentTrainer, ...]:
    """Check that the trainers match the surface: each names a component it serves, and a surface serving
    no component has exactly one trainer, bound to ``records``."""
    if not trainers or any(not isinstance(bound, ComponentTrainer) for bound in trainers):
        raise ReefError("a scenario requires at least one ComponentTrainer")
    names = [bound.component for bound in trainers]
    if len(set(names)) != len(names):
        raise ReefError(f"scenario trainers must evolve distinct components, not {names}")
    if not surface.names:
        if names != [RECORDS_COMPONENT]:
            raise ReefError(
                f"a scenario serving no component has one trainer bound to {RECORDS_COMPONENT!r}, not {names}"
            )
        return trainers
    unknown = [name for name in names if name not in surface.names]
    if unknown:
        raise ReefError(f"scenario trainers name components the surface does not serve: {unknown}")
    dispatched = []
    for bound in trainers:
        backend = bound.trainer.candidate_backend
        if backend is not None and backend.dispatched:
            dispatched.append(bound.component)
    if len(dispatched) > 1:
        raise ReefError(f"at most one component runs on the training runtime, not {dispatched}")
    return trainers


__all__ = [
    "SCENARIO_METADATA_KEY",
    "ReleaseNotRestorable",
    "Scenario",
    "StaleTrainingResultError",
    "validate_component_trainers",
]
