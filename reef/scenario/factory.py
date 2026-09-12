"""Construction and durable recovery of scenario aggregates."""

from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactNotFound,
    ArtifactPublicationError,
    ArtifactRef,
    LiveWeightArtifactRef,
)
from reef.artifact.repository import (
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendFactory,
    StagedReleaseRepositoryBackend,
)
from reef.core.errors import ReefError
from reef.observability import ExperimentLogger, ExperimentTracker
from reef.recipe.base import Recipe
from reef.records import RecordRetention, RecordStore
from reef.scenario.binding import ScenarioBinding
from reef.scenario.model_config import ScenarioModelConfig
from reef.scenario.scenario import Scenario
from reef.scenario.snapshot import SCENARIO_SNAPSHOT_METADATA_KEY, parse_snapshot_metadata, snapshot_metadata_for
from reef.scenario.state import CommitRecord, ScenarioSnapshot
from reef.scenario.store import ScenarioStore, ScenarioStoreFactory
from reef.surface.base import ArtifactActivator, Surface
from reef.train.trainer import Trainer


@dataclass(frozen=True)
class _RecoveredHead:
    """The committed state a scenario resumes from after recovery.

    Built either from the store's recovered head record or, when nothing
    was committed beyond the checkpoint head, from the snapshot metadata.
    """

    step: int
    algorithm_state: Mapping[str, Any] | None
    #: The committed artifact ref, or None when recovery starts from the
    #: checkpoint head alone.
    artifact_ref: ArtifactRef | None
    #: (high_water_sequence, high_water_offset), or None when the snapshot
    #: pinned no record progress.
    high_water: tuple[int, int] | None

    @classmethod
    def from_commit_record(cls, record: CommitRecord) -> _RecoveredHead:
        return cls(
            step=record.step,
            algorithm_state=record.algorithm_state,
            artifact_ref=record.artifact_ref,
            high_water=(record.high_water_sequence, record.high_water_offset),
        )

    @classmethod
    def from_snapshot(cls, snapshot: ScenarioSnapshot) -> _RecoveredHead:
        # Nothing committed beyond the checkpoint head (a fresh scenario or
        # an in-memory deployment): recover from the snapshot metadata.
        # Checkpoints still pin record consumption progress through it.
        progress = snapshot.record_progress
        return cls(
            step=snapshot.scenario_step,
            algorithm_state=snapshot.algorithm_state,
            artifact_ref=None,
            high_water=(None if progress is None else (progress.high_water_sequence, progress.high_water_offset)),
        )


def _consumed_by_committed_steps(
    store: ScenarioStore,
    head_record: CommitRecord | None,
) -> frozenset[str]:
    """The rows every committed step's batch consumed.

    Rehydration must skip these rows: retention may keep a consumed row stored
    (audit-only retention is contract-legal), and re-ingesting one would train
    it twice. Consumption is permanent, so the union over the whole log is the
    exclusion set.
    """
    records = store.history()
    if not records and head_record is not None:
        # No durable log: the head adopted from checkpoint metadata is the
        # only committed step there is.
        records = (head_record,)
    consumed: set[str] = set()
    for record in records:
        consumed |= record.consumed_ids
    return frozenset(consumed)


class ScenarioFactory:
    """Build a complete scenario from the served recipe and artifact backend."""

    def __init__(
        self,
        recipe: Recipe,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        agent_record_dir: Path | None = None,
        experiment_tracker: ExperimentTracker,
        scenario_store_factory: ScenarioStoreFactory,
    ) -> None:
        self._recipe = recipe
        self._model_configs: dict[str, ScenarioModelConfig] = {}
        self._backend_factory = backend_factory
        self._local_artifact_dir = local_artifact_dir
        self._agent_record_dir = None if agent_record_dir is None else Path(agent_record_dir)
        self._experiment_tracker = experiment_tracker
        self._store_factory = scenario_store_factory
        if self._agent_record_dir is not None:
            self._agent_record_dir.mkdir(parents=True, exist_ok=True)

    def model_config(self, scenario: str) -> ScenarioModelConfig:
        if scenario not in self._model_configs:
            path = (
                None
                if self._agent_record_dir is None
                else self._agent_record_dir / f"{self._scenario_key(scenario)}-model.json"
            )
            self._model_configs[scenario] = ScenarioModelConfig(path)
        return self._model_configs[scenario]

    def forget_model_config(self, scenario: str) -> None:
        self._model_configs.pop(scenario, None)

    def configure_model(self, scenario: str, value: object) -> None:
        config = ScenarioModelConfig()
        config.save(value)
        self._recipe.with_model_config(config)
        self.model_config(scenario).save(value)

    def has_registration(self, scenario: str) -> bool:
        """True when the scenario is durably registered with the backend."""
        return isinstance(
            self._backend_factory, RegistrationAwareRepositoryBackendFactory
        ) and self._backend_factory.has_registration(scenario)

    def load_or_create(
        self,
        scenario: str,
        release_id: str | None = None,
    ) -> Scenario:
        """Create or recover a scenario in this deployment's repository."""
        backend = self._backend_factory(scenario)
        if self._store_factory.durable and not isinstance(backend, StagedReleaseRepositoryBackend):
            raise ArtifactPublicationError(
                "scenarios with durable commit storage require a backend implementing StagedReleaseRepositoryBackend"
            )
        metadata = backend.metadata()
        snapshot_data = None if metadata is None else metadata.get(SCENARIO_SNAPSHOT_METADATA_KEY)
        if snapshot_data is not None:
            return self._recover(
                scenario,
                backend,
                snapshot_data,
                release_id=release_id,
            )

        selected = backend.resolve_release(release_id)
        backend.fork(
            selected.release_id,
            metadata={
                SCENARIO_SNAPSHOT_METADATA_KEY: snapshot_metadata_for(
                    name=scenario,
                    base_artifact=selected,
                )
            },
        )

        # fork() is the atomic registration point. Another caller may have
        # won it, so always rebuild from the durable registration instead of
        # assuming this creation attempt won.
        persisted_metadata = backend.metadata()
        persisted_snapshot = (
            None if persisted_metadata is None else persisted_metadata.get(SCENARIO_SNAPSHOT_METADATA_KEY)
        )
        if persisted_snapshot is None:
            raise ReefError(f"scenario backend did not persist registration metadata for {scenario!r}")
        return self._recover(
            scenario,
            backend,
            persisted_snapshot,
            # Freeze moving selectors such as "head" at the release resolved
            # for this create attempt. If another creator won, its persisted
            # base must still match the version this caller observed.
            release_id=selected.release_id,
        )

    def validate_existing(
        self,
        current: Scenario,
        release_id: str | None,
    ) -> None:
        self._validate_release_selector(
            current.name,
            current.repository.base_artifact,
            current.repository.backend,
            release_id,
        )

    def _recover(
        self,
        scenario: str,
        backend: RepositoryBackend,
        snapshot_data: object,
        *,
        release_id: str | None,
    ) -> Scenario:
        if not isinstance(snapshot_data, Mapping):
            raise ValueError(f"invalid scenario snapshot for {scenario!r}")
        snapshot = parse_snapshot_metadata(snapshot_data)
        if snapshot.scenario != scenario:
            raise ValueError(f"scenario snapshot is for {snapshot.scenario!r}, not {scenario!r}")
        base_artifact = backend.resolve_release(snapshot.base_artifact.release_id)
        self._validate_release_selector(
            scenario,
            base_artifact,
            backend,
            release_id,
        )
        recipe_definition = self._recipe.with_model_config(self.model_config(scenario))
        surface = recipe_definition.build_surface(scenario)
        runtime = recipe_definition.runtime
        checkpoint_head = backend.current()
        store = self._store_factory.open(scenario)
        recovered: Scenario | None = None
        try:
            head_record = store.recover(snapshot=snapshot, checkpoint_head=checkpoint_head)
            head = (
                _RecoveredHead.from_commit_record(head_record)
                if head_record is not None
                else _RecoveredHead.from_snapshot(snapshot)
            )

            # Publication stages durable bytes before the commit record is durable, while
            # the backend's head is only a post-commit mirror. A crash between the
            # two leaves the commit log's checkpoint ahead of that pointer.
            if store.durable:
                checkpoints = [
                    record
                    for record in store.history()
                    if record.checkpoint and not record.pending and record.step >= snapshot.scenario_step
                ]
                if checkpoints:
                    checkpoint_head = checkpoints[-1].artifact_ref

            current_artifact = (
                checkpoint_head
                if surface.loader is None
                else surface.loader.recover(head.artifact_ref, checkpoint_head, runtime)
            )

            repository = Repository(
                backend,
                base_artifact,
                current_artifact=current_artifact,
                checkpoint_artifact=checkpoint_head,
                local_dir=self._local_artifact_dir,
            )
            repository.synchronize_checkpoint()
            if isinstance(surface.loader, ArtifactActivator) and not isinstance(
                current_artifact, LiveWeightArtifactRef
            ):
                # Traffic must not reach a recovered scenario before its committed
                # head is servable; a failed activation leaves the scenario unloaded.
                surface.loader.activate(Artifact(current_artifact, repository), runtime)
            recovered = self._build(
                scenario,
                recipe_definition,
                surface,
                repository,
                scenario_step=head.step,
                algorithm_state=head.algorithm_state,
                store=store,
                recovered_head_record=head_record,
            )
            # The store has repaired interrupted compaction. Rebuild
            # processor memory from the retained rows behind the high-water mark
            # (issue #344: the cursor passes rows of the next, still-incomplete
            # step), and resume consumption at the mark so consumed rows are not
            # re-ingested and trained twice.
            if head.high_water is not None:
                consumed = _consumed_by_committed_steps(store, head_record)
                recovered.reingest(up_to_sequence=head.high_water[0], consumed_ids=consumed)
                recovered.restore_record_progress(
                    after_sequence=head.high_water[0],
                    offset=head.high_water[1],
                )
            return recovered
        except BaseException:
            if recovered is not None:
                recovered.close()
            else:
                store.close()
            raise

    def _scenario_key(self, scenario: str) -> str:
        return hashlib.sha256(scenario.encode("utf-8")).hexdigest()

    def _model_path(self, scenario: str) -> Path | None:
        return (
            None
            if self._agent_record_dir is None
            else self._agent_record_dir / f"{self._scenario_key(scenario)}-model.json"
        )

    def archive_store(self, scenario: str) -> tuple[str, ...]:
        """Retire closed record/commit storage and this deployment's model settings."""
        archived = self._store_factory.archive(scenario)
        model_path = self._model_path(scenario)
        if model_path is not None and model_path.exists():
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = model_path.parent / "archived" / f"{self._scenario_key(scenario)}-{stamp}-{uuid.uuid4().hex}"
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / model_path.name
            shutil.move(str(model_path), str(target))
            archived = (*archived, str(target))
        return archived

    def prune_records(self, retention: RecordRetention) -> int:
        return self._store_factory.prune(days=retention.days, max_bytes=retention.max_bytes)

    def close(self) -> None:
        """Close factory resources after all opened scenario sessions are closed."""
        self._store_factory.close()

    @property
    def agent_record_dir(self) -> Path | None:
        return self._agent_record_dir

    def _build(
        self,
        scenario: str,
        recipe_definition: Recipe,
        surface: Surface,
        repository: Repository,
        *,
        scenario_step: int = 0,
        algorithm_state: Mapping[str, Any] | None = None,
        store: ScenarioStore,
        recovered_head_record: CommitRecord | None = None,
    ) -> Scenario:
        records = store.records
        experiment_logger = self._experiment_tracker.bind_scenario(
            scenario=scenario,
            recipe=recipe_definition.name,
            source_artifact_ref=repository.require_current_artifact(),
            run_segment=max(
                (record.step for record in store.history() if record.operation in ("rollback", "promote")),
                default=0,
            ),
        )
        trainer = self._build_recipe_trainer(
            recipe_definition,
            scenario,
            records,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )
        try:
            return Scenario(
                name=scenario,
                model_config=self.model_config(scenario),
                binding=ScenarioBinding(
                    surface=surface,
                    runtime=recipe_definition.runtime,
                    inference_backend=recipe_definition.inference_backend,
                    artifact_validator=recipe_definition.build_artifact_validator(),
                    report_type=trainer.report_type,
                ),
                repository=repository,
                checkpoint_strategy=recipe_definition.checkpoint_strategy,
                trainer=trainer,
                scenario_step=scenario_step,
                store=store,
                recovered_head_record=recovered_head_record,
            )
        except BaseException:
            trainer.close()
            raise

    @staticmethod
    def _build_recipe_trainer(
        recipe: Recipe,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger,
    ) -> Trainer:
        """Build a recipe trainer with the complete current recipe contract."""
        trainer = recipe.build(
            scenario,
            records,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )
        if trainer.training_mode != recipe.training_mode:
            trainer.close()
            raise ValueError("recipe.build must pass its training_mode to Trainer.build")
        return trainer

    def _artifact_selector_matches(
        self,
        base_artifact: ArtifactRef,
        selector: str,
        backend: RepositoryBackend,
    ) -> bool:
        if selector == base_artifact.release_id:
            return True
        try:
            return backend.resolve_release(selector).release_id == base_artifact.release_id
        except ArtifactNotFound:
            return False

    def _validate_release_selector(
        self,
        scenario: str,
        base_artifact: ArtifactRef,
        backend: RepositoryBackend,
        release_id: str | None,
    ) -> None:
        """Refuse a release selector that conflicts with the existing binding."""
        if release_id is not None and not self._artifact_selector_matches(
            base_artifact,
            release_id,
            backend,
        ):
            raise ArtifactConflict(
                f"scenario {scenario!r} is already bound to release {base_artifact.release_id!r}, not {release_id!r}"
            )
