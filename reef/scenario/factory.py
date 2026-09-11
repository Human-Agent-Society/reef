"""Construction and durable recovery of scenario aggregates."""

from __future__ import annotations

import hashlib
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
from reef.records import RecordStore
from reef.scenario.binding import ScenarioBinding
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.scenario.commit_protocol import ScenarioCommitProtocol
from reef.scenario.model_config import ScenarioModelConfig
from reef.scenario.scenario import Scenario
from reef.scenario.snapshot import (
    SCENARIO_SNAPSHOT_METADATA_KEY,
    ScenarioSnapshot,
    parse_snapshot_metadata,
    snapshot_metadata_for,
)
from reef.surface.base import ArtifactActivator, Surface
from reef.train.trainer import Trainer


@dataclass(frozen=True)
class _RecoveredHead:
    """The committed state a scenario resumes from after recovery.

    Built either from the commit log's recovered head record or, when nothing
    was committed beyond the checkpoint head, from the snapshot metadata.
    """

    step: int
    algorithm_state: Mapping[str, Any] | None
    #: The committed artifact ref, or None when recovery starts from the
    #: checkpoint head alone.
    artifact_ref: ArtifactRef | None
    compacted_ids: frozenset[str]
    #: (high_water_sequence, high_water_offset), or None when the snapshot
    #: pinned no record progress.
    high_water: tuple[int, int] | None

    @classmethod
    def from_commit_record(cls, record: CommitRecord) -> _RecoveredHead:
        return cls(
            step=record.step,
            algorithm_state=record.algorithm_state,
            artifact_ref=record.artifact_ref,
            compacted_ids=record.compacted_ids,
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
            compacted_ids=frozenset() if progress is None else progress.compacted_ids,
            high_water=(None if progress is None else (progress.high_water_sequence, progress.high_water_offset)),
        )


def _consumed_by_committed_steps(
    commit_log: CommitLog | None,
    head_record: CommitRecord | None,
) -> frozenset[str]:
    """The rows every committed step's batch consumed.

    Rehydration must skip these rows: retention may keep a consumed row stored
    (audit-only retention is contract-legal), and re-ingesting one would train
    it twice. Consumption is permanent, so the union over the whole log is the
    exclusion set.
    """
    records = commit_log.records() if commit_log is not None else ()
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
    ) -> None:
        self._recipe = recipe
        self._model_configs: dict[str, ScenarioModelConfig] = {}
        self._backend_factory = backend_factory
        self._local_artifact_dir = local_artifact_dir
        self._agent_record_dir = None if agent_record_dir is None else Path(agent_record_dir)
        self._experiment_tracker = experiment_tracker
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
        if self._agent_record_dir is not None and not isinstance(backend, StagedReleaseRepositoryBackend):
            raise ArtifactPublicationError(
                "scenarios with a commit log require a backend implementing StagedReleaseRepositoryBackend"
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
        commit_log = self._commit_log_for(scenario)
        head_record = ScenarioCommitProtocol.recover_head(
            scenario,
            commit_log,
            snapshot_step=snapshot.scenario_step,
            snapshot_state=snapshot.algorithm_state,
            snapshot_record_progress=snapshot.record_progress,
            snapshot_training_job_id=snapshot.training_job_id,
            snapshot_metrics=snapshot.metrics,
            snapshot_operation=snapshot.operation,
            snapshot_rollback_target_release_id=snapshot.rollback_target_release_id,
            checkpoint_head=checkpoint_head,
        )
        head = (
            _RecoveredHead.from_commit_record(head_record)
            if head_record is not None
            else _RecoveredHead.from_snapshot(snapshot)
        )

        # Publication stages durable bytes before the commit record is durable, while
        # the backend's head is only a post-commit mirror. A crash between the
        # two leaves the commit log's checkpoint ahead of that pointer.
        if commit_log is not None:
            checkpoints = [
                record
                for record in commit_log.records()
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
        if isinstance(surface.loader, ArtifactActivator) and not isinstance(current_artifact, LiveWeightArtifactRef):
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
            commit_log=commit_log,
            recovered_head_record=head_record,
        )
        # Derive the record store and trainer progress from the recovered
        # head: re-apply any compaction the crash interrupted, rebuild
        # processor memory from the retained rows behind the high-water mark
        # (issue #344: the cursor passes rows of the next, still-incomplete
        # step), and resume consumption at the mark so consumed rows are not
        # re-ingested and trained twice.
        if head.compacted_ids:
            recovered.records.compact(scenario, head.compacted_ids)
        if head.high_water is not None:
            consumed = _consumed_by_committed_steps(commit_log, head_record)
            recovered.reingest(up_to_sequence=head.high_water[0], consumed_ids=consumed)
            recovered.restore_record_progress(
                after_sequence=head.high_water[0],
                offset=head.high_water[1],
            )
        return recovered

    def _scenario_key(self, scenario: str) -> str:
        return hashlib.sha256(scenario.encode("utf-8")).hexdigest()

    def state_paths(self, scenario: str) -> tuple[Path, ...]:
        """The files under ``agent_record_dir`` that are this scenario's alone: its record store and its commit log."""
        if self._agent_record_dir is None:
            return ()
        key = self._scenario_key(scenario)
        return tuple(
            self._agent_record_dir / name
            for name in (
                f"{key}.sqlite3",
                f"{key}.sqlite3-wal",
                f"{key}.sqlite3-shm",
                f"{key}.commits.jsonl",
                f"{key}-model.json",
            )
        )

    @property
    def agent_record_dir(self) -> Path | None:
        return self._agent_record_dir

    def _commit_log_for(self, scenario: str) -> CommitLog | None:
        if self._agent_record_dir is None:
            return None
        return CommitLog(self._agent_record_dir / f"{self._scenario_key(scenario)}.commits.jsonl")

    def _build(
        self,
        scenario: str,
        recipe_definition: Recipe,
        surface: Surface,
        repository: Repository,
        *,
        scenario_step: int = 0,
        algorithm_state: Mapping[str, Any] | None = None,
        commit_log: CommitLog | None = None,
        recovered_head_record: CommitRecord | None = None,
    ) -> Scenario:
        database = None
        if self._agent_record_dir is not None:
            database = self._agent_record_dir / f"{self._scenario_key(scenario)}.sqlite3"
        records = RecordStore(database)
        experiment_logger = self._experiment_tracker.bind_scenario(
            scenario=scenario,
            recipe=recipe_definition.name,
            source_artifact_ref=repository.require_current_artifact(),
            run_segment=max(
                (
                    record.step
                    for record in (() if commit_log is None else commit_log.records())
                    if record.operation in ("rollback", "promote")
                ),
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
            records=records,
            trainer=trainer,
            scenario_step=scenario_step,
            commit_log=commit_log,
            recovered_head_record=recovered_head_record,
        )

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
