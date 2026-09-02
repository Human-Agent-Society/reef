"""Construction and durable recovery of scenario aggregates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactNotFound, ArtifactRef, LiveWeightArtifactRef
from reef.artifact.repository import (
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendFactory,
)
from reef.core.errors import ReefError
from reef.observability import ExperimentLogger, ExperimentTracker
from reef.recipe.base import Recipe
from reef.recipe.errors import ScenarioRecipeConflict
from reef.recipe.registry import RecipeRegistry
from reef.records import RecordStore
from reef.scenario.binding import ScenarioBinding
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.scenario.commit_protocol import ScenarioCommitProtocol
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
    """Build a complete scenario from recipe and artifact registries."""

    def __init__(
        self,
        recipes: RecipeRegistry,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        agent_record_dir: Path | None = None,
        experiment_tracker: ExperimentTracker,
    ) -> None:
        self._recipes = recipes
        self._backend_factory = backend_factory
        self._local_artifact_dir = local_artifact_dir
        self._agent_record_dir = None if agent_record_dir is None else Path(agent_record_dir)
        self._experiment_tracker = experiment_tracker
        if self._agent_record_dir is not None:
            self._agent_record_dir.mkdir(parents=True, exist_ok=True)

    def has_registration(self, scenario: str) -> bool:
        """True when the scenario is durably registered with the backend."""
        return isinstance(
            self._backend_factory, RegistrationAwareRepositoryBackendFactory
        ) and self._backend_factory.has_registration(scenario)

    def load_or_create(
        self,
        scenario: str,
        recipe: str | None = None,
        release_id: str | None = None,
    ) -> Scenario:
        """Create or recover a scenario.

        Creation binds the scenario to the deployment's served recipe; an
        in-process caller may name the recipe instead, which a multi-recipe
        registry (tests) requires.
        """
        if recipe is None:
            recipe = self._recipes.served_recipe
        if (
            recipe is None
            and isinstance(self._backend_factory, RegistrationAwareRepositoryBackendFactory)
            and not self._backend_factory.has_registration(scenario)
        ):
            raise ReefError(f"no served recipe to bind scenario {scenario!r} to")
        backend = self._backend_factory(scenario)
        metadata = backend.metadata()
        snapshot_data = None if metadata is None else metadata.get(SCENARIO_SNAPSHOT_METADATA_KEY)
        if snapshot_data is not None:
            return self._recover(
                scenario,
                backend,
                snapshot_data,
                recipe=recipe,
                release_id=release_id,
            )

        if recipe is None:
            raise ReefError(f"no served recipe to bind scenario {scenario!r} to")
        self._recipes.resolve(recipe)
        selected = backend.resolve_release(release_id)
        backend.fork(
            selected.release_id,
            metadata={
                SCENARIO_SNAPSHOT_METADATA_KEY: snapshot_metadata_for(
                    name=scenario,
                    recipe=recipe,
                    base_artifact=selected,
                )
            },
        )

        # fork() is the atomic registration point. Another caller may have
        # won it, so always rebuild from the durable binding instead of the
        # recipe this caller proposed.
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
            recipe=recipe,
            # Freeze moving selectors such as "head" at the release resolved
            # for this create attempt. If another creator won, its persisted
            # base must still match the version this caller observed.
            release_id=selected.release_id,
        )

    def validate_existing(
        self,
        current: Scenario,
        recipe: str | None,
        release_id: str | None,
    ) -> None:
        self._validate_binding_selectors(
            current.name,
            current.recipe,
            current.repository.base_artifact,
            current.repository.backend,
            recipe,
            release_id,
        )

    def _recover(
        self,
        scenario: str,
        backend: RepositoryBackend,
        snapshot_data: object,
        *,
        recipe: str | None,
        release_id: str | None,
    ) -> Scenario:
        if not isinstance(snapshot_data, Mapping):
            raise ValueError(f"invalid scenario snapshot for {scenario!r}")
        snapshot = parse_snapshot_metadata(snapshot_data)
        if snapshot.scenario != scenario:
            raise ValueError(f"scenario snapshot is for {snapshot.scenario!r}, not {scenario!r}")
        base_artifact = backend.resolve_release(snapshot.base_artifact.release_id)
        self._validate_binding_selectors(
            scenario,
            snapshot.recipe,
            base_artifact,
            backend,
            recipe,
            release_id,
        )
        recipe_definition = self._recipes.resolve(snapshot.recipe)
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
        if isinstance(surface.loader, ArtifactActivator) and not isinstance(current_artifact, LiveWeightArtifactRef):
            # Traffic must not reach a recovered scenario before its committed
            # head is servable; a failed activation leaves the scenario unloaded.
            surface.loader.activate(Artifact(current_artifact, repository), runtime)
        recovered = self._build(
            scenario,
            snapshot.recipe,
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

    def _commit_log_for(self, scenario: str) -> CommitLog | None:
        if self._agent_record_dir is None:
            return None
        return CommitLog(self._agent_record_dir / f"{self._scenario_key(scenario)}.commits.jsonl")

    def _build(
        self,
        scenario: str,
        recipe: str,
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
            recipe=recipe,
            source_artifact_ref=repository.require_current_artifact(),
            run_segment=max(
                (
                    record.step
                    for record in (() if commit_log is None else commit_log.records())
                    if record.operation == "rollback"
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
            binding=ScenarioBinding(
                name=recipe,
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
        return recipe.build(
            scenario,
            records,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )

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

    def _validate_binding_selectors(
        self,
        scenario: str,
        bound_recipe: str,
        base_artifact: ArtifactRef,
        backend: RepositoryBackend,
        recipe: str | None,
        release_id: str | None,
    ) -> None:
        """Refuse request selectors that conflict with the existing binding."""
        if recipe is not None and recipe != bound_recipe:
            raise ScenarioRecipeConflict(
                f"scenario {scenario!r} is already bound to recipe {bound_recipe!r}, not {recipe!r}"
            )
        if release_id is not None and not self._artifact_selector_matches(
            base_artifact,
            release_id,
            backend,
        ):
            raise ArtifactConflict(
                f"scenario {scenario!r} is already bound to release "
                f"{base_artifact.release_id!r}, not {release_id!r}"
            )
