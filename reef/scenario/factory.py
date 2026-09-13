"""Create and restore scenarios from durable registration and committed state.

Storage repair precedes checkpoint synchronization and activation. The trainer
is then built from committed state and retained records are replayed. Any
failure closes the trainer and opened storage session owned by this attempt.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

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
from reef.observability import ExperimentTracker
from reef.recipe.base import Recipe
from reef.runtime.model_config import ModelConfig
from reef.scenario.binding import ScenarioBinding
from reef.scenario.scenario import Scenario
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord, parse_scenario_metadata, scenario_metadata_for
from reef.storage.scenario import ScenarioStorage, ScenarioStore
from reef.surface.base import ArtifactActivator
from reef.train.trainer import Trainer


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
    """Register, assemble, and recover complete scenario instances."""

    def __init__(
        self,
        recipe: Recipe,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        experiment_tracker: ExperimentTracker,
        scenario_storage: ScenarioStorage,
    ) -> None:
        self._recipe = recipe
        self._backend_factory = backend_factory
        self._local_artifact_dir = local_artifact_dir
        self._experiment_tracker = experiment_tracker
        self._storage = scenario_storage

    def has_registration(self, scenario: str) -> bool:
        """True when the scenario is durably registered with the backend."""
        return isinstance(
            self._backend_factory, RegistrationAwareRepositoryBackendFactory
        ) and self._backend_factory.has_registration(scenario)

    def load_or_create(
        self,
        scenario: str,
        release_id: str | None = None,
        *,
        model_config: ModelConfig,
    ) -> Scenario:
        """Create or recover a scenario in this deployment's repository."""
        backend = self._backend_factory(scenario)
        if self._storage.durable and not isinstance(backend, StagedReleaseRepositoryBackend):
            raise ArtifactPublicationError(
                "scenarios with durable commit storage require a backend implementing StagedReleaseRepositoryBackend"
            )
        metadata = backend.metadata()
        registration = None if metadata is None else metadata.get(SCENARIO_METADATA_KEY)
        if registration is None:
            selected = backend.resolve_release(release_id)
            backend.fork(
                selected.release_id,
                metadata={
                    SCENARIO_METADATA_KEY: scenario_metadata_for(
                        name=scenario,
                        base_artifact=selected,
                    )
                },
            )

            # fork() is the atomic registration point. Another caller may have
            # won it, so always rebuild from the durable registration instead of
            # assuming this creation attempt won.
            metadata = backend.metadata()
            registration = None if metadata is None else metadata.get(SCENARIO_METADATA_KEY)
            if registration is None:
                raise ReefError(f"scenario backend did not persist registration metadata for {scenario!r}")
            # Freeze moving selectors such as "head" at the release resolved
            # for this attempt, even if another creator won registration.
            release_id = selected.release_id

        return self._recover(scenario, backend, registration, release_id=release_id, model_config=model_config)

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
        name: str,
        backend: RepositoryBackend,
        registration: object,
        *,
        release_id: str | None,
        model_config: ModelConfig,
    ) -> Scenario:
        if not isinstance(registration, Mapping):
            raise ValueError(f"invalid scenario metadata for {name!r}")
        checkpoint_head = backend.current()
        registered_name, base_artifact, checkpoint = parse_scenario_metadata(
            registration, checkpoint_head=checkpoint_head
        )
        if registered_name != name:
            raise ValueError(f"scenario metadata is for {registered_name!r}, not {name!r}")
        base_artifact = backend.resolve_release(base_artifact.release_id)
        checkpoint_step = 0 if checkpoint is None else checkpoint.step
        self._validate_release_selector(
            name,
            base_artifact,
            backend,
            release_id,
        )
        recipe = self._recipe.with_model_config(model_config)
        surface = recipe.build_surface(name)
        runtime = recipe.runtime
        store = self._storage.open(name)
        scenario: Scenario | None = None
        trainer: Trainer | None = None
        try:
            head_record = store.recover(checkpoint=checkpoint)
            committed_artifact: ArtifactRef | None
            if head_record is None:
                scenario_step = 0
                algorithm_state = None
                committed_artifact = None
                high_water = None
            else:
                scenario_step = head_record.step
                algorithm_state = head_record.algorithm_state
                committed_artifact = head_record.artifact_ref
                high_water = (head_record.high_water_sequence, head_record.high_water_offset)

            # Publication stages durable bytes before the commit record is durable, while
            # the backend's head is only a post-commit mirror. A crash between the
            # two leaves the commit log's checkpoint ahead of that pointer.
            if store.durable:
                checkpoints = [
                    record
                    for record in store.history()
                    if record.checkpoint and not record.pending and record.step >= checkpoint_step
                ]
                if checkpoints:
                    checkpoint_head = checkpoints[-1].artifact_ref

            current_artifact = (
                checkpoint_head
                if surface.loader is None
                else surface.loader.recover(committed_artifact, checkpoint_head, runtime)
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

            experiment_logger = self._experiment_tracker.bind_scenario(
                scenario=name,
                recipe=recipe.name,
                source_artifact_ref=repository.require_current_artifact(),
                run_segment=max(
                    (record.step for record in store.history() if record.operation in ("rollback", "promote")),
                    default=0,
                ),
            )
            trainer = recipe.build(
                name,
                store.records,
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
            )
            if trainer.training_mode != recipe.training_mode:
                raise ValueError("recipe.build must pass its training_mode to Trainer.build")
            scenario = Scenario(
                name=name,
                model_config=model_config,
                binding=ScenarioBinding(
                    surface=surface,
                    runtime=runtime,
                    training_runtime=recipe.training_runtime,
                    inference_backend=recipe.inference_backend,
                    artifact_validator=recipe.build_artifact_validator(),
                    report_type=trainer.report_type,
                ),
                repository=repository,
                checkpoint_strategy=recipe.checkpoint_strategy,
                trainer=trainer,
                scenario_step=scenario_step,
                store=store,
                recovered_head_record=head_record,
            )
            # Replay retained, unconsumed rows behind the watermark before resuming
            # the cursor. Retention may keep already-consumed rows for audit.
            if high_water is not None:
                consumed = _consumed_by_committed_steps(store, head_record)
                scenario.reingest(up_to_sequence=high_water[0], consumed_ids=consumed)
                scenario.restore_record_progress(after_sequence=high_water[0], offset=high_water[1])
            return scenario
        except BaseException:
            if scenario is not None:
                scenario.close()
            else:
                try:
                    if trainer is not None:
                        trainer.close()
                finally:
                    store.close()
            raise

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
