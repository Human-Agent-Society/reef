"""Create and restore scenarios from durable registration and committed state.

Storage repair precedes checkpoint synchronization and activation. The trainer
is then built from committed state and retained records are replayed. Any
failure closes the trainer and opened storage session owned by this attempt.
"""

from __future__ import annotations

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
from reef.core.components import (
    COMPONENTS_METADATA_KEY,
    RECORDS_COMPONENT,
    ComponentEntry,
    ReleaseComponents,
    release_components,
)
from reef.core.errors import ReefError
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentTracker
from reef.recipe.base import Recipe
from reef.scenario.binding import ScenarioBinding
from reef.scenario.scenario import Scenario
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord, parse_scenario_metadata, scenario_metadata_for
from reef.storage.scenario import ScenarioStorage, ScenarioStore
from reef.surface.base import Surface
from reef.surface.files import REPOSITORY_FILES
from reef.train.trainer import ComponentTrainer


def _consumed_by_committed_steps(records: tuple[CommitRecord, ...]) -> frozenset[str]:
    """The rows every committed step's batch consumed.

    Rehydration must skip these rows: retention may keep a consumed row stored
    (audit-only retention is contract-legal), and re-ingesting one would train
    it twice. Consumption is permanent, so the union over the whole log is the
    exclusion set.
    """
    consumed: set[str] = set()
    for record in records:
        consumed |= record.consumed_ids
    return frozenset(consumed)


@dataclass(frozen=True)
class _RecoveredTrainerState:
    """What one component's trainer recovers from its own commits: state, cursor, and consumed rows."""

    algorithm_state: Mapping[str, Any] | None
    high_water: tuple[int, int] | None
    consumed_ids: frozenset[str]


def _recovered_trainer_states(
    store: ScenarioStore, head_record: CommitRecord | None, surface: Surface
) -> dict[str, _RecoveredTrainerState]:
    """What each component's trainer recovers from its own commits.

    The only trainer of a one-component (or record-only) scenario also owns
    every commit that named no trainer: the records made before commits
    carried a component, and rollbacks, which carry its state.
    """
    records = store.history()
    if not records and head_record is not None:
        # No durable log: the head adopted from checkpoint metadata is the
        # only committed step there is.
        records = (head_record,)
    components = surface.names or (RECORDS_COMPONENT,)
    states: dict[str, _RecoveredTrainerState] = {}
    for component in components:
        own = tuple(
            record
            for record in records
            if record.component == component or (len(components) == 1 and record.component is None)
        )
        last = own[-1] if own else None
        states[component] = _RecoveredTrainerState(
            algorithm_state=None if last is None else last.algorithm_state,
            high_water=None if last is None else (last.high_water_sequence, last.high_water_offset),
            consumed_ids=_consumed_by_committed_steps(own),
        )
    return states


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
        recipe = self._recipe.with_model_config(model_config)
        surface = recipe.build_surface(scenario)
        if registration is None:
            selected = backend.resolve_release(release_id)
            registration_metadata: dict[str, object] = {
                SCENARIO_METADATA_KEY: scenario_metadata_for(
                    name=scenario,
                    base_artifact=selected,
                )
            }
            if surface.names:
                registration_metadata[COMPONENTS_METADATA_KEY] = self._base_manifest(
                    backend, selected, surface
                ).to_dict()
            backend.fork(selected.release_id, metadata=registration_metadata)

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

        return self._recover(
            scenario,
            backend,
            registration,
            release_id=release_id,
            model_config=model_config,
            surface=surface,
            registered_components=release_components(metadata),
        )

    def _base_manifest(self, backend: RepositoryBackend, selected: ArtifactRef, surface: Surface) -> ReleaseComponents:
        """Name the base release's components, so every later step carries the unchanged ones forward.

        A flat release is its one component. A composed base keeps its own
        manifest, which must bind every component the surface serves. A base
        without one keeps one directory per component; a component the base
        seeds nothing for starts empty. Files outside every component
        directory, a model snapshot laid out at the root say, belong to no
        component and would be carried forward by none: such a base is refused.
        """
        if surface.single:
            return ReleaseComponents({name: ComponentEntry(selected.content_id) for name in surface.names})
        base = backend.materialize(selected)
        manifest = base.components
        if manifest is not None:
            missing = [name for name in surface.names if name not in manifest.entries]
            if missing:
                raise ReefError(
                    f"base release {selected.release_id!r} binds components {list(manifest.names)}; "
                    f"the recipe also serves {missing}"
                )
            return manifest
        local_path = base.local_path
        if local_path is None:
            raise ReefError(f"base release {selected.release_id!r} has no local tree to name components in")
        # A component directory, or nothing at all, is what a component may keep at the root.
        stray = sorted(
            entry.name
            for entry in local_path.iterdir()
            if (entry.name not in surface.names or not entry.is_dir())
            and entry.name not in REPOSITORY_FILES
            and entry.name != ".git"
        )
        if stray:
            raise ReefError(
                f"base release {selected.release_id!r} keeps {stray} outside its components {list(surface.names)}: "
                "a release serving several components keeps one directory per component"
            )
        return ReleaseComponents({name: ComponentEntry(f"{selected.content_id}:{name}") for name in surface.names})

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
        surface: Surface,
        registered_components: ReleaseComponents | None,
    ) -> Scenario:
        if not isinstance(registration, Mapping):
            raise ValueError(f"invalid scenario metadata for {name!r}")
        if not surface.single:
            # A registration names the components its releases bind. One made by a
            # recipe serving a single component, or none, has releases with no
            # component directories to carry forward; refuse rather than compose
            # the whole flat tree into every component.
            if registered_components is None:
                raise ReefError(
                    f"scenario {name!r} was registered without a component manifest; a recipe serving components "
                    f"{list(surface.names)} needs a scenario registered with that layout"
                )
            missing = [component for component in surface.names if component not in registered_components.entries]
            if missing:
                raise ReefError(
                    f"scenario {name!r} was registered with components {list(registered_components.names)}; "
                    f"the recipe also serves {missing}"
                )
        elif registered_components is not None and not registered_components.single:
            # The reverse mismatch: a flat recipe would serve the composed root as its one tree and its next
            # step would publish a head without the other components.
            raise ReefError(
                f"scenario {name!r} was registered with components {list(registered_components.names)}; "
                "a recipe serving one component cannot reopen it"
            )
        checkpoint_head = backend.current()
        registered_name, base_artifact, checkpoint = parse_scenario_metadata(
            registration,
            checkpoint_head=checkpoint_head,
            components=(
                None
                if surface.single or registered_components is None
                else {name: entry.content_id for name, entry in registered_components.entries.items()}
            ),
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
        runtime = recipe.runtime
        store = self._storage.open(name)
        scenario: Scenario | None = None
        trainers: tuple[ComponentTrainer, ...] = ()
        try:
            head_record = store.recover(checkpoint=checkpoint)
            committed_artifact: ArtifactRef | None
            if head_record is None:
                scenario_step = 0
                committed_artifact = None
            else:
                scenario_step = head_record.step
                committed_artifact = head_record.artifact_ref
            recovered_states = _recovered_trainer_states(store, head_record, surface)

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

            current_artifact = surface.recover(committed_artifact, checkpoint_head, runtime)

            repository = Repository(
                backend,
                base_artifact,
                current_artifact=current_artifact,
                checkpoint_artifact=checkpoint_head,
                local_dir=self._local_artifact_dir,
            )
            repository.synchronize_checkpoint()
            if not isinstance(current_artifact, LiveWeightArtifactRef):
                # Traffic must not reach a recovered scenario before its committed
                # head is servable; a failed activation leaves the scenario unloaded.
                surface.activate(Artifact(current_artifact, repository), runtime)

            experiment_logger = self._experiment_tracker.bind_scenario(
                scenario=name,
                recipe=recipe.name,
                source_artifact_ref=repository.require_current_artifact(),
                run_segment=max(
                    (record.step for record in store.history() if record.operation in ("rollback", "promote")),
                    default=0,
                ),
            )
            trainers = recipe.build_trainers(
                name,
                store.records,
                surface=surface,
                algorithm_states={component: state.algorithm_state for component, state in recovered_states.items()},
                experiment_logger=experiment_logger,
            )
            if any(bound.trainer.training_mode != recipe.training_mode for bound in trainers):
                raise ValueError("recipe.build must pass its training_mode to Trainer.build")
            if len(trainers) > 1 and not store.durable:
                raise ReefError(
                    f"scenario {name!r} runs a trainer per component: each recovers from its own commits, "
                    "which needs durable commit storage"
                )
            scenario = Scenario(
                name=name,
                model_config=model_config,
                binding=ScenarioBinding(
                    surface=surface,
                    runtime=runtime,
                    training_runtime=recipe.training_runtime,
                    inference_handler=recipe.inference_handler,
                    # The recipe's own contract, not the first trainer's: a composite agrees
                    # one across its components, whatever order they are listed in.
                    report_type=recipe.report_type,
                ),
                repository=repository,
                checkpoint_strategy=recipe.checkpoint_strategy,
                trainers=trainers,
                scenario_step=scenario_step,
                store=store,
                recovered_head_record=head_record,
            )
            # Replay retained, unconsumed rows behind each trainer's watermark
            # before resuming its cursor. Retention may keep already-consumed
            # rows for audit.
            for bound in trainers:
                recovered = recovered_states.get(bound.component)
                if recovered is None or recovered.high_water is None:
                    continue
                scenario.reingest(
                    up_to_sequence=recovered.high_water[0],
                    consumed_ids=recovered.consumed_ids,
                    component=bound.component,
                )
                scenario.restore_record_progress(
                    after_sequence=recovered.high_water[0], offset=recovered.high_water[1], component=bound.component
                )
            # A scenario created or last stepped by an older Reef serves that Reef's shipped content (the harness
            # requests extension, for one) until it is republished; every later step builds on what it serves.
            scenario.publish_shipped_content()
            return scenario
        except BaseException:
            if scenario is not None:
                scenario.close()
            else:
                try:
                    for bound in trainers:
                        bound.trainer.close()
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
