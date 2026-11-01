"""Create and restore scenarios from durable registration and committed state.

A new scenario owns its root release: the recipe's default start, or an artifact item of its owner. Storage repair
precedes checkpoint synchronization and activation. The trainer
is then built from committed state and retained records are replayed. Any
failure closes the trainer and opened storage session owned by this attempt.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactPublicationError
from reef.artifact.composite import compose_release
from reef.artifact.repository import (
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendFactory,
    StagedReleaseRepositoryBackend,
)
from reef.artifact.sources import SourceFiles
from reef.core.artifact_ref import RootSource
from reef.core.components import COMPONENTS_METADATA_KEY, RECORDS_COMPONENT, ReleaseComponents, release_components
from reef.core.errors import ReefError
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.observability.commits import CommitExperimentObserver, ExperimentReportQueue
from reef.recipe.base import Recipe
from reef.runtime.jobs import ScenarioRoot
from reef.scenario.binding import ScenarioBinding
from reef.scenario.release_files import ScenarioReleaseFiles
from reef.scenario.scenario import Scenario
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    CommitRecord,
    active_artifact_ref,
    parse_scenario_metadata,
    scenario_metadata_for,
    served_artifact_ref,
)
from reef.storage.scenario import ScenarioStorage, ScenarioStore
from reef.surface.base import Surface
from reef.train.instructions import InstructionQueue
from reef.train.trainer import ComponentTrainer


def committed_records(store: ScenarioStore, head_record: CommitRecord | None) -> tuple[CommitRecord, ...]:
    records = store.history()
    if not records and head_record is not None:
        # No durable log: the head adopted from checkpoint metadata is the
        # only committed step there is.
        records = (head_record,)
    return records


@dataclass(frozen=True)
class ItemStart:
    """The artifact item that a new scenario's root release holds.

    Values: ``item_id`` is the item's non-empty ID, and ``uri`` its non-empty ``source_uri``. ``files`` is the
    ``SourceFiles`` that ``uri`` names, not yet checked. ``component`` is the component that the item fills, or
    ``None`` for the scenario's one component.

    Purpose: ``ArtifactImports.create_from_item`` makes it for ``POST /reef/scenarios`` with ``artifact_id``.
    ``ScenarioFactory.register`` checks it and writes it as the root release.
    """

    item_id: str
    uri: str
    files: SourceFiles
    component: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id or not isinstance(self.uri, str) or not self.uri:
            raise ValueError("an item start names a non-empty item_id and uri")
        if not isinstance(self.files, SourceFiles):
            raise TypeError("an item start's files are SourceFiles")
        if self.component is not None and (not isinstance(self.component, str) or not self.component):
            raise ValueError("an item start's component is a non-empty string or None")


@dataclass(frozen=True)
class RecoveredTrainerState:
    """What one component's trainer recovers from its own commits and drops: state, cursor, and consumed rows."""

    algorithm_state: Mapping[str, Any] | None
    high_water: tuple[int, int] | None
    consumed_ids: frozenset[str]


def recovered_trainer_states(
    store: ScenarioStore, scenario: str, head_record: CommitRecord | None, surface: Surface
) -> dict[str, RecoveredTrainerState]:
    """What each component's trainer recovers from its own commits and its own stale drops.

    The only trainer of a one-component (or record-only) scenario owns every
    training commit and consumption receipt, which name no component.
    Activations and new training starts carry no trainer state, so no
    trainer recovers from them. The records that a trainer's stored samples
    consumed (``SampleStore.consumed_record_ids``) count as consumed too, so a
    replay never makes their samples again. A drop is a step without
    a commit that an earlier Reef version recorded as a consumption receipt:
    its rows count as consumed, and the state and the cursor stay the last
    commit's.

    Rehydration must skip consumed rows: retention may keep a consumed row
    stored (audit-only retention is contract-legal), and pushing one again would
    train it twice. Consumption is permanent, so the union over the whole log
    and every receipt is the exclusion set; with several trainers each keeps
    its own share, the rows of its own commits and of its own stale drops.
    """
    records = committed_records(store, head_record)
    consumed_by_component: dict[str | None, set[str]] = {}
    for record in records:
        consumed_by_component.setdefault(record.component, set()).update(record.consumed_ids)
    for receipt in store.records.consumption_receipts(scenario):
        named = receipt["metadata"].get("component")
        owner = named if isinstance(named, str) else None
        consumed_by_component.setdefault(owner, set()).update(receipt["consumed_ids"])
    components = surface.names or (RECORDS_COMPONENT,)
    alone = len(components) == 1
    states: dict[str, RecoveredTrainerState] = {}
    for component in components:
        own = tuple(
            record
            for record in records
            if record.operation == "training"
            and (record.component == component or (alone and record.component is None))
        )
        last = own[-1] if own else None
        # The records that the trainer's stored samples consumed: its sample store key is "" when it is alone.
        stored = store.samples.consumed_record_ids("" if alone else component)
        states[component] = RecoveredTrainerState(
            algorithm_state=None if last is None else last.algorithm_state,
            high_water=None if last is None else (last.high_water_sequence, last.high_water_offset),
            consumed_ids=stored
            | (
                frozenset().union(*consumed_by_component.values())
                if alone
                else frozenset(consumed_by_component.get(component, ()))
            ),
        )
    return states


class ScenarioFactory:
    """Register, assemble, and recover complete scenario instances.

    ``activates_runtime`` is whether a recovered scenario's active artifact is activated on the runtime. Only the
    process that runs control activates: a process of another service that loads a scenario on demand (samples,
    evaluation, data) must not load weights onto the engines that control manages.
    """

    def __init__(
        self,
        recipe: Recipe,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        experiment_tracker: ExperimentTracker,
        scenario_storage: ScenarioStorage,
        experiment_reports: ExperimentReportQueue | None = None,
        activates_runtime: bool = True,
        instructions: InstructionQueue | None = None,
    ) -> None:
        self._recipe = recipe
        #: The training instructions of every scenario; None builds scenarios that take no instruction.
        self._instructions = instructions
        self._activates_runtime = activates_runtime
        self._backend_factory = backend_factory
        self._local_artifact_dir = local_artifact_dir
        self._experiment_tracker = experiment_tracker
        self._storage = scenario_storage
        self.experiment_reports = (
            experiment_reports if experiment_reports is not None else ExperimentReportQueue(experiment_tracker)
        )

    def has_registration(self, scenario: str) -> bool:
        """True when the scenario is durably registered with the backend."""
        return isinstance(
            self._backend_factory, RegistrationAwareRepositoryBackendFactory
        ) and self._backend_factory.has_registration(scenario)

    def load_or_create(self, scenario: str, start: ItemStart | None = None) -> Scenario:
        """Recover the scenario ``scenario``, or create it when the backend has no registration of it.

        A new scenario gets its own root release (``register``): the artifact item ``start``, or the recipe's
        default start when ``start`` is ``None``. An existing scenario ignores ``start``.
        """
        backend = self._backend_factory(scenario)
        if self._storage.durable and not isinstance(backend, StagedReleaseRepositoryBackend):
            raise ArtifactPublicationError(
                "scenarios with durable commit storage require a backend implementing StagedReleaseRepositoryBackend"
            )
        metadata = backend.metadata()
        registration = None if metadata is None else metadata.get(SCENARIO_METADATA_KEY)
        surface = self._recipe.serving_surface(scenario)
        if registration is None:
            self.register(scenario, backend, surface, start)
            # register() is the atomic registration point. Another caller may have won it, so always rebuild from
            # the durable registration instead of assuming this creation attempt won.
            metadata = backend.metadata()
            registration = None if metadata is None else metadata.get(SCENARIO_METADATA_KEY)
            if registration is None:
                raise ReefError(f"scenario backend did not persist registration metadata for {scenario!r}")
        return self._recover(
            scenario,
            backend,
            registration,
            surface=surface,
            registered_components=release_components(metadata),
        )

    def register(self, scenario: str, backend: RepositoryBackend, surface: Surface, start: ItemStart | None) -> None:
        """Write the root release of the new scenario ``scenario`` and register it with ``backend``.

        The root holds the artifact item ``start`` in its component, or the recipe's default start
        (``Recipe.default_start``) when ``start`` is ``None``. A root of several components holds each other
        component's default start; a default start with files outside the component directories raises
        ``ReefError``. The registration metadata records the root's ``RootSource`` and, for several components, the
        manifest of the components that the root makes. An item that the component's layout or admission check
        refuses raises ``ValueError``, and so does an item root that any of the recipe's training components cannot
        start from (``Recipe.check_root_start``). Nothing is registered after a refusal.
        """
        if self._local_artifact_dir is not None:
            self._local_artifact_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="reef-root-", dir=self._local_artifact_dir) as work:
            root, source = self.root_release(surface, start, Path(work))
            if start is not None:
                self._recipe.check_root_start(ScenarioRoot(root, source))
            registration_metadata: dict[str, object] = {
                SCENARIO_METADATA_KEY: scenario_metadata_for(name=scenario, source=source)
            }
            if not surface.single:
                registration_metadata[COMPONENTS_METADATA_KEY] = root.metadata[COMPONENTS_METADATA_KEY]
            backend.register(root, metadata=registration_metadata)

    def root_release(self, surface: Surface, start: ItemStart | None, work: Path) -> tuple[Artifact, RootSource]:
        """The unpublished root release of a new scenario that ``surface`` serves, under ``work``, and its source."""
        item: tuple[str, Artifact] | None = None
        if start is None:
            default = self._recipe.default_start()
            source = RootSource("default", inputs=default.inputs)
        else:
            component, layout = surface.import_target(start.component)
            directory = work / "item"
            directory.mkdir()
            stored = layout.stored(start.files, directory)
            source = RootSource("item", item=start.item_id, uri=start.uri, version=start.files.version)
            # The admission checks of an import: the release's own with one component, else the component's here and
            # the release's on the composed root.
            if surface.single:
                surface.validate(stored)
                return stored, source
            surface.components[component].validator.validate(stored)
            item = (component, stored)
            default = self._recipe.default_start()
        tree = work / "default"
        tree.mkdir()
        default.write(tree)
        if surface.single:
            return Artifact.local(tree, content_id=default.content_id), source
        # Files outside every component directory, a model laid out at the root say, belong to no component.
        stray = sorted(entry.name for entry in tree.iterdir() if entry.name not in surface.names or not entry.is_dir())
        if stray:
            raise ReefError(
                f"the default start keeps {stray} outside its components {list(surface.names)}: a release serving "
                "several components keeps one directory per component"
            )
        components = {
            name: Artifact.local(tree / name, content_id=f"{default.content_id}:{name}") for name in surface.names
        }
        if item is None:
            return compose_release(components, directory=work / "root"), source
        components[item[0]] = item[1]
        root = compose_release(components, directory=work / "root")
        surface.validator.validate(root)
        return root, source

    def _recover(
        self,
        name: str,
        backend: RepositoryBackend,
        registration: object,
        *,
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
            unserved = [component for component in registered_components.names if component not in surface.names]
            if unserved:
                # A step carries forward the components the recipe serves: one it does not serve would leave
                # every later release, and the head would then refuse the recipe that registered them.
                raise ReefError(
                    f"scenario {name!r} was registered with components {list(registered_components.names)}; "
                    f"the recipe does not serve {unserved}"
                )
        elif registered_components is not None and not registered_components.single:
            # The reverse mismatch: a flat recipe would serve the composed root as its one tree and its next
            # step would publish a head without the other components.
            raise ReefError(
                f"scenario {name!r} was registered with components {list(registered_components.names)}; "
                "a recipe serving one component cannot reopen it"
            )
        checkpoint_head = backend.current()
        registered_name, start_artifact, root_source, checkpoint = parse_scenario_metadata(
            registration,
            checkpoint_head=checkpoint_head,
            components=(
                None
                if surface.single or registered_components is None
                else registered_components.artifact_ids(checkpoint_head.release_id)
            ),
        )
        if registered_name != name:
            raise ValueError(f"scenario metadata is for {registered_name!r}, not {name!r}")
        # What served after the artifact head's commit; the commit history's activations come after it.
        served_at_head = served_artifact_ref(registration, checkpoint_head=checkpoint_head)
        start_artifact = backend.resolve_release(start_artifact.release_id)
        checkpoint_step = 0 if checkpoint is None else checkpoint.step
        runtime = self._recipe.runtime
        store = self._storage.open(name)
        scenario: Scenario | None = None
        trainers: tuple[ComponentTrainer, ...] = ()
        try:
            head_record = store.recover(checkpoint=checkpoint)
            scenario_step = 0 if head_record is None else head_record.step
            recovered_states = recovered_trainer_states(store, name, head_record, surface)

            # Publication stages durable bytes before the commit record is durable, while
            # the backend's head is only a post-commit mirror. A crash between the
            # two leaves the commit history's checkpoint ahead of that pointer.
            if store.durable:
                checkpoints = [
                    record for record in store.history() if record.trained_artifact and record.step >= checkpoint_step
                ]
                if checkpoints:
                    checkpoint_head = checkpoints[-1].artifact_ref
            # The active artifact is the newest activation in the log; the chain tip, the parent of the next
            # publication, stays the checkpoint. Without a durable log, the artifact head's metadata names it.
            current_artifact = active_artifact_ref(committed_records(store, head_record), served_at_head)

            repository = Repository(
                backend,
                start_artifact,
                current_artifact=current_artifact,
                checkpoint_artifact=checkpoint_head,
                local_dir=self._local_artifact_dir,
            )
            repository.synchronize_checkpoint()
            # Traffic must not reach a recovered scenario before its committed
            # head is servable; a failed activation leaves the scenario unloaded.
            if self._activates_runtime:
                surface.activate(Artifact(current_artifact, repository), runtime)

            experiment_logger = self._experiment_tracker.bind_scenario(
                scenario=name,
                recipe=self._recipe.name,
                source_artifact_ref=repository.require_current_artifact(),
                run_segment=store.training_run_position()[0] if store.durable else 0,
            )
            trainers = self._recipe.build_trainers(
                name,
                store.records,
                surface=surface,
                algorithm_states={component: state.algorithm_state for component, state in recovered_states.items()},
                experiment_logger=experiment_logger,
                task_storage=store.tasks,
                releases=ScenarioReleaseFiles(name=name, store=store, repository=repository, surface=surface),
                samples=store.samples,
                root=ScenarioRoot(Artifact(start_artifact, repository), root_source),
            )
            if any(
                component_trainer.trainer.trains
                and component_trainer.trainer.pipeline.training_mode != self._recipe.training_mode
                for component_trainer in trainers
            ):
                # A trainer that runs no step keeps no mode; every stepping trainer runs the recipe's.
                raise ValueError("recipe.build must pass its training_mode to Trainer.build")
            if len(trainers) > 1 and not store.durable:
                raise ReefError(
                    f"scenario {name!r} runs a trainer per component: each recovers from its own commits, "
                    "which needs durable commit storage"
                )
            experiment_observer = None
            if not isinstance(self._experiment_tracker, NullExperimentTracker):
                experiment_observer = CommitExperimentObserver(
                    self._experiment_tracker,
                    scenario=name,
                    recipe=self._recipe.name,
                    source_artifact_ref=repository.require_current_artifact(),
                    scenario_step=scenario_step,
                    store=store,
                    reports=self.experiment_reports,
                )
                store = experiment_observer.store
            scenario = Scenario(
                name=name,
                binding=ScenarioBinding(
                    surface=surface,
                    runtime=runtime,
                    training_component=self._recipe.training_component,
                    activation_rule=self._recipe.activation_rule(),
                    # The recipe's own contract, not the first trainer's: a composite agrees
                    # one across its components, whatever order they are listed in.
                    report_type=self._recipe.report_type,
                ),
                repository=repository,
                root_source=root_source,
                trainers=trainers,
                scenario_step=scenario_step,
                store=store,
                recovered_head_record=head_record,
                experiment_observer=experiment_observer,
                instructions=None if self._instructions is None else self._instructions.of(name),
            )
            # Replay retained, unconsumed rows behind each trainer's watermark
            # before resuming its cursor. Retention may keep already-consumed
            # rows for audit.
            for component_trainer in trainers:
                # Scenario() checked that every trainer names a component the surface serves: each has a state.
                recovered = recovered_states[component_trainer.component]
                if recovered.high_water is not None:
                    scenario.replay_retained(
                        up_to_sequence=recovered.high_water[0],
                        consumed_ids=recovered.consumed_ids,
                        component=component_trainer.component,
                    )
                    scenario.restore_record_progress(
                        after_sequence=recovered.high_water[0],
                        offset=recovered.high_water[1],
                        component=component_trainer.component,
                    )
                elif recovered.consumed_ids:
                    scenario.replay_retained(
                        up_to_sequence=0, consumed_ids=recovered.consumed_ids, component=component_trainer.component
                    )
            scenario.resume_training_starts()
            return scenario
        except BaseException:
            if scenario is not None:
                scenario.close()
            else:
                try:
                    for component_trainer in trainers:
                        component_trainer.trainer.close()
                finally:
                    store.close()
            raise
