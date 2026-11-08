"""Create and restore scenarios from durable registration and committed state.

A new scenario owns its root snapshot: the recipe's default start, or an artifact item of its owner. Storage repair
precedes checkpoint synchronization and activation. The trainer
is then built from committed state and retained records are replayed. Any
failure closes the trainer and opened storage session owned by this attempt.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.artifact import (
    Artifact,
    ArtifactNotFound,
    ArtifactPublicationError,
    SnapshotRef,
    artifact_components,
)
from reef.artifact.components import ScenarioComponents
from reef.artifact.composite import compose_snapshot
from reef.artifact.repository import (
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendFactory,
    StagedSnapshotRepositoryBackend,
    selection_digest,
)
from reef.artifact.sources import SourceFiles
from reef.core.artifact_selection import ArtifactSelection
from reef.core.components import COMPONENTS_METADATA_KEY, RECORDS_COMPONENT, SnapshotComponents, snapshot_components
from reef.core.errors import ReefError
from reef.core.snapshot import SELECTION_VIEW_PREFIX, ItemIdentity, RootSource
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.observability.commits import CommitExperimentObserver, ExperimentReportQueue, training_starts
from reef.recipe.base import Recipe
from reef.scenario.binding import ScenarioBinding
from reef.scenario.scenario import Scenario
from reef.scenario.snapshot_files import ScenarioArtifactFiles
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    UNLOCATED_SNAPSHOT,
    CommitHistoryError,
    CommitRecord,
    CommitSnapshots,
    active_snapshot,
    parse_scenario_metadata,
    scenario_metadata_for,
    served_snapshot,
)
from reef.storage.scenario import ScenarioStorage, ScenarioStore
from reef.storage.served_models import ServedModels
from reef.train.component import ScenarioRoot
from reef.train.trainer import ComponentTrainer
from reef.train.training_requests import TrainingRequestQueue


class RepositoryCommitSnapshots(CommitSnapshots):
    """Finds each record's snapshot from its selection and the backend's bindings of artifact IDs.

    Values: ``backend`` is the scenario's backend, and ``start`` is its root snapshot. Bindings never change, so one
    lookup per artifact ID is enough. The ``snapshot`` of a record without one is:

    - a step or import that made an artifact: the binding of the artifact that this record names first;
    - a training step that made none: the snapshot that was active then;
    - any other record of one component, or a new training start: the binding of its artifact;
    - any other record of several components: the local view of its selection.
    """

    def __init__(self, backend: RepositoryBackend, start: SnapshotRef) -> None:
        self.backend = backend
        self.start = start
        self.bindings: dict[str, SnapshotRef] = {}
        self.start_ids: frozenset[str] | None = None

    def binding(self, artifact_id: str) -> SnapshotRef:
        if artifact_id not in self.bindings:
            self.bindings[artifact_id] = self.backend.artifact_snapshot(artifact_id)
        return self.bindings[artifact_id]

    def made_artifact(self, record: CommitRecord, seen: set[str]) -> str:
        """The artifact that ``record`` made. A lone trainer of several components names no component."""
        artifact_ids = record.artifact_ids or {}
        if record.component is not None:
            return artifact_ids[record.component]
        if len(artifact_ids) == 1:
            return next(iter(artifact_ids.values()))
        if self.start_ids is None:
            self.start_ids = frozenset(artifact_components(self.backend.materialize(self.start)))
        made = [id_ for id_ in artifact_ids.values() if id_ not in seen and id_ not in self.start_ids]
        if not made:
            raise CommitHistoryError(f"commit step {record.step} made an artifact, but names no new artifact ID")
        return made[0]

    def snapshot(self, record: CommitRecord, seen: set[str], active: SnapshotRef) -> SnapshotRef:
        selection = record.selection
        if selection is None:
            raise CommitHistoryError(f"commit step {record.step} has no selection")
        if record.checkpoint:
            return self.binding(self.made_artifact(record, seen))
        if record.operation == "training":
            return active
        if len(selection.artifact_ids) == 1:
            return self.binding(next(iter(selection.artifact_ids.values())))
        if record.operation == "train_from" and record.component is not None:
            return self.binding(selection.artifact_ids[record.component])
        return SnapshotRef(f"{SELECTION_VIEW_PREFIX}{selection_digest(selection)}", None)

    def locate(self, records: Sequence[CommitRecord]) -> tuple[CommitRecord, ...]:
        wanted = {
            artifact_id
            for record in records
            if record.snapshot == UNLOCATED_SNAPSHOT and (record.checkpoint or record.operation != "training")
            for artifact_id in (record.artifact_ids or {}).values()
            if artifact_id not in self.bindings
        }
        if wanted:
            self.bindings.update(self.backend.artifact_snapshots(wanted))
        seen: set[str] = set()
        active = self.start
        located = []
        for record in records:
            if record.snapshot == UNLOCATED_SNAPSHOT:
                try:
                    record = replace(record, snapshot=self.snapshot(record, seen, active))
                except ArtifactNotFound as exc:
                    raise CommitHistoryError(f"commit step {record.step} names a missing artifact: {exc}") from exc
            seen.update((record.artifact_ids or {}).values())
            if record.activates:
                active = record.snapshot
            located.append(record)
        return tuple(located)


def committed_records(store: ScenarioStore, head_record: CommitRecord | None) -> tuple[CommitRecord, ...]:
    records = store.history()
    if not records and head_record is not None:
        # No durable log: the head adopted from checkpoint metadata is the
        # only committed step there is.
        records = (head_record,)
    return records


@dataclass(frozen=True)
class ItemStart:
    """The artifact item that a new scenario's root snapshot holds.

    Values: ``item`` is the item's ``ItemIdentity``, and ``uri`` its non-empty ``source_uri``. ``files`` is the
    ``SourceFiles`` that ``uri`` names, not yet checked. ``component`` is the component that the item fills, or
    ``None`` for the scenario's one component. The root's artifact of ``component`` keeps the item's artifact ID.

    Purpose: ``ArtifactImports.create_from_item`` makes it for ``POST /reef/scenarios`` with ``artifact_id``.
    ``ScenarioFactory.register`` checks it and writes it as the root snapshot.
    """

    item: ItemIdentity
    uri: str
    files: SourceFiles
    component: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item, ItemIdentity):
            raise TypeError("an item start's item is an ItemIdentity")
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("an item start names a non-empty uri")
        if not isinstance(self.files, SourceFiles):
            raise TypeError("an item start's files are SourceFiles")
        if self.component is not None and (not isinstance(self.component, str) or not self.component):
            raise ValueError("an item start's component is a non-empty string or None")


@dataclass(frozen=True)
class RecoveredTrainerState:
    """What one component's trainer recovers from its own commits: state and consumed rows.

    ``committed_ids`` holds the IDs that its commits named. ``consumed_ids`` holds those and the records that
    its stored samples consumed.
    """

    algorithm_state: Mapping[str, Any] | None
    consumed_ids: frozenset[str]
    committed_ids: frozenset[str]


def recovered_trainer_states(
    store: ScenarioStore, scenario: str, head_record: CommitRecord | None, components: ScenarioComponents
) -> dict[str, RecoveredTrainerState]:
    """What each component's trainer recovers from its own commits.

    The only trainer of a one-component (or record-only) scenario owns every
    training commit, which names no component.
    Activations and new training starts carry no trainer state, so no
    trainer recovers from them. The records that a trainer's stored samples
    consumed (``TrainingState.consumed_record_ids``) count as consumed too, so a
    replay never makes their samples again.

    Rehydration must skip consumed rows: retention may keep a consumed row
    stored (audit-only retention is contract-legal), and pushing one again would
    train it twice. Consumption is permanent, so the union over the whole log
    is the exclusion set; with several trainers each keeps its own share, the rows
    of its own commits.
    """
    records = committed_records(store, head_record)
    consumed_by_component: dict[str | None, set[str]] = {}
    for record in records:
        consumed_by_component.setdefault(record.component, set()).update(record.consumed_ids)
    names = components.names or (RECORDS_COMPONENT,)
    alone = len(names) == 1
    states: dict[str, RecoveredTrainerState] = {}
    for component in names:
        own = tuple(
            record
            for record in records
            if record.operation == "training"
            and (record.component == component or (alone and record.component is None))
        )
        last = own[-1] if own else None
        # The records that the trainer's stages consumed: its key is "" when it is alone.
        stored = store.training_state.consumed_record_ids("" if alone else component)
        committed = (
            frozenset().union(*consumed_by_component.values())
            if alone
            else frozenset(consumed_by_component.get(component, ()))
        )
        states[component] = RecoveredTrainerState(
            algorithm_state=None if last is None else last.algorithm_state,
            consumed_ids=stored | committed,
            committed_ids=committed,
        )
    return states


class ScenarioFactory:
    """Register, assemble, and recover complete scenario instances.

    Inference loads each selected release. ``served_models`` supplies the actual serving artifact for training.
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
        served_models: ServedModels | None = None,
        training_requests: TrainingRequestQueue | None = None,
    ) -> None:
        self._recipe = recipe
        #: The training requests of every scenario; None builds scenarios that take no request.
        self._training_requests = training_requests
        self.served_models = served_models
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

        A new scenario gets its own root snapshot (``register``): the artifact item ``start``, or the recipe's
        default start when ``start`` is ``None``. An existing scenario ignores ``start``.
        """
        backend = self._backend_factory(scenario)
        if self._storage.durable and not isinstance(backend, StagedSnapshotRepositoryBackend):
            raise ArtifactPublicationError(
                "scenarios with durable commit storage require a backend implementing StagedSnapshotRepositoryBackend"
            )
        metadata = backend.metadata()
        registration = None if metadata is None else metadata.get(SCENARIO_METADATA_KEY)
        components = self._recipe.scenario_components(scenario)
        if registration is None:
            self.register(scenario, backend, components, start)
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
            components=components,
            registered_components=snapshot_components(metadata),
        )

    def register(
        self, scenario: str, backend: RepositoryBackend, components: ScenarioComponents, start: ItemStart | None
    ) -> None:
        """Write the root snapshot of the new scenario ``scenario`` and register it with ``backend``.

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
            root, source = self.root_snapshot(components, start, Path(work))
            if start is not None:
                self._recipe.check_root_start(ScenarioRoot(root, source))
            registration_metadata: dict[str, object] = {
                SCENARIO_METADATA_KEY: scenario_metadata_for(name=scenario, source=source)
            }
            names = components.names or (RECORDS_COMPONENT,)
            if components.single:
                registration_metadata[COMPONENTS_METADATA_KEY] = {
                    names[0]: {"artifact_id": root.artifact_id, "metadata": {}}
                }
            else:
                registration_metadata[COMPONENTS_METADATA_KEY] = root.metadata[COMPONENTS_METADATA_KEY]
            backend.register(root, metadata=registration_metadata)

    def root_snapshot(
        self, components: ScenarioComponents, start: ItemStart | None, work: Path
    ) -> tuple[Artifact, RootSource]:
        """The unpublished root snapshot of a new scenario that ``components`` serves, under ``work``, and its source."""
        item: tuple[str, Artifact] | None = None
        if start is None:
            default = self._recipe.default_start()
            source = RootSource("default", inputs=default.inputs)
        else:
            component, layout = components.import_target(start.component)
            directory = work / "item"
            directory.mkdir()
            stored = layout.stored(start.files, directory, artifact_id=start.item.artifact_id)
            source = RootSource("item", item=start.item.item_id, uri=start.uri, version=start.files.version)
            # The admission checks of an import: the snapshot's own with one component, else the component's here and
            # the snapshot's on the composed root.
            if components.single:
                components.validate(stored)
                return stored, source
            components.components[component].validator.validate(stored)
            item = (component, stored)
            default = self._recipe.default_start()
        tree = work / "default"
        tree.mkdir()
        default.write(tree)
        if components.single:
            return Artifact.local(tree), source
        # Files outside every component directory, a model laid out at the root say, belong to no component.
        stray = sorted(
            entry.name for entry in tree.iterdir() if entry.name not in components.names or not entry.is_dir()
        )
        if stray:
            raise ReefError(
                f"the default start keeps {stray} outside its components {list(components.names)}: a snapshot serving "
                "several components keeps one directory per component"
            )
        artifacts = {name: Artifact.local(tree / name) for name in components.names}
        if item is None:
            return compose_snapshot(artifacts, directory=work / "root"), source
        artifacts[item[0]] = item[1]
        root = compose_snapshot(artifacts, directory=work / "root")
        components.validator.validate(root)
        return root, source

    def _recover(
        self,
        name: str,
        backend: RepositoryBackend,
        registration: object,
        *,
        components: ScenarioComponents,
        registered_components: SnapshotComponents | None,
    ) -> Scenario:
        if not isinstance(registration, Mapping):
            raise ValueError(f"invalid scenario metadata for {name!r}")
        if not components.single:
            # A registration names the components its snapshots bind. One made by a
            # recipe serving a single component, or none, has releases with no
            # component directories to carry forward; refuse rather than compose
            # the whole flat tree into every component.
            if registered_components is None:
                raise ReefError(
                    f"scenario {name!r} was registered without a component manifest; a recipe serving components "
                    f"{list(components.names)} needs a scenario registered with that layout"
                )
            missing = [component for component in components.names if component not in registered_components.entries]
            if missing:
                raise ReefError(
                    f"scenario {name!r} was registered with components {list(registered_components.names)}; "
                    f"the recipe also serves {missing}"
                )
            unserved = [component for component in registered_components.names if component not in components.names]
            if unserved:
                # A step carries forward the components the recipe serves: one it does not serve would leave
                # every later snapshot, and the head would then refuse the recipe that registered them.
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
                None if registered_components is None else registered_components.artifact_ids(checkpoint_head.revision)
            ),
        )
        if registered_name != name:
            raise ValueError(f"scenario metadata is for {registered_name!r}, not {name!r}")
        # What served after the artifact head's commit; the commit history's activations come after it.
        served_at_head = served_snapshot(registration, checkpoint_head=checkpoint_head)
        start_artifact = backend.resolve_snapshot(start_artifact.revision)
        checkpoint_step = 0 if checkpoint is None else checkpoint.step
        runtime = self._recipe.runtime
        store = self._storage.open(name)
        store.use_snapshots(RepositoryCommitSnapshots(backend, start_artifact))
        scenario: Scenario | None = None
        trainers: tuple[ComponentTrainer, ...] = ()
        try:
            head_record = store.recover(checkpoint=checkpoint)
            scenario_step = 0 if head_record is None else head_record.step
            recovered_states = recovered_trainer_states(store, name, head_record, components)

            # Publication stages durable bytes before the commit record is durable, while
            # the backend's head is only a post-commit mirror. A crash between the
            # two leaves the commit history's checkpoint ahead of that pointer.
            if store.durable:
                checkpoints = [
                    record for record in store.history() if record.trained_artifact and record.step >= checkpoint_step
                ]
                if checkpoints:
                    checkpoint_head = checkpoints[-1].snapshot
            # The active artifact is the newest activation in the log; the chain tip, the parent of the next
            # publication, stays the checkpoint. Without a durable log, the artifact head's metadata names it.
            current_artifact = active_snapshot(committed_records(store, head_record), served_at_head)

            repository = Repository(
                backend,
                start_artifact,
                current_artifact=current_artifact,
                checkpoint_artifact=checkpoint_head,
                local_dir=self._local_artifact_dir,
            )
            active_selection = registration.get("active_selection")
            records = committed_records(store, head_record)
            if active_selection is not None:
                repository.remember_selection(served_at_head, ArtifactSelection.from_dict(active_selection))
            for record in records:
                if record.selection is not None:
                    repository.remember_selection(record.snapshot, record.selection)
            repository.synchronize_checkpoint()

            root_artifacts = repository.selection(start_artifact, components.names or (RECORDS_COMPONENT,))
            training_artifacts = root_artifacts
            for record in committed_records(store, head_record):
                training_artifacts = training_starts(training_artifacts, record)
            experiment_logger = self._experiment_tracker.bind_scenario(
                scenario=name,
                recipe=self._recipe.name,
                source_artifacts=training_artifacts,
                run_segment=store.training_run_position()[0] if store.durable else 0,
            )
            trainers = self._recipe.build_trainers(
                name,
                store.records,
                components=components,
                algorithm_states={component: state.algorithm_state for component, state in recovered_states.items()},
                experiment_logger=experiment_logger,
                task_storage=store.tasks,
                artifacts=ScenarioArtifactFiles(name=name, store=store, repository=repository, components=components),
                samples=self._storage.sample_store(name, store),
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
                    source_artifacts=root_artifacts,
                    scenario_step=scenario_step,
                    store=store,
                    reports=self.experiment_reports,
                )
                store = experiment_observer.store
            scenario = Scenario(
                name=name,
                binding=ScenarioBinding(
                    components=components,
                    runtime=runtime,
                    training_component=self._recipe.training_component,
                    activation_rule=self._recipe.activation_rule(),
                    served_models=self.served_models,
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
                training_requests=None if self._training_requests is None else self._training_requests.of(name),
            )
            # Each trainer reads its records from the start again and skips the consumed ones. A commit's record
            # progress is not a cursor here: it can name a sequence of a record store that came before the record
            # items (RFC #1219).
            for component_trainer in trainers:
                # Scenario() checked that every trainer names a component the scenario serves: each has a state.
                recovered = recovered_states[component_trainer.component]
                component_trainer.trainer.restore_committed_records(recovered.committed_ids)
                component_trainer.trainer.restore_consumed_records(recovered.consumed_ids)
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
