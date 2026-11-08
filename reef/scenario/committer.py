"""Scenario committer coordinating training and artifact publication.

Artifact storage and head movement live in ``reef.artifact``; the durable
record values and checkpoint metadata formats live in ``commits``. This
module owns the scenario-specific ordering across trainer state, store
settlement, runtime activation, and artifact operations.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactNotFound,
    ArtifactPublicationError,
    SnapshotRef,
    training_starts_of,
    with_training_starts,
)
from reef.artifact.components import ArtifactLayout
from reef.artifact.snapshot_chain import ArtifactSnapshotChain, SnapshotNotRestorable
from reef.artifact.sources import SourceFiles
from reef.core.activation import MadeArtifact
from reef.core.artifact_selection import ArtifactSelection
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, SnapshotComponents, validate_component_name
from reef.core.errors import CommitPending, ReefError
from reef.core.snapshot import ItemIdentity, RootSource
from reef.observability.base import TrainingExperimentContext
from reef.observability.commits import CommitExperimentObserver
from reef.scenario.binding import ScenarioBinding
from reef.scenario.snapshots import ScenarioSnapshots, ServedBesideArtifact
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    CommitOperation,
    CommitRecord,
    RecordProgress,
    active_record,
    active_snapshot,
    scenario_metadata_for,
    to_json_value,
)
from reef.storage.scenario import ScenarioStore, ScenarioStoreConflict
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import NoArtifactPublication, PreparedCommit, SavedArtifactPublication, TrainingCommit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentArtifact:
    """The selected artifact of one component, returned by scenario operations.

    ``component`` is a valid component name; ``artifact_id`` is a nonempty
    immutable artifact identity. The service returns these values to callers
    without exposing the storage snapshot used to materialize their files.
    """

    component: str
    artifact_id: str

    def __post_init__(self) -> None:
        validate_component_name(self.component)
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("artifact_id must be a nonempty string")


@dataclass(frozen=True)
class ArtifactHeadSync:
    """The result of synchronizing the backend head with committed scenario state.

    ``state`` is synchronized, pending, or conflict. ``error`` is None on
    success, otherwise failure text. The committer produces this status;
    the service exposes it without leaking a storage locator.
    """

    state: Literal["synchronized", "pending", "conflict"]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.state not in ("synchronized", "pending", "conflict"):
            raise ValueError("unknown snapshot synchronization state")
        if self.state == "synchronized":
            if self.error is not None:
                raise ValueError("a synchronized snapshot has no error")
        elif not isinstance(self.error, str):
            raise ValueError("an unsynchronized snapshot requires error text")


class ScenarioCommitter:
    """Order one scenario's commits, activations, new training starts, imports, syncs and retries; delegate release
    queries."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        artifacts: ArtifactSnapshotChain,
        trainers: tuple[ComponentTrainer, ...],
        root_source: RootSource,
        scenario_step: int = 0,
        store: ScenarioStore,
        recovered_head_record: CommitRecord | None = None,
        experiment_observer: CommitExperimentObserver | None = None,
    ) -> None:
        if not isinstance(scenario_step, int) or scenario_step < 0:
            raise ValueError("scenario_step must be non-negative")
        if store.durable:
            artifacts.repository.require_staged_commit_support()
        if not trainers:
            raise ValueError("a scenario commits through at least one trainer")
        self._name = name
        self._binding = binding
        #: Decides whether a training step's artifact serves. It starts as the binding's rule.
        self.activation_rule = binding.activation_rule
        self._artifacts = artifacts
        #: Where the scenario's root snapshot came from; every snapshot that a training step makes records it.
        self.root_source = root_source
        self.trainers = trainers
        self._step = scenario_step
        self._store = store
        self.experiment_observer = experiment_observer
        self._lock = RLock()
        # A lone trainer's preparation holds the operation lock across proposer
        # calls and evaluation rollouts, neither of which changes committed
        # releases. With several trainers a preparation runs outside it; the
        # dispatcher still runs one step of the scenario at a time.
        # Readers share only the publication lock with commit and activation.
        # Writers must acquire the operation lock first; readers never take it.
        self._publication_lock = RLock()
        #: Read-only release queries. They share the publication lock with commit and activation.
        self.snapshot_queries = ScenarioSnapshots(
            name=name,
            artifacts=artifacts,
            store=store,
            publication_lock=self._publication_lock,
            root_source=root_source,
        )
        records = (
            (() if recovered_head_record is None else (recovered_head_record,))
            if not store.durable
            else store.history()
        )
        self._latest_training_record = next(
            (record for record in reversed(records) if record.operation == "training"),
            None,
        )
        self._artifact_head_sync = ArtifactHeadSync("synchronized")
        self._commit_status = (scenario_step, self._latest_training_record, self._artifact_head_sync)

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def step(self) -> int:
        return self._step

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """Current step and latest training outcome read without blocking."""
        step, record, head_sync = self._commit_status
        return {
            "scenario_step": step,
            "artifact_head_sync": asdict(head_sync),
            "last_committed_step": (
                None
                if record is None
                else {
                    "step": record.step,
                    "recorded_at": record.recorded_at,
                    "metrics": None if record.metrics is None else deepcopy(record.metrics),
                }
            ),
        }

    def advance_to(self, step: int) -> None:
        if step != self._step + 1:
            raise ValueError(f"scenario step must advance from {self._step} to {self._step + 1}")
        self._step = step
        self._commit_status = (step, self._latest_training_record, self._artifact_head_sync)

    def component_trainer(self, component: str | None) -> ComponentTrainer:
        """Return the component trainer of ``component``. ``None`` selects the first, for scenario-wide operations."""
        if component is None:
            return self.trainers[0]
        for component_trainer in self.trainers:
            if component_trainer.component == component:
                return component_trainer
        raise ReefError(f"scenario {self._name!r} has no trainer for component {component!r}")

    def trainer_for(self, component: str | None) -> Trainer:
        return self.component_trainer(component).trainer

    def component_on_record(self, component: str | None) -> str | None:
        """The trainer a record or receipt names: ``component`` with several, none with one.

        A scenario of one trainer writes its commit records, release metadata,
        consumption receipts and tracking as it did before components, so
        every write path names no component for it.
        """
        return component if len(self.trainers) > 1 else None

    def own_record(self, record: CommitRecord, component: str | None) -> bool:
        """Whether ``component``'s trainer made the training record ``record``; one naming no trainer is the lone trainer's.

        ``component`` is the name on record (``component_on_record``). An
        activation or a new training start is no trainer's own record: it
        carries no trainer state.
        """
        return record.operation == "training" and record.component == component

    def starts_trainer(self, record: CommitRecord, component: str) -> bool:
        """Whether ``record`` sets the training start of ``component``'s trainer.

        It does when it ``sets_training_start`` and names that trainer: a
        training or ``"train_from"`` record names it as
        ``component_on_record`` does, and a ``"sync"`` record names
        ``component`` or, in a scenario of one component, no component.
        """
        if not record.sets_training_start:
            return False
        if record.operation == "sync":
            return record.component in (None, component)
        return record.component == self.component_on_record(component)

    def artifact_status(self) -> dict[str, dict[str, str | None]]:
        """Each component's active artifact and training start, by ``artifact_id``; ``None`` where none is known.

        The active artifact is the component's artifact in the active
        selection. A trainer's training start is the artifact of its newest
        record that sets it (``starts_trainer``): a ``"train_from"`` or
        ``"sync"`` record, or the newest artifact it made, and the creation's
        artifact before any of them. A component whose
        artifact the records do not name, or that no trainer trains, reads
        ``None``.
        """
        names = self.component_names()
        queries = self.snapshot_queries
        records = queries.records()
        creation = queries.creation_artifact.revision

        def bound(revision: str, name: str) -> str | None:
            """The ``artifact_id`` of ``name`` in the snapshot ``revision``, as the records name it."""
            return queries.bound_artifact_ids(revision, names, self._step).get(name)

        active = active_record(records)
        active_revision = creation if active is None else active.snapshot.revision
        recovered_active: Mapping[str, str] = {}
        if active is None and self._artifacts.current != queries.creation_artifact:
            repository = self._artifacts.repository
            recovered_active = repository.selection(self._artifacts.current, names).artifact_ids
        trainers = {component_trainer.component: component_trainer for component_trainer in self.trainers}
        status: dict[str, dict[str, str | None]] = {}
        for name in names:
            start: str | None = None
            if name in trainers:
                latest = next((record for record in reversed(records) if self.starts_trainer(record, name)), None)
                start = bound(creation if latest is None else latest.snapshot.revision, name)
            status[name] = {
                "active": recovered_active.get(name, bound(active_revision, name)),
                "training_start": start,
            }
        return status

    def last_record_for(self, component: str | None) -> CommitRecord | None:
        """The newest durable training commit made by ``component``'s trainer."""
        records = self._store.history() if self._store.durable else ()
        name = self.component_on_record(self.component_trainer(component).component)
        return next((record for record in reversed(records) if self.own_record(record, name)), None)

    def snapshot_manifest(self, artifact: Artifact) -> SnapshotComponents:
        """The manifest a snapshot carries: its own, or the one-component manifest a flat snapshot implies."""
        manifest = artifact.components
        if manifest is not None:
            return manifest
        components = self._binding.components
        if not components.single:
            raise ReefError(
                f"scenario {self._name!r} serves components {list(components.names)} but release "
                f"{artifact.ref.revision!r} carries no component manifest"
            )
        name = components.names[0] if components.names else self.trainers[0].component
        return SnapshotComponents({name: ComponentEntry(artifact.artifact_id)})

    def require_components(self, artifact: Artifact) -> None:
        """Refuse a snapshot that does not bind every component this scenario serves.

        Reading such a snapshot as composed would hand the whole tree, or
        nothing, to each component; every snapshot a composed scenario
        registers or publishes carries a full manifest, so this is a
        misdirected repository, not a supported input.
        """
        components = self._binding.components
        if components.single:
            return
        manifest = self.snapshot_manifest(artifact)
        missing = [name for name in components.names if name not in manifest.entries]
        if missing:
            raise ReefError(
                f"scenario {self._name!r} serves components {list(components.names)} but release "
                f"{artifact.ref.revision!r} binds {list(manifest.names)}"
            )

    def recorded_components(self, artifact: Artifact, revision: str) -> dict[str, str] | None:
        """The ``artifact_id`` of each component that ``artifact``, published as ``revision``, binds; ``None`` for a scenario of one component."""
        if self._binding.components.single:
            return None
        return self.snapshot_manifest(artifact).artifact_ids(revision)

    def import_target(self, component: str | None) -> tuple[str, ArtifactLayout]:
        """The component an import into ``component`` lands in, and the layout that checks its directory
        (``ScenarioComponents.import_target``). Purpose: the import service calls it before it downloads anything."""
        return self._binding.components.import_target(component)

    def component_names(self) -> tuple[str, ...]:
        """The components whose artifacts this scenario names: the declared ones, or the lone trainer's when it serves none."""
        return self._binding.components.names or (self.trainers[0].component,)

    def require_no_pending_commit(self, action: str) -> None:
        """Raise ``CommitPending`` while any trainer's result waits for its commit; ``action`` names what waits."""
        if any(component_trainer.trainer.has_pending_commit for component_trainer in self.trainers):
            raise CommitPending(f"cannot {action} while a training result waits for its commit")

    def artifact_snapshot(self, artifact_id: str, component: str | None) -> tuple[str, SnapshotRef]:
        """The component that ``artifact_id`` names, and the ref of the durable snapshot that minted it.

        It reads the commit history alone and does not materialize the snapshot. ``component`` and ``artifact_id``
        are as for :meth:`resolve_artifact`, which raises the same errors, except for the check of the snapshot's
        components.
        """
        names = self.component_names()
        if component is None:
            if len(names) > 1:
                raise ValueError(f"scenario {self._name!r} has components {list(names)}: name the component")
            component = names[0]
        if component not in names:
            raise ValueError(f"scenario {self._name!r} has no component {component!r}")
        listed = self.snapshot_queries.artifacts(self._step, names)
        row = next((row for row in listed if (row["artifact_id"], row["component"]) == (artifact_id, component)), None)
        if row is None:
            raise ArtifactNotFound(
                f"scenario {self._name!r} has no artifact {artifact_id!r} of component {component!r}"
            )
        found = self.snapshot_queries.find_snapshot(row["snapshot"].revision)
        if found is None or not found[1]:
            raise SnapshotNotRestorable(f"scenario {self._name!r} artifact {artifact_id!r} has no durable bytes")
        return component, found[0]

    def resolve_artifact(self, artifact_id: str, component: str | None) -> tuple[ComponentArtifact, Artifact]:
        """The artifact ``artifact_id`` names as ``component``'s content, and the resolved release that minted it.

        With one component, ``component`` may be ``None``. With several,
        ``component`` is required. ``artifact_id`` is an opaque identity from
        ``ScenarioSnapshots.artifacts``, independent of its storage revision. Raise
        ``ValueError`` for a missing or unknown component,
        ``ArtifactNotFound`` for an ``artifact_id`` the scenario does not
        list, and ``SnapshotNotRestorable`` for a snapshot without durable
        bytes or without every component the scenario serves.
        """
        component, ref = self.artifact_snapshot(artifact_id, component)
        snapshot = self._artifacts.resolve(ref)
        try:
            self.require_components(snapshot)
        except ReefError as exc:
            raise SnapshotNotRestorable(str(exc)) from exc
        return ComponentArtifact(component, artifact_id), snapshot

    def activate(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Make ``artifact_id`` the active artifact of ``component``; return its component and ID.

        The selected artifact replaces only its component in the scenario's
        active selection. No artifact is published. The checkpoint and every
        training start stay. Inference loads the snapshot after the commit.
        The ``"activate"`` record lands at the next step, then the active selection moves.
        A crash before the record leaves the active selection that the log names. Activating the active artifact
        changes nothing. Raise as :meth:`resolve_artifact` does, and
        ``CommitPending`` while a training result waits for its commit.
        """
        with self._lock, self._publication_lock:
            selected, snapshot = self.resolve_artifact(artifact_id, component)
            components = self._binding.components
            current_ref = self._artifacts.current
            served = Artifact(current_ref, self._artifacts.repository)
            if components.component_artifact(served, selected.component).artifact_id == selected.artifact_id:
                return selected
            self.require_no_pending_commit("activate an artifact")
            target = self.snapshot_with(served, selected, snapshot)
            record = self.append_operation_record(
                "activate",
                target.ref,
                component=None if components.single else selected.component,
                components=self.recorded_components(target, target.ref.revision),
            )
            self._artifacts.activate(target.ref, expected=current_ref)
            self.advance_to(record.step)
            return selected

    def snapshot_with(self, active: Artifact, selected: ComponentArtifact, snapshot: Artifact) -> Artifact:
        """A validated local view of ``active`` with one existing artifact replaced; publish nothing."""
        components = self._binding.components
        if components.single:
            components.validate(snapshot)
            return snapshot
        repository = self._artifacts.repository
        artifact_ids = dict(repository.selection(active.ref, self.component_names()).components)
        artifact_ids[selected.component] = selected.artifact_id
        cached = set(repository.selections)
        target = repository.select(ArtifactSelection(artifact_ids))
        try:
            components.validator.validate(target)
        except BaseException:
            if target.ref.revision not in cached:
                repository.discard(target)
            raise
        return target

    def restore(self, step: int) -> Mapping[str, str]:
        """Restore the complete active component mapping after ``step`` without changing artifacts or training starts.

        Zero selects the scenario's creation. Other steps require enough operation history to reconstruct their
        active selection. Validation and runtime activation precede the new restore record; recovery reads that
        record's complete selection.
        """
        with self._lock, self._publication_lock:
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError("restore step must be a non-negative integer")
            records = self.snapshot_queries.records()
            if step and not any(record.step == step for record in records):
                raise ValueError(f"scenario {self._name!r} has no operation at step {step}")
            self.require_no_pending_commit("restore a selection")
            preceding = tuple(record for record in records if record.step <= step)
            previous = active_record(preceding)
            if step and previous is None and preceding[0].step > 1:
                raise ValueError(f"cannot restore step {step}: earlier operation history is unavailable")
            ref = self.snapshot_queries.creation_artifact if previous is None else previous.snapshot
            repository = self._artifacts.repository
            selection = (
                previous.selection
                if previous is not None and previous.selection is not None
                else repository.selection(ref, self.component_names())
            )
            current_ref = self._artifacts.current
            if repository.selection(current_ref, self.component_names()).artifact_ids == selection.artifact_ids:
                return selection.artifact_ids
            cached = set(repository.selections)
            target = repository.select(selection)
            try:
                components = self._binding.components
                if components.single:
                    components.validate(target)
                else:
                    components.validator.validate(target)
            except BaseException:
                if target.ref.selection_view and target.ref.revision not in cached:
                    repository.discard(target)
                raise
            record = self.append_operation_record(
                "restore", target.ref, component=None, components=selection.artifact_ids, selection=selection
            )
            self._artifacts.activate(target.ref, expected=current_ref)
            self.advance_to(record.step)
            return selection.artifact_ids

    def serve(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Serve ``artifact_id`` as ``component`` beside the active selection; return it with the snapshot that serves it.

        The snapshot that serves it is the active selection with the artifact in
        its component (:meth:`snapshot_with`).
        Then the ``"serve"`` record lands at the next step. The active
        selection, the chain tip and every training start stay. Serving a
        served artifact changes nothing. An artifact of the active selection can
        be served too, so that it keeps serving after another activation.

        Raise as :meth:`resolve_artifact` does, and ``CommitPending`` while a
        training result waits for its commit. Raise ``ReefError`` when the
        scenario keeps no durable commit history. Raise it also when the
        runtime answers only from the snapshot that it has loaded
        (``InferenceRuntime.serves_any_artifact`` is False).
        """
        with self._lock, self._publication_lock:
            if not self._store.durable:
                raise ReefError(
                    f"scenario {self._name!r} keeps no durable commit history, so it serves its active selection alone"
                )
            selected, snapshot = self.resolve_artifact(artifact_id, component)
            served = self.snapshot_queries.served_beside(self._step, self.component_names()).get(selected.artifact_id)
            if served is not None:
                return selected
            self.require_no_pending_commit("serve an artifact")
            components = self._binding.components
            current = Artifact(self._artifacts.current, self._artifacts.repository)
            loaded = components.loaded_component
            runtime = self._binding.runtime
            # The calls use the artifact's own weights, or the active selection's weights for an artifact of another
            # component. The runtime must answer each call from the snapshot that the call names.
            if loaded is not None and runtime is not None and not runtime.serves_any_artifact:
                raise ReefError(
                    f"scenario {self._name!r}'s runtime answers only from the snapshot it has loaded, so it "
                    f"cannot serve artifact {selected.artifact_id!r} beside the active selection"
                )
            target = self.snapshot_with(current, selected, snapshot)
            record = self.append_operation_record(
                "serve",
                target.ref,
                component=None if components.single else selected.component,
                components=self.recorded_components(target, target.ref.revision),
            )
            self.advance_to(record.step)
            return selected

    def stop_serving(self, artifact_id: str) -> ServedBesideArtifact:
        """Stop serving ``artifact_id`` beside the active selection; return it with the snapshot that served it.

        The ``"stop_serving"`` record names the snapshot, the component and the
        manifest of the ``"serve"`` record that it ends. The active selection
        stays, so an artifact that it binds still answers. Raise
        ``ArtifactNotFound`` when the scenario does not serve ``artifact_id``
        beside its active selection. Raise ``CommitPending`` while a training
        result waits for its commit.
        """
        with self._lock, self._publication_lock:
            served = self.snapshot_queries.served_beside(self._step, self.component_names()).get(artifact_id)
            if served is None:
                raise ArtifactNotFound(
                    f"scenario {self._name!r} does not serve artifact {artifact_id!r} beside its active selection"
                )
            self.require_no_pending_commit("stop serving an artifact")
            if self._binding.components.single:
                recorded_component, components = None, None
            else:
                recorded_component = served.component
                components = served.selection.artifact_ids
            view = self._artifacts.repository.select(served.selection)
            record = self.append_operation_record(
                "stop_serving",
                view.ref,
                component=recorded_component,
                components=components,
                selection=served.selection,
            )
            self.advance_to(record.step)
            return served

    def train_from(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Set ``component``'s training start to ``artifact_id``; return the artifact.

        The trainer's running job is cancelled, and its next job or step
        starts from the artifact. The active selection does not change.
        ``component`` may be left out when the scenario has one component.
        The trainer takes the start first, then the ``"train_from"`` record
        lands at the next step; a restart before the trainer's next training
        commit takes the start again from that record. Raise as
        :meth:`resolve_artifact` does, ``TrainingStartRefused`` when the
        trainer cannot start from the artifact, and ``CommitPending`` while
        a training result waits for its commit.
        """
        # The start can import weights for minutes: readers of the snapshots wait only for the record.
        with self._lock:
            selected, snapshot = self.resolve_artifact(artifact_id, component)
            component_trainer = self.component_trainer(selected.component)
            self.require_no_pending_commit("set the training start")
            self.start_trainer(component_trainer, snapshot)
            with self._publication_lock:
                record = self.append_operation_record(
                    "train_from", snapshot.ref, component=self.component_on_record(component_trainer.component)
                )
                self.advance_to(record.step)
            return selected

    def import_artifact(
        self,
        files: SourceFiles,
        component: str | None,
        *,
        source: Mapping[str, str],
        directory: Path,
        item: ItemIdentity | None = None,
    ) -> ComponentArtifact:
        """Publish ``files``, which an import's URI names, as a new artifact of ``component``; return it.

        ``component`` may be left out when the scenario has one component.
        The component's layout (``ComponentSpec.layout``) checks ``files``
        and stores them in the empty ``directory``, outside the locks, since
        the check can ask a remote service (``ArtifactLayout.stored``). The snapshot's
        own admission check runs on what is stored, as at a training
        publication. The imported artifact receives an independent ID,
        including when its bytes match an existing artifact, unless
        the files are those of the artifact item ``item`` with an artifact ID (RFC #1218): the artifact
        keeps that ID, and when ``component`` lists that ID already, this call
        returns it and reads nothing. The snapshot binds the artifact as ``component`` and every other
        component as the active selection binds it; its parent is the chain tip,
        and it is published beside the chain. An ``"import"`` record lands at
        the next step with ``source``, the ``{"uri", "version"}`` that the
        files were read from (and the ``item`` of an artifact item's import), as its metrics. The active selection, the chain
        tip, every training start and the activation rule are left alone.
        Raise ``ValueError`` for an unknown component, a component that takes
        no import, or files that the layout or the admission check refuses;
        ``ArtifactSourceError`` when the copy fails; ``CommitPending`` while a
        training result waits for its commit. Nothing changes then.
        """
        component, layout = self.import_target(component)
        artifact_id = None if item is None else item.artifact_id
        if artifact_id is not None:
            with self._publication_lock:
                listed = self.snapshot_queries.artifacts(self._step, self.component_names())
            if any((row["artifact_id"], row["component"]) == (artifact_id, component) for row in listed):
                return ComponentArtifact(component, artifact_id)
        components = self._binding.components
        spec = components.components[component]
        imported = layout.stored(files, directory, artifact_id=artifact_id)
        with self._lock, self._publication_lock:
            self.require_no_pending_commit("import an artifact")
            next_step = self._step + 1
            if components.single:
                components.validate(imported)
                staged = self._artifacts.stage(next_step, imported, parent=self._artifacts.checkpoint)
                metadata = dict(staged.metadata)
            else:
                spec.validator.validate(imported)
                active = self._artifacts.resolve(self._artifacts.current)
                self.require_components(active)
                staged = self.stage_combination(next_step, active, component, imported)
                metadata = {**dict(staged.metadata), COMPONENTS_METADATA_KEY: self.snapshot_manifest(staged).to_dict()}
            # Published beside the chain: the chain tip, the backend head and the active selection stay.
            published_ref = self._artifacts.publish(
                staged, expected_parent=self._artifacts.checkpoint, metadata=metadata, advance_heads=False
            )
            published = self._artifacts.resolve(published_ref)
            record = self.append_operation_record(
                "import",
                published_ref,
                component=None if components.single else component,
                components=self.recorded_components(published, published_ref.revision),
                metrics={"source": dict(source)},
            )
            self.advance_to(record.step)
            return ComponentArtifact(
                component,
                published.component(component).artifact_id,
            )

    def sync_artifact(
        self, artifact: Artifact, *, component: str, expected_selection: ArtifactSelection, metrics: Mapping[str, Any]
    ) -> ComponentArtifact:
        """Publish ``artifact``, a person's merged tree of ``component``, as the active selection and its training start.

        The snapshot binds ``artifact`` as ``component`` and every other
        component as the active selection binds it. Its parent is the chain
        tip, and it is published beside the chain, as an import is. The
        snapshot's admission check runs on the staged snapshot first. The
        component's trainer then takes the snapshot as its training start (``Trainer.start_from``).
        The ``"sync"`` record lands with ``metrics``, and inference loads the snapshot after it. The active selection
        moves to it last; a crash before the record leaves the active selection
        that the log names. The chain tip and the activation rule stay.

        ``expected_selection`` names every active component artifact used for the merge. Raise ``ArtifactConflict``, and change nothing, while
        a step holds job data or a training job runs (the step was prepared
        against the training start that the sync replaces) or when the active
        selection no longer equals ``expected_selection``. Raise ``ReefError`` when no trainer
        trains ``component``, the admission check's error when it refuses the
        tree, and ``TrainingStartRefused`` when the trainer cannot start from
        it.
        """
        with self._lock, self._publication_lock:
            if any(
                component_trainer.trainer.has_pending_commit or component_trainer.trainer.job_running
                for component_trainer in self.trainers
            ):
                raise ArtifactConflict(f"scenario {self._name!r} is running a step; sync again when it ends")
            current_ref, checkpoint = self._artifacts.current, self._artifacts.checkpoint
            current_selection = self._artifacts.repository.selection(current_ref, self.component_names())
            if current_selection != expected_selection:
                raise ArtifactConflict(
                    f"scenario {self._name!r} changed its active artifacts during the sync; sync again"
                )
            component_trainer = self.component_trainer(component)
            next_step = self._step + 1
            staged = self.stage_publication(next_step, SavedArtifactPublication(artifact, component), checkpoint)
            try:
                # Published beside the chain: the chain tip and the backend head stay.
                published_ref = self._artifacts.publish(
                    staged,
                    expected_parent=checkpoint,
                    metadata={
                        **dict(staged.metadata),
                        COMPONENTS_METADATA_KEY: self.snapshot_manifest(staged).to_dict(),
                    },
                    advance_heads=False,
                )
                published = self._artifacts.resolve(published_ref)
                self.start_trainer(component_trainer, published)
                record = self.append_operation_record(
                    "sync",
                    published_ref,
                    component=None if self._binding.components.single else component,
                    components=self.recorded_components(published, published_ref.revision),
                    metrics=metrics,
                )
            except Exception:
                # A lost append acknowledgment can leave the record on file: its bytes must survive for the retry.
                if not any(row.snapshot == staged.ref for row in self._store.history()):
                    self._artifacts.discard(staged)
                raise
            self._artifacts.activate(published_ref, expected=current_ref)
            self.advance_to(record.step)
            return ComponentArtifact(component, published.artifact_id)

    def resume_training_starts(self) -> None:
        """After a restart, give each trainer its newest ``"train_from"`` or ``"sync"`` start that its training
        records do not hold.

        The trainer's next training commit records the start in its state;
        a restart before that commit recovers the state of the earlier
        commit, so the start is taken again here.
        """
        records = self._store.history() if self._store.durable else ()
        for component_trainer in self.trainers:
            last = self.last_record_for(component_trainer.component)
            last_step = 0 if last is None else last.step
            start = next(
                (
                    record
                    for record in reversed(records)
                    if record.step > last_step
                    and record.operation in ("train_from", "sync")
                    and self.starts_trainer(record, component_trainer.component)
                ),
                None,
            )
            if start is not None:
                self.start_trainer(component_trainer, self._artifacts.resolve(start.snapshot))

    def start_trainer(self, component_trainer: ComponentTrainer, snapshot: Artifact) -> None:
        """Start ``component_trainer`` from its component of ``release``."""
        artifact = self._binding.components.component_artifact(snapshot, component_trainer.component).materialize()
        component_trainer.trainer.start_from(artifact)

    def commit(self, result: TrainingCommit, *, component: str | None = None) -> Any:
        """Commit ``component``'s pending training result as one atomic version record.

        The trainers of one scenario run one step at a time, so no other
        trainer's commit lands while this step runs. The head is the snapshot
        that the step was prepared against.
        """
        with self._lock:
            with self._publication_lock:
                component_trainer = self.component_trainer(component)
                trainer, component = component_trainer.trainer, component_trainer.component
                next_step = self._step + 1
                # A step that is in the commit history already is a retry after a lost acknowledgment.
                records = self._store.history()
                retrying = bool(records) and records[-1].step == next_step
            # The operation lock keeps the step and source stable while tracker
            # I/O runs; published artifact readers need only the publication lock.
            correlation_metrics = None if retrying else self.prepare_experiment(trainer, next_step, component, result)
            with self._publication_lock:
                prepared = trainer.prepare_commit(result, correlation_metrics)
                # The activation rule decides whether the step's artifact serves. The training record keeps the result.
                activated = not isinstance(result.publication, NoArtifactPublication) and (
                    self.activation_rule.activates(MadeArtifact(component, result.changed_kinds))
                )
                recorded = self.recorded_retry(prepared, next_step, activated=activated, component=component)
                if recorded is not None:
                    self.finish_recorded_retry(recorded, prepared, trainer)
                    return result.state

                if self._store.durable:
                    self._synchronize_checkpoint()
                publication = result.publication
                if isinstance(publication, NoArtifactPublication):
                    self.commit_without_artifact(prepared, trainer, component)
                    return result.state

                self.commit_saved_artifact(publication, prepared, trainer, component, activated=activated)
                return result.state

    def prepare_experiment(
        self, trainer: Trainer, step: int, component: str | None, result: TrainingCommit
    ) -> Mapping[str, Any] | None:
        """The experiment tracker's correlation fields for the commit of ``result``, or None without a tracker."""
        observer = self.experiment_observer
        if observer is None:
            return None
        job_loop = trainer.job_loop
        backend_config: dict[str, Any] | None = None
        if job_loop is not None:
            try:
                backend_config = dict(trainer.pipeline.sample2job.experiment_config())
            except Exception:
                logger.exception("training backend failed to describe experiment configuration")
        backend_name = None if job_loop is None else job_loop.component.name
        context = TrainingExperimentContext(
            scenario=self._name,
            recipe=observer.recipe,
            step=step,
            source_artifacts=observer.start_artifacts,
            run_segment=observer.run_segment,
            run_step=observer.run_step,
            backend=backend_name,
            backend_config=backend_config,
            component=self.component_on_record(component),
        )
        artifact = result.artifact
        return observer.prepare(
            context,
            checkpoint_path=None if artifact is None or artifact.local_path is None else str(artifact.local_path),
        )

    def commit_without_artifact(self, prepared: PreparedCommit, trainer: Trainer, component: str | None) -> None:
        next_step = self._step + 1
        record = self._append_commit_record(
            step=next_step,
            snapshot=self._artifacts.current,
            checkpoint=False,
            prepared=prepared,
            component=component,
        )
        self.finish_committed_step(
            prepared, record, trainer, expected=self._artifacts.current, expected_checkpoint=self._artifacts.checkpoint
        )

    def commit_saved_artifact(
        self,
        publication: SavedArtifactPublication,
        prepared: PreparedCommit,
        trainer: Trainer,
        component: str | None,
        *,
        activated: bool,
    ) -> None:
        """Publish and record a training artifact. When ``activated`` is true, select it for serving too.

        Every artifact that training makes becomes the chain tip, the parent of the next publication. Only an
        ``activated`` artifact becomes the active artifact. Any other artifact stays inactive until an activation.
        """
        artifacts = self._artifacts
        next_step = self._step + 1
        checkpoint = artifacts.checkpoint
        head = artifacts.current
        durable = self._store.durable
        if publication.component is None:
            publication = SavedArtifactPublication(publication.artifact, component)
        publication = self.with_training_starts(publication)
        local_artifact = self.stage_publication(next_step, publication, checkpoint)
        try:
            # The durable snapshot must exist before inference can take the committed selection.
            published_ref = self.published_attempt(publication, checkpoint) if durable else None
            if published_ref is None:
                published_ref = artifacts.publish(
                    local_artifact,
                    expected_parent=checkpoint,
                    metadata=self.snapshot_metadata(
                        local_artifact,
                        prepared,
                        next_step,
                        component=component,
                        active_artifact=None if activated else head,
                    ),
                    advance_heads=not durable,
                    activate=False,
                )
            else:
                artifacts.discard(local_artifact)
            record = self._append_commit_record(
                step=next_step,
                snapshot=published_ref,
                checkpoint=True,
                prepared=prepared,
                activated=activated,
                component=component,
                components=self.recorded_components(local_artifact, published_ref.revision),
            )
        except Exception:
            # A lost append acknowledgment can leave a committed local release.
            # Its bytes must survive so retry can settle that exact record.
            if not any(row.snapshot == local_artifact.ref for row in self._store.history()):
                artifacts.discard(local_artifact)
            raise

        self.finish_committed_step(prepared, record, trainer, expected=head, expected_checkpoint=checkpoint)

    def with_training_starts(self, publication: SavedArtifactPublication) -> SavedArtifactPublication:
        """``publication``, whose artifact names its training starts (``training_starts_of``): its trainer's training
        start, then that start's own training starts (RFC #1263). An inference runtime serves a training artifact
        as the nearest of them that has a serving setting. A publication without a start stays as it is.
        """
        component = publication.component
        if component is None:
            return publication
        # A component that the scenario does not serve has no start; ``stage_publication`` refuses it.
        start = self.artifact_status().get(component, {}).get("training_start")
        if start is None:
            return publication
        _, snapshot = self.resolve_artifact(start, component)
        starts = (start, *training_starts_of(self._binding.components.component_artifact(snapshot, component)))
        return SavedArtifactPublication(with_training_starts(publication.artifact, starts), component)

    def published_attempt(self, publication: SavedArtifactPublication, checkpoint: SnapshotRef) -> SnapshotRef | None:
        """The snapshot that an earlier attempt of ``publication`` stored, or None when no attempt stored one.

        A retry after a failed commit names the same artifact, so it publishes nothing new, and the artifact ID
        names exactly one snapshot. Raise ``ArtifactConflict`` when that snapshot has another parent: then it is
        no attempt of this publication.
        """
        artifact_id = publication.artifact.artifact_id
        try:
            stored = self._artifacts.repository.backend.artifact_snapshot(artifact_id)
        except ArtifactNotFound:
            return None
        if stored.parent_revision != checkpoint.revision:
            raise ArtifactConflict(f"artifact {artifact_id!r} is already in release {stored.revision!r}")
        return stored

    def snapshot_metadata(
        self,
        artifact: Artifact,
        prepared: PreparedCommit,
        scenario_step: int,
        *,
        component: str | None = None,
        active_artifact: SnapshotRef | None = None,
    ) -> dict[str, Any]:
        """The metadata a committed snapshot carries.

        It is the staged snapshot's own metadata (the publication's own for a
        flat snapshot), the manifest naming its components, and the scenario
        metadata of the commit ``prepared`` describes.
        """
        scenario_metadata = scenario_metadata_for(
            name=self._name,
            source=self.root_source,
            start_artifact=self._artifacts.start,
            scenario_step=scenario_step,
            algorithm_state=prepared.algorithm_state,
            record_progress=RecordProgress(
                high_water_sequence=prepared.high_water_sequence,
                high_water_offset=prepared.high_water_offset,
                consumed_ids=prepared.consumed_ids,
            ),
            metrics=prepared.metrics,
            training_job_id=prepared.training_job_id,
            component=self.component_on_record(component),
            active_artifact=active_artifact,
            active_selection=(
                None
                if active_artifact is None
                else self._artifacts.repository.selections.get(active_artifact.revision)
            ),
        )
        return {
            **dict(artifact.metadata),
            COMPONENTS_METADATA_KEY: self.snapshot_manifest(artifact).to_dict(),
            SCENARIO_METADATA_KEY: scenario_metadata,
        }

    def _install_committed_checkpoint(
        self, ref: SnapshotRef, *, expected: SnapshotRef, expected_checkpoint: SnapshotRef, activate: bool
    ) -> None:
        """Install the refs already committed in the store, then synchronize artifact storage."""
        self._artifacts.install_checkpoint(
            ref, expected=expected, expected_checkpoint=expected_checkpoint, activate=activate
        )
        self._refresh_committed_checkpoint()

    def _refresh_committed_checkpoint(self) -> None:
        # The step is already committed. Expose the pointer failure in
        # commit_status without making the caller repeat a successful step.
        # A new commit must synchronize first and propagates any failure.
        with suppress(ArtifactPublicationError, ArtifactConflict):
            self._synchronize_checkpoint()

    def _synchronize_checkpoint(self) -> None:
        try:
            self._artifacts.repository.synchronize_checkpoint()
        except ArtifactConflict as exc:
            self._artifact_head_sync = ArtifactHeadSync("conflict", str(exc))
            raise
        except ArtifactPublicationError as exc:
            self._artifact_head_sync = ArtifactHeadSync("pending", str(exc))
            raise
        else:
            self._artifact_head_sync = ArtifactHeadSync("synchronized")
        finally:
            self._commit_status = (self._step, self._latest_training_record, self._artifact_head_sync)

    def stage_publication(
        self, next_step: int, publication: SavedArtifactPublication, checkpoint: SnapshotRef
    ) -> Artifact:
        """Admit the published component and stage the snapshot it belongs to, with ``checkpoint`` as its parent.

        A flat scenario stages the artifact as published. A multi-component
        scenario binds the named component to the publication and carries
        every other component from the active selection
        (:meth:`stage_combination`), so a step changes one component and
        never brings back what an activation replaced.
        """
        components = self._binding.components
        component = publication.component
        if component is None:
            raise ReefError(f"scenario {self._name!r}: a publication names the component it replaces")
        if components.names and component not in components.names:
            raise ReefError(f"scenario {self._name!r} serves no component {component!r}")
        if components.single:
            components.validate(publication.artifact)
            return self._artifacts.stage(next_step, publication.artifact, parent=checkpoint)
        components.components[component].validator.validate(publication.artifact)
        active = self._artifacts.resolve(self._artifacts.current)
        self.require_components(active)
        return self.stage_combination(next_step, active, component, publication.artifact)

    def stage_combination(self, next_step: int, carried: Artifact, component: str, content: Artifact) -> Artifact:
        """Stage the snapshot ``carried`` with ``component`` bound to ``content``, on the chain tip as its parent.

        The snapshot's own check runs on the staged snapshot; the carried
        components were admitted when they were published. A refused release
        is discarded.
        """
        components = self._binding.components
        artifacts = {name: carried.component(name) for name in components.names}
        artifacts[component] = content
        staged = self._artifacts.stage_composed(next_step, artifacts, parent=self._artifacts.checkpoint)
        try:
            components.validator.validate(staged)
        except BaseException:
            self._artifacts.discard(staged)
            raise
        return staged

    def finish_committed_step(
        self,
        prepared: PreparedCommit,
        record: CommitRecord,
        trainer: Trainer,
        *,
        expected: SnapshotRef,
        expected_checkpoint: SnapshotRef,
    ) -> None:
        """Complete a recorded step before exposing trainer state; callers hold both commit locks.

        Install heads and settle the trainer before advancing the scenario step.
        Failed completion retries the same record. A step that made no
        artifact records the active one and moves no head. A made artifact
        becomes the chain tip. It also becomes the active artifact when the
        record is ``activated``.
        """
        if record.trained_artifact:
            if self._artifacts.checkpoint == record.snapshot:
                if self._store.durable:
                    self._refresh_committed_checkpoint()
                elif record.activated:
                    # Publication moved the chain tip alone; the snapshot serves now that its record landed.
                    self._artifacts.activate(record.snapshot, expected=expected)
            else:
                self._install_committed_checkpoint(
                    record.snapshot,
                    expected=expected,
                    expected_checkpoint=expected_checkpoint,
                    activate=record.activated,
                )
        trainer.settle_commit(prepared)
        if not self._store.durable:
            self._artifact_head_sync = ArtifactHeadSync("synchronized")
        if record.operation == "training":
            self._latest_training_record = record
        self.advance_to(record.step)

    def finish_recorded_retry(self, record: CommitRecord, prepared: PreparedCommit, trainer: Trainer) -> CommitRecord:
        """Retry storage settlement, then complete the exact release already on record."""
        record = self._store.commit_step(expected_step=self._step, commit=record)
        previous = active_snapshot(self._store.history()[:-1], self.snapshot_queries.creation_artifact)
        self.finish_committed_step(
            prepared, record, trainer, expected=previous, expected_checkpoint=self._artifacts.checkpoint
        )
        return record

    def recorded_retry(
        self,
        prepared: PreparedCommit,
        next_step: int,
        *,
        activated: bool,
        component: str | None = None,
    ) -> CommitRecord | None:
        """The training record an earlier attempt of this commit left at ``next_step``, or ``None`` when there is none."""
        records = self._store.history()
        if not records or records[-1].step != next_step:
            return None
        record = records[-1]
        if (
            record.activated != activated
            or not self.own_record(record, self.component_on_record(component))
            or not self._record_matches_prepared(record, prepared)
        ):
            raise ScenarioStoreConflict(f"commit step {next_step} conflicts with the pending training commit")
        return record

    @staticmethod
    def _record_matches_prepared(record: CommitRecord, prepared: PreparedCommit) -> bool:
        # A record holds JSON values, where the prepared commit can hold tuples.
        return (
            record.algorithm_state == to_json_value(prepared.algorithm_state)
            and record.high_water_sequence == prepared.high_water_sequence
            and record.high_water_offset == prepared.high_water_offset
            and record.consumed_ids == prepared.consumed_ids
            and record.metrics == to_json_value(prepared.metrics)
            and record.training_job_id == prepared.training_job_id
        )

    def recorded_selection(self, ref: SnapshotRef) -> ArtifactSelection | None:
        """Record component identities for durable artifacts independently of storage snapshot identities."""
        if ref.local:
            return None
        repository = self._artifacts.repository
        return repository.selection(ref, self.component_names())

    def append_operation_record(
        self,
        operation: CommitOperation,
        snapshot: SnapshotRef,
        *,
        component: str | None,
        components: Mapping[str, str] | None = None,
        metrics: Mapping[str, Any] | None = None,
        selection: ArtifactSelection | None = None,
    ) -> CommitRecord:
        """Append a record that carries no trainer state at the next step: an ``"activate"``, ``"train_from"``,
        ``"import"``, ``"sync"``, ``"serve"`` or ``"stop_serving"`` record.

        ``components`` is the manifest of a snapshot that an activation, an import, a sync or a serve published.
        ``component`` then names the component that it activated, imported, synced or served. ``metrics`` is an
        import's source or a sync's own metrics.
        """
        record = CommitRecord(
            scenario=self._name,
            step=self._step + 1,
            snapshot=snapshot,
            checkpoint=operation in ("import", "sync"),
            metrics=metrics,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
            operation=operation,
            component=component,
            components=components,
            selection=selection or self.recorded_selection(snapshot),
        )
        return self._store.commit_step(expected_step=self._step, commit=record)

    def _append_commit_record(
        self,
        *,
        step: int,
        snapshot: SnapshotRef,
        checkpoint: bool,
        prepared: PreparedCommit,
        activated: bool = False,
        component: str | None = None,
        components: Mapping[str, str] | None = None,
    ) -> CommitRecord:
        record = CommitRecord(
            scenario=self._name,
            step=step,
            snapshot=snapshot,
            checkpoint=checkpoint,
            activated=activated,
            algorithm_state=prepared.algorithm_state,
            high_water_sequence=prepared.high_water_sequence,
            high_water_offset=prepared.high_water_offset,
            consumed_ids=prepared.consumed_ids,
            metrics=prepared.metrics,
            training_job_id=prepared.training_job_id,
            component=self.component_on_record(component),
            components=components,
            selection=self.recorded_selection(snapshot),
        )
        return self._store.commit_step(expected_step=self._step, commit=record)


__all__ = ["ComponentArtifact", "ScenarioCommitter"]
