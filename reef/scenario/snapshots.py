"""Read scenario releases and their committed training metadata.

Release queries share the committer's publication lock. They never acquire
its operation lock, so long-running training preparation cannot block serving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactError, ArtifactNotFound, SnapshotRef
from reef.artifact.snapshot_chain import ArtifactSnapshotChain
from reef.core.artifact_selection import ArtifactSelection
from reef.core.components import validate_component_name
from reef.core.snapshot import RootSource
from reef.storage.commits import CommitRecord, active_record
from reef.storage.scenario import ScenarioStore


@dataclass(frozen=True)
class ServedBesideArtifact:
    """An artifact served at its own route with a complete component selection.

    ``selection`` maps component names to immutable artifact IDs. It contains
    ``component`` with value ``artifact_id``. Scenario history builds these
    values; serving workers materialize the selection to answer that route.
    """

    component: str
    artifact_id: str
    selection: ArtifactSelection

    def __post_init__(self) -> None:
        validate_component_name(self.component)
        if not isinstance(self.selection, ArtifactSelection):
            raise TypeError("a served artifact requires an ArtifactSelection")
        if self.selection.components.get(self.component) != self.artifact_id:
            raise ValueError("a served artifact must occur in its selection")


class ScenarioSnapshots:
    """Query committed snapshots against one scenario's artifact chain."""

    def __init__(
        self,
        *,
        name: str,
        artifacts: ArtifactSnapshotChain,
        store: ScenarioStore,
        publication_lock: RLock,
        root_source: RootSource,
    ) -> None:
        self._name = name
        self._artifacts = artifacts
        self._store = store
        self.publication_lock = publication_lock
        #: Where the creation artifact, the scenario's root snapshot, came from; its row lists it as ``source``.
        self.root_source = root_source
        self.recorded_creation_components: Mapping[str, str] | None = None
        self.creation_components_read = False
        self.creation_read_failed_at: int | None = None
        #: The scenario step that ``served_beside`` last read, and what it returned then.
        self.served_read: tuple[int, Mapping[str, ServedBesideArtifact]] | None = None
        #: The scenario step and current revision that ``manifests`` last read, and what it returned then.
        self.manifests_read: tuple[int, str, dict[str, Mapping[str, str]]] | None = None

    @property
    def creation_artifact(self) -> SnapshotRef:
        """The scenario's root snapshot, which the scenario was created with."""
        return self._artifacts.start

    def creation_components(self, scenario_step: int, *, retry: bool = False) -> Mapping[str, str] | None:
        """The ``artifact_id`` of each component the creation artifact binds; ``None`` without a manifest or when it cannot be read.

        The creation artifact has no commit record to carry its manifest, so
        it is read from the snapshot once and kept. A read that failed is
        asked again after the next commit, or at once when ``retry`` says so:
        on a remote backend a read is a fetch, which a page polled every few
        seconds must not repeat while the remote is away.
        """
        if not self.creation_components_read:
            if self.creation_read_failed_at == scenario_step and not retry:
                return None
            try:
                manifest = self._artifacts.resolve(self.creation_artifact).components
            except ArtifactError:
                self.creation_read_failed_at = scenario_step
                return None
            self.creation_components_read = True
            if manifest is not None:
                self.recorded_creation_components = manifest.artifact_ids(self.creation_artifact.revision)
        return self.recorded_creation_components

    def records(self) -> tuple[CommitRecord, ...]:
        """The durable commit history; empty for a store that keeps none."""
        return self._store.history() if self._store.durable else ()

    def snapshot_rows(self, scenario_step: int) -> tuple[dict[str, Any], ...]:
        """List the creation, every training step, every import and every sync, newest first; activations and new
        training starts are not listed. The creation row is the scenario's root snapshot, with its ``source``.

        ``active`` is true on the row that made the active selection (see
        :func:`active_record`), and ``activated`` on each row whose snapshot
        has been active at some point; the creation always has been. A step
        that made no artifact names the snapshot it ran on. Without a durable
        log a scenario past step zero lists its creation and a ``recovery``
        row for the active selection.
        """
        with self.publication_lock:
            records = self.records()
            # Without a durable log nothing names the active selection but the chain: a recovery row stands for it.
            recovered = not records and scenario_step > 0
            activating = [record for record in records if record.activates]
            active_id = activating[-1].snapshot.revision if activating else self._artifacts.current.revision
            activated = {self.creation_artifact.revision, active_id} | {
                record.snapshot.revision for record in activating
            }
            creation = self.snapshot_row(
                step=0,
                snapshot=self.creation_artifact,
                checkpoint=True,
                recorded_at=None,
                operation="creation",
                active=not recovered and active_id == self.creation_artifact.revision,
                activated=True,
            )
            rows = [{**creation, "source": self.root_source.to_dict()}]
            rows.extend(
                self.snapshot_row(
                    step=record.step,
                    snapshot=record.snapshot,
                    checkpoint=record.checkpoint,
                    recorded_at=record.recorded_at,
                    operation=record.operation,
                    active=record.checkpoint and record.snapshot.revision == active_id,
                    activated=record.snapshot.revision in activated,
                    high_water_sequence=record.high_water_sequence,
                    high_water_offset=record.high_water_offset,
                    metrics=record.metrics,
                    component=record.component,
                    components=record.components,
                )
                for record in records
                if record.operation in ("training", "import", "sync")
            )
            if recovered:
                rows.append(
                    self.snapshot_row(
                        step=scenario_step,
                        snapshot=self._artifacts.current,
                        checkpoint=True,
                        recorded_at=None,
                        operation="recovery",
                        active=True,
                        activated=True,
                    )
                )
            return tuple(reversed(rows))

    def manifests(self, scenario_step: int) -> dict[str, Mapping[str, str]]:
        """The ``artifact_id`` of each component, by release, for every snapshot whose manifest is known.

        A training record that made a snapshot and an activation that
        published one carry their manifest; the creation's is read from the
        snapshot. Without a durable journal, the current snapshot supplies its own manifest.
        Each ``scenario_step`` and current revision is read once.
        """
        current_revision = self._artifacts.current.revision
        if self.manifests_read is not None and self.manifests_read[:2] == (scenario_step, current_revision):
            return self.manifests_read[2]
        known: dict[str, Mapping[str, str]] = {}
        creation = self.creation_components(scenario_step)
        if creation is not None:
            known[self.creation_artifact.revision] = creation
        for record in self.records():
            if record.artifact_ids is not None:
                known[record.snapshot.revision] = record.artifact_ids
        if not self._store.durable and self._artifacts.current.revision not in known:
            current = self._artifacts.resolve(self._artifacts.current)
            if current.components is not None:
                known[current.ref.revision] = current.components.artifact_ids(current.ref.revision)
        self.manifests_read = (scenario_step, current_revision, known)
        return known

    def active_components(self, scenario_step: int) -> Mapping[str, str] | None:
        """The ``artifact_id`` of each component the active selection binds; ``None`` for a flat snapshot or when unknown."""
        record = active_record(self.records())
        if record is None:
            if self._artifacts.current != self.creation_artifact:
                artifact = self._artifacts.resolve(self._artifacts.current)
                manifest = artifact.components
                return None if manifest is None else manifest.artifact_ids(artifact.ref.revision)
            return self.creation_components(scenario_step)
        return record.artifact_ids

    def artifact_parents(self, scenario_step: int, components: tuple[str, ...]) -> dict[str, str | None]:
        """The parent artifact of each output, derived from that component's training start or synced active tree.

        Creation and external imports have no local parent. Training follows its component's training start,
        including explicit ``train_from`` operations. A sync follows its previously active artifact. Storage
        commit ancestry never substitutes for these component relationships.
        """
        active = self.bound_artifact_ids(self.creation_artifact.revision, components, scenario_step)
        starts = dict(active)
        parents: dict[str, str | None] = dict.fromkeys(active.values())
        for record in self.records():
            bound = record.artifact_ids or self.bound_artifact_ids(record.snapshot.revision, components, scenario_step)
            changed = components if record.component is None else (record.component,)
            if record.made_artifact:
                for name in changed:
                    artifact_id = bound.get(name)
                    if artifact_id is None or artifact_id in parents:
                        continue
                    if record.operation == "training":
                        parent = starts.get(name)
                    elif record.operation == "sync":
                        parent = active.get(name)
                    else:
                        parent = None
                    parents[artifact_id] = parent
            if record.sets_training_start:
                starts.update({name: bound[name] for name in changed if name in bound})
            if record.activates:
                active = dict(bound)
        return parents

    def artifacts(self, scenario_step: int, components: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        """List the artifacts of each component in ``components``, newest first, with each component's active one.

        Each artifact appears once, at the first retained publication that
        names it. Later snapshots carrying that component do not create another
        row. Creation and import rows retain their source information.

        ``activated`` marks an artifact that has been active at some time, and
        ``served`` one that the scenario serves beside its active selection. An
        import's row has ``source``: where its content was read from.
        """
        with self.publication_lock:
            made = [row for row in self.snapshot_rows(scenario_step) if row["checkpoint"]]
            beside = self.served_beside(scenario_step, components)
            active = self.active_components(scenario_step) or self.bound_artifact_ids(
                self._artifacts.current.revision, components, scenario_step
            )
            activated = self.activated_artifacts(scenario_step, components)
            seen: set[tuple[str, str]] = set()
            listed: list[dict[str, Any]] = []
            for row in reversed(made):
                bound = self.bound_artifact_ids(row["snapshot"].revision, components, scenario_step)
                for component in components:
                    artifact_id = bound.get(component)
                    if artifact_id is None or (component, artifact_id) in seen:
                        continue
                    seen.add((component, artifact_id))
                    listed.append(
                        self.artifact_row(
                            row,
                            component,
                            artifact_id,
                            active.get(component) == artifact_id,
                            artifact_id in activated[component],
                            artifact_id in beside,
                        )
                    )
            return tuple(reversed(listed))

    def bound_artifact_ids(self, revision: str, components: tuple[str, ...], scenario_step: int) -> dict[str, str]:
        """The ``artifact_id`` of each of ``components`` that the snapshot ``revision`` binds, as the records name it.

        ``components`` are all the components of the scenario. A snapshot with a known manifest gives its entries;
        a component that the manifest does not name is left out. Otherwise, the result is empty.
        """
        manifest = self.manifests(scenario_step).get(revision)
        if manifest is None:
            return {}
        return {name: manifest[name] for name in components if name in manifest}

    def served_history(self, scenario_step: int, component: str, components: tuple[str, ...]) -> tuple[str, ...]:
        """The artifacts of ``component`` that the active releases of the scenario bound, oldest first, with no two
        neighbors equal: the creation's, then each activating record's. Empty for a store without a durable history."""
        records = self.records()
        if not records:
            return ()
        history: list[str] = []
        revisions = (self.creation_artifact.revision, *(r.snapshot.revision for r in records if r.activates))
        for revision in revisions:
            artifact_id = self.bound_artifact_ids(revision, components, scenario_step).get(component)
            if artifact_id is not None and (not history or history[-1] != artifact_id):
                history.append(artifact_id)
        return tuple(history)

    def served_beside(self, scenario_step: int, components: tuple[str, ...]) -> Mapping[str, ServedBesideArtifact]:
        """The artifacts that the scenario serves beside its active selection, by ``artifact_id``.

        The result is the fold of the commit history. A ``"serve"`` record adds its artifact, and a
        ``"stop_serving"`` record removes it. An activation changes nothing here, so the result can hold an
        artifact of the active selection. A store without a durable history serves none. Each ``scenario_step``
        is read once.
        """
        if self.served_read is not None and self.served_read[0] == scenario_step:
            return self.served_read[1]
        served: dict[str, ServedBesideArtifact] = {}
        for record in self.records():
            if record.operation in ("serve", "stop_serving"):
                component = components[0] if record.component is None else record.component
                bound = record.components or self.bound_artifact_ids(
                    record.snapshot.revision, components, scenario_step
                )
                artifact_id = bound[component]
                if record.operation == "serve":
                    served[artifact_id] = ServedBesideArtifact(component, artifact_id, ArtifactSelection(bound))
                else:
                    served.pop(artifact_id, None)
        self.served_read = (scenario_step, served)
        return served

    def activated_artifacts(self, scenario_step: int, components: tuple[str, ...]) -> dict[str, set[str]]:
        """The ``artifact_id`` values that each of ``components`` has served.

        These are the creation's, each activating record's, and the active ones. ``components`` are all the
        components of the scenario, so a one-component tuple means a flat scenario.
        """
        served: dict[str, set[str]] = {name: set() for name in components}
        selections = [self.creation_components(scenario_step) or {}, self.active_components(scenario_step) or {}]
        selections.extend(
            self.bound_artifact_ids(record.snapshot.revision, components, scenario_step)
            for record in self.records()
            if record.activates
        )
        for selection in selections:
            for name, artifact_id in selection.items():
                if name in served:
                    served[name].add(artifact_id)
        return served

    @staticmethod
    def artifact_row(
        row: Mapping[str, Any],
        component: str,
        artifact_id: str,
        active: bool,
        activated: bool,
        served: bool,
    ) -> dict[str, Any]:
        """One artifact row: the snapshot row's fields that describe the step that made it, under its own identity.

        ``served`` is true when the scenario serves the artifact beside its active selection (``served_beside``).
        """
        listed: dict[str, Any] = {
            "artifact_id": artifact_id,
            "component": component,
            "snapshot": row["snapshot"],
            "operation": row["operation"],
            "recorded_at": row["recorded_at"],
            "active": active,
            "activated": activated,
            "served": served,
        }
        if row["operation"] == "training":
            listed["record_progress"] = dict(row["record_progress"])
        if row["operation"] == "import":
            # An import's metrics hold only where its content was read from.
            listed["source"] = row["metrics"]["source"]
        elif row["operation"] == "creation":
            listed["source"] = row["source"]
        elif "metrics" in row:
            listed["metrics"] = dict(row["metrics"])
        return listed

    def find_snapshot(self, revision: str) -> tuple[SnapshotRef, bool] | None:
        """The snapshot that ``revision`` names and whether it has durable bytes; ``None`` when no record names it.

        A training step that made the artifact, or the creation, gives it
        durable bytes. Other records (a step that made nothing, an
        activation, a new training start) only name an artifact, so the
        newest of them answers without bytes when nothing made it.
        """
        records = self.records()
        found: tuple[SnapshotRef, bool] | None = None
        for record in reversed(records):
            if record.snapshot.revision == revision:
                if record.made_artifact:
                    return record.snapshot, True
                if found is None:
                    found = (record.snapshot, False)
        if self.creation_artifact.revision == revision:
            # A first step that made nothing records the creation without a checkpoint; the creation has its own bytes.
            return self.creation_artifact, True
        if found is not None:
            return found
        current = self._artifacts.current
        if current.revision == revision:
            # A logless scenario still publishes durable snapshots; local selection views have no stored bytes.
            return current, current.durable
        return None

    def artifact_for_version(self, revision: str) -> Artifact:
        """Materialize a scenario release for a read-only content serve.

        The read-side counterpart of ``activate``: the same snapshot lookup,
        but no head moves and no commit is written; the caller only wants the
        version's file tree. Every failure is ``ArtifactNotFound`` naming the
        version (not ``SnapshotNotRestorable``, which answers a rejected
        write): to a reader, a version whose bytes are gone and a version
        that was recorded but not kept are the same absent content.
        """
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        revision = revision.strip()
        with self.publication_lock:
            found = self.find_snapshot(revision)
        if found is None:
            raise ArtifactNotFound(f"scenario {self._name!r} has no snapshot {revision!r}")
        ref, _ = found
        try:
            return self._artifacts.resolve(ref)
        except ArtifactError as exc:
            raise ArtifactNotFound(f"scenario {self._name!r} cannot restore release {revision!r}: {exc}") from exc

    @staticmethod
    def snapshot_row(
        *,
        step: int,
        snapshot: SnapshotRef,
        checkpoint: bool,
        recorded_at: float | None,
        operation: str,
        active: bool,
        activated: bool,
        high_water_sequence: int = 0,
        high_water_offset: int = 0,
        metrics: Mapping[str, Any] | None = None,
        component: str | None = None,
        components: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "step": step,
            "snapshot": snapshot,
            "checkpoint": checkpoint,
            "restorable": checkpoint,
            "recorded_at": recorded_at,
            "operation": operation,
            "active": active,
            "activated": activated,
            "record_progress": {
                "high_water_sequence": high_water_sequence,
                "high_water_offset": high_water_offset,
            },
        }
        if metrics is not None:
            row["metrics"] = dict(metrics)
        if component is not None:
            row["component"] = component
        if components is not None:
            row["components"] = dict(components)
        return row
