"""Read scenario releases and their committed training metadata.

Release queries share the committer's publication lock. They never acquire
its operation lock, so long-running training preparation cannot block serving.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactError, ArtifactNotFound, ArtifactRef, is_local_release
from reef.artifact.release_chain import ArtifactReleaseChain
from reef.storage.scenario import ScenarioStore


class ScenarioReleases:
    """Query committed releases against one scenario's artifact chain."""

    def __init__(
        self,
        *,
        name: str,
        artifacts: ArtifactReleaseChain,
        store: ScenarioStore,
        publication_lock: RLock,
        scenario_step: int,
    ) -> None:
        self._name = name
        self._artifacts = artifacts
        self._store = store
        self.publication_lock = publication_lock
        self._creation_artifact = self._resolve_creation_artifact(scenario_step)
        self.recorded_creation_components: Mapping[str, str] | None = None
        self.creation_components_read = False
        self.creation_read_failed_at: int | None = None

    @property
    def creation_artifact(self) -> ArtifactRef:
        return self._creation_artifact

    def creation_components(self, scenario_step: int, *, retry: bool = False) -> Mapping[str, str] | None:
        """The content id of each component the creation artifact binds; ``None`` without a manifest or when it cannot be read.

        The creation artifact has no commit record to carry its manifest, so
        it is read from the release once and kept. A read that failed is
        asked again after the next commit, or at once when ``retry`` says so:
        on a remote backend a read is a fetch, which a page polled every few
        seconds must not repeat while the remote is away.
        """
        if not self.creation_components_read:
            if self.creation_read_failed_at == scenario_step and not retry:
                return None
            try:
                manifest = self._artifacts.resolve(self._creation_artifact).components
            except ArtifactError:
                self.creation_read_failed_at = scenario_step
                return None
            self.creation_components_read = True
            if manifest is not None:
                self.recorded_creation_components = manifest.content_ids
        return self.recorded_creation_components

    def releases(self, scenario_step: int) -> tuple[dict[str, Any], ...]:
        """List the creation and every training step, newest first; activations and new training starts are not listed.

        ``active`` is true on exactly one row when the active artifact is
        listed: the row that made it. A step that made no artifact records
        the active artifact's id but is never ``active``; nor is a pending
        artifact until a person activates it. Without a durable log a
        recovered scenario lists its creation and a ``recovery`` row for the
        artifact it serves.
        """
        with self.publication_lock:
            records = () if not self._store.durable else self._store.history()
            active = self._artifacts.current
            rows = [
                self._release_row(
                    artifact_ref=self._creation_artifact,
                    checkpoint=True,
                    recorded_at=None,
                    operation="creation",
                    active=self._creation_artifact.release_id == active.release_id,
                )
            ]
            rows.extend(
                self._release_row(
                    artifact_ref=record.artifact_ref,
                    checkpoint=record.checkpoint,
                    recorded_at=record.recorded_at,
                    operation=record.operation,
                    active=record.made_artifact and record.artifact_ref.release_id == active.release_id,
                    high_water_sequence=record.high_water_sequence,
                    high_water_offset=record.high_water_offset,
                    metrics=record.metrics,
                    component=record.component,
                    components=record.components,
                )
                for record in records
                if record.operation == "training"
            )
            if not records and scenario_step > 0:
                rows.append(
                    self._release_row(
                        artifact_ref=active,
                        checkpoint=True,
                        recorded_at=None,
                        operation="recovery",
                        active=True,
                    )
                )
            return tuple(reversed(rows))

    def active_components(self, scenario_step: int) -> Mapping[str, str] | None:
        """The content id of each component the active release binds; ``None`` for a flat release or when unknown.

        The newest record that activated a release names its manifest: an
        ``"activate"`` record that published a release, or a training record
        that made a release and is not pending. With none, the active
        release is the creation, whose manifest is read from the release.
        """
        records = () if not self._store.durable else self._store.history()
        for record in reversed(records):
            if record.operation == "activate" or (record.made_artifact and not record.pending):
                return record.components
        return self.creation_components(scenario_step)

    def artifacts(self, scenario_step: int, components: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        """List the artifacts of each component in ``components``, newest first; ``active`` marks each component's.

        An artifact is one component's content. In a scenario of one
        component each release a training step made, and the creation, is
        one artifact whose ``artifact_id`` is its ``release_id``. In a
        scenario of several components a release holds a new artifact of a
        component when that component's content id differs from the one its
        parent release binds (the creation holds one of each), and its
        ``artifact_id`` is ``<release_id>:<component>``. A component whose
        content ids the records do not name is listed for the component of
        the trainer that made the release.
        """
        with self.publication_lock:
            rows = list(reversed(self.releases(scenario_step)))
            made = [row for row in rows if row["checkpoint"]]
            if len(components) == 1:
                return tuple(
                    self.artifact_row(row, components[0], row["release_id"], row["content_id"], row["active"])
                    for row in reversed(made)
                )
            creation = self.creation_components(scenario_step)
            active = self.active_components(scenario_step)
            manifests = {
                row["release_id"]: (creation if row["operation"] == "creation" else row.get("components"))
                for row in made
            }
            listed: list[dict[str, Any]] = []
            for row in made:
                manifest = manifests[row["release_id"]]
                parent = manifests.get(row["parent_release_id"]) if row["operation"] != "creation" else None
                for component in components:
                    content_id = None if manifest is None else manifest.get(component)
                    if row["operation"] == "creation" or content_id is None or parent is None:
                        new = row["operation"] == "creation" or row.get("component") in (None, component)
                    else:
                        new = parent.get(component) != content_id
                    if not new:
                        continue
                    listed.append(
                        self.artifact_row(
                            row,
                            component,
                            f"{row['release_id']}:{component}",
                            content_id or row["content_id"],
                            active is not None and content_id is not None and active.get(component) == content_id,
                        )
                    )
            return tuple(reversed(listed))

    @staticmethod
    def artifact_row(
        row: Mapping[str, Any], component: str, artifact_id: str, content_id: str, active: bool
    ) -> dict[str, Any]:
        """One artifact row: the release row's fields that describe the step that made it, under its own identity."""
        listed: dict[str, Any] = {
            "artifact_id": artifact_id,
            "component": component,
            "release_id": row["release_id"],
            "parent_release_id": row["parent_release_id"],
            "content_id": content_id,
            "operation": row["operation"],
            "recorded_at": row["recorded_at"],
            "active": active,
        }
        if row["operation"] == "training":
            listed["record_progress"] = dict(row["record_progress"])
        if "metrics" in row:
            listed["metrics"] = dict(row["metrics"])
        return listed

    def ref_for_version(self, release_id: str) -> ArtifactRef | None:
        """The ref of a scenario release without materializing it; ``None`` when the catalog has no such release."""
        with self.publication_lock:
            found = self.find_release(release_id)
        return None if found is None else found[0]

    def find_release(self, release_id: str) -> tuple[ArtifactRef, bool] | None:
        """The artifact ``release_id`` names and whether it has durable bytes; ``None`` when no record names it.

        A training step that made the artifact, or the creation, gives it
        durable bytes. Other records (a step that made nothing, an
        activation, a new training start) only name an artifact, so the
        newest of them answers without bytes when nothing made it.
        """
        records = () if not self._store.durable else self._store.history()
        found: tuple[ArtifactRef, bool] | None = None
        for record in reversed(records):
            if record.artifact_ref.release_id == release_id:
                if record.made_artifact:
                    return record.artifact_ref, True
                if found is None:
                    found = (record.artifact_ref, False)
        if self._creation_artifact.release_id == release_id:
            # A first step that made nothing records the creation without a checkpoint; the creation has its own bytes.
            return self._creation_artifact, True
        if found is not None:
            return found
        current = self._artifacts.current
        if current.release_id == release_id:
            # Without a durable log the served release is known by the chain alone.
            return current, False
        return None

    def _resolve_creation_artifact(self, scenario_step: int) -> ArtifactRef:
        """The artifact the scenario was forked from, for the release list.

        A fresh scenario is still on it, so the chain head is the creation
        artifact. After a recovery the fork point is reconstructed from the
        first commit record: normally the durable parent of step 1's artifact
        (a checkpoint or local ref still knows its parent version); if
        step 1 never recorded a durable parent (a plain non-checkpoint ref),
        that ref itself is the earliest version the release list can show. When
        the parent has since disappeared from the backend, fall back to the
        repository base artifact.
        """
        if scenario_step == 0:
            return self._artifacts.current
        records = () if not self._store.durable else self._store.history()
        if records and records[0].step == 1:
            first = records[0]
            ref = first.artifact_ref
            if (first.checkpoint or is_local_release(ref.release_id)) and ref.parent_release_id is not None:
                try:
                    return self._artifacts.repository.backend.resolve_release(ref.parent_release_id)
                except ArtifactNotFound:
                    pass
            elif not first.checkpoint:
                return ref
        return self._artifacts.base

    def metrics_for_version(self, release_id: str) -> Mapping[str, Any] | None:
        """Metrics of the training step that published ``release_id``, if logged."""
        if not self._store.durable:
            return None
        for record in self._store.history():
            if record.artifact_ref.release_id == release_id and record.operation == "training":
                return record.metrics
        return None

    def entries_for_version(self, release_id: str) -> tuple[Mapping[str, Any], ...] | None:
        """The composition entries of the training step that made ``release_id``, if logged; ``None`` for the creation."""
        if not self._store.durable:
            return None
        record = next(
            (
                item
                for item in self._store.history()
                if item.made_artifact and item.artifact_ref.release_id == release_id
            ),
            None,
        )
        if record is None:
            return None
        entries = (record.algorithm_state or {}).get("entries")
        if isinstance(entries, Sequence) and not isinstance(entries, str):
            return tuple(dict(entry) for entry in entries if isinstance(entry, Mapping))
        return None

    def artifact_for_version(self, release_id: str) -> Artifact:
        """Materialize a scenario release for a read-only content serve.

        The read-side counterpart of ``activate``: the same release lookup,
        but no head moves and no commit is written; the caller only wants the
        version's file tree. Every failure is ``ArtifactNotFound`` naming the
        version (not ``ReleaseNotRestorable``, which answers a rejected
        write): to a reader, a version whose bytes are gone and a version
        that was recorded but not kept are the same absent content.
        """
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        release_id = release_id.strip()
        with self.publication_lock:
            found = self.find_release(release_id)
        if found is None:
            raise ArtifactNotFound(f"scenario {self._name!r} has no release {release_id!r}")
        ref, _ = found
        try:
            return self._artifacts.resolve(ref)
        except ArtifactError as exc:
            raise ArtifactNotFound(f"scenario {self._name!r} cannot restore release {release_id!r}: {exc}") from exc

    def artifact_with_metrics(
        self,
        release_id: str | None = None,
    ) -> tuple[Artifact, Mapping[str, Any] | None]:
        """Capture one artifact and its metrics without waiting for preparation."""
        with self.publication_lock:
            artifact = (
                Artifact(self._artifacts.current, self._artifacts.repository)
                if release_id is None
                else self.artifact_for_version(release_id)
            )
            metrics = self.metrics_for_version(artifact.ref.release_id)
            return artifact, metrics

    @staticmethod
    def _release_row(
        *,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        recorded_at: float | None,
        operation: str,
        active: bool,
        high_water_sequence: int = 0,
        high_water_offset: int = 0,
        metrics: Mapping[str, Any] | None = None,
        component: str | None = None,
        components: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "release_id": artifact_ref.release_id,
            "parent_release_id": artifact_ref.parent_release_id,
            "content_id": artifact_ref.content_id,
            "checkpoint": checkpoint,
            "restorable": checkpoint,
            "recorded_at": recorded_at,
            "operation": operation,
            "active": active,
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
