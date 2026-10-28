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
from reef.storage.commits import CommitRecord, active_record
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

    def records(self) -> tuple[CommitRecord, ...]:
        """The durable commit history; empty for a store that keeps none."""
        return self._store.history() if self._store.durable else ()

    def releases(self, scenario_step: int) -> tuple[dict[str, Any], ...]:
        """List the creation and every training step, newest first; activations and new training starts are not listed.

        ``active`` is true on the row that made the active release (see
        :func:`active_record`), and ``activated`` on each row whose release
        has been active at some point; the creation always has been. A step
        that made no artifact names the release it ran on. Without a durable
        log a scenario past step zero lists its creation and a ``recovery``
        row for the active release.
        """
        with self.publication_lock:
            records = self.records()
            # Without a durable log nothing names the active release but the chain: a recovery row stands for it.
            recovered = not records and scenario_step > 0
            activating = [record for record in records if record.activates]
            active_id = activating[-1].artifact_ref.release_id if activating else self._creation_artifact.release_id
            activated = {self._creation_artifact.release_id} | {
                record.artifact_ref.release_id for record in activating
            }
            rows = [
                self._release_row(
                    artifact_ref=self._creation_artifact,
                    checkpoint=True,
                    recorded_at=None,
                    operation="creation",
                    active=not recovered and active_id == self._creation_artifact.release_id,
                    activated=True,
                )
            ]
            rows.extend(
                self._release_row(
                    artifact_ref=record.artifact_ref,
                    checkpoint=record.checkpoint,
                    recorded_at=record.recorded_at,
                    operation=record.operation,
                    active=record.checkpoint and record.artifact_ref.release_id == active_id,
                    activated=record.artifact_ref.release_id in activated,
                    high_water_sequence=record.high_water_sequence,
                    high_water_offset=record.high_water_offset,
                    metrics=record.metrics,
                    component=record.component,
                    components=record.components,
                )
                for record in records
                if record.operation == "training"
            )
            if recovered:
                rows.append(
                    self._release_row(
                        artifact_ref=self._artifacts.current,
                        checkpoint=True,
                        recorded_at=None,
                        operation="recovery",
                        active=True,
                        activated=True,
                    )
                )
            return tuple(reversed(rows))

    def manifests(self, scenario_step: int) -> dict[str, Mapping[str, str]]:
        """The content id of each component, by release, for every release whose manifest is known.

        A training record that made a release and an activation that
        published one carry their manifest; the creation's is read from the
        release. A flat scenario's releases carry none.
        """
        known: dict[str, Mapping[str, str]] = {}
        creation = self.creation_components(scenario_step)
        if creation is not None:
            known[self._creation_artifact.release_id] = creation
        for record in self.records():
            if record.components is not None:
                known[record.artifact_ref.release_id] = record.components
        return known

    def active_components(self, scenario_step: int) -> Mapping[str, str] | None:
        """The content id of each component the active release binds; ``None`` for a flat release or when unknown."""
        record = active_record(self.records())
        if record is None:
            return self.creation_components(scenario_step)
        return record.components

    def artifacts(self, scenario_step: int, components: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        """List the artifacts of each component in ``components``, newest first, with each component's active one.

        An artifact is one component's content. In a scenario of one
        component each release a training step made, and the creation, is
        one artifact whose ``artifact_id`` is its ``release_id``. In a
        scenario of several components a release mints an artifact of a
        component when no earlier release binds that component's content id
        (the creation mints one of each), and the ``artifact_id`` is
        ``<release_id>:<component>``. When the release's manifest is unknown,
        it mints an artifact of the component its trainer named, with
        ``content_id`` ``None``.
        ``activated`` marks an artifact that has been active at some point.
        """
        with self.publication_lock:
            made = [row for row in self.releases(scenario_step) if row["checkpoint"]]
            if len(components) == 1:
                return tuple(
                    self.artifact_row(
                        row, components[0], row["release_id"], row["content_id"], row["active"], row["activated"]
                    )
                    for row in made
                )
            manifests = self.manifests(scenario_step)
            active = self.active_components(scenario_step) or {}
            activated = {name: self.activated_contents(name, scenario_step) for name in components}
            seen: dict[str, set[str]] = {name: set() for name in components}
            listed: list[dict[str, Any]] = []
            for row in reversed(made):
                manifest = manifests.get(row["release_id"])
                for component in components:
                    content_id = None if manifest is None else manifest.get(component)
                    if content_id is None:
                        minted = row["operation"] == "creation" or row.get("component") in (None, component)
                    else:
                        minted = content_id not in seen[component]
                        seen[component].add(content_id)
                    if minted:
                        listed.append(
                            self.artifact_row(
                                row,
                                component,
                                f"{row['release_id']}:{component}",
                                content_id,
                                content_id is not None and active.get(component) == content_id,
                                content_id in activated[component],
                            )
                        )
            return tuple(reversed(listed))

    def activated_contents(self, component: str, scenario_step: int) -> set[str]:
        """The content ids that ``component`` has served: the creation's, and each activating record's."""
        creation = self.creation_components(scenario_step) or {}
        served = {record.components[component] for record in self.records() if record.activates and record.components}
        return served | ({creation[component]} if component in creation else set())

    @staticmethod
    def artifact_row(
        row: Mapping[str, Any],
        component: str,
        artifact_id: str,
        content_id: str | None,
        active: bool,
        activated: bool,
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
            "activated": activated,
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
        records = self.records()
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
            # Without a durable log the active release is known by the chain alone.
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
        records = self.records()
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
        for record in self.records():
            if record.artifact_ref.release_id == release_id and record.operation == "training":
                return record.metrics
        return None

    def entries_for_version(self, release_id: str) -> tuple[Mapping[str, Any], ...] | None:
        """The composition entries of the training step that made ``release_id``, if logged; ``None`` for the creation."""
        record = next(
            (item for item in self.records() if item.made_artifact and item.artifact_ref.release_id == release_id),
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
        activated: bool,
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
