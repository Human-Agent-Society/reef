"""Read scenario releases and their committed training metadata.

Release queries share the committer's publication lock. They never acquire
its operation lock, so long-running training preparation cannot block serving.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any

from reef.artifact.artifact import (
    Artifact,
    ArtifactError,
    ArtifactNotFound,
    ArtifactRef,
    LiveWeightArtifactRef,
    is_local_release,
)
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
        self._publication_lock = publication_lock
        self._creation_artifact = self._resolve_creation_artifact(scenario_step)

    @property
    def creation_artifact(self) -> ArtifactRef:
        return self._creation_artifact

    def releases(self, scenario_step: int) -> tuple[dict[str, Any], ...]:
        """List committed releases newest first."""
        with self._publication_lock:
            records = () if not self._store.durable else self._store.history()
            rows = [
                self._release_row(
                    artifact_ref=self._creation_artifact,
                    checkpoint=True,
                    recorded_at=None,
                    operation="creation",
                    current=scenario_step == 0,
                )
            ]
            rows.extend(
                self._release_row(
                    artifact_ref=record.artifact_ref,
                    checkpoint=record.checkpoint,
                    recorded_at=record.recorded_at,
                    operation=record.operation,
                    current=record.step == scenario_step,
                    rollback_target_release_id=record.rollback_target_release_id,
                    high_water_sequence=record.high_water_sequence,
                    high_water_offset=record.high_water_offset,
                    metrics=record.metrics,
                    pending=record.pending,
                )
                for record in records
            )
            if not records and scenario_step > 0:
                rows.append(
                    self._release_row(
                        artifact_ref=self._artifacts.current,
                        checkpoint=True,
                        recorded_at=None,
                        operation="recovery",
                        current=True,
                    )
                )
            return tuple(reversed(rows))

    def find_release(self, release_id: str) -> tuple[ArtifactRef, bool] | None:
        records = () if not self._store.durable else self._store.history()
        for record in reversed(records):
            if record.artifact_ref.release_id == release_id:
                return record.artifact_ref, record.checkpoint
        if self._creation_artifact.release_id == release_id:
            return self._creation_artifact, True
        return None

    def _resolve_creation_artifact(self, scenario_step: int) -> ArtifactRef:
        """The artifact the scenario was forked from, for the release list.

        A fresh scenario is still on it, so the chain head is the creation
        artifact. After a recovery the fork point is reconstructed from the
        first commit record: normally the durable parent of step 1's artifact
        (a checkpoint, live, or local ref still knows its parent version); if
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
            if (
                first.checkpoint or isinstance(ref, LiveWeightArtifactRef) or is_local_release(ref.release_id)
            ) and ref.parent_release_id is not None:
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
        """The composition entries the training step that published ``release_id`` committed, if logged."""
        if not self._store.durable:
            return None
        for record in self._store.history():
            if record.artifact_ref.release_id == release_id and record.operation == "training":
                entries = (record.algorithm_state or {}).get("entries")
                if isinstance(entries, Sequence) and not isinstance(entries, str):
                    return tuple(dict(entry) for entry in entries if isinstance(entry, Mapping))
                return None
        return None

    def artifact_for_version(self, release_id: str) -> Artifact:
        """Materialize a scenario release for a read-only content serve.

        The read-side counterpart of ``rollback``: the same release lookup,
        but no head moves and no commit is written; the caller only wants the
        version's file tree. Every failure is ``ArtifactNotFound`` naming the
        version (not ``ReleaseNotRestorable``, which answers a rejected
        write): to a reader, a version whose bytes are gone and a version
        that was recorded but not kept are the same absent content.
        """
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        release_id = release_id.strip()
        with self._publication_lock:
            found = self.find_release(release_id)
        if found is None:
            raise ArtifactNotFound(f"scenario {self._name!r} has no release {release_id!r}")
        ref, _ = found
        if isinstance(ref, LiveWeightArtifactRef):
            raise ArtifactNotFound(
                f"scenario {self._name!r} release {release_id!r} is live weights and has no file tree"
            )
        try:
            return self._artifacts.resolve(ref)
        except ArtifactError as exc:
            raise ArtifactNotFound(f"scenario {self._name!r} cannot restore release {release_id!r}: {exc}") from exc

    def artifact_with_metrics(
        self,
        release_id: str | None = None,
    ) -> tuple[Artifact, Mapping[str, Any] | None]:
        """Capture one artifact and its metrics without waiting for preparation."""
        with self._publication_lock:
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
        current: bool,
        rollback_target_release_id: str | None = None,
        high_water_sequence: int = 0,
        high_water_offset: int = 0,
        metrics: Mapping[str, Any] | None = None,
        pending: bool = False,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "release_id": artifact_ref.release_id,
            "parent_release_id": artifact_ref.parent_release_id,
            "content_id": artifact_ref.content_id,
            "content_kind": "live_weights" if isinstance(artifact_ref, LiveWeightArtifactRef) else "saved_artifact",
            "checkpoint": checkpoint,
            "restorable": checkpoint and not isinstance(artifact_ref, LiveWeightArtifactRef),
            "recorded_at": recorded_at,
            "operation": operation,
            "pending": pending,
            "current": current,
            "record_progress": {
                "high_water_sequence": high_water_sequence,
                "high_water_offset": high_water_offset,
            },
        }
        if rollback_target_release_id is not None:
            row["rollback_target_release_id"] = rollback_target_release_id
        if isinstance(artifact_ref, LiveWeightArtifactRef):
            row["runtime_load_id"] = artifact_ref.runtime_load_id
        if metrics is not None:
            row["metrics"] = dict(metrics)
        return row
