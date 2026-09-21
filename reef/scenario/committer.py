"""Scenario committer coordinating training and artifact publication.

Artifact storage and head movement live in ``reef.artifact``; the durable
record values and checkpoint metadata formats live in ``commits``. This
module owns the scenario-specific ordering across trainer state, store
settlement, checkpoint policy, surfaces, and artifact
operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactNotFound,
    ArtifactPublicationError,
    ArtifactRef,
    LiveWeightArtifactRef,
)
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, ReleaseComponents
from reef.core.errors import ReefError
from reef.recipe.checkpoint_strategy import CheckpointStrategy
from reef.scenario.binding import ScenarioBinding
from reef.scenario.releases import ScenarioReleases
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord, RecordProgress, scenario_metadata_for
from reef.storage.scenario import ScenarioStore, ScenarioStoreConflict
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import (
    DurableWeightsPublication,
    LiveWeightPublication,
    NoArtifactPublication,
    PreparedCommit,
    SavedArtifactPublication,
    TrainStepResult,
)


@dataclass(frozen=True)
class _ArtifactHeadSync:
    state: Literal["synchronized", "pending", "conflict"]
    release_id: str
    error: str | None = None


class StaleTrainingResultError(ReefError):
    """A result was prepared against a release that another component's commit has since replaced.

    The result is not attached to the newer combination. The caller keeps
    the batch, drops the result, and prepares it again against the release
    served now.
    """


class ScenarioCommitter:
    """Order one scenario's commit, rollback, and retry effects; delegate release queries."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        artifacts: ArtifactReleaseChain,
        checkpoint_strategy: CheckpointStrategy,
        trainers: tuple[ComponentTrainer, ...],
        scenario_step: int = 0,
        store: ScenarioStore,
        recovered_head_record: CommitRecord | None = None,
    ) -> None:
        if not isinstance(scenario_step, int) or scenario_step < 0:
            raise ValueError("scenario_step must be non-negative")
        if store.durable:
            artifacts.repository.require_staged_commit_support()
        if not trainers:
            raise ValueError("a scenario commits through at least one trainer")
        self._name = name
        self._binding = binding
        self._artifacts = artifacts
        self._checkpoint_strategy = checkpoint_strategy
        self._trainers = trainers
        # The first trainer answers commits made on the scenario's behalf, such as rollback.
        self._trainer = trainers[0].trainer
        self._step = scenario_step
        self._store = store
        self._lock = RLock()
        # Preparation holds the operation lock across proposer calls and
        # evaluation episodes, neither of which changes committed releases.
        # Readers share only the publication lock with commit and rollback.
        # Writers must acquire the operation lock first; readers never take it.
        self._publication_lock = RLock()
        self._releases = ScenarioReleases(
            name=name,
            artifacts=artifacts,
            store=store,
            publication_lock=self._publication_lock,
            scenario_step=scenario_step,
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
        self._artifact_head_sync = _ArtifactHeadSync("synchronized", artifacts.checkpoint.release_id)
        self._commit_status = (scenario_step, self._latest_training_record, self._artifact_head_sync)

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def artifacts(self) -> ArtifactReleaseChain:
        return self._artifacts

    @property
    def checkpoint_strategy(self) -> CheckpointStrategy:
        return self._checkpoint_strategy

    @property
    def store(self) -> ScenarioStore:
        return self._store

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

    def releases(self) -> tuple[dict[str, Any], ...]:
        with self._publication_lock:
            return self._releases.releases(self._step)

    def _bound_trainer(self, component: str | None) -> ComponentTrainer:
        """The trainer bound to ``component``; ``None`` selects the first, for scenario-wide operations."""
        if component is None:
            return self._trainers[0]
        for bound in self._trainers:
            if bound.component == component:
                return bound
        raise ReefError(f"scenario {self._name!r} has no trainer for component {component!r}")

    def _trainer_for(self, component: str | None) -> Trainer:
        return self._bound_trainer(component).trainer

    def _own_record(self, record: CommitRecord, component: str | None) -> bool:
        """Whether ``component``'s trainer made ``record``; a record naming no trainer belongs to the only one."""
        if record.component == component:
            return True
        return record.component is None and record.operation == "training" and len(self._trainers) == 1

    def _compactable_for(self, component: str | None) -> frozenset[str] | None:
        """The rows every other trainer has released; ``None`` when there is no other trainer.

        Trainers of one scenario consume the same records, so a row is retired
        only when no trainer still needs it.
        """
        others = [bound.trainer for bound in self._trainers if bound.component != component]
        if not others:
            return None
        released: frozenset[str] | None = None
        for trainer in others:
            ids = trainer.releasable_agent_record_ids()
            released = ids if released is None else released & ids
        return frozenset() if released is None else released

    def reject_pending(self, component: str | None, metrics: Mapping[str, Any] | None = None) -> None:
        """Drop ``component``'s reserved batch, retiring only rows every trainer has released."""
        with self._lock:
            bound = self._bound_trainer(component)
            bound.trainer.reject_pending(metrics, compactable=self._compactable_for(bound.component))

    def last_record_for(self, component: str | None) -> CommitRecord | None:
        """The newest durable commit made by ``component``'s trainer."""
        records = self._store.history() if self._store.durable else ()
        name = self._bound_trainer(component).component
        return next((record for record in reversed(records) if self._own_record(record, name)), None)

    def record_is_current(self, record: CommitRecord) -> bool:
        """Whether ``record`` still describes what is served.

        A lone trainer's newest record is current only at the scenario step:
        a rollback after it moves the step on. With several trainers the step
        also moves on every other trainer's commit, so the record stays current
        until a rollback or promote lands after it.
        """
        if len(self._trainers) == 1:
            return record.step == self._step
        records = self._store.history() if self._store.durable else ()
        return all(later.operation == "training" for later in records if later.step > record.step)

    def _release_manifest(self, artifact: Artifact) -> ReleaseComponents:
        """The manifest a release carries: its own, or the one-component manifest a flat release implies."""
        manifest = artifact.components
        if manifest is not None:
            return manifest
        surface = self._binding.surface
        name = surface.names[0] if surface.names else self._trainers[0].component
        return ReleaseComponents({name: ComponentEntry(artifact.ref.content_id)})

    def rollback(self, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        """Publish a durable copy of an older version as a new fenced commit; promote uses the same path."""
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        release_id = release_id.strip()
        with self._lock, self._publication_lock:
            current_ref = self._artifacts.current
            if current_ref.release_id == release_id:
                return current_ref
            target = self._releases.find_release(release_id)
            if target is None:
                raise ArtifactNotFound(f"scenario {self._name!r} has no release {release_id!r}")
            target_ref, target_checkpoint = target
            if not target_checkpoint or isinstance(target_ref, LiveWeightArtifactRef):
                raise ReleaseNotRestorable(
                    f"scenario {self._name!r} release {release_id!r} has no durable checkpoint bytes"
                )
            if any(bound.trainer.pending_batch is not None for bound in self._trainers):
                raise ReefError("cannot rollback while a training result is pending commit")

            artifacts = self._artifacts
            checkpoint = artifacts.checkpoint
            next_step = self._step + 1
            prepared = self._trainer.prepare_commit(None)
            recorded = self._recorded_operation_retry(prepared, operation, release_id, next_step)
            if recorded is not None:
                recorded = self._store.commit_step(expected_step=self._step, commit=recorded)
                self._reconcile_recorded_artifact(recorded)
                self._settle_trainer_commit(prepared, recorded, next_step, self._trainer)
                self._resume_restored_weights()
                return recorded.artifact_ref
            if self._store.durable:
                self._synchronize_checkpoint()
            source = artifacts.resolve(target_ref)
            durable = self._store.durable
            surface = self._binding.surface
            surface.validate(source)
            # The runtime-loaded component is restored only when the engine serves other content.
            loaded_component = surface.loader_component
            served = Artifact(current_ref, artifacts.repository)
            restore_weights = loaded_component is not None and surface.component_changed(
                source, served, loaded_component
            )
            if self._binding.training_runtime is not None and loaded_component is not None and restore_weights:
                if self._binding.runtime is None:
                    raise ReefError("training checkpoint restore requires an inference runtime")
                self._binding.runtime.pause_admission()
                self._binding.training_runtime.restore_checkpoint(surface.component_artifact(source, loaded_component))
            if restore_weights:
                surface.load(source, self._binding.runtime)
            staged = artifacts.stage(next_step, source, parent=checkpoint)
            try:
                commit_metadata = scenario_metadata_for(
                    name=self._name,
                    base_artifact=artifacts.base,
                    scenario_step=next_step,
                    algorithm_state=prepared.algorithm_state,
                    record_progress=RecordProgress(
                        high_water_sequence=prepared.high_water_sequence,
                        high_water_offset=prepared.high_water_offset,
                        compacted_ids=prepared.compacted_ids,
                        consumed_ids=prepared.consumed_ids,
                    ),
                    metrics=prepared.metrics,
                    training_job_id=prepared.training_job_id,
                    operation=operation,
                    rollback_target_release_id=release_id,
                )
                commit_metadata["rollback"] = {
                    "target_release_id": release_id,
                }
                published_ref = artifacts.publish(
                    staged,
                    expected_parent=checkpoint,
                    metadata={
                        **dict(source.metadata),
                        COMPONENTS_METADATA_KEY: self._release_manifest(source).to_dict(),
                        SCENARIO_METADATA_KEY: commit_metadata,
                    },
                    advance_heads=not durable,
                )
                surface.activate(
                    artifacts.resolve(published_ref), self._binding.runtime, source=source, previous=served
                )
                record = self._append_commit_record(
                    step=next_step,
                    artifact_ref=published_ref,
                    checkpoint=True,
                    prepared=prepared,
                    operation=operation,
                    rollback_target_release_id=release_id,
                )
                if durable:
                    self._install_committed_checkpoint(
                        published_ref, expected=current_ref, expected_checkpoint=checkpoint
                    )
            except Exception:
                artifacts.discard(staged)
                raise
            self._settle_trainer_commit(prepared, record, next_step, self._trainer)
            self._resume_restored_weights()
            return published_ref

    def publish_shipped_content(self) -> ArtifactRef | None:
        """Commit the backend's update of the content this Reef ships, when the served release is stale.

        The update is a training commit with no batch behind it: it consumes no
        records and runs no evaluation, since its content is Reef's own. It is
        served at once rather than held for review for the same reason. Returns the
        new head, or ``None`` when the served release already carries the content.
        """
        with self._lock, self._publication_lock:
            current_ref = self._artifacts.current
            if isinstance(current_ref, LiveWeightArtifactRef):
                return None
            if self._store.durable:
                self._synchronize_checkpoint()
            surface = self._binding.surface
            served = self._artifacts.resolve(current_ref)
            files_component = surface.files_component
            published_tree = (
                served if files_component is None else surface.component_artifact(served, files_component)
            ).local_path
            if published_tree is None:
                return None
            component = self._trainers[0].component if files_component is None else files_component
            trainer = self._trainer_for(component)
            result = trainer.shipped_content_update(published_tree)
            if result is None:
                return None
            publication = result.publication
            if not isinstance(publication, SavedArtifactPublication) or result.pending or result.state is None:
                raise ReefError("a shipped content update must publish durable bytes at once, with its state")
            prepared = replace(
                trainer.prepare_commit(None), algorithm_state=dict(result.state), metrics=dict(result.metrics)
            )
            if publication.component is None:
                publication = SavedArtifactPublication(publication.artifact, component)
            self._commit_saved_artifact(result, publication, prepared, trainer, component)
            trainer.apply_committed_state(result.state)
            return self._artifacts.current

    def _resume_restored_weights(self) -> None:
        if self._binding.training_runtime is not None and self._binding.runtime is not None:
            self._binding.runtime.mark_published()
            self._binding.runtime.resume_admission()

    def commit(self, result: TrainStepResult, *, component: str | None = None) -> Any:
        """Commit ``component``'s pending training result as one atomic version record.

        Several trainers of one scenario meet here: the scenario lock serializes
        their commits, and a result whose batch was reserved against a release
        that another trainer has since replaced is refused as
        :class:`StaleTrainingResultError` rather than attached to a combination it
        was never evaluated with.
        """
        with self._lock, self._publication_lock:
            next_step = self._step + 1
            bound = self._bound_trainer(component)
            trainer, component = bound.trainer, bound.component
            surface = self._binding.surface
            # Refuse a stale base before the trainer acknowledges its batch, so
            # the batch stays whole for another preparation. Only another
            # trainer can move the head under a reserved batch: a lone trainer
            # finds its own failed attempt's head, and a step already in the
            # log is a retry after a lost acknowledgment. Neither is stale.
            records = self._store.history()
            retrying = bool(records) and records[-1].step == next_step
            base = trainer.pending_base_release_id
            served = self._artifacts.current.release_id
            if not retrying and len(self._trainers) > 1 and base is not None and base != served:
                raise StaleTrainingResultError(
                    f"scenario {self._name!r} component {component!r} prepared its result against release "
                    f"{base!r} but {served!r} is served now"
                )
            prepared = trainer.prepare_commit(result, compactable=self._compactable_for(component))
            recorded = self._recorded_training_retry(prepared, result, next_step, component)
            if recorded is not None:
                recorded = self._store.commit_step(expected_step=self._step, commit=recorded)
                self._reconcile_recorded_artifact(recorded)
                self._settle_trainer_commit(prepared, recorded, next_step, trainer)
                return result.state

            if self._store.durable:
                self._synchronize_checkpoint()
            publication = result.publication
            if isinstance(publication, DurableWeightsPublication):
                # Checkpoint policy lives here, so a backend that exported
                # weights hands over both options and this is the only place
                # that picks one.
                if self._should_checkpoint(result):
                    publication = SavedArtifactPublication(
                        Artifact.local(
                            Path(publication.checkpoint_path),
                            metadata={"runtime_load_id": publication.runtime_load_id},
                        ),
                        surface.loader_component,
                    )
                else:
                    publication = LiveWeightPublication(publication.runtime_load_id)

            if isinstance(publication, LiveWeightPublication):
                if not surface.single:
                    # A live release names an engine load and nothing else, so it
                    # cannot carry the other components; every step must checkpoint.
                    raise ReefError(
                        f"scenario {self._name!r} serves components {list(surface.names)}: "
                        "live weights cannot be published without a checkpoint"
                    )
                return self._commit_live_weights(result, publication, prepared, trainer)
            if isinstance(publication, NoArtifactPublication):
                return self._commit_without_artifact(result, prepared, trainer, component)
            if publication.component is None:
                publication = SavedArtifactPublication(publication.artifact, component)
            return self._commit_saved_artifact(result, publication, prepared, trainer, component)

    def _should_checkpoint(self, result: TrainStepResult) -> bool:
        return self._checkpoint_strategy.should_checkpoint(self._name, self._step + 1, result)

    def _commit_live_weights(
        self,
        result: TrainStepResult,
        publication: LiveWeightPublication,
        prepared: PreparedCommit,
        trainer: Trainer,
    ) -> Any:
        artifacts = self._artifacts
        next_step = self._step + 1
        if self._should_checkpoint(result):
            raise ReefError(
                "checkpoint-selected training results must include the artifact returned by execute_training_job"
            )
        if result.pending:
            # Live weights have no durable bytes to promote later, so holding them back is not expressible.
            raise ReefError("a pending release requires a durable checkpoint; this step publishes live weights only")
        head, live_ref = artifacts.prepare_live(step=next_step, runtime_load_id=publication.runtime_load_id)
        if not self._store.durable:
            artifacts.advance(live_ref, expected=head)
        record = self._append_commit_record(
            step=next_step,
            artifact_ref=live_ref,
            checkpoint=False,
            prepared=prepared,
        )
        if self._store.durable:
            artifacts.advance(live_ref, expected=head)
        self._settle_trainer_commit(prepared, record, next_step, trainer)
        return result.state

    def _commit_without_artifact(
        self, result: TrainStepResult, prepared: PreparedCommit, trainer: Trainer, component: str | None
    ) -> Any:
        # No pending check: a pending step carries durable bytes by construction, so it never lands here.
        next_step = self._step + 1
        record = self._append_commit_record(
            step=next_step,
            artifact_ref=self._artifacts.current,
            checkpoint=False,
            prepared=prepared,
            component=component,
        )
        self._settle_trainer_commit(prepared, record, next_step, trainer)
        return result.state

    def _commit_saved_artifact(
        self,
        result: TrainStepResult,
        publication: SavedArtifactPublication,
        prepared: PreparedCommit,
        trainer: Trainer,
        component: str | None = None,
    ) -> Any:
        artifacts = self._artifacts
        next_step = self._step + 1
        checkpoint = artifacts.checkpoint
        head = artifacts.current
        pending = result.pending
        durable = self._store.durable
        checkpointed = pending or self._should_checkpoint(result)
        local_artifact = self._stage_publication(next_step, publication, checkpoint)
        try:
            # A pending release is recorded but never activated and moves no head.
            # The engine must confirm the new revision before anything moves
            # the served head: the staged bytes load first, and the version
            # minted by publication then aliases them.
            if not pending:
                self._activate(local_artifact)
            if checkpointed:
                commit_metadata = scenario_metadata_for(
                    name=self._name,
                    base_artifact=artifacts.base,
                    scenario_step=next_step,
                    algorithm_state=prepared.algorithm_state,
                    record_progress=RecordProgress(
                        high_water_sequence=prepared.high_water_sequence,
                        high_water_offset=prepared.high_water_offset,
                        compacted_ids=prepared.compacted_ids,
                        consumed_ids=prepared.consumed_ids,
                    ),
                    metrics=prepared.metrics,
                    training_job_id=prepared.training_job_id,
                    component=component,
                    base_release_id=prepared.base_release_id,
                )
                published_ref = artifacts.publish(
                    local_artifact,
                    expected_parent=checkpoint,
                    # The staged release's metadata (the publication's own for a
                    # flat release) plus the manifest naming its components.
                    metadata={
                        **dict(local_artifact.metadata),
                        COMPONENTS_METADATA_KEY: self._release_manifest(local_artifact).to_dict(),
                        SCENARIO_METADATA_KEY: commit_metadata,
                    },
                    advance_heads=not pending and not durable,
                )
                if not pending:
                    self._activate(artifacts.resolve(published_ref), source=local_artifact)
                record = self._append_commit_record(
                    step=next_step,
                    artifact_ref=published_ref,
                    checkpoint=True,
                    prepared=prepared,
                    pending=pending,
                    component=component,
                )
            else:
                if not durable:
                    # Without a durable commit point, reject a changed serving
                    # head before recording or compacting the prepared step.
                    artifacts.advance(local_artifact.ref, expected=head)
                record = self._append_commit_record(
                    step=next_step,
                    artifact_ref=local_artifact.ref,
                    checkpoint=False,
                    prepared=prepared,
                    component=component,
                )
            if not pending:
                if checkpointed and durable:
                    self._install_committed_checkpoint(published_ref, expected=head, expected_checkpoint=checkpoint)
                elif not checkpointed and durable:
                    artifacts.advance(local_artifact.ref, expected=head)
        except Exception:
            # A lost append acknowledgment can leave a committed local release.
            # Its bytes must survive so retry can settle that exact record.
            if not any(row.artifact_ref == local_artifact.ref for row in self._store.history()):
                artifacts.discard(local_artifact)
            raise

        self._settle_trainer_commit(prepared, record, next_step, trainer)
        return result.state

    def _install_committed_checkpoint(
        self, ref: ArtifactRef, *, expected: ArtifactRef, expected_checkpoint: ArtifactRef
    ) -> None:
        """Install the refs already committed in the store, then synchronize artifact storage."""
        self._artifacts.install_checkpoint(ref, expected=expected, expected_checkpoint=expected_checkpoint)
        self._refresh_committed_checkpoint()

    def _refresh_committed_checkpoint(self) -> None:
        # The step is already committed. Expose the pointer failure in
        # commit_status without making the caller repeat a successful step.
        # A new commit must synchronize first and propagates any failure.
        with suppress(ArtifactPublicationError, ArtifactConflict):
            self._synchronize_checkpoint()

    def _synchronize_checkpoint(self) -> None:
        checkpoint = self._artifacts.checkpoint
        try:
            self._artifacts.repository.synchronize_checkpoint()
        except ArtifactConflict as exc:
            self._artifact_head_sync = _ArtifactHeadSync("conflict", checkpoint.release_id, str(exc))
            raise
        except ArtifactPublicationError as exc:
            self._artifact_head_sync = _ArtifactHeadSync("pending", checkpoint.release_id, str(exc))
            raise
        else:
            self._artifact_head_sync = _ArtifactHeadSync("synchronized", checkpoint.release_id)
        finally:
            self._commit_status = (self._step, self._latest_training_record, self._artifact_head_sync)

    def _stage_publication(
        self, next_step: int, publication: SavedArtifactPublication, checkpoint: ArtifactRef
    ) -> Artifact:
        """Admit the published component and stage the release it belongs to.

        A flat scenario stages the artifact as published. A multi-component
        scenario replaces the named component and carries every other
        component forward from the checkpoint, so one step changes one
        component and the release still binds the whole combination.
        """
        surface = self._binding.surface
        component = publication.component
        if component is None:
            raise ReefError(f"scenario {self._name!r}: a publication names the component it replaces")
        if surface.names and component not in surface.names:
            raise ReefError(f"scenario {self._name!r} serves no component {component!r}")
        if surface.single:
            surface.validate(publication.artifact)
            return self._artifacts.stage(next_step, publication.artifact, parent=checkpoint)
        surface.components[component].validator.validate(publication.artifact)
        carried = self._artifacts.resolve(checkpoint)
        components = {name: carried.component(name) for name in surface.names}
        components[component] = publication.artifact
        return self._artifacts.stage_composed(next_step, components, parent=checkpoint)

    def _activate(self, artifact: Artifact, *, source: Artifact | None = None) -> None:
        # A component the engine already serves is not activated again, so a
        # harness step never reloads weights; the served head is compared by
        # identity and materialized only when a composed manifest is needed.
        previous = Artifact(self._artifacts.current, self._artifacts.repository)
        self._binding.surface.activate(artifact, self._binding.runtime, source=source, previous=previous)

    def _settle_trainer_commit(
        self, prepared: PreparedCommit, record: CommitRecord, next_step: int, trainer: Trainer
    ) -> None:
        """Finish recoverable effects before exposing the prepared state."""
        trainer.commit_applied(prepared.algorithm_state)
        # Retired rows leave every trainer's processor memory, not only the committing one's.
        for bound in self._trainers:
            bound.trainer.compaction_applied(prepared.compacted_ids)
        trainer.commit(prepared)
        if not self._store.durable:
            self._artifact_head_sync = _ArtifactHeadSync("synchronized", self._artifacts.checkpoint.release_id)
        if record.operation == "training":
            self._latest_training_record = record
        self.advance_to(next_step)

    def _recorded_training_retry(
        self,
        prepared: PreparedCommit,
        result: TrainStepResult,
        next_step: int,
        component: str | None,
    ) -> CommitRecord | None:
        records = self._store.history()
        if not records or records[-1].step != next_step:
            return None
        record = records[-1]
        matches = (
            record.operation == "training"
            and record.pending == result.pending
            and self._own_record(record, component)
            and self._record_matches_prepared(record, prepared)
        )
        if not matches:
            raise ScenarioStoreConflict(f"commit step {next_step} conflicts with the pending training result")
        return record

    def _recorded_operation_retry(
        self,
        prepared: PreparedCommit,
        operation: str,
        target_release_id: str,
        next_step: int,
    ) -> CommitRecord | None:
        records = self._store.history()
        if not records or records[-1].step != next_step:
            return None
        record = records[-1]
        if (
            record.operation != operation
            or record.rollback_target_release_id != target_release_id
            or not self._record_matches_prepared(record, prepared)
        ):
            raise ScenarioStoreConflict(f"commit step {next_step} conflicts with the pending {operation}")
        return record

    @staticmethod
    def _record_matches_prepared(record: CommitRecord, prepared: PreparedCommit) -> bool:
        return (
            record.algorithm_state == prepared.algorithm_state
            and record.high_water_sequence == prepared.high_water_sequence
            and record.high_water_offset == prepared.high_water_offset
            and record.compacted_ids == prepared.compacted_ids
            and record.consumed_ids == prepared.consumed_ids
            and record.metrics == prepared.metrics
            and record.training_job_id == prepared.training_job_id
        )

    def _reconcile_recorded_artifact(self, record: CommitRecord) -> None:
        current = self._artifacts.current
        if record.pending:
            return
        if current == record.artifact_ref:
            if record.checkpoint:
                self._refresh_committed_checkpoint()
            return
        records = self._store.history()
        previous = next(
            (prior.artifact_ref for prior in reversed(records[:-1]) if not prior.pending),
            self._releases.creation_artifact,
        )
        if record.checkpoint:
            self._install_committed_checkpoint(
                record.artifact_ref, expected=previous, expected_checkpoint=self._artifacts.checkpoint
            )
        else:
            self._artifacts.advance(record.artifact_ref, expected=previous)

    def _append_commit_record(
        self,
        *,
        step: int,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        prepared: PreparedCommit,
        operation: str = "training",
        rollback_target_release_id: str | None = None,
        pending: bool = False,
        component: str | None = None,
    ) -> CommitRecord:
        record = CommitRecord(
            scenario=self._name,
            step=step,
            artifact_ref=artifact_ref,
            checkpoint=checkpoint,
            pending=pending,
            algorithm_state=prepared.algorithm_state,
            high_water_sequence=prepared.high_water_sequence,
            high_water_offset=prepared.high_water_offset,
            compacted_ids=prepared.compacted_ids,
            consumed_ids=prepared.consumed_ids,
            operation=operation,
            rollback_target_release_id=rollback_target_release_id,
            metrics=prepared.metrics,
            training_job_id=prepared.training_job_id,
            component=component,
            base_release_id=prepared.base_release_id if operation == "training" else None,
        )
        return self._store.commit_step(expected_step=self._step, commit=record)

    def metrics_for_version(self, release_id: str) -> Mapping[str, Any] | None:
        return self._releases.metrics_for_version(release_id)

    def entries_for_version(self, release_id: str) -> tuple[Mapping[str, Any], ...] | None:
        return self._releases.entries_for_version(release_id)

    def artifact_for_version(self, release_id: str) -> Artifact:
        return self._releases.artifact_for_version(release_id)

    def artifact_with_metrics(
        self,
        release_id: str | None = None,
    ) -> tuple[Artifact, Mapping[str, Any] | None]:
        return self._releases.artifact_with_metrics(release_id)


__all__ = [
    "ScenarioCommitter",
    "StaleTrainingResultError",
]
