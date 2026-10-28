"""Scenario committer coordinating training and artifact publication.

Artifact storage and head movement live in ``reef.artifact``; the durable
record values and checkpoint metadata formats live in ``commits``. This
module owns the scenario-specific ordering across trainer state, store
settlement, surfaces, and artifact operations.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Any, Literal

from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactNotFound, ArtifactPublicationError, ArtifactRef
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.core.activation import MadeArtifact
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, ReleaseComponents
from reef.core.errors import CommitPending, ReefError
from reef.observability.base import TrainingExperimentContext
from reef.observability.commits import CommitExperimentObserver
from reef.scenario.binding import ScenarioBinding
from reef.scenario.releases import ScenarioReleases
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    CommitOperation,
    CommitRecord,
    RecordProgress,
    active_artifact_ref,
    scenario_metadata_for,
    to_json_value,
)
from reef.storage.scenario import ScenarioStore, ScenarioStoreConflict
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import NoArtifactPublication, PreparedCommit, SavedArtifactPublication, TrainStepResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentArtifact:
    """One component's artifact, as an activation or a new training start names it, and a release that binds it.

    Values: ``component`` is a component the scenario names (``ScenarioCommitter.component_names``).
    ``artifact_id`` is the artifact's identity: ``release_id`` of the release that made it in a scenario of one
    component, ``<release_id>:<component>`` in a scenario of several. ``content_id`` is the artifact's content id:
    the release's own with one component, the component entry's with several. ``release_id`` names a release that
    binds the artifact as ``component``: the active release after an activation, and the release that minted the
    artifact for a new training start. Every field is a non-empty string.

    Purpose: ``ScenarioCommitter.activate`` and ``train_from`` make it, and the service routes answer with it.
    """

    component: str
    artifact_id: str
    content_id: str
    release_id: str

    def __post_init__(self) -> None:
        values = (self.component, self.artifact_id, self.content_id, self.release_id)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("component, artifact_id, content_id and release_id must be non-empty strings")


@dataclass(frozen=True)
class _ArtifactHeadSync:
    state: Literal["synchronized", "pending", "conflict"]
    release_id: str
    error: str | None = None


class ScenarioCommitter:
    """Order one scenario's commits, activations, new training starts and retries; delegate release queries."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        artifacts: ArtifactReleaseChain,
        trainers: tuple[ComponentTrainer, ...],
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
        self._artifacts = artifacts
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
        self.release_queries = ScenarioReleases(
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
        every write path names no component and no base release for it.
        """
        return component if len(self.trainers) > 1 else None

    def own_record(self, record: CommitRecord, component: str | None) -> bool:
        """Whether ``component``'s trainer made the training record ``record``; one naming no trainer is the lone trainer's.

        ``component`` is the name on record (``component_on_record``). An
        activation or a new training start is no trainer's own record: it
        carries no trainer state.
        """
        return record.operation == "training" and record.component == component

    def last_record_for(self, component: str | None) -> CommitRecord | None:
        """The newest durable training commit made by ``component``'s trainer."""
        records = self._store.history() if self._store.durable else ()
        name = self.component_on_record(self.component_trainer(component).component)
        return next((record for record in reversed(records) if self.own_record(record, name)), None)

    def release_manifest(self, artifact: Artifact) -> ReleaseComponents:
        """The manifest a release carries: its own, or the one-component manifest a flat release implies."""
        manifest = artifact.components
        if manifest is not None:
            return manifest
        surface = self._binding.surface
        if not surface.single:
            raise ReefError(
                f"scenario {self._name!r} serves components {list(surface.names)} but release "
                f"{artifact.ref.release_id!r} carries no component manifest"
            )
        name = surface.names[0] if surface.names else self.trainers[0].component
        return ReleaseComponents({name: ComponentEntry(artifact.ref.content_id)})

    def require_components(self, artifact: Artifact) -> None:
        """Refuse a release that does not bind every component this scenario serves.

        Reading such a release as composed would hand the whole tree, or
        nothing, to each component; every release a composed scenario
        registers or publishes carries a full manifest, so this is a
        misdirected repository, not a supported input.
        """
        surface = self._binding.surface
        if surface.single:
            return
        manifest = self.release_manifest(artifact)
        missing = [name for name in surface.names if name not in manifest.entries]
        if missing:
            raise ReefError(
                f"scenario {self._name!r} serves components {list(surface.names)} but release "
                f"{artifact.ref.release_id!r} binds {list(manifest.names)}"
            )

    def recorded_components(self, artifact: Artifact) -> dict[str, str] | None:
        """The content id of each component a release binds, for its commit record; ``None`` for a flat release."""
        if self._binding.surface.single:
            return None
        return self.release_manifest(artifact).content_ids

    def component_names(self) -> tuple[str, ...]:
        """The components whose artifacts this scenario names: the surface's, or the lone trainer's when it serves none."""
        return self._binding.surface.names or (self.trainers[0].component,)

    def require_no_pending_commit(self, action: str) -> None:
        """Raise ``CommitPending`` while any trainer's result waits for its commit; ``action`` names what waits."""
        if any(component_trainer.trainer.has_pending_commit for component_trainer in self.trainers):
            raise CommitPending(f"cannot {action} while a training result waits for its commit")

    def resolve_artifact(self, artifact_id: str, component: str | None) -> tuple[ComponentArtifact, Artifact]:
        """The artifact ``artifact_id`` names as ``component``'s content, and the resolved release that minted it.

        With one component, ``component`` may be ``None`` and
        ``artifact_id`` is the ``release_id`` of the release that made the
        artifact. With several, ``component`` is required and ``artifact_id``
        is ``<release_id>:<component>`` for a release that minted that
        component's content (``ScenarioReleases.artifacts``). Raise
        ``ValueError`` for a missing or unknown component,
        ``ArtifactNotFound`` for an ``artifact_id`` the scenario does not
        list, and ``ReleaseNotRestorable`` for a release without durable
        bytes or without every component the scenario serves.
        """
        names = self.component_names()
        if component is None:
            if len(names) > 1:
                raise ValueError(f"scenario {self._name!r} has components {list(names)}: name the component")
            component = names[0]
        if component not in names:
            raise ValueError(f"scenario {self._name!r} has no component {component!r}")
        listed = self.release_queries.artifacts(self._step, names)
        row = next((row for row in listed if (row["artifact_id"], row["component"]) == (artifact_id, component)), None)
        if row is None:
            raise ArtifactNotFound(
                f"scenario {self._name!r} has no artifact {artifact_id!r} of component {component!r}"
            )
        found = self.release_queries.find_release(row["release_id"])
        if found is None or not found[1]:
            raise ReleaseNotRestorable(f"scenario {self._name!r} artifact {artifact_id!r} has no durable bytes")
        release = self._artifacts.resolve(found[0])
        try:
            self.require_components(release)
        except ReefError as exc:
            raise ReleaseNotRestorable(str(exc)) from exc
        content_id = self._binding.surface.component_artifact(release, component).ref.content_id
        return ComponentArtifact(component, artifact_id, content_id, release.ref.release_id), release

    def activate(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Make ``artifact_id`` the active artifact of ``component``; return it with the release that is active after.

        With one component the release that made the artifact becomes the
        active release: no artifact or release is written. With several, the
        activation publishes a release: the active release with
        ``component`` bound to the artifact. It reuses the components'
        content, and the chain tip, the parent of the next training
        publication, stays. No training start changes. The runtime activates
        the release first, then the ``"activate"`` record lands at the next
        step, then the active release moves; a crash before the record leaves
        the active release that the log names. Activating the active artifact
        changes nothing. Raise as :meth:`resolve_artifact` does, and
        ``CommitPending`` while a training result waits for its commit.
        """
        with self._lock, self._publication_lock:
            selected, release = self.resolve_artifact(artifact_id, component)
            surface = self._binding.surface
            current_ref = self._artifacts.current
            served = Artifact(current_ref, self._artifacts.repository)
            if surface.component_artifact(served, selected.component).ref.content_id == selected.content_id:
                return replace(selected, release_id=current_ref.release_id)
            self.require_no_pending_commit("activate an artifact")
            if surface.single:
                # The release's own admission check, as at its publication.
                surface.validate(release)
                target, recorded_component, components = release, None, None
            else:
                staged = self.stage_combination(
                    self._step + 1, served, selected.component, release.component(selected.component)
                )
                manifest = self.release_manifest(staged)
                # Published beside the chain: the chain tip and the backend head stay.
                published_ref = self._artifacts.publish(
                    staged,
                    expected_parent=self._artifacts.checkpoint,
                    metadata={**dict(staged.metadata), COMPONENTS_METADATA_KEY: manifest.to_dict()},
                    advance_heads=False,
                )
                target, recorded_component, components = (
                    self._artifacts.resolve(published_ref),
                    selected.component,
                    manifest.content_ids,
                )
            surface.activate(target, self._binding.runtime, previous=served)
            record = self.append_operation_record(
                "activate", target.ref, component=recorded_component, components=components
            )
            self._artifacts.activate(target.ref, expected=current_ref)
            self.advance_to(record.step)
            return replace(selected, release_id=target.ref.release_id)

    def train_from(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Set ``component``'s training start to ``artifact_id``; return the artifact.

        The trainer's running job is cancelled, and its next job or step
        starts from the artifact. The active release does not change.
        ``component`` may be left out when the scenario has one component.
        The trainer takes the start first, then the ``"train_from"`` record
        lands at the next step; a restart before the trainer's next training
        commit takes the start again from that record. Raise as
        :meth:`resolve_artifact` does, ``TrainingStartRefused`` when the
        trainer cannot start from the artifact, and ``CommitPending`` while
        a training result waits for its commit.
        """
        # The start can import weights for minutes: readers of the releases wait only for the record.
        with self._lock:
            selected, release = self.resolve_artifact(artifact_id, component)
            component_trainer = self.component_trainer(selected.component)
            self.require_no_pending_commit("set the training start")
            self.start_trainer(component_trainer, release)
            with self._publication_lock:
                record = self.append_operation_record(
                    "train_from", release.ref, component=self.component_on_record(component_trainer.component)
                )
                self.advance_to(record.step)
            return selected

    def resume_training_starts(self) -> None:
        """After a restart, give each trainer its newest ``"train_from"`` start that its training records do not hold.

        The trainer's next training commit records the start in its state;
        a restart before that commit recovers the state of the earlier
        commit, so the start is taken again here.
        """
        records = self._store.history() if self._store.durable else ()
        for component_trainer in self.trainers:
            name = self.component_on_record(component_trainer.component)
            last = self.last_record_for(component_trainer.component)
            last_step = 0 if last is None else last.step
            start = next(
                (
                    record
                    for record in reversed(records)
                    if record.step > last_step and record.operation == "train_from" and record.component == name
                ),
                None,
            )
            if start is not None:
                self.start_trainer(component_trainer, self._artifacts.resolve(start.artifact_ref))

    def start_trainer(self, component_trainer: ComponentTrainer, release: Artifact) -> None:
        """Start ``component_trainer`` from its component of ``release``, with the entries its commit recorded."""
        artifact = self._binding.surface.component_artifact(release, component_trainer.component).materialize()
        component_trainer.trainer.start_from(
            artifact, self.release_queries.entries_for_version(release.ref.release_id)
        )

    def publish_shipped_content(self) -> ArtifactRef | None:
        """Commit the backend's update of the content this Reef ships, when the active release is stale.

        The update is a training commit with no job data behind it: it consumes no
        records and runs no evaluation, since its content is Reef's own. It is
        activated at once for the same reason. Returns the new active release, or
        ``None`` when the active release already carries the content.
        """
        with self._lock, self._publication_lock:
            current_ref = self._artifacts.current
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
            component = self.trainers[0].component if files_component is None else files_component
            trainer = self.trainer_for(component)
            result = trainer.shipped_content_update(published_tree)
            if result is None:
                return None
            publication = result.publication
            if not isinstance(publication, SavedArtifactPublication) or result.state is None:
                raise ReefError("a shipped content update must publish durable bytes at once, with its state")
            prepared = replace(
                trainer.prepare_commit(None), algorithm_state=dict(result.state), metrics=dict(result.metrics)
            )
            # The content that this Reef ships replaces the served tree whatever the activation rule says.
            self.commit_saved_artifact(publication, prepared, trainer, component, activated=True)
            trainer.apply_committed_state(result.state)
            return self._artifacts.current

    def commit(self, result: TrainStepResult, *, component: str | None = None) -> Any:
        """Commit ``component``'s pending training result as one atomic version record.

        The trainers of one scenario run one step at a time, so no other
        trainer's commit lands while this step runs. The head is the release
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
            if not retrying:
                result = self.prepare_experiment(result, trainer, next_step, component)
            with self._publication_lock:
                prepared = trainer.prepare_commit(result)
                # The activation rule decides whether the step's artifact serves. The training record keeps the result.
                activated = not isinstance(result.publication, NoArtifactPublication) and (
                    self._binding.activation_rule.activates(MadeArtifact(component, result.changed_kinds))
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
        self, result: TrainStepResult, trainer: Trainer, step: int, component: str | None
    ) -> TrainStepResult:
        observer = self.experiment_observer
        if observer is None:
            return result
        backend = trainer.candidate_backend
        job_loop = trainer.job_loop
        try:
            if backend is not None:
                backend_config: dict[str, Any] | None = dict(backend.experiment_config())
            elif job_loop is not None:
                backend_config = dict(job_loop.plan.experiment_config())
            else:
                backend_config = None
        except Exception:
            logger.exception("training backend failed to describe experiment configuration")
            backend_config = None
        if backend is not None:
            backend_name: str | None = type(backend).__name__
        elif job_loop is not None:
            backend_name = job_loop.component.name
        else:
            backend_name = None
        context = TrainingExperimentContext(
            scenario=self._name,
            recipe=observer.recipe,
            step=step,
            source_artifact_ref=observer.start_artifact_ref,
            run_segment=observer.run_segment,
            run_step=observer.run_step,
            backend=backend_name,
            backend_config=backend_config,
            component=self.component_on_record(component),
        )
        artifact = result.artifact
        correlation = observer.prepare(
            context,
            checkpoint_path=None if artifact is None or artifact.local_path is None else str(artifact.local_path),
        )
        if correlation:
            try:
                return trainer.add_commit_metrics(result, correlation)
            except RuntimeError:
                logger.exception("experiment correlation could not annotate the pending result")
        return result

    def commit_without_artifact(self, prepared: PreparedCommit, trainer: Trainer, component: str | None) -> None:
        next_step = self._step + 1
        record = self._append_commit_record(
            step=next_step,
            artifact_ref=self._artifacts.current,
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
        """Publish a training step's artifact on the chain and record it. When ``activated`` is true, serve it too.

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
        local_artifact = self.stage_publication(next_step, publication, checkpoint)
        try:
            # The engine must confirm the new revision before anything moves
            # the active release: the staged bytes load first, and the version
            # minted by publication then aliases them.
            if activated:
                self._activate(local_artifact)
            published_ref = artifacts.publish(
                local_artifact,
                expected_parent=checkpoint,
                metadata=self.release_metadata(
                    local_artifact,
                    prepared,
                    next_step,
                    component=component,
                    active_artifact=None if activated else head,
                ),
                advance_heads=not durable,
                activate=activated,
            )
            if activated:
                self._activate(artifacts.resolve(published_ref), source=local_artifact)
            record = self._append_commit_record(
                step=next_step,
                artifact_ref=published_ref,
                checkpoint=True,
                prepared=prepared,
                activated=activated,
                component=component,
                components=self.recorded_components(local_artifact),
            )
        except Exception:
            # A lost append acknowledgment can leave a committed local release.
            # Its bytes must survive so retry can settle that exact record.
            if not any(row.artifact_ref == local_artifact.ref for row in self._store.history()):
                artifacts.discard(local_artifact)
            raise

        self.finish_committed_step(prepared, record, trainer, expected=head, expected_checkpoint=checkpoint)

    def release_metadata(
        self,
        artifact: Artifact,
        prepared: PreparedCommit,
        scenario_step: int,
        *,
        component: str | None = None,
        active_artifact: ArtifactRef | None = None,
    ) -> dict[str, Any]:
        """The metadata a committed release carries.

        It is the staged release's own metadata (the publication's own for a
        flat release), the manifest naming its components, and the scenario
        metadata of the commit ``prepared`` describes.
        """
        scenario_metadata = scenario_metadata_for(
            name=self._name,
            base_artifact=self._artifacts.base,
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
        )
        return {
            **dict(artifact.metadata),
            COMPONENTS_METADATA_KEY: self.release_manifest(artifact).to_dict(),
            SCENARIO_METADATA_KEY: scenario_metadata,
        }

    def _install_committed_checkpoint(
        self, ref: ArtifactRef, *, expected: ArtifactRef, expected_checkpoint: ArtifactRef, activate: bool
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

    def stage_publication(
        self, next_step: int, publication: SavedArtifactPublication, checkpoint: ArtifactRef
    ) -> Artifact:
        """Admit the published component and stage the release it belongs to, with ``checkpoint`` as its parent.

        A flat scenario stages the artifact as published. A multi-component
        scenario binds the named component to the publication and carries
        every other component from the active release
        (:meth:`stage_combination`), so a step changes one component and
        never brings back what an activation replaced.
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
        active = self._artifacts.resolve(self._artifacts.current)
        self.require_components(active)
        return self.stage_combination(next_step, active, component, publication.artifact)

    def stage_combination(self, next_step: int, carried: Artifact, component: str, content: Artifact) -> Artifact:
        """Stage the release ``carried`` with ``component`` bound to ``content``, on the chain tip as its parent.

        The release's own check runs on the staged release; the carried
        components were admitted when they were published. A refused release
        is discarded.
        """
        surface = self._binding.surface
        components = {name: carried.component(name) for name in surface.names}
        components[component] = content
        staged = self._artifacts.stage_composed(next_step, components, parent=self._artifacts.checkpoint)
        try:
            surface.validator.validate(staged)
        except BaseException:
            self._artifacts.discard(staged)
            raise
        return staged

    def _activate(self, artifact: Artifact, *, source: Artifact | None = None) -> None:
        # A component the engine already serves is not activated again, so a
        # harness step never reloads weights; the active release is compared by
        # identity and materialized only when a composed manifest is needed.
        previous = Artifact(self._artifacts.current, self._artifacts.repository)
        self._binding.surface.activate(artifact, self._binding.runtime, source=source, previous=previous)

    def finish_committed_step(
        self,
        prepared: PreparedCommit,
        record: CommitRecord,
        trainer: Trainer,
        *,
        expected: ArtifactRef,
        expected_checkpoint: ArtifactRef,
    ) -> None:
        """Complete a recorded step before exposing trainer state; callers hold both commit locks.

        Install heads and settle the trainer before advancing the scenario step.
        Failed completion retries the same record. A step that made no
        artifact records the active one and moves no head. A made artifact
        becomes the chain tip. It also becomes the active artifact when the
        record is ``activated``.
        """
        if record.made_artifact:
            if self._artifacts.checkpoint == record.artifact_ref:
                if self._store.durable:
                    self._refresh_committed_checkpoint()
            else:
                self._install_committed_checkpoint(
                    record.artifact_ref,
                    expected=expected,
                    expected_checkpoint=expected_checkpoint,
                    activate=record.activated,
                )
        trainer.commit_applied(prepared.algorithm_state)
        trainer.commit(prepared)
        if not self._store.durable:
            self._artifact_head_sync = _ArtifactHeadSync("synchronized", self._artifacts.checkpoint.release_id)
        if record.operation == "training":
            self._latest_training_record = record
        self.advance_to(record.step)

    def finish_recorded_retry(self, record: CommitRecord, prepared: PreparedCommit, trainer: Trainer) -> CommitRecord:
        """Retry storage settlement, then complete the exact release already on record."""
        record = self._store.commit_step(expected_step=self._step, commit=record)
        previous = active_artifact_ref(self._store.history()[:-1], self.release_queries.creation_artifact)
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

    def append_operation_record(
        self,
        operation: CommitOperation,
        artifact_ref: ArtifactRef,
        *,
        component: str | None,
        components: Mapping[str, str] | None = None,
    ) -> CommitRecord:
        """Append an ``"activate"`` or ``"train_from"`` record at the next step; it carries no trainer state.

        ``components`` is the manifest of a release that an activation published, and ``component`` then names
        the activated component.
        """
        record = CommitRecord(
            scenario=self._name,
            step=self._step + 1,
            artifact_ref=artifact_ref,
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
            operation=operation,
            component=component,
            components=components,
        )
        return self._store.commit_step(expected_step=self._step, commit=record)

    def _append_commit_record(
        self,
        *,
        step: int,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        prepared: PreparedCommit,
        activated: bool = False,
        component: str | None = None,
        components: Mapping[str, str] | None = None,
    ) -> CommitRecord:
        record = CommitRecord(
            scenario=self._name,
            step=step,
            artifact_ref=artifact_ref,
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
        )
        return self._store.commit_step(expected_step=self._step, commit=record)


__all__ = ["ComponentArtifact", "ScenarioCommitter"]
