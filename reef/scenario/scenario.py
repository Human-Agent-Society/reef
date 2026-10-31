"""Scenario aggregate and its durable recovery metadata."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.artifact.repository import Repository
from reef.artifact.sources import SourceFiles
from reef.core.activation import ActivationRule
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError
from reef.core.reports import ReportBase
from reef.inference.model_config import ModelConfig
from reef.observability.commits import CommitExperimentObserver
from reef.observability.operations import OperationMetrics
from reef.runtime.interfaces import InferenceRuntime
from reef.runtime.jobs import TrainingComponent
from reef.scenario.binding import ScenarioBinding
from reef.scenario.committer import ComponentArtifact, ScenarioCommitter
from reef.scenario.releases import ServedBesideArtifact
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord
from reef.storage.records import RecordStore
from reef.storage.scenario import ScenarioStore
from reef.surface.base import Surface
from reef.train.instructions import ScenarioInstructions
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import TrainStepResult


@dataclass(frozen=True)
class ScenarioCapabilities:
    """The artifact operations that one scenario can run, as its store, runtime and trainers allow them.

    Values:

    - ``serve_beside`` is True exactly when ``Scenario.serve`` can serve an artifact beside the active release: the
      store keeps a durable commit history, and the scenario has no runtime-loaded component, no runtime, or a
      runtime that serves artifacts on request (``InferenceRuntime.serves_artifacts_on_request``).
    - ``evaluate`` holds the components whose artifacts an artifact evaluation can run, in surface order: each
      component with a kind, when no component is a ``harness_plugin`` and the runtime serves artifacts on request;
      else it is empty.
    - ``train_from`` holds the components whose trainer can start from an artifact (``Trainer.can_train_from``), in
      trainer order.

    Each name occurs at most once in its tuple.

    Purpose: ``Scenario.capabilities`` makes it, and the contract route reports it, so that a client offers only the
    operations that the scenario can run. An operation that it allows can still refuse one artifact, for example a
    release whose weights the training component cannot import.
    """

    serve_beside: bool
    evaluate: tuple[str, ...]
    train_from: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.serve_beside, bool):
            raise TypeError("serve_beside must be a bool")
        for field_name, names in (("evaluate", self.evaluate), ("train_from", self.train_from)):
            if not isinstance(names, tuple) or not all(isinstance(name, str) and name for name in names):
                raise TypeError(f"{field_name} must be a tuple of component names")
            if len(set(names)) != len(names):
                raise ValueError(f"{field_name} names a component twice")

    def to_json(self) -> dict[str, object]:
        """``{serve_beside, evaluate, train_from}``, with the tuples as lists."""
        return {
            "serve_beside": self.serve_beside,
            "evaluate": list(self.evaluate),
            "train_from": list(self.train_from),
        }


class Scenario:
    """Scenario aggregate owning one runtime binding, one release chain, and one trainer per component."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        repository: Repository,
        store: ScenarioStore,
        trainers: tuple[ComponentTrainer, ...],
        model_config: ModelConfig | None = None,
        scenario_step: int = 0,
        recovered_head_record: CommitRecord | None = None,
        experiment_observer: CommitExperimentObserver | None = None,
        instructions: ScenarioInstructions | None = None,
    ) -> None:
        #: The scenario's training instructions; None for a scenario built without an instructions queue, which takes
        #: no instruction.
        self.instructions = instructions
        self.operations = OperationMetrics(
            ("ingest/write",),
            counters=(
                "ingest/accepted_total",
                "ingest/duplicates_total",
                "ingest/rejected_report_total",
                "ingest/rejected_conflict_total",
                "ingest/rejected_request_total",
            ),
        )
        self._name = name
        self._binding = binding
        self.model_config = model_config or ModelConfig()
        self._surface = binding.surface
        self._store = store
        self._closed = False
        #: Every trainer of this scenario with the component it evolves, in recipe order.
        self.trainers = validate_component_trainers(trainers, binding.surface)
        self._artifact_chain = ArtifactReleaseChain(repository)
        self._committer = ScenarioCommitter(
            name=name,
            binding=binding,
            artifacts=self._artifact_chain,
            trainers=self.trainers,
            scenario_step=scenario_step,
            store=store,
            recovered_head_record=recovered_head_record,
            experiment_observer=experiment_observer,
        )
        #: Read-only queries of committed releases and their content. They take only the publication lock.
        self.release_queries = self._committer.release_queries

    @property
    def name(self) -> str:
        return self._name

    @property
    def training_component(self) -> TrainingComponent | None:
        """The training component the scenario's weight trainer submits jobs to, if it trains weights."""
        return self._binding.training_component

    @property
    def runtime(self) -> InferenceRuntime | None:
        """The inference runtime that serves this scenario, if any."""
        return self.model_config.runtime or self._binding.runtime

    @property
    def activation_rule(self) -> ActivationRule:
        """The rule that decides whether an artifact that training makes serves."""
        return self._committer.activation_rule

    def set_activation_rule(self, rule: ActivationRule) -> None:
        """Select the rule for the artifacts that training makes from now on.

        An artifact made before the change keeps the decision it got.
        """
        self._committer.activation_rule = rule

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The recipe's declared report contract, enforced at ingress when set."""
        return self._binding.report_type

    @property
    def repository(self) -> Repository:
        """The release chain's scenario-scoped artifact repository.

        The public read path to the repository base, the active release (``current``) and the chain tip
        (``checkpoint``);
        mutation goes through Scenario methods so it stays serialized with
        activation and commit.
        """
        return self._artifact_chain.repository

    @property
    def records(self) -> RecordStore:
        return self._store.records

    @property
    def store(self) -> ScenarioStore:
        """The scenario's record and commit storage session."""
        return self._store

    @property
    def trainer(self) -> Trainer:
        """Inspection-only view of the trainer that trains the scenario; the only one of a flat scenario.

        The first trainer that trains, else the first: the status, the
        contract and the operational metrics speak of the trainer that
        trains, not of a component that only carries files. Every mutating path goes
        through a Scenario method so it is serialized against activation and
        commit by the committer lock. The one exception is the preparation of
        a local step in a scenario with several trainers; the dispatcher still
        runs one step of the scenario at a time. Reading state that is
        not part of a transaction (objective identity, consumption
        watermarks, processor schema) is safe here; do not reserve job data
        or replace results through this handle.
        """
        return self.stepping_trainers()[0].trainer

    def trainer_for(self, component: str | None) -> Trainer:
        """The trainer evolving ``component``; ``None`` selects the first trainer, for scenario-wide operations."""
        return self._committer.trainer_for(component)

    @property
    def scenario_step(self) -> int:
        return self._committer.step

    @property
    def surface(self) -> Surface:
        """The serving surface built for this scenario."""
        return self._surface

    @property
    def training_mode(self) -> str:
        """The mode the scenario's stepping trainers run in; a trainer that runs no step keeps none."""
        return self.stepping_trainers()[0].trainer.pipeline.training_mode

    def evaluation_refusal(self, component: str) -> str | None:
        """Why this scenario cannot evaluate an artifact of ``component``, a component that its surface declares
        with a kind; None when it can.

        Reef evaluates a model artifact (``RFC #959``): the artifacts hold no harness, and the runtime answers each
        call from the artifact that the call names (``InferenceRuntime.serves_artifacts_on_request``).
        """
        kinds = {name: declared.kind for name, declared in self._surface.components.items()}
        if kinds[component] == "harness_plugin":
            return "Reef does not evaluate a harness artifact yet (RFC #959, step 2)"
        if "harness_plugin" in kinds.values():
            return "Reef evaluates a model artifact only in a scenario whose artifacts hold no harness"
        runtime = self.runtime
        if runtime is None or not runtime.serves_artifacts_on_request:
            return (
                f"scenario {self.name!r}'s runtime serves only its current artifact, so it cannot evaluate an "
                "artifact on request"
            )
        return None

    def capabilities(self) -> ScenarioCapabilities:
        """The artifact operations that this scenario can run (``ScenarioCapabilities``)."""
        surface, runtime = self._surface, self.runtime
        on_request = runtime is not None and runtime.serves_artifacts_on_request
        return ScenarioCapabilities(
            serve_beside=self._store.durable and (surface.loader_component is None or runtime is None or on_request),
            evaluate=tuple(
                name
                for name, component in surface.components.items()
                if component.kind is not None and self.evaluation_refusal(name) is None
            ),
            train_from=tuple(
                component_trainer.component
                for component_trainer in self.trainers
                if component_trainer.trainer.can_train_from
            ),
        )

    def stepping_trainers(self) -> tuple[ComponentTrainer, ...]:
        """The trainers that train; the first trainer when none does."""
        stepping = tuple(component_trainer for component_trainer in self.trainers if component_trainer.trainer.trains)
        return stepping or self.trainers[:1]

    def set_training_mode(self, training_mode: str, *, fill_manual_buffer: bool = False) -> None:
        """Select future job data without waiting for a running backend step.

        ``fill_manual_buffer`` is for a person's switch: a switch from ``auto`` to ``manual`` then starts each
        trainer's manual buffer from what auto would train next (``TrainingPipeline.set_training_mode``).

        Every trainer that runs a step switches or none does: a scenario whose
        components run in different modes would accept instructions for some
        and drop them for others. A trainer without a step builds no job data,
        so it has no mode to switch.
        """
        if training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        stepping = self.stepping_trainers()
        unsupported = [
            f"{component_trainer.component} ({type(component_trainer.trainer.pipeline.sample2job).__name__})"
            for component_trainer in stepping
            if training_mode not in component_trainer.trainer.pipeline.supported_training_modes
        ]
        if unsupported:
            raise NotImplementedError(f"{', '.join(unsupported)} does not implement training_mode={training_mode!r}")
        for component_trainer in stepping:
            component_trainer.trainer.set_training_mode(training_mode, fill_manual_buffer=fill_manual_buffer)

    def produce_samples(self) -> bool:
        """Do the samples service's step for every trainer: make the samples of the new records
        (``Trainer.produce_samples``). Return whether background work of a pipeline still runs. A process that
        steps the scenario itself, such as a recipe's driver, calls it before ``prepare_training_step``."""
        running = False
        for component_trainer in self.trainers:
            running = component_trainer.trainer.produce_samples().background or running
        return running

    def prepare_training_step(self, component: str | None = None) -> TrainStepResult | None:
        """Prepare ``component``'s next commit: a job's submission, or a running job's checkpoint or end.

        The caller runs one step of the scenario at a time, as the
        dispatcher's cycle lock does. Thus no training commit of another
        trainer lands while the step runs, and an activation refuses to run
        while the job data is reserved. A lone trainer also holds the commit
        lock. With several trainers the step runs outside that lock, so a
        trainer without a backend advances its record cursor without waiting
        for a long evaluation.
        """
        trainer = self.trainer_for(component)
        self.refresh_instructions()
        with self._committer.lock if len(self.trainers) == 1 else nullcontext():
            # The staleness check measures data against the weights that serve; a scenario that loads no weights
            # serves one fixed model, so its data never goes stale.
            weights_load = self._binding.surface.loader_component is not None and self.runtime is not None
            serving = self.serving_artifact_id() if weights_load else None
            return trainer.run_once(serving_artifact_id=serving)

    def pending_instructions(self) -> int:
        """The queued instructions that no trainer's commit consumed yet; 0 for a scenario without instructions."""
        if self.instructions is None:
            return 0
        ids = [request.id for request in self.instructions.queued()]
        consumed: set[str] = set()
        for component_trainer in self.trainers:
            consumed |= component_trainer.trainer.consumed_among(ids)
        return len(set(ids) - consumed)

    def refresh_instructions(self) -> None:
        """Give every trainer the queued instructions that no commit consumed (``ScenarioInstructions.give``)."""
        if self.instructions is not None:
            self.instructions.give(component_trainer.trainer for component_trainer in self.trainers)

    def served_component(self) -> str | None:
        """The component whose artifact the runtime serves: the component that the runtime loads, or the only
        component; None when the scenario has several components and the runtime loads none of them."""
        loader = self._binding.surface.loader_component
        if loader is not None:
            return loader
        names = self._committer.component_names()
        return names[0] if len(names) == 1 else None

    def serving_artifact_id(self) -> str:
        """The artifact that answers this scenario's requests now: the active artifact of ``served_component``,
        or the active release for a scenario whose model is no Reef artifact or whose store keeps no commit
        history to name the active artifact."""
        component = self.served_component()
        active = None if component is None or not self.store.durable else self.artifact_status()[component]["active"]
        return active or self.current_artifact_ref().release_id

    def replay_retained(
        self, *, up_to_sequence: int, consumed_ids: frozenset[str], component: str | None = None
    ) -> None:
        """Rebuild processor memory from retained rows behind a recovered watermark."""
        with self._committer.lock:
            self.trainer_for(component).replay_retained(up_to_sequence=up_to_sequence, consumed_ids=consumed_ids)

    def restore_record_progress(self, *, after_sequence: int, offset: int, component: str | None = None) -> None:
        """Resume record consumption at a recovered commit's high-water mark."""
        with self._committer.lock:
            self.trainer_for(component).restore_record_progress(after_sequence=after_sequence, offset=offset)

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """The non-blocking committed step, training outcome, and artifact-head sync status."""
        return self._committer.commit_status

    def last_commit_for(self, component: str | None) -> CommitRecord | None:
        """The newest durable commit made by ``component``'s trainer, read from the commit history without waiting on a commit."""
        return self._committer.last_record_for(component)

    def releases(self) -> tuple[dict[str, Any], ...]:
        """Committed releases, newest first; the scenario step is read under the same lock as the commit history."""
        with self.release_queries.publication_lock:
            return self.release_queries.releases(self._committer.step)

    def artifacts(self) -> tuple[dict[str, Any], ...]:
        """Each component's artifacts, newest first, with each component's active one marked."""
        with self.release_queries.publication_lock:
            return self.release_queries.artifacts(self._committer.step, self._committer.component_names())

    def current_artifact_ref(self) -> ArtifactRef:
        """The active release."""
        return self._artifact_chain.current

    def resume_training_starts(self) -> None:
        """Give each trainer the training start of a ``train_from`` record that its own training commits do not hold yet."""
        with self._committer.lock:
            self._committer.resume_training_starts()

    def activate(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Make ``artifact_id`` serve as ``component``; see :meth:`ScenarioCommitter.activate`."""
        return self._committer.activate(artifact_id, component)

    def serve(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Serve ``artifact_id`` beside the active release; see :meth:`ScenarioCommitter.serve`."""
        return self._committer.serve(artifact_id, component)

    def stop_serving(self, artifact_id: str) -> ServedBesideArtifact:
        """Stop serving ``artifact_id`` beside the active release; see :meth:`ScenarioCommitter.stop_serving`."""
        return self._committer.stop_serving(artifact_id)

    def served_beside(self) -> Mapping[str, ServedBesideArtifact]:
        """The artifacts that the scenario serves beside its active release, by ``artifact_id``; see
        :meth:`ScenarioReleases.served_beside`."""
        with self.release_queries.publication_lock:
            return self.release_queries.served_beside(self._committer.step, self._committer.component_names())

    def train_from(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Start ``component``'s trainer from ``artifact_id``; see :meth:`ScenarioCommitter.train_from`."""
        return self._committer.train_from(artifact_id, component)

    def import_target(self, component: str | None) -> str:
        """The component an import into ``component`` lands in; see :meth:`ScenarioCommitter.import_target`."""
        return self._committer.import_target(component)[0]

    def import_artifact(
        self, files: SourceFiles, component: str | None, *, source: Mapping[str, str], directory: Path
    ) -> ComponentArtifact:
        """Publish ``files`` as a new artifact of ``component``; see :meth:`ScenarioCommitter.import_artifact`."""
        return self._committer.import_artifact(files, component, source=source, directory=directory)

    def artifact_status(self) -> dict[str, dict[str, str | None]]:
        """Each component's active artifact and training start; see :meth:`ScenarioCommitter.artifact_status`."""
        with self.release_queries.publication_lock:
            return self._committer.artifact_status()

    def artifact_release_ref(self, artifact_id: str, component: str | None = None) -> tuple[str, ArtifactRef]:
        """The component that ``artifact_id`` names and the ref of the release that minted it, without materializing
        the release; see :meth:`ScenarioCommitter.artifact_release_ref`."""
        with self.release_queries.publication_lock:
            return self._committer.artifact_release_ref(artifact_id, component)

    def resolve_artifact(self, artifact_id: str, component: str | None = None) -> tuple[ComponentArtifact, Artifact]:
        """The artifact ``artifact_id`` names and a release that binds it; see :meth:`ScenarioCommitter.resolve_artifact`."""
        with self.release_queries.publication_lock:
            return self._committer.resolve_artifact(artifact_id, component)

    def sync_release(
        self, artifact: Artifact, *, component: str, expected_release_id: str, metrics: Mapping[str, Any]
    ) -> ArtifactRef:
        """Make a person's merged ``component`` tree active and the training start; see
        :meth:`ScenarioCommitter.sync_release`."""
        return self._committer.sync_release(
            artifact, component=component, expected_release_id=expected_release_id, metrics=metrics
        )

    def commit(self, result: TrainStepResult, *, component: str | None = None) -> Any:
        """Commit ``component``'s pending result as one atomic version record."""
        return self._committer.commit(result, component=component)

    def close(self) -> None:
        """Tear down what this scenario instance owns: trainers, then its storage session.

        The dispatcher calls this on shutdown and when a durable reload
        replaces the instance — the one guarantee stages with background
        workers rely on (see :meth:`TrainingPipeline.close`). Safe to call more
        than once; the trainers close first so no stage thread can touch
        the record store after it closes.
        """
        with self._committer.lock:
            if self._closed:
                return
            self._closed = True
            try:
                for component_trainer in self.trainers:
                    component_trainer.trainer.close()
            finally:
                self._store.close()


def validate_component_trainers(
    trainers: tuple[ComponentTrainer, ...], surface: Surface
) -> tuple[ComponentTrainer, ...]:
    """Check that the trainers match the surface: each names a component it serves, and a surface serving
    no component has exactly one trainer, bound to ``records``."""
    if not trainers or any(not isinstance(component_trainer, ComponentTrainer) for component_trainer in trainers):
        raise ReefError("a scenario requires at least one ComponentTrainer")
    names = [component_trainer.component for component_trainer in trainers]
    if len(set(names)) != len(names):
        raise ReefError(f"scenario trainers must evolve distinct components, not {names}")
    if not surface.names:
        if names != [RECORDS_COMPONENT]:
            raise ReefError(
                f"a scenario serving no component has one trainer bound to {RECORDS_COMPONENT!r}, not {names}"
            )
        return trainers
    unknown = [name for name in names if name not in surface.names]
    if unknown:
        raise ReefError(f"scenario trainers name components the surface does not serve: {unknown}")
    return trainers


__all__ = [
    "SCENARIO_METADATA_KEY",
    "ReleaseNotRestorable",
    "Scenario",
    "ScenarioCapabilities",
    "validate_component_trainers",
]
