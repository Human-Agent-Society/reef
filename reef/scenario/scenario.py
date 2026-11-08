"""Scenario aggregate and its durable recovery metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact, SnapshotRef
from reef.artifact.components import ScenarioComponents
from reef.artifact.repository import Repository
from reef.artifact.snapshot_chain import ArtifactSnapshotChain, SnapshotNotRestorable
from reef.artifact.sources import SourceFiles
from reef.core.activation import ActivationRule
from reef.core.artifact_selection import ArtifactSelection
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord
from reef.core.reports import ReportBase
from reef.core.served_models import ScenarioModel
from reef.core.serving_config import ServingConfig
from reef.core.snapshot import ItemIdentity, RootSource
from reef.core.values import JsonValue
from reef.inference.interfaces import InferenceRuntime
from reef.observability.commits import CommitExperimentObserver
from reef.observability.operations import OperationMetrics
from reef.record2sample.service import ProducedSamples, Selection
from reef.scenario.binding import ScenarioBinding
from reef.scenario.committer import ComponentArtifact, ScenarioCommitter
from reef.scenario.snapshots import ServedBesideArtifact
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord
from reef.storage.records import RecordStore
from reef.storage.scenario import ScenarioStore
from reef.train.component import TrainingComponent
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.training_requests import ScenarioTrainingRequests
from reef.train.types import TrainingCommit

#: The keys of a ``Scenario.harness_snapshot_rows`` row that hold private storage locators.
SNAPSHOT_ROW_KEYS = frozenset({"snapshot", "artifact_snapshot", "parent_snapshot"})


class Scenario:
    """Scenario aggregate owning one runtime binding, one snapshot chain, and one trainer per component."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        repository: Repository,
        root_source: RootSource,
        store: ScenarioStore,
        trainers: tuple[ComponentTrainer, ...],
        scenario_step: int = 0,
        recovered_head_record: CommitRecord | None = None,
        experiment_observer: CommitExperimentObserver | None = None,
        training_requests: ScenarioTrainingRequests | None = None,
    ) -> None:
        #: The scenario's training requests; None for a scenario built without a training request queue, which takes
        #: no request.
        self.training_requests = training_requests
        self.operations = OperationMetrics(
            ("ingest/write", "serve/request"),
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
        self._components = binding.components
        self._serving_config = ServingConfig()
        self._store = store
        self._closed = False
        #: The newest step that each trainer has read (``Trainer.read_commits``), by component; the step that this
        #: instance loaded before that. A new pipeline reads the last job end from its own state.
        self.loaded_step = scenario_step
        self.job_ends_read_step: dict[str, int] = {}
        #: Every trainer of this scenario with the component it evolves, in recipe order.
        self.trainers = validate_component_trainers(trainers, binding.components)
        self._artifact_chain = ArtifactSnapshotChain(repository)
        self._committer = ScenarioCommitter(
            name=name,
            binding=binding,
            artifacts=self._artifact_chain,
            trainers=self.trainers,
            root_source=root_source,
            scenario_step=scenario_step,
            store=store,
            recovered_head_record=recovered_head_record,
            experiment_observer=experiment_observer,
        )
        #: Read-only queries of committed snapshots and their content. They take only the publication lock.
        self.snapshot_queries = self._committer.snapshot_queries

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
        return self._binding.runtime

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
    def serving_config(self) -> ServingConfig:
        """The serving fields that this scenario sets over the deployment's values (RFC #1097)."""
        return self._serving_config

    def effective_serving_config(self, config: ServingConfig) -> dict[str, dict[str, JsonValue]]:
        """The serving fields of each component that has some, with their effective values under ``config``.

        Only the component that the scenario's inference runtime serves (``ScenarioComponents.loaded_component``) has serving
        fields; ``InferenceRuntime.serving_config`` gives them. ``ValueError`` when ``config`` sets a field of
        another component, or a field or value that the runtime refuses.
        """
        component = self._components.loaded_component
        runtime = self._binding.runtime
        effective: dict[str, dict[str, JsonValue]] = {}
        if component is not None and runtime is not None and runtime.serves_requests:
            effective[component] = runtime.serving_config(config.component(component))
        unknown = sorted(set(config.overrides) - set(effective))
        if unknown:
            raise ValueError(f"no serving fields for components {unknown}: the scenario serves {sorted(effective)}")
        return effective

    def set_serving_config(self, config: ServingConfig) -> None:
        """Serve with ``config`` from the next serving request on; ``ValueError`` as ``effective_serving_config``
        gives, and then the scenario keeps its config."""
        self.effective_serving_config(config)
        self._serving_config = config

    def served_serving_config(self) -> dict[str, JsonValue]:
        """The override that a serving request carries to inference: the one of the served component."""
        component = self._components.loaded_component
        return {} if component is None else self._serving_config.component(component)

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The recipe's declared report contract, enforced at ingress when set."""
        return self._binding.report_type

    @property
    def repository(self) -> Repository:
        """The snapshot chain's scenario-scoped artifact repository.

        The public read path to the root snapshot (``start_artifact``), the active selection (``current``) and the
        chain tip (``checkpoint``); mutation goes through Scenario methods so it stays serialized with activation
        and commit.
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
    def components(self) -> ScenarioComponents:
        """The components that every snapshot of this scenario binds (``Recipe.build_components``)."""
        return self._components

    @property
    def training_mode(self) -> str:
        """The mode the scenario's stepping trainers run in; a trainer that runs no step keeps none."""
        return self.stepping_trainers()[0].trainer.pipeline.training_mode

    def evaluation_refusal(self, component: str) -> str | None:
        """Why this scenario cannot evaluate an artifact of ``component``, a component that the scenario declares
        with a kind; None when it can.

        Reef evaluates a model artifact (``RFC #959``): the artifacts hold no harness, and the runtime answers each
        call from the artifact that the call names (``InferenceRuntime.serves_any_artifact``).
        """
        kinds = {name: declared.kind for name, declared in self._components.components.items()}
        if kinds[component] == "harness_plugin":
            return "Reef does not evaluate a harness artifact yet (RFC #959, step 2)"
        if "harness_plugin" in kinds.values():
            return "Reef evaluates a model artifact only in a scenario whose artifacts hold no harness"
        runtime = self.runtime
        if runtime is None or not runtime.serves_any_artifact:
            return (
                f"scenario {self.name!r}'s runtime serves only its current artifact, so it cannot evaluate an "
                "artifact on request"
            )
        return None

    def stepping_trainers(self) -> tuple[ComponentTrainer, ...]:
        """The trainers that train; the first trainer when none does."""
        stepping = tuple(component_trainer for component_trainer in self.trainers if component_trainer.trainer.trains)
        return stepping or self.trainers[:1]

    def set_training_mode(self, training_mode: str) -> None:
        """Select future job data without waiting for a running backend step.

        Every trainer that runs a step switches or none does: a scenario whose
        components run in different modes would accept requests for some
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
            component_trainer.trainer.set_training_mode(training_mode)

    def produce_samples(self, component: str | None = None, selection: Selection | None = None) -> ProducedSamples:
        """Do the samples service's step for ``component``'s trainer: read its newer commits, then make the samples of
        the new records (``Trainer.produce_samples``).

        The commit history gives the job ends and the records that the commits named, since control can run in another
        process. The caller runs one step of the trainer at a time.
        """
        component_trainer = self._committer.component_trainer(component)
        name = self._committer.component_on_record(component_trainer.component)
        records = self._store.history() if self._store.durable else ()
        read_step = self.job_ends_read_step.get(component_trainer.component, self.loaded_step)
        new = [record for record in records if record.step > read_step and self._committer.own_record(record, name)]
        component_trainer.trainer.read_commits(new)
        if new:
            self.job_ends_read_step[component_trainer.component] = new[-1].step
        return component_trainer.trainer.produce_samples(selection)

    def next_commit(self, component: str | None = None) -> TrainingCommit | None:
        """Prepare ``component``'s next commit: a job's submission, or a running job's checkpoint or end.

        The caller runs one step of the scenario at a time, as the dispatcher's cycle lock does. Thus no training
        commit of another trainer lands while the step runs, and an activation refuses to run while the job data is
        reserved. A running job's step reads only the job's state, so it takes no commit lock and reads no pipeline.
        Another step of a lone trainer holds the commit lock. With several trainers it runs outside that lock.
        """
        trainer = self.trainer_for(component)
        # The staleness check measures data against the weights that serve; a scenario that loads no weights
        # serves one fixed model, so its data never goes stale.
        runtime = self.runtime
        weights_load = self._binding.components.loaded_component is not None and runtime is not None
        serving = self.served_artifact_id() if weights_load else None
        # Automatic training waits until the selected artifact serves, so that it does not train the samples of the
        # artifact before. A request names its samples (``TrainingPipeline.request_entries``), so manual training
        # does not wait, also when the selected artifact does not start.
        if (
            trainer.pipeline.training_mode == "auto"
            and weights_load
            and runtime is not None
            and runtime.serves_requests
            and self._binding.served_models is not None
            and serving != self.serving_artifact_id()
        ):
            return None
        if trainer.job_running:
            return trainer.advance_job()
        if len(self.trainers) > 1:
            return self.prepare_job(component, serving)
        with self._committer.lock:
            return self.prepare_job(component, serving)

    def prepare_job(self, component: str | None, serving: str | None) -> TrainingCommit | None:
        """Give ``component``'s trainer the requests and the served artifacts of the commit history, and take its next
        step: a job's submission when its data is ready."""
        component_trainer = self._committer.component_trainer(component)
        trainer = component_trainer.trainer
        self.refresh_requests()
        served_component = self.served_component()
        if serving is not None and served_component is not None:
            history = self.snapshot_queries.served_history(
                self._committer.step, served_component, self._committer.component_names()
            )
            if history:
                trainer.restore_served_artifacts(history, self.served_beside())
        # Only a job's submission needs the start, and it reads the commit history.
        train_from_artifact_id = (
            self.artifact_status()[component_trainer.component]["training_start"] if trainer.job_due() else None
        )
        return trainer.advance_job(serving_artifact_id=serving, train_from_artifact_id=train_from_artifact_id)

    def fail_pending_request(self, error: str, component: str | None = None) -> bool:
        """Store that ``component``'s reserved request failed with ``error``, so that its next job data is a skip row,
        also after a reload or a restart; return whether a request was reserved."""
        request_id = self.trainer_for(component).fail_pending_request(error)
        if request_id is None:
            return False
        if self.training_requests is not None:
            self.training_requests.mark_failed(request_id, error)
        return True

    def pending_requests(self) -> int:
        """The queued requests that no trainer's commit consumed yet; 0 for a scenario without requests."""
        if self.training_requests is None:
            return 0
        ids = [request.id for request in self.training_requests.queued()]
        consumed: set[str] = set()
        for component_trainer in self.trainers:
            consumed |= component_trainer.trainer.consumed_among(ids)
        return len(set(ids) - consumed)

    def refresh_requests(self) -> None:
        """Give every trainer the queued requests that no commit consumed (``ScenarioTrainingRequests.give``)."""
        if self.training_requests is not None:
            self.training_requests.give(component_trainer.trainer for component_trainer in self.trainers)

    def served_component(self) -> str | None:
        """The loaded component, the pulled file component, or the sole component; None when several could serve."""
        loaded = self._binding.components.loaded_component
        if loaded is not None:
            return loaded
        files = self._binding.components.files_component
        if files is not None:
            return files
        names = self._committer.component_names()
        return names[0] if len(names) == 1 else None

    def serving_artifact_id(self) -> str:
        """The selected artifact of the served component; several components cannot name one artifact."""
        component = self.served_component()
        if component is None:
            raise ReefError(f"scenario {self.name!r} has no unique component for inference")
        active = self.artifact_status()[component]["active"] if self.store.durable else None
        if active is not None:
            return active
        return self.repository.resolve(self.current_snapshot()).component(component).artifact_id

    def served_artifact_id(self) -> str | None:
        """The artifact that serves requests, or None before inference publishes a route.

        Training uses this identity while a newly selected artifact still starts.
        """
        routes = self._binding.served_models
        if routes is None:
            return None
        served = routes.read(ScenarioModel(self.name))
        return None if served is None else served.artifact_id

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """The non-blocking committed step, training outcome, and artifact-head sync status."""
        return self._committer.commit_status

    def last_commit_for(self, component: str | None) -> CommitRecord | None:
        """The newest durable commit made by ``component``'s trainer, read from the commit history without waiting on a commit."""
        return self._committer.last_record_for(component)

    def artifact_rows(self) -> tuple[dict[str, Any], ...]:
        """Internal artifact catalog, including the storage locators needed to materialize each artifact."""
        with self.snapshot_queries.publication_lock:
            return self.snapshot_queries.artifacts(self._committer.step, self._committer.component_names())

    def artifacts(self) -> tuple[dict[str, Any], ...]:
        """Public artifact catalog, with opaque artifact identities and component-specific parent links."""
        with self.snapshot_queries.publication_lock:
            parents = self.snapshot_queries.artifact_parents(self._committer.step, self._committer.component_names())
            return tuple(
                {
                    **{key: value for key, value in row.items() if key != "snapshot"},
                    "parent_artifact_id": parents.get(row["artifact_id"]),
                }
                for row in self.artifact_rows()
            )

    def harness_rows(self) -> list[dict[str, Any]]:
        """Public harness history, newest first: ``harness_snapshot_rows`` without ``SNAPSHOT_ROW_KEYS``."""
        return [
            {key: value for key, value in row.items() if key not in SNAPSHOT_ROW_KEYS}
            for row in self.harness_snapshot_rows()
        ]

    def harness_snapshot_rows(self) -> list[dict[str, Any]]:
        """Internal harness history, newest first: publication and training rows with private storage locators.

        Each row identifies its harness artifact and its component parent. Other
        components' publications and job-submission events do not add harness rows.
        The locators (``SNAPSHOT_ROW_KEYS``) serve the legacy HTTP catalog only, as ``artifact_rows`` does.
        """
        with self.snapshot_queries.publication_lock:
            component = self.components.files_component
            harness = [row for row in self.artifact_rows() if self.components.single or row["component"] == component]
            by_snapshot = {row["snapshot"].revision: row for row in harness}
            by_artifact = {row["artifact_id"]: row for row in harness}
            manifests = self.snapshot_queries.manifests(self._committer.step)

            def artifact_of(revision: str | None) -> dict[str, Any] | None:
                if revision is None or revision in by_snapshot:
                    return None if revision is None else by_snapshot[revision]
                return by_artifact.get((manifests.get(revision) or {}).get(str(component)))

            parents = {row["artifact_id"]: row["parent_artifact_id"] for row in self.artifacts()}
            kept: list[dict[str, Any]] = []
            for row in self.snapshot_queries.snapshot_rows(self._committer.step):
                metrics = row.get("metrics") or {}
                job = metrics.get("training_job")
                if isinstance(job, Mapping) and (
                    job.get("event") == "submitted" or not set(metrics) - {"training_job", "training_request"}
                ):
                    continue
                if row["checkpoint"] and row["snapshot"].revision not in by_snapshot:
                    continue
                if not row["checkpoint"] and row.get("component") not in (None, component):
                    continue
                artifact = artifact_of(row["snapshot"].revision)
                if artifact is None:
                    continue
                parent = artifact_of(artifact["snapshot"].parent_revision)
                kept.append(
                    {
                        **row,
                        "artifact_id": artifact["artifact_id"],
                        "parent_artifact_id": parents[artifact["artifact_id"]],
                        "component": artifact["component"],
                        "active": row["checkpoint"] and artifact["active"],
                        "activated": artifact["activated"],
                        "artifact_snapshot": artifact["snapshot"],
                        "parent_snapshot": None if parent is None else parent["snapshot"],
                    }
                )
            return kept

    def current_snapshot(self) -> SnapshotRef:
        """The private storage or local-view locator of the current component selection."""
        return self._artifact_chain.current

    def artifact_selection(self, ref: SnapshotRef | None = None) -> ArtifactSelection:
        """Return the complete component-to-artifact-ID mapping for independent serving workers."""
        with self.snapshot_queries.publication_lock:
            return self.repository.selection(
                self.current_snapshot() if ref is None else ref, self._committer.component_names()
            )

    def resume_training_starts(self) -> None:
        """Give each trainer the training start of a ``train_from`` record that its own training commits do not hold yet."""
        with self._committer.lock:
            self._committer.resume_training_starts()

    def activate(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Make ``artifact_id`` serve as ``component``; see :meth:`ScenarioCommitter.activate`."""
        return self._committer.activate(artifact_id, component)

    def restore(self, step: int) -> Mapping[str, str]:
        """Restore the full component selection after ``step`` without changing training starts."""
        return self._committer.restore(step)

    def serve(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Serve ``artifact_id`` beside the active selection; see :meth:`ScenarioCommitter.serve`."""
        return self._committer.serve(artifact_id, component)

    def stop_serving(self, artifact_id: str) -> ServedBesideArtifact:
        """Stop serving ``artifact_id`` beside the active selection; see :meth:`ScenarioCommitter.stop_serving`."""
        return self._committer.stop_serving(artifact_id)

    def served_beside(self) -> Mapping[str, ServedBesideArtifact]:
        """The artifacts that the scenario serves beside its active selection, by ``artifact_id``; see
        :meth:`ScenarioSnapshots.served_beside`."""
        with self.snapshot_queries.publication_lock:
            return self.snapshot_queries.served_beside(self._committer.step, self._committer.component_names())

    def train_from(self, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Start ``component``'s trainer from ``artifact_id``; see :meth:`ScenarioCommitter.train_from`."""
        return self._committer.train_from(artifact_id, component)

    def import_target(self, component: str | None) -> str:
        """The component an import into ``component`` lands in; see :meth:`ScenarioCommitter.import_target`."""
        return self._committer.import_target(component)[0]

    def import_artifact(
        self,
        files: SourceFiles,
        component: str | None,
        *,
        source: Mapping[str, str],
        directory: Path,
        item: ItemIdentity | None = None,
    ) -> ComponentArtifact:
        """Publish ``files`` as an artifact of ``component``; see :meth:`ScenarioCommitter.import_artifact`."""
        return self._committer.import_artifact(files, component, source=source, directory=directory, item=item)

    def used_record(self, record_id: str) -> AgentRecord | None:
        """The record item of another scenario that this scenario uses, read as its record, or None."""
        return self.stepping_trainers()[0].trainer.used_record(record_id)

    def artifact_status(self) -> dict[str, dict[str, str | None]]:
        """Each component's active artifact and training start; see :meth:`ScenarioCommitter.artifact_status`."""
        with self.snapshot_queries.publication_lock:
            return self._committer.artifact_status()

    def artifact_snapshot(self, artifact_id: str, component: str | None = None) -> tuple[str, SnapshotRef]:
        """The component that ``artifact_id`` names and the ref of the snapshot that minted it, without materializing
        the snapshot; see :meth:`ScenarioCommitter.artifact_snapshot`."""
        with self.snapshot_queries.publication_lock:
            return self._committer.artifact_snapshot(artifact_id, component)

    def resolve_artifact(self, artifact_id: str, component: str | None = None) -> tuple[ComponentArtifact, Artifact]:
        """The artifact ``artifact_id`` names and a snapshot that binds it; see :meth:`ScenarioCommitter.resolve_artifact`."""
        with self.snapshot_queries.publication_lock:
            return self._committer.resolve_artifact(artifact_id, component)

    def sync_artifact(
        self, artifact: Artifact, *, component: str, expected_selection: ArtifactSelection, metrics: Mapping[str, Any]
    ) -> ComponentArtifact:
        """Make a person's merged ``component`` tree active and the training start; see
        :meth:`ScenarioCommitter.sync_artifact`."""
        return self._committer.sync_artifact(
            artifact, component=component, expected_selection=expected_selection, metrics=metrics
        )

    def commit(self, result: TrainingCommit, *, component: str | None = None) -> Any:
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
    trainers: tuple[ComponentTrainer, ...], components: ScenarioComponents
) -> tuple[ComponentTrainer, ...]:
    """Check that the trainers match ``components``: each names a declared component, and a scenario that declares
    no component has exactly one trainer, bound to ``records``."""
    if not trainers or any(not isinstance(component_trainer, ComponentTrainer) for component_trainer in trainers):
        raise ReefError("a scenario requires at least one ComponentTrainer")
    names = [component_trainer.component for component_trainer in trainers]
    if len(set(names)) != len(names):
        raise ReefError(f"scenario trainers must evolve distinct components, not {names}")
    if not components.names:
        if names != [RECORDS_COMPONENT]:
            raise ReefError(
                f"a scenario serving no component has one trainer bound to {RECORDS_COMPONENT!r}, not {names}"
            )
        return trainers
    unknown = [name for name in names if name not in components.names]
    if unknown:
        raise ReefError(f"scenario trainers name components the scenario does not serve: {unknown}")
    return trainers


__all__ = [
    "SCENARIO_METADATA_KEY",
    "Scenario",
    "SnapshotNotRestorable",
    "validate_component_trainers",
]
