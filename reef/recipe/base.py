"""Base recipe contracts shared by every concrete recipe.

``Recipe`` is both the default record-only recipe and the base class for every
specialized recipe; ``WeightTrainingRecipe`` narrows the contract for recipes whose
backend updates model weights.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.artifact.components import AcceptAnyArtifact, ComponentSpec, ScenarioComponents
from reef.artifact.repository import DefaultStart
from reef.core.activation import ActivationRule
from reef.core.components import RECORDS_COMPONENT
from reef.core.config import config_arguments
from reef.core.plugins import PluginConfigError, PluginReference
from reef.core.reports import ReportBase
from reef.evaluation.activation import activation_reference, build_activation_rule
from reef.harness.tasks.model_binding import ModelBinding
from reef.inference.http import resolve_proxy_runtime
from reef.inference.interfaces import InferenceRuntime
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.recipe.config_fields import config_field, resolve_config_field_values
from reef.recipe.errors import RecipeConfigError
from reef.record2sample import Record2Sample
from reef.sample2job import NoJobData
from reef.storage.data import DataStore
from reef.storage.records import RecordStore
from reef.storage.samples import SAMPLE_KINDS, SampleKind, SampleStore, TrainerSamples
from reef.storage.tasks import TaskStorage
from reef.train.backend import CommittedArtifacts, ComponentArtifacts
from reef.train.component import ScenarioRoot, TrainingArtifactLayout, TrainingComponent
from reef.train.jobs import JobLoop
from reef.train.pipeline import PipelineConfig, PipelineSpec, TrainingPipeline
from reef.train.trainer import ComponentTrainer, Trainer


@dataclass(frozen=True)
class EveryCheck(ArtifactValidator):
    """Admit an artifact only when every check admits it, in order."""

    checks: tuple[ArtifactValidator, ...]

    def validate(self, artifact: Artifact) -> None:
        for check in self.checks:
            check.validate(artifact)


def every_check(*checks: ArtifactValidator) -> ArtifactValidator:
    """One check that admits an artifact only when each of ``checks`` does; a check that admits anything is left out."""
    kept = tuple(check for check in checks if not isinstance(check, AcceptAnyArtifact))
    if not kept:
        return AcceptAnyArtifact()
    return kept[0] if len(kept) == 1 else EveryCheck(kept)


@dataclass(frozen=True)
class ServedEndpoint:
    """Where this Reef answers inference itself: what a recipe's own evaluation calls target.

    ``url`` is the service's base URL and ``token`` a bearer token it accepts.
    ``component`` names the snapshot component the recipe evolves when it is
    one of several: its evaluation calls then leave that component's served
    hooks out, since the rollout runs a candidate of it.
    """

    url: str
    token: str | None = None
    component: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("served endpoint requires a url")
        object.__setattr__(self, "url", self.url.strip().rstrip("/"))


def check_job_components(recipe: Recipe, components: Mapping[str, TrainingComponent]) -> None:
    """Raise ``ValueError`` unless ``components`` names exactly the job components of ``recipe``."""
    names = sorted(component.name for component in recipe.job_components())
    if sorted(components) != names:
        raise ValueError(f"the job components of {type(recipe).__name__} are {names}, not {sorted(components)}")


@dataclass(frozen=True, kw_only=True)
class Recipe:
    """Default inference recipe and base contract for update recipes.

    Instantiating ``Recipe`` directly records traffic without producing job
    data. Specialized recipes are frozen dataclasses over this base: they
    add their own fields, override ``build`` and, when needed, narrow the
    ``runtime`` field.
    """

    config_sections: ClassVar[tuple[str, ...]] = ()
    """Opaque config sections validated by the recipe alongside its declared fields."""

    name: str = "recipe"
    runtime: InferenceRuntime | None = None
    training_component: TrainingComponent | None = None
    #: The store of the job data directories that the trainers write for their jobs; the dispatcher binds it
    #: (``with_job_components``).
    data_store: DataStore | None = field(default=None, repr=False, compare=False)

    training_mode: str = config_field("auto")
    #: The deployment's ``pipeline`` section over :meth:`pipeline_spec`; None is the recipe's stages with no config.
    pipeline_stages: PipelineConfig | None = field(default=None, repr=False, compare=False)
    #: The deployment's ``activation`` section: the rule that activates new artifacts; None is ``always``.
    activation: PluginReference | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        # The stages' classes fail the deployment here, not the first scenario.
        self.stages()

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        """The recipe's default stage classes. The default passes records through and makes no job data."""
        return PipelineSpec()

    def activation_rule(self, reference: PluginReference | None = None) -> ActivationRule:
        """The rule that decides whether an artifact that the recipe's trainers make serves: the one ``reference``
        names, such as a scenario's own rule, else the ``activation`` section's."""
        try:
            return build_activation_rule(self.activation if reference is None else reference)
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def stages(self) -> PipelineConfig:
        """The pipeline's stage classes and their configs."""
        if self.pipeline_stages is not None:
            return self.pipeline_stages
        try:
            return PipelineConfig.from_setting(None, spec=type(self).pipeline_spec(), environ={})
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def build_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        """Build the training pipeline of one scenario from :meth:`stages`, in pipeline order."""
        try:
            return self.stages().build(context)
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """Directories that belong to one scenario alone, archived when the scenario is deleted.

        The record-to-sample stage names them.
        """
        stage = self.stages().record2sample
        return Record2Sample.referenced_class(stage).scenario_state_dirs(stage.config, scenario)

    def job_components(self) -> tuple[TrainingComponent, ...]:
        """The training components whose jobs this recipe's trainers submit, each under its own ``name``: the
        weight trainer's ``training_component``, and the ones that a recipe adds, such as a harness's."""
        return () if self.training_component is None else (self.training_component,)

    def with_job_components(self, components: Mapping[str, TrainingComponent], data_store: DataStore) -> Recipe:
        """This recipe whose trainers submit the jobs of each job component to ``components[name]``, such as the
        jobs queue's view of it, and write the data of each job to ``data_store``. Raise ``ValueError`` unless
        ``components`` names exactly the job components."""
        check_job_components(self, components)
        if self.training_component is None:
            return replace(self, data_store=data_store)
        return replace(self, training_component=components[self.training_component.name], data_store=data_store)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        runtime: InferenceRuntime | None = None,
        training_component: TrainingComponent | None = None,
    ) -> Recipe:
        values = os.environ if environ is None else environ
        settings = config or {}
        data = settings.get("data", {})
        resolved = resolve_config_field_values(cls, data, values)
        return cls.from_resolved_config(
            settings, resolved, environ=values, runtime=runtime, training_component=training_component
        )

    @classmethod
    def from_resolved_config(
        cls,
        config: Mapping[str, Any],
        field_values: Mapping[str, Any],
        *,
        environ: Mapping[str, str],
        runtime: InferenceRuntime | None = None,
        training_component: TrainingComponent | None = None,
    ) -> Recipe:
        """Construct from values resolved before runtime allocation.

        ``field_values`` comes from ``resolve_config_field_values``. Domain
        validation still runs in the constructor; no environment is re-read.
        """
        try:
            return cls(
                **cls._recipe_kwargs(config, environ),
                runtime=cls._resolve_runtime(environ, runtime),
                training_component=training_component,
                **field_values,
            )
        except ValueError as exc:
            raise RecipeConfigError(f"invalid {cls.__name__} configuration: {exc}") from exc

    @classmethod
    def _resolve_runtime(cls, values: Mapping[str, str], runtime: InferenceRuntime | None) -> InferenceRuntime | None:
        return resolve_proxy_runtime(values, runtime)

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        """Recipe-specific keyword arguments resolved from config sections: the ``pipeline`` section's stages and
        the ``activation`` section's rule."""
        kwargs: dict[str, Any] = {}
        pipeline = settings.get("pipeline")
        activation = settings.get("activation")
        try:
            if pipeline is not None:
                kwargs["pipeline_stages"] = PipelineConfig.from_setting(
                    pipeline, spec=cls.pipeline_spec(), environ=values
                )
            if activation is not None:
                kwargs["activation"] = activation_reference(activation)
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc
        return kwargs

    @classmethod
    def select_weight_training(
        cls, config: Mapping[str, Any]
    ) -> tuple[type[WeightTrainingRecipe], Mapping[str, Any]] | None:
        """The weight training recipe class a deployment of this class trains with, and that recipe's own config.

        Read from ``config`` alone, before anything is constructed: the service
        decides from it whether to connect a training component and launch a
        training backend. ``None`` for a recipe that trains no weights.
        """
        return None

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> Recipe:
        """This recipe told where the service answers inference; the default has no calls of its own to point there."""
        return self

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The typed external-report contract this recipe accepts.

        Recipes that consume reports override this property with their
        concrete :class:`~reef.core.reports.ReportBase` subclass. ``None`` keeps
        ingress open for record-only and computed-feedback recipes. This is a
        runtime capability, not a dataclass field or recipe configuration.
        """
        return None

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        artifacts: ComponentArtifacts | None = None,
        samples: TrainerSamples | None = None,
        root: ScenarioRoot | None = None,
    ) -> Trainer:
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=self.build_pipeline,
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            samples=samples,
        )

    def build_trainers(
        self,
        scenario: str,
        records: RecordStore,
        *,
        components: ScenarioComponents,
        algorithm_states: Mapping[str, Mapping[str, Any] | None],
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        artifacts: CommittedArtifacts | None = None,
        samples: SampleStore | None = None,
        root: ScenarioRoot | None = None,
    ) -> tuple[ComponentTrainer, ...]:
        """Build every trainer of the named scenario, each bound to the component it evolves.

        The default builds the one trainer :meth:`build` returns and binds it
        to the scenario's only component (``records`` when the scenario serves
        none). A recipe that declares several components overrides
        this to return one trainer per component. The scenario runs them as
        separate workers that take turns, one step at a time.
        ``algorithm_states`` carries each one's recovered state under its
        component name. ``task_storage`` opens the generated task stores of
        the scenario session, and ``samples`` is the data items that it trains
        on. Each trainer gives them to its pipeline as its own
        (``SampleStore.for_trainer``, ``""`` for the only trainer).
        ``releases`` reads the scenario's committed snapshots; each trainer
        reads those of its own component. ``root`` is the scenario's root
        snapshot, where a new trainer starts; a recipe of several components
        gives each trainer its own component of it
        (``ScenarioComponents.component_artifact``).
        """
        if len(components.names) > 1:
            raise RecipeConfigError(
                f"{type(self).__name__} serves components {list(components.names)}: override build_trainers "
                "to bind one trainer per component"
            )
        component = components.names[0] if components.names else RECORDS_COMPONENT
        return (
            ComponentTrainer(
                component,
                self.build(
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                    task_storage=task_storage,
                    artifacts=None if artifacts is None else ComponentArtifacts(artifacts, component),
                    samples=None if samples is None else samples.for_trainer(""),
                    root=root,
                ),
            ),
        )

    def build_components(self, scenario: str) -> ScenarioComponents:
        """The components that every snapshot of the named scenario binds.

        Each ``ComponentSpec`` gives the component's kind, the admission check
        run before it is published or restored, its import layout and the
        files a client pulls. The default declares none: the scenario records
        traffic only. Most recipes ignore ``scenario``.
        """
        return ScenarioComponents()

    def build_artifact_validator(self) -> ArtifactValidator:
        """An admission check run beside the component's own before the recipe's snapshot is published or restored.

        Kept for recipes written before admission moved onto the component
        (``ComponentSpec.validator``, where a new recipe binds it):
        it joins the check of the one component the recipe serves, and on a
        recipe that serves none it admits the snapshot as a whole; inside a
        composite that recipe's snapshot is its component, so the check joins
        that component's. A recipe that overrides it while serving several
        components is refused at build, since the check could not say which
        component it admits.
        """
        return AcceptAnyArtifact()

    def scenario_components(self, scenario: str) -> ScenarioComponents:
        """``build_components`` with ``build_artifact_validator`` joined to the served component's check."""
        components = self.build_components(scenario)
        if type(self).build_artifact_validator is Recipe.build_artifact_validator:
            return components
        if not components.components:
            # A snapshot with no component is admitted as a whole, as it was before components existed.
            return replace(components, validator=every_check(components.validator, self.build_artifact_validator()))
        if len(components.components) != 1:
            raise RecipeConfigError(
                f"{type(self).__name__} overrides build_artifact_validator but serves components "
                f"{list(components.names)}: bind the check on each component's ComponentSpec.validator instead"
            )
        ((name, component),) = components.components.items()
        checks = every_check(component.validator, self.build_artifact_validator())
        return replace(components, components={name: replace(component, validator=checks)})

    def check_root_start(self, root: ScenarioRoot) -> None:
        """Check that training can start from ``root`` before an item start is registered.

        The default calls ``training_component.root_start`` when a training component is bound. A recipe with
        several trainers checks each trainer's component and root view. Raise ``ValueError`` on refusal.
        """
        if self.training_component is not None:
            self.training_component.root_start(root)

    def default_start(self) -> DefaultStart:
        """The content of a new scenario's root snapshot when no artifact item starts the scenario; empty by
        default."""
        return DefaultStart()

    def harness_state_dir(self, scenario: str, component: str) -> Path | None:
        """The directory of ``component``'s harness evolution state in ``scenario``: its history, its step records
        and the running step's progress (``reef.train.harness_backend.component``); ``None`` for a recipe that
        evolves no harness."""
        return None

    def harness_served_model(self, scenario: str, component: str) -> ModelBinding | None:
        """The model that ``component``'s harness in ``scenario`` calls: a client's layer asks Reef for this model
        by name and dialect. ``None`` for a recipe that evolves no harness."""
        return None

    @property
    def harness_adapter(self) -> str | None:
        """The adapter this recipe's harness artifacts render for, as ``GET /reef/harness`` serves them.

        ``None`` for a recipe that serves no harness tree. Every scenario of
        the process shares it, so the scenario list names it before a
        scenario loads.
        """
        return None

    def serving_status(self) -> Mapping[str, Any] | None:
        """Runtime-wide serving state this recipe owns, for ``/reef/status``.

        ``None`` when the recipe has nothing beyond per-scenario training
        state to report.
        """
        return None


#: The component of a weight-training recipe's weights.
WEIGHTS_COMPONENT = "weights"


@dataclass(frozen=True, kw_only=True)
class WeightTrainingRecipe(Recipe):
    """Shared recipe contract for backend algorithms that update weights.

    Requires a :class:`~reef.train.component.TrainingComponent` in
    ``training_component`` (the first positional argument of every training
    recipe), which runs the recipe's training jobs.

    A job's ``parameters`` belong to its training component, and Reef reads
    none of them. :meth:`default_parameters` gives the recipe's own default
    parameters by sample kind, normally the method as ``objective``. The class
    method keeps the dataclass fields as the recipe's instance configuration.
    :meth:`configured_parameters` adds the ``job_parameters`` config; the service
    passes it to the backend when it connects the training component, so a backend
    that starts with one method reads it there.

    ``pipeline_stages`` is framework-owned configuration derived from an
    optional top-level ``pipeline`` section: each stage's class, the recipe's
    :meth:`pipeline_spec` unless the section names another, and its config.

    Configuration is declared as fields: a config field (see
    :mod:`reef.recipe.config_fields`) is simultaneously the YAML key, the
    environment fallback, and the type-aware parser. :meth:`service_config`
    and :meth:`from_environment` derive everything from those fields, so a
    recipe never repeats a setting's name or type anywhere else.
    """

    training_component: TrainingComponent = field(kw_only=False)
    runtime: InferenceRuntime = field(kw_only=True)
    job_parameters: Mapping[str, Any] | None = config_field(
        None,
        help=(
            "Job parameters that the recipe sends with every job, over its default parameters; the training "
            "component refuses a parameter it does not take."
        ),
    )

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        """The recipe's own parameters of a training job, by the kind of the job's samples; empty by default.

        Each value belongs to the training component, and Reef passes it unread. A recipe normally names its method
        here, as the ``objective`` that the backends register. A kind can have its own method, and the training
        component switches the objective from one job to the next.
        """
        return {}

    @classmethod
    def configured_parameters(cls, job_parameters: Mapping[str, Any] | None) -> Mapping[SampleKind, Mapping[str, Any]]:
        """The parameters of each training job, by the kind of the job's samples, for the ``job_parameters`` config.

        The default is :meth:`default_parameters` with ``job_parameters`` over it, key by key, for every kind. A kind
        that a result leaves out trains only for a request that gives its parameters. The service reads this hook
        before it constructs the recipe, and passes the result to the backend when it connects the training component.
        """
        defaults = cls.default_parameters()
        return {kind: {**defaults.get(kind, {}), **(job_parameters or {})} for kind in SAMPLE_KINDS}

    def parameters_by_kind(self) -> Mapping[SampleKind, Mapping[str, Any]]:
        """This recipe's ``configured_parameters``.

        The pipeline's sample-to-job stage gives a job the parameters of its samples' kind, with its request's
        parameters merged in (``Sample2Job.job_parameters``). The component's own defaults fill the rest, such as its
        learning rate. The result must be the same at every call: it is part of each job's identity.
        """
        return type(self).configured_parameters(self.job_parameters)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.resolve_training_component(self.training_component)
        if not isinstance(self.runtime, InferenceRuntime):
            raise RecipeConfigError("weight recipes require an independent inference runtime")

    def build_components(self, scenario: str) -> ScenarioComponents:
        weights = ComponentSpec(
            kind=self.runtime.served_weight_kind, layout=TrainingArtifactLayout(self.training_component)
        )
        return ScenarioComponents({WEIGHTS_COMPONENT: weights})

    def default_start(self) -> DefaultStart:
        """The training component's untrained checkpoint (``TrainingComponent.default_start``)."""
        return self.training_component.default_start()

    def serving_status(self) -> Mapping[str, Any] | None:
        """The engine's adapter residency on a runtime that serves per-scenario adapters."""
        status = self.runtime.adapter_residency_status()
        return None if status is None else {"adapters": status}

    @classmethod
    def resolve_training_component(cls, component: TrainingComponent | None) -> TrainingComponent:
        if component is None:
            raise RecipeConfigError(
                f"{cls.__name__} requires a training component; pass one via the 'training_component' argument"
            )
        if not isinstance(component, TrainingComponent):
            raise TypeError(f"{cls.__name__} requires a TrainingComponent, got {type(component).__name__}")
        return component

    @classmethod
    def select_weight_training(
        cls, config: Mapping[str, Any]
    ) -> tuple[type[WeightTrainingRecipe], Mapping[str, Any]] | None:
        return cls, config

    @classmethod
    def service_config(cls, settings: Mapping[str, Any], *, model_path: str) -> dict[str, Any]:
        """Assemble this recipe's ``from_environment`` config from flat service settings.

        ``settings`` is the recipe-owned slice of the deployment's flat
        ``reef`` section — the caller (``reef.service.assembly``) removes the
        service's own keys first. Only keys the operator actually set are
        forwarded, so every default lives with the recipe's own config fields,
        never in the service layer. This step only translates the layout; shared
        field resolution performs type conversion before construction. A key this recipe does not declare
        is a loud error: the operator set a value nothing would consume.
        """
        field_names = {argument.name for argument in config_arguments(cls)}
        data: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in settings.items():
            if key in field_names:
                # A YAML key left empty (None) is unset, not a value to parse.
                if value is not None:
                    data[key] = value
            else:
                unknown.append(key)
        if unknown:
            known = ", ".join(f"reef.{name}" for name in sorted(field_names))
            raise RecipeConfigError(
                f"{cls.__name__} does not consume {', '.join(f'reef.{key}' for key in sorted(unknown))}; "
                f"the recipe would silently run with its defaults instead. "
                f"Settings {cls.__name__} consumes: {known}. "
                f"A new setting is declared with config_field() on the recipe dataclass "
                f"(see reef.recipe.config_fields)."
            )
        return {"data": data, "model": {"path": model_path}}

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        cls._validate_config(settings)
        return super()._recipe_kwargs(settings, values)

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        """Internal hook: reject config sections this recipe refuses to own.

        Config field values never come through here; they resolve from
        their field declarations. This exists only for structural rules such
        as SAO refusing an ``optimization`` section.
        """

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        artifacts: ComponentArtifacts | None = None,
        samples: TrainerSamples | None = None,
        root: ScenarioRoot | None = None,
    ) -> Trainer:
        """Build a trainer from this recipe's pipeline, job parameters, and report contract.

        Override with ``Trainer.build`` for bespoke wiring such as a local
        training backend, and pass :attr:`report_type` and ``task_storage``
        through there too.
        """
        return self._build_trainer(
            scenario,
            records,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            task_storage=task_storage,
            artifacts=artifacts,
            samples=samples,
            root=root,
        )

    def _build_trainer(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        artifacts: ComponentArtifacts | None = None,
        samples: TrainerSamples | None = None,
        root: ScenarioRoot | None = None,
    ) -> Trainer:
        """Build the shared weight trainer with this recipe's report contract."""
        if type(self).pipeline_spec().sample2job is NoJobData:
            raise TypeError(
                f"{type(self).__name__} declares no sample2job: return a PipelineSpec with a Sample2Job subclass "
                f"from pipeline_spec() or override build() for bespoke trainer wiring"
            )
        # A weight trainer's job size has no default: the deployment sets pipeline.sample2job.config.job_samples.
        try:
            return Trainer.build(
                scenario,
                records,
                pipeline_factory=self.build_pipeline,
                job_loop=JobLoop(
                    self.training_component,
                    scenario=scenario,
                    scenario_component=None if artifacts is None else artifacts.component,
                    data_store=self.data_store,
                ),
                root=root,
                job_parameters=self.parameters_by_kind(),
                algorithm_state=algorithm_state,
                report_type=self.report_type,
                experiment_logger=experiment_logger,
                training_mode=self.training_mode,
                task_storage=task_storage,
                samples=samples,
            )
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc


def close_runtimes(
    training_components: Sequence[TrainingComponent], runtime: InferenceRuntime | None
) -> list[BaseException]:
    """Close a recipe's training components and inference runtime, and return their close errors in that order.

    Admission closes first, so that no new request starts. Then the training
    components close, and then the runtime shuts down. A failed close does not
    stop the next one. ``None`` stands for an absent runtime.
    """
    if runtime is not None:
        runtime.close_admission()
    errors: list[BaseException] = []
    for training_component in training_components:
        try:
            training_component.close()
        except BaseException as exc:
            errors.append(exc)
    if runtime is not None:
        try:
            runtime.shutdown()
        except BaseException as exc:
            errors.append(exc)
    return errors
