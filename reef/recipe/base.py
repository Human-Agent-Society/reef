"""Base recipe contracts shared by every concrete recipe.

``Recipe`` is both the default record-only recipe and the base class for every
specialized recipe; ``WeightTrainingRecipe`` narrows the contract for recipes whose
backend updates model weights.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.core.components import RECORDS_COMPONENT
from reef.core.config import config_arguments
from reef.core.plugins import PluginConfigError
from reef.core.reports import ReportBase
from reef.evaluation import CandidateEvaluationConfig, CandidateEvaluationConfigError, build_candidate_evaluation
from reef.evaluation.evaluators import AlwaysSelectCheckpointPlugin
from reef.inference.http import resolve_proxy_runtime
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.recipe.config_fields import config_field, resolve_config_field_values
from reef.recipe.errors import RecipeConfigError
from reef.record2sample import Record2Sample
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, MultimodalRelay, TrainingMethod
from reef.runtime.jobs import TrainingComponent
from reef.sample2job import NoJobData
from reef.storage.records import RecordStore
from reef.storage.tasks import TaskStorage
from reef.surface.base import AcceptAnyArtifact, Surface
from reef.surface.weights import create_weight_surface
from reef.train.algos.registry import resolve_objective
from reef.train.jobs import ArtifactEvaluation, FixedJobPlan, JobLoop, JobPlan, JobRequest
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
    ``component`` names the release component the recipe evolves when it is
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

    training_mode: str = config_field("auto")
    #: The deployment's ``pipeline`` section over :meth:`pipeline_spec`; None is the recipe's stages with no config.
    pipeline_stages: PipelineConfig | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        # The stages' classes fail the deployment here, not the first scenario.
        self.stages()

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        """The recipe's default stage classes. The default passes records through and makes no job data."""
        return PipelineSpec()

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

    def with_model_config(self, config: ModelConfig) -> Recipe:
        """Bind model settings supplied for this scenario."""
        if config.runtime is not None and self.training_component is not None:
            raise RecipeConfigError("model overrides require an inference-only runtime")
        return self

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
        """Recipe-specific keyword arguments resolved from config sections: the ``pipeline`` section's stages."""
        pipeline = settings.get("pipeline")
        if pipeline is None:
            return {}
        try:
            return {"pipeline_stages": PipelineConfig.from_setting(pipeline, spec=cls.pipeline_spec(), environ=values)}
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

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
        )

    def build_trainers(
        self,
        scenario: str,
        records: RecordStore,
        *,
        surface: Surface,
        algorithm_states: Mapping[str, Mapping[str, Any] | None],
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
    ) -> tuple[ComponentTrainer, ...]:
        """Build every trainer of the named scenario, each bound to the component it evolves.

        The default builds the one trainer :meth:`build` returns and binds it
        to the surface's only component (``records`` when the surface serves
        none). A recipe whose surface declares several components overrides
        this to return one trainer per component. The scenario runs them as
        separate workers that take turns, one step at a time.
        ``algorithm_states`` carries each one's recovered state under its
        component name. ``task_storage`` opens the generated task stores of
        the scenario session. Each trainer gives it to its pipeline.
        """
        if len(surface.names) > 1:
            raise RecipeConfigError(
                f"{type(self).__name__} serves components {list(surface.names)}: override build_trainers "
                "to bind one trainer per component"
            )
        component = surface.names[0] if surface.names else RECORDS_COMPONENT
        return (
            ComponentTrainer(
                component,
                self.build(
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                    task_storage=task_storage,
                ),
            ),
        )

    @property
    def inference_handler(self) -> InferenceHandler | None:
        """The inference backend composed by this recipe's runtime, if any.

        Use ``runtime`` for the runtime itself.
        """
        runtime = self.runtime
        return runtime.inference_handler if runtime is not None else None

    @property
    def multimodal_relay(self) -> MultimodalRelay | None:
        """Where this recipe's scenarios send multimodal calls, or ``None`` when it offers none (the default)."""
        return None

    def build_surface(self, scenario: str) -> Surface:
        """Build the serving surface for the named scenario.

        The surface names the release's components and binds each one's
        capabilities, including the admission check run before that component
        is published or restored. Most recipes ignore ``scenario``; recipes
        whose serving state is scenario-specific (an adapter on a shared
        engine) route by it.
        """
        return Surface()

    def build_artifact_validator(self) -> ArtifactValidator:
        """An admission check run beside the component's own before the recipe's release is published or restored.

        Kept for recipes written before admission moved onto the component
        surface (``ComponentSurface.validator``, where a new recipe binds it):
        it joins the check of the one component the recipe serves, and on a
        recipe that serves none it admits the release as a whole; inside a
        composite that recipe's release is its component, so the check joins
        that component's. A recipe that overrides it while serving several
        components is refused at build, since the check could not say which
        component it admits.
        """
        return AcceptAnyArtifact()

    def serving_surface(self, scenario: str) -> Surface:
        """``build_surface`` with ``build_artifact_validator`` joined to the served component's check."""
        surface = self.build_surface(scenario)
        if type(self).build_artifact_validator is Recipe.build_artifact_validator:
            return surface
        if not surface.components:
            # A release with no component is admitted as a whole, as it was before components existed.
            return replace(surface, validator=every_check(surface.validator, self.build_artifact_validator()))
        if len(surface.components) != 1:
            raise RecipeConfigError(
                f"{type(self).__name__} overrides build_artifact_validator but serves components "
                f"{list(surface.names)}: bind the check on each component's ComponentSurface.validator instead"
            )
        ((name, component),) = surface.components.items()
        checks = every_check(component.validator, self.build_artifact_validator())
        return replace(surface, components={name: replace(component, validator=checks)})

    def base_artifact_files(self) -> Mapping[str, str] | None:
        """The files a fresh scenario's base artifact starts with, or ``None`` for a recipe with no tree."""
        return None

    @property
    def harness_adapter(self) -> str | None:
        """The adapter a client installs this recipe's harness tree with (``reef-<adapter>``).

        ``None`` for a recipe that serves no harness tree. Every scenario of
        the process shares it, so the scenario list names it before a
        scenario loads.
        """
        return None

    def bootstrap_artifact_component(self) -> str | None:
        """The release component a bootstrap model snapshot belongs to; ``None`` places it at the release root."""
        return None

    def serving_status(self) -> Mapping[str, Any] | None:
        """Runtime-wide serving state this recipe owns, for ``/reef/status``.

        ``None`` when the recipe has nothing beyond per-scenario training
        state to report.
        """
        return None


@dataclass(frozen=True)
class WeightTrainingSpec:
    """Static machinery a weight-training recipe binds to its configuration.

    Keeping the binding behind one explicit hook leaves the recipe dataclass
    to describe instance configuration only. The service can still inspect a
    recipe class without constructing it.

    ``objective`` is the recipe's objective: a registered name or a dotted
    ``package.module:Objective`` reference. Every job trains with it unless
    the recipe overrides :meth:`WeightTrainingRecipe.job_plan`. How a job's
    data is cut into optimizer steps belongs to the training component.
    """

    objective: str

    @property
    def loss_family(self) -> str:
        """The selected objective owns backend loss selection."""
        return resolve_objective(self.objective).loss_family


@dataclass(frozen=True, kw_only=True)
class WeightTrainingRecipe(Recipe):
    """Shared recipe contract for backend algorithms that update weights.

    Requires a :class:`~reef.runtime.jobs.TrainingComponent` in
    ``training_component`` (the first positional argument of every training
    recipe), which runs the recipe's training jobs.

    :meth:`training_spec` binds the training objective, which declares its
    backend loss family. Keeping that static machinery in one structured
    return value means the dataclass fields remain the recipe's instance
    configuration. The service reads the selected class from the same
    deployment config and obtains its loss family from this hook; deployments
    do not repeat that binding in an environment variable.

    ``candidate_evaluation`` is framework-owned configuration derived from an
    optional top-level ``evaluation`` section. The plugin class it names builds one
    candidate evaluator per scenario with ``from_config``, replacing the backend's default
    metrics-only evaluator and selection policy without changing the recipe's
    method-specific config fields.

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
    candidate_evaluation: CandidateEvaluationConfig | None = field(default=None, repr=False, compare=False)
    job_parameters: Mapping[str, Any] | None = config_field(
        None,
        help=(
            "Job parameters that the default job plan sends with every job, beside method; the training "
            "component refuses a parameter it does not take."
        ),
    )

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        """Return the objective binding for this recipe.

        Concrete weight recipes override this hook to name their objective.
        """
        return WeightTrainingSpec(objective="")

    def job_plan(self) -> JobPlan:
        """How this recipe chooses each training job.

        The default submits one job each time job data is ready to the recipe's
        training component, with ``training_spec()``'s objective (the
        parameter ``method``) and the ``job_parameters`` config, such as a
        ``scheduling`` or a ``save_interval``. The component's own defaults
        fill the rest, such as its learning rate. A recipe that switches methods
        within one run (a supervised phase before policy training, a
        distillation phase after it) returns its own plan, built from its
        config fields; the training component switches the objective from one
        job to the next.
        """
        spec = type(self).training_spec()
        parameters = dict(self.job_parameters or {})
        parameters["method"] = TrainingMethod(spec.objective).to_dict()
        return FixedJobPlan(JobRequest(parameters))

    def artifact_evaluation(self, task_storage: TaskStorage | None) -> ArtifactEvaluation | None:
        """The evaluation job of every artifact that the recipe's trainer publishes, or ``None`` for none.

        ``task_storage`` opens the scenario's task stores, or is ``None`` for a trainer without a scenario
        session. The default evaluates no artifact.
        """
        return None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.job_parameters is not None and "method" in self.job_parameters:
            raise RecipeConfigError("job_parameters cannot set method: the recipe's objective is the job's method")
        self.resolve_training_component(self.training_component)
        if not isinstance(self.runtime, InferenceRuntime):
            raise RecipeConfigError("weight recipes require an independent inference runtime")

    def build_surface(self, scenario: str) -> Surface:
        if self.runtime.per_scenario_adapters:
            # Every scenario owns an adapter on the shared base: route by scenario.
            return create_weight_surface(scenario=scenario, runtime=self.runtime)
        return create_weight_surface()

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
        kwargs = super()._recipe_kwargs(settings, values)
        evaluation = settings.get("evaluation")
        if evaluation is not None:
            if not isinstance(evaluation, Mapping):
                raise RecipeConfigError("evaluation must be an object")
            try:
                kwargs["candidate_evaluation"] = CandidateEvaluationConfig.from_dict(evaluation, environ=values)
            except CandidateEvaluationConfigError as exc:
                raise RecipeConfigError(str(exc)) from exc
        return kwargs

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
    ) -> Trainer:
        """Build a trainer from this recipe's pipeline, objective, and report contract.

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
        )

    def _build_trainer(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
    ) -> Trainer:
        """Build the shared weight trainer with this recipe's report contract."""
        spec = type(self).training_spec()
        if type(self).pipeline_spec().sample2job is NoJobData:
            raise TypeError(
                f"{type(self).__name__} declares no sample2job: return a PipelineSpec with a Sample2Job subclass "
                f"from pipeline_spec() or override build() for bespoke trainer wiring"
            )
        if not spec.objective:
            raise TypeError(
                f"{type(self).__name__} declares no objective: return a registered objective name "
                f"(see reef.train.algos) or a dotted 'module:Objective' path from training_spec(), "
                f"or override build()"
            )
        # Resolve the objective before starting workers, so an unknown reference fails the build.
        resolve_objective(spec.objective)
        evaluator = (
            AlwaysSelectCheckpointPlugin()
            if self.candidate_evaluation is None
            else build_candidate_evaluation(self.candidate_evaluation, runtime=self.runtime, scenario=scenario)
        )
        # A weight trainer's job size has no default: the deployment sets pipeline.sample2job.config.job_samples.
        try:
            return Trainer.build(
                scenario,
                records,
                pipeline_factory=self.build_pipeline,
                job_loop=JobLoop(
                    self.training_component,
                    self.job_plan(),
                    evaluator,
                    scenario=scenario,
                    artifact_evaluation=self.artifact_evaluation(task_storage),
                ),
                algorithm_state=algorithm_state,
                report_type=self.report_type,
                experiment_logger=experiment_logger,
                training_mode=self.training_mode,
                task_storage=task_storage,
            )
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc
