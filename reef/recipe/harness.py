"""The harness evolution recipe: one recipe for every harness evolution method.

``HarnessEvolutionRecipe`` reads the ``evolution`` config section. Its
trainer submits each step as a job of its ``HarnessTrainingComponent``
(``reef.train.harness_backend.component``), which runs the step in the
training service on one ``HarnessBackend`` per scenario component, with
the evaluator and the selection policy around it. A method is a
``reef.train.harness_backend.Method`` that the deployment names in
``evolution.method``; a method needs no recipe of its own. A recipe subclass
is for a method that changes more than the method, such as the delivery
surface::

    evolution:
      adapter: pi
      seed: seed/
      method: recipes.gepa:GEPA
      models: {reflection: {url: ..., model: ..., api_key_env: OPENAI_API_KEY}}
      evaluation:
        tasks: [...]
        scorer: harness.aime:AIMEAnswerScorer
        policy: compare
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.core.activation import ActivationRule
from reef.core.config import Settings, config_option
from reef.core.errors import ReefError
from reef.core.evaluation import CandidateEvaluationPlugin
from reef.core.job_data import JobData, TaskSample
from reef.core.model_metadata import ModelMetadata
from reef.core.plugins import PluginReference
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import TaskSplitError, manifest_task_paths
from reef.evaluation.activation import AlwaysActivate
from reef.evaluation.harness import (
    ComparePolicy,
    EvaluationSettings,
    EvaluationTasks,
    FloorPolicy,
    HarnessTaskEvaluator,
    History,
    NoEvaluationPolicy,
    TaskCollector,
    TaskScorer,
    evaluation_selection,
)
from reef.execution.sandbox.config import build_executor
from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, DescriptorError
from reef.harness.sync import create_harness_surface
from reef.harness.tasks.harbor_task import can_run_in_task_container
from reef.harness.tasks.model_binding import ModelBinding, ModelBindings
from reef.harness.tasks.vendor_install import AdapterProgram
from reef.harness.tree import HarnessTree, validate_tree
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe, ServedEndpoint, check_job_components
from reef.recipe.errors import RecipeConfigError
from reef.record2record import ReportRecord2Record
from reef.record2sample.traces import TraceSamples
from reef.runtime.executor.config import ExecutorSettings, role_executor_settings
from reef.runtime.jobs import TrainingComponent
from reef.sample2job.traces import TraceJobData, TrainTaskJobData
from reef.storage.records import RecordStore
from reef.storage.samples import TrainerSamples
from reef.storage.tasks import TaskStorage
from reef.surface.base import ArtifactLayout, Surface
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.agent.settings import AgentSettings, agent_settings
from reef.train.harness_backend.backend import HarnessBackend, StepLimits
from reef.train.harness_backend.component import HISTORY_DIR, STEPS_DIR, HarnessSteps, HarnessTrainingComponent
from reef.train.harness_backend.contract import Method
from reef.train.harness_backend.records import StepRecord
from reef.train.jobs import JobLoop, JobLoopState, SubmitParameters
from reef.train.pipeline import PipelineSpec
from reef.train.trainer import Trainer

#: The directory of the harness training component's jobs and starts under ``evolution.state_dir``.
TRAINING_DIR = ".training"


@dataclass(frozen=True)
class EvolutionSettings(Settings):
    """The parsed ``evolution`` section of a harness evolution deployment, without its ``evaluation`` section.

    Values: ``adapter`` names an adapter ``reef.harness.adapters.get_adapter``
    resolves. ``seed`` is the directory of the first tree, ``None`` for the
    empty tree. ``method`` is a plugin setting that names a ``Method``.
    ``models`` maps a name other than ``served`` to a model section (``url``,
    ``model``, ``api_key_env``); ``model_metadata`` maps a model name to its
    metadata. ``state_dir`` is the directory of every scenario's history
    and step records. ``batch_size`` ≥ 1 is the traces of one
    automatic step. Every other count is ≥ 0, where 0 turns its limit off.
    The retired ``publish`` and ``review_kinds`` are refused: the top-level
    ``activation`` section decides whether a kept tree serves.
    ``executor`` and ``sandbox`` choose where tasks run
    (``reef.execution.sandbox.config.build_executor``), and ``agent`` which coding
    agent a method runs and where (``reef.train.harness_backend.agent.settings``),
    ``None`` for none.
    Purpose: ``HarnessEvolutionRecipe`` parses it from the deployment and
    builds the backend from it.
    """

    method: str | Mapping[str, Any] | None = config_option(None, help="The Method that evolves the tree.")
    adapter: str = config_option("pi", help="The harness adapter the tree renders for.")
    seed: str | None = config_option(None, help="The directory of the first tree; none starts empty.")
    binary: str | None = config_option(None, help="The harness binary; none installs the adapter's pinned one.")
    models: Mapping[str, Any] = config_option(
        default_factory=dict, help="The method's own models by name: url, model and api_key_env."
    )
    model_metadata: Mapping[str, Any] = config_option(default_factory=dict, help="Model metadata by model name.")
    state_dir: str = config_option(".reef/harness", help="Each scenario's history and step records.")
    batch_size: int = config_option(1, help="The traces of one automatic step.")
    max_steps: int = config_option(0, help="The most automatic steps. 0 sets no limit.")
    max_failure_streak: int = config_option(
        0, help="Skip automatic steps after this many rejections in a row. 0 sets no limit."
    )
    max_model_calls_per_step: int = config_option(0, help="The most model calls of one step. 0 sets no limit.")
    max_runs_per_step: int = config_option(0, help="The most runs a method runs in one step. 0 sets no limit.")
    executor: str = config_option("local", help="Where tasks run: local, sandbox or e2b.")
    sandbox: Mapping[str, Any] | None = config_option(None, help="The sandbox settings of the run executor.")
    agent: Mapping[str, Any] | None = config_option(
        None, help="The coding agent a method runs: adapter, binary, sandbox and timeout_s."
    )

    def __post_init__(self) -> None:
        if not self.state_dir.strip():
            raise ValueError("evolution.state_dir must be a non-empty path")
        if self.seed is not None and not self.seed.strip():
            raise ValueError("evolution.seed must be the path of a tree directory when it is set")
        if self.batch_size < 1:
            raise ValueError("evolution.batch_size must be at least 1")
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("max_runs_per_step", self.max_runs_per_step),
        ):
            if value < 0:
                raise ValueError(f"evolution.{label} must be at least 0 (0 turns the limit off)")
        if "served" in self.models:
            raise ValueError("evolution.models may not name a model 'served'; that is the model under test")


def plugin_name(setting: str | Mapping[str, Any]) -> str:
    """A stable text form of a plugin setting: the reference, or its JSON with sorted keys."""
    return setting if isinstance(setting, str) else json.dumps(dict(setting), sort_keys=True, default=str)


@dataclass(frozen=True)
class HarnessLayout(ArtifactLayout):
    """The layout of a harness artifact for the adapter ``descriptor``, which an import must meet (RFC #980).

    A directory meets it when its files, read as UTF-8 text, form a
    ``HarnessTree`` that passes ``validate_tree`` for ``descriptor``: the
    files a release of the harness component stores.
    Purpose: ``HarnessEvolutionRecipe.build_surface`` binds it on the harness
    component; the committer checks an imported harness with it.
    """

    descriptor: AdapterDescriptor

    def check(self, files: ArtifactFiles) -> None:
        tree = HarnessTree({relative: files.read_bytes(relative).decode("utf-8") for relative in files.paths()})
        validation = validate_tree(tree, self.descriptor)
        if not validation.passed:
            raise ValueError(f"the harness tree fails validation: {validation.message()}")


@dataclass(frozen=True)
class HarnessJobParameters(SubmitParameters):
    """The parameters of a scenario component's harness job that only the scenario knows when the job is submitted.

    Values: ``releases`` reads the component's head, whose release is the job's start. ``model`` is the scenario's
    model selection, read at each submit, so a model change reaches the next job. ``eval_samples`` is the sample
    store whose imported eval split the job evaluates on, or ``None`` for the configured tasks.
    ``parameters(data, state)`` is ``{component, head_release_id, evaluation_tasks, scenario_model,
    previous_job}``: the component's name, its head release, the eval split's task paths in arrival order or
    ``None``, the scenario's model override without its key (``InferenceProxyRuntime.model_config``) or ``None``,
    and the ID of the last job that the trainer committed the end of (``state.last_job``), whose harness state
    the job continues, or ``None`` before the first.

    Purpose: ``HarnessEvolutionRecipe.build`` gives one to its trainer's ``JobLoop``;
    ``HarnessTrainingComponent`` reads the parameters.
    """

    releases: ComponentReleases
    model: ModelConfig | None
    eval_samples: TrainerSamples | None

    def parameters(self, data: JobData, state: JobLoopState) -> Mapping[str, Any]:
        tasks = (
            None
            if self.eval_samples is None
            else [
                str(stored.sample.task_path)
                for stored in self.eval_samples.imported_tasks(split="eval")
                if isinstance(stored.sample, TaskSample)
            ]
        )
        runtime = None if self.model is None else self.model.runtime
        return {
            "component": self.releases.component,
            "head_release_id": self.releases.head().release_id,
            "evaluation_tasks": tasks,
            "scenario_model": None if runtime is None else runtime.model_config(),
            "previous_job": None if state.last_job is None else state.last_job.job_id,
        }


@dataclass(frozen=True)
class RecipeSteps(HarnessSteps):
    """The ``HarnessSteps`` of ``recipe``: its seed, its state directories, its backend and its selection policy.

    Values: ``recipe`` is the recipe whose component reads it. ``backend`` binds a scenario's model override to the
    deployment's provider key, the key of ``recipe.runtime``.
    Purpose: ``HarnessEvolutionRecipe`` gives one to its ``HarnessTrainingComponent``.
    """

    recipe: HarnessEvolutionRecipe

    @property
    def seed(self) -> HarnessTree:
        return self.recipe.seed

    def state_root(self, scenario: str, component: str) -> Path:
        return self.recipe.state_root(scenario) / component

    def backend(self, scenario: str, component: str, scenario_model: Mapping[str, Any] | None) -> HarnessBackend:
        runtime = self.recipe.runtime
        model = (
            None
            if scenario_model is None
            else ModelConfig.from_value(scenario_model, api_key=None if runtime is None else runtime.provider_key)
        )
        return self.recipe.build_backend(scenario, component, scenario_model=model)

    def policy(self, backend: HarnessBackend) -> CandidateEvaluationPlugin:
        return self.recipe.build_policy(backend)

    def check_tree(self, tree: HarnessTree) -> None:
        validation = validate_tree(tree, get_adapter(self.recipe.evolution.adapter))
        if not validation.passed:
            raise ValueError(f"the harness tree fails validation: {validation.message()}")


@dataclass(frozen=True)
class HarnessEvolutionRecipe(Recipe):
    """Harness evolution of one adapter's harness tree by one method.

    Values: ``method`` is the deployment's ``Method``. ``evolution`` and
    ``evaluation`` are the parsed sections. ``scorer`` scores each run,
    and ``collector`` picks the prompts that collection adds: each is the
    plugin that ``evaluation`` names, or the instance that a recipe built in
    Python receives, and ``None`` when there is neither. ``seed`` is the first
    tree of a new scenario. ``models`` and ``model_metadata`` are the parsed
    ``evolution`` entries. ``executor`` runs the runs;
    ``worker_executor`` places the run workers (``execution.evolution``).
    ``agent`` is ``evolution.agent`` read, ``None`` when no coding agent
    runs; a method that ``runs_agent`` needs one.
    The model under test is the recipe's inference runtime. It reaches the
    method as ``services.models.served``, and the adapter's ``model_binding``
    templates write it into each evaluation run, so neither the seed nor
    a kept tree ever carries a provider.
    ``harness_component`` is the training component that runs the steps: the recipe's own
    ``HarnessTrainingComponent`` over ``RecipeSteps`` of this recipe, under ``<evolution.state_dir>/.training``,
    which each copy of the recipe makes again, or the jobs queue's view of one (``with_job_components``).
    Purpose: a deployment names it as its recipe; the service builds it once
    from the deployment config, and ``build`` makes each scenario's trainer,
    which submits each step as a job of ``harness_component``.
    """

    method: Method
    evolution: EvolutionSettings
    evaluation: EvaluationSettings
    seed: HarnessTree = field(default_factory=lambda: HarnessTree({}))
    scorer: TaskScorer | None = None
    collector: TaskCollector | None = None
    models: Mapping[str, ModelBinding] = field(default_factory=dict)
    model_metadata: Mapping[str, ModelMetadata] = field(default_factory=dict)
    executor: ProgramExecutor = field(default_factory=lambda: build_executor(None))
    worker_executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    agent: AgentSettings | None = None
    model_name: str | None = None
    #: Where this Reef answers inference: evaluation runs sample the release it serves through it.
    served_endpoint: ServedEndpoint | None = None
    config_sections: ClassVar[tuple[str, ...]] = ("evolution",)

    name: str = field(default="harness_evolution", kw_only=True)
    scenario_model: ModelConfig | None = field(default=None, repr=False, kw_only=True)
    harness_component: TrainingComponent | None = field(default=None, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.method, Method):
            raise ValueError("harness evolution requires a Method")
        if self.method.runs_agent and self.agent is None:
            raise RecipeConfigError(
                f"{type(self.method).__name__} runs a coding agent: set evolution.agent, with a sandbox this host "
                "can run (sandbox: none runs the agent without isolation)"
            )
        if self.scorer is None and self.evaluation.policy != "none":
            raise ValueError("harness evolution requires a scorer unless the evaluation policy is none")
        try:
            descriptor = get_adapter(self.evolution.adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        # The activation section's review_kinds name parts of this tree; a name the tree lacks is refused here.
        self.activation_rule()
        seed_problems = validate_tree(self.seed, descriptor).problems
        if seed_problems:
            raise RecipeConfigError(f"evolution.seed fails validation: {'; '.join(seed_problems)}")
        offline = self.runs_offline()
        task_directories = self.task_directories(descriptor)
        if task_directories and not descriptor.takes_task_directory:
            # An adapter whose command takes text plays each Harbor task in the task's own container.
            if not can_run_in_task_container(descriptor):
                raise RecipeConfigError(
                    f"adapter {descriptor.name!r} cannot play Harbor task directories: Reef installs the agent in "
                    "each task's container with npm, and the adapter declares no npm install"
                )
            if self.evolution.executor != "local":
                raise RecipeConfigError(
                    "Harbor task directories run in their own containers, which are the runs' boundary: "
                    "set evolution.executor to local"
                )
            if offline and self.evaluation.tasks:
                raise RecipeConfigError(
                    "offline runs play task directories, so the evaluation tasks must be task directories "
                    "too: use evaluation.imported_tasks or evaluation.task_manifest instead of evaluation.tasks"
                )
        if self.evaluation.collect_failures and task_directories:
            raise RecipeConfigError("evaluation.collect_failures adds prompts, and these tasks are task directories")
        if offline and self.scorer is None:
            # An offline step plays the imported train tasks and scores each run.
            raise RecipeConfigError("offline runs score each train task, so they need evaluation.scorer")
        if self.scorer is not None:
            evaluation_selection(self.scorer, self.worker_executor)
        # A task manifest is read now, so a deployment with an unreadable one refuses to start.
        self.configured_tasks()
        # The recipe's own component reads this recipe, so each copy of the recipe makes its own; a component that
        # the recipe was given, the jobs queue's view of one, stays.
        if self.harness_component is None or isinstance(self.harness_component, HarnessTrainingComponent):
            root = Path(self.evolution.state_dir).expanduser().resolve() / TRAINING_DIR
            object.__setattr__(self, "harness_component", HarnessTrainingComponent(RecipeSteps(self), root))

    def job_components(self) -> tuple[TrainingComponent, ...]:
        """The harness component, whose jobs are this recipe's steps."""
        return () if self.harness_component is None else (self.harness_component,)

    def with_job_components(self, components: Mapping[str, TrainingComponent]) -> HarnessEvolutionRecipe:
        check_job_components(self, components)
        return replace(self, harness_component=next(iter(components.values())))

    def runs_offline(self) -> bool:
        """Whether the steps roll out the imported train tasks (``TrainTaskJobData``)."""
        return issubclass(self.stages().sample2job.plugin_class, TrainTaskJobData)

    def task_directories(self, descriptor: AdapterDescriptor) -> bool:
        """Whether the runs play Harbor task directories rather than prompts: always for an adapter whose
        command takes a task directory, else when the evaluation reads a task manifest or the imported tasks, or the
        steps roll out the imported train tasks."""
        return (
            descriptor.takes_task_directory
            or self.evaluation.task_manifest is not None
            or self.evaluation.imported_tasks
            or self.runs_offline()
        )

    def configured_tasks(self) -> tuple[str, ...]:
        """The configured evaluation tasks: ``evaluation.tasks``, or the eval tasks the task manifest names."""
        evaluation = self.evaluation
        if evaluation.task_manifest is None or evaluation.tasks_root is None:
            return evaluation.tasks
        try:
            paths = manifest_task_paths(
                Path(evaluation.task_manifest).expanduser(), Path(evaluation.tasks_root).expanduser(), "eval"
            )
        except TaskSplitError as exc:
            raise RecipeConfigError(str(exc)) from exc
        if not paths:
            raise RecipeConfigError(
                f"evolution.evaluation.task_manifest {evaluation.task_manifest} names no eval tasks"
            )
        return tuple(str(path) for path in paths)

    def with_model_config(self, config: ModelConfig) -> HarnessEvolutionRecipe:
        super().with_model_config(config)
        return replace(self, scenario_model=config)

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=ReportRecord2Record, record2sample=TraceSamples, sample2job=TraceJobData)

    @property
    def harness_adapter(self) -> str | None:
        return self.evolution.adapter

    @property
    def batch_size(self) -> int:
        return self.evolution.batch_size

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        section = settings.get("evolution")
        if not isinstance(section, Mapping):
            raise RecipeConfigError("harness evolution requires an 'evolution' config section")
        try:
            for retired in ("publish", "review_kinds"):
                if retired in section:
                    raise ValueError(
                        f"evolution.{retired} is gone: the activation section decides whether a kept tree serves "
                        "(activation: {module: always, config: {review_kinds: [...]}}, or module: never)"
                    )
            evaluation_section = section.get("evaluation", {})
            if isinstance(evaluation_section, Mapping) and "task_store" in evaluation_section:
                raise ValueError(
                    "evolution.evaluation.task_store is gone: imported tasks are in the scenario's sample store; "
                    "set evolution.evaluation.imported_tasks: true"
                )
            evolution = EvolutionSettings.from_config(
                {key: value for key, value in section.items() if key != "evaluation"}, setting="evolution"
            )
            evaluation = EvaluationSettings.from_config(evaluation_section, setting="evolution.evaluation")
            if evolution.method is None:
                raise ValueError("evolution.method must name the Method that evolves the tree")
            method = Method.from_setting(evolution.method, setting="evolution.method")
            scorer = (
                None
                if evaluation.scorer is None
                else TaskScorer.from_setting(evaluation.scorer, setting="evolution.evaluation.scorer")
            )
            collector = (
                None
                if evaluation.collector is None
                else TaskCollector.from_setting(evaluation.collector, setting="evolution.evaluation.collector")
            )
            models = {
                name: ModelBinding.from_config(section, values, where=f"evolution.models.{name}")
                for name, section in evolution.models.items()
            }
            model_metadata = {
                name: ModelMetadata.from_config(section) for name, section in evolution.model_metadata.items()
            }
            executor = build_executor(section, environ=values)
            worker_executor = role_executor_settings(settings, "evolution")
            agent = agent_settings(evolution.agent, values, state_dir=Path(evolution.state_dir).expanduser())
        except (TypeError, ValueError, ReefError) as exc:
            raise RecipeConfigError(str(exc)) from exc
        try:
            seed = HarnessTree({}) if evolution.seed is None else HarnessTree.read(Path(evolution.seed).expanduser())
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.seed: {exc}") from exc
        model = settings.get("model")
        model_name = model.get("path") if isinstance(model, Mapping) else None
        return {
            **super()._recipe_kwargs(settings, values),
            "method": method,
            "evolution": evolution,
            "evaluation": evaluation,
            "seed": seed,
            "scorer": scorer,
            "collector": collector,
            "models": models,
            "model_metadata": model_metadata,
            "executor": executor,
            "worker_executor": worker_executor,
            "agent": agent,
            "model_name": model_name if isinstance(model_name, str) and model_name else None,
        }

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> HarnessEvolutionRecipe:
        return replace(self, served_endpoint=endpoint)

    def model_binding(self, scenario: str | None = None) -> ModelBinding:
        """The served model's endpoint.

        With the service known and a scenario named, this Reef's own
        evaluation route for that scenario: a run then samples the
        release the scenario serves, weights, request defaults and all, and
        its calls are kept by nobody. Otherwise the runtime's own endpoint.
        """
        if self.runtime is None:
            raise RecipeConfigError(
                "harness evolution requires an inference runtime: set reef.upstream_url (and reef.upstream_model) "
                "in the deployment config"
            )
        try:
            binding = ModelBinding.from_runtime(self.runtime, model=self.model_name)
        except ValueError as exc:
            raise RecipeConfigError(str(exc)) from exc
        return self.served_through_service(self.bind_model_metadata(binding), scenario)

    def bind_model_metadata(self, binding: ModelBinding) -> ModelBinding:
        """Only Codex needs discovery; keep other adapters' startup independent of /models."""
        if self.evolution.adapter == "codex":
            return binding.with_metadata(self.model_metadata.get(binding.model))
        return binding

    def served_through_service(self, binding: ModelBinding, scenario: str | None) -> ModelBinding:
        """``binding`` routed through this Reef's evaluation route for ``scenario``, when the service is known.

        A component of a composed release names itself in the route, so the
        release's hooks for the component a candidate replaces stay out.
        """
        if self.served_endpoint is None or scenario is None:
            return binding
        # The name is free form: quoted, so a slash or a space stays one segment.
        route = f"{self.served_endpoint.url}/reef/scenarios/{quote(scenario, safe='')}"
        component = self.served_endpoint.component
        if component is not None:
            route += f"/components/{quote(component, safe='')}"
        return replace(binding, base_url=f"{route}/evaluation", api_key=self.served_endpoint.token)

    def model_bindings(self, scenario: str | None = None, model: ModelConfig | None = None) -> ModelBindings:
        """The model bindings of ``scenario``'s runs on ``model``, the scenario's model selection, else on the
        recipe's ``scenario_model``.

        A model with a runtime is served by this Reef too, so a run reaches it through the same route, and every
        named model is that model. Without a model, or for a model with no runtime, the bindings are the recipe's
        served and named models.
        """
        model = self.scenario_model if model is None else model
        if model is None or model.runtime is None:
            return ModelBindings(served=self.model_binding(scenario), named=dict(self.models))
        binding = self.bind_model_metadata(ModelBinding.from_runtime(model.runtime))
        served = self.served_through_service(binding, scenario)
        return ModelBindings(served=served, named=dict.fromkeys(self.models, served))

    def activation_rule(self, reference: PluginReference | None = None) -> ActivationRule:
        """The rule ``Recipe.activation_rule`` names, with the adapter's parts checked.

        Under ``always``, every name in ``review_kinds`` must be a part of the adapter's tree, and a change to a
        part with ``review: true`` also waits for a person. ``RecipeConfigError`` for a name the tree does not have.
        """
        rule = super().activation_rule(reference)
        if not isinstance(rule, AlwaysActivate):
            return rule
        layout = get_adapter(self.evolution.adapter).layout
        unknown = sorted(rule.review_kinds - set(layout.parts))
        if unknown:
            raise RecipeConfigError(
                f"activation.config.review_kinds names {', '.join(unknown)}, not a part of the "
                f"{self.evolution.adapter} tree: {', '.join(layout.parts)}"
            )
        reviewed = frozenset(name for name, part in layout.parts.items() if part.review)
        return AlwaysActivate(rule.review_kinds | reviewed)

    def build_surface(self, scenario: str) -> Surface:
        descriptor = get_adapter(self.evolution.adapter)
        return create_harness_surface(descriptor, layout=HarnessLayout(descriptor))

    def base_artifact_source(self) -> BaseArtifactSource:
        """The seed tree as the base harness, which a new scenario serves and the first step builds on.

        The files are the seed tree's files; the empty seed makes a base
        harness with no files. The input is the seed tree's checksum, so a
        scenario opens only on a base made from the same seed.
        """
        return BaseArtifactSource(files=dict(self.seed.files), inputs={"seed": self.seed.checksum()})

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's history and step records: what a delete archives beside the record store."""
        return (self.state_root(scenario),)

    def harness_served_model(self, scenario: str, component: str, model: ModelConfig) -> ModelBinding | None:
        return self.model_bindings(scenario, model).served

    def harness_state_dir(self, scenario: str, component: str) -> Path | None:
        return self.state_root(scenario) / component

    def state_root(self, scenario: str) -> Path:
        """The scenario's directory under ``evolution.state_dir``, made absolute."""
        return Path(self.evolution.state_dir).expanduser().resolve() / scenario

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        releases: ComponentReleases | None = None,
        samples: TrainerSamples | None = None,
    ) -> Trainer:
        """The scenario's trainer: it submits each step as a job of ``harness_component``, whose start is the
        component's head release in ``releases``."""
        if releases is None:
            raise RecipeConfigError("harness evolution reads its head tree from the scenario's committed releases")
        if self.evaluation.imported_tasks and samples is None:
            raise RecipeConfigError("evolution.evaluation.imported_tasks needs the sample store of a scenario session")
        if self.harness_component is None:
            raise RecipeConfigError("harness evolution has no training component to run its steps")
        submitted = HarnessJobParameters(
            releases, self.scenario_model, samples if self.evaluation.imported_tasks else None
        )
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=self.build_pipeline,
            job_loop=JobLoop(self.harness_component, scenario=scenario, submitted=submitted),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            samples=samples,
            default_job_samples=self.batch_size,
        )

    def build_policy(self, backend: HarnessBackend) -> CandidateEvaluationPlugin:
        """The publication policy that ``evolution.evaluation.policy`` names, measuring through ``backend``."""
        evaluation = self.evaluation
        if evaluation.policy == "compare":
            return ComparePolicy(backend, statistic=evaluation.statistic, min_margin=evaluation.min_margin)
        if evaluation.policy == "floor":
            return FloorPolicy(backend, floor_score=evaluation.floor_score)
        return NoEvaluationPolicy()

    def build_backend(
        self,
        scenario: str,
        component: str,
        *,
        scenario_model: ModelConfig | None = None,
    ) -> HarnessBackend:
        """The backend of ``component`` in ``scenario``: the binary installed, the history, the evaluator, the
        tasks, the records and the coding agent bound. ``scenario_model`` is the scenario's model selection, else
        the recipe's own (``scenario_model``) applies.
        """
        descriptor = get_adapter(self.evolution.adapter)
        # Each component of a composed scenario keeps its own history and records.
        root = self.state_root(scenario) / component
        try:
            history = History.open(root / HISTORY_DIR)
        except OSError as exc:
            raise RecipeConfigError(f"evolution.state_dir cannot hold {root}: {exc}") from exc
        models = self.model_bindings(scenario, scenario_model)
        evaluator = None if self.scorer is None else self.build_evaluator(descriptor, models, history, self.scorer)
        records = StepRecord(root / STEPS_DIR)
        return HarnessBackend(
            descriptor=descriptor,
            method=self.method,
            # Only a method that runs the agent through its services gets it bound here.
            agent=self.agent.bind() if self.agent is not None and self.method.runs_agent else None,
            evaluator=evaluator,
            tasks=EvaluationTasks(
                seed_tasks=self.configured_tasks(),
                collect_failures=self.evaluation.collect_failures,
                collector=self.collector,
                max_collected=self.evaluation.max_collected_tasks,
                max_collected_per_client=self.evaluation.max_collected_per_client,
            ),
            history=history,
            models=models,
            limits=StepLimits(
                max_steps=self.evolution.max_steps,
                max_failure_streak=self.evolution.max_failure_streak,
                max_model_calls_per_step=self.evolution.max_model_calls_per_step,
                max_runs_per_step=self.evolution.max_runs_per_step,
            ),
            records=records,
            task_directories=self.task_directories(descriptor),
        )

    def build_evaluator(
        self, descriptor: AdapterDescriptor, models: ModelBindings, history: History, scorer: TaskScorer
    ) -> HarnessTaskEvaluator:
        """The run evaluator: the adapter's binary installed, the worker pool placed, ``scorer`` bound."""
        executor, binary = self.executor.for_program(AdapterProgram(descriptor), binary=self.evolution.binary)
        executor.preflight()
        selection, requirements = evaluation_selection(scorer, self.worker_executor)
        # The evaluation context names the scorer by its setting. A scorer built in Python has no setting: its
        # class and the instance name it, so no other scorer reuses its scores, and a restart reuses none of them.
        setting = self.evaluation.scorer
        if setting is None:
            scorer_name = f"{type(scorer).__module__}:{type(scorer).__qualname__}@{id(scorer):x}"
        else:
            scorer_name = plugin_name(setting)
        return HarnessTaskEvaluator(
            descriptor=descriptor,
            scorer=scorer,
            scorer_name=scorer_name,
            models=models,
            binary=binary,
            binary_setting=self.evolution.binary,
            executor_setting=json.dumps(
                {
                    "executor": self.evolution.executor,
                    "sandbox": None if self.evolution.sandbox is None else dict(self.evolution.sandbox),
                },
                sort_keys=True,
                default=str,
            ),
            task_timeout_s=self.evaluation.task_timeout_s,
            task_repeats=self.evaluation.repeats,
            executor=executor,
            forbid_residue=self.evaluation.forbid_residue,
            worker_selection=selection,
            worker_requirements=requirements,
            history=history,
            task_directories=self.task_directories(descriptor),
            agent_host=self.evaluation.agent_host,
        )


__all__ = ["EvolutionSettings", "HarnessEvolutionRecipe", "plugin_name"]
