"""The harness evolution recipe: one recipe for every harness evolution method.

``HarnessEvolutionRecipe`` reads the ``evolution`` config section and builds
one ``HarnessBackend`` per scenario, with the evaluator, the selection
policy and the surface around it. A method is a
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
from reef.core.config import Settings, config_option
from reef.core.errors import ReefError
from reef.core.evaluation import CandidateEvaluationPlugin
from reef.core.model_metadata import ModelMetadata
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import TaskSplitError, manifest_task_paths
from reef.evaluation.harness import (
    ComparePolicy,
    EvaluationSettings,
    EvaluationTasks,
    FloorPolicy,
    HarnessRolloutEvaluator,
    History,
    NoEvaluationPolicy,
    Promoter,
    RolloutScorer,
    evaluation_selection,
)
from reef.execution.sandbox.config import build_executor
from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, DescriptorError
from reef.harness.delivery import create_harness_surface
from reef.harness.rollouts.model_binding import ModelBinding, ModelBindings, ModelBindingsResolver
from reef.harness.rollouts.vendor_install import AdapterProgram
from reef.harness.tree import HarnessTree, validate_tree
from reef.inference.http import InferenceProxyRuntime
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe, ServedEndpoint
from reef.recipe.errors import RecipeConfigError
from reef.record2record import ReportRecord2Record
from reef.record2sample.traces import TraceSamples
from reef.runtime.executor.config import ExecutorSettings, role_executor_settings
from reef.sample2job.traces import TraceJobData, TrainTaskRollouts
from reef.storage.records import RecordStore
from reef.storage.samples import TrainerSamples
from reef.storage.tasks import TaskStorage
from reef.surface.base import ArtifactLayout, Surface
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import HarnessBackend, StepLimits
from reef.train.harness_backend.contract import Method
from reef.train.harness_backend.inbox import ProposalInbox
from reef.train.harness_backend.records import StepRecord
from reef.train.pipeline import PipelineSpec
from reef.train.trainer import Trainer

#: The directories of one component's state under ``<evolution.state_dir>/<scenario>/<component>``.
PROPOSALS_DIR = "proposals"
HISTORY_DIR = "history"
STEPS_DIR = "steps"


@dataclass(frozen=True)
class EvolutionSettings(Settings):
    """The parsed ``evolution`` section of a harness evolution deployment, without its ``evaluation`` section.

    Values: ``adapter`` names an adapter ``reef.harness.adapters.get_adapter``
    resolves. ``seed`` is the directory of the first tree, ``None`` for the
    empty tree. ``method`` is a plugin setting that names a ``Method``.
    ``models`` maps a name other than ``served`` to a model section (``url``,
    ``model``, ``api_key_env``); ``model_metadata`` maps a model name to its
    metadata. ``state_dir`` is the directory of every scenario's proposals,
    history and step records. ``batch_size`` ≥ 1 is the traces of one
    automatic step. Every other count is ≥ 0, where 0 turns its limit off.
    The retired ``publish`` and ``review_kinds`` are refused: the top-level
    ``activation`` section decides whether a kept tree serves.
    ``executor`` and ``sandbox`` choose where rollouts run
    (``reef.execution.sandbox.config.build_executor``).
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
    state_dir: str = config_option(".reef/harness", help="Each scenario's proposals, history and step records.")
    batch_size: int = config_option(1, help="The traces of one automatic step.")
    max_steps: int = config_option(0, help="The most automatic steps. 0 sets no limit.")
    max_failure_streak: int = config_option(
        0, help="Skip automatic steps after this many rejections in a row. 0 sets no limit."
    )
    max_model_calls_per_step: int = config_option(0, help="The most model calls of one step. 0 sets no limit.")
    max_rollouts_per_step: int = config_option(0, help="The most rollouts a method runs in one step. 0 sets no limit.")
    max_pending_proposals: int = config_option(8, help="The most client proposals that wait for a step.")
    client_models: tuple[str, ...] = config_option((), help="Models an installed client may switch to.")
    executor: str = config_option("local", help="Where rollouts run: local, sandbox or e2b.")
    sandbox: Mapping[str, Any] | None = config_option(None, help="The sandbox settings of the rollout executor.")

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
            ("max_rollouts_per_step", self.max_rollouts_per_step),
        ):
            if value < 0:
                raise ValueError(f"evolution.{label} must be at least 0 (0 turns the limit off)")
        if self.max_pending_proposals < 1:
            raise ValueError("evolution.max_pending_proposals must be at least 1")
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
class ScenarioModels(ModelBindingsResolver):
    """The model bindings of a scenario that overrides the deployment's model.

    Values: ``config`` is the scenario's model configuration; ``recipe`` is
    the recipe whose bindings it overrides; ``scenario`` is the scenario a
    call that names none resolves for.
    Purpose: the backend resolves its bindings through it at each step, so a
    model change of the scenario reaches the next step.
    """

    config: ModelConfig
    recipe: HarnessEvolutionRecipe
    scenario: str | None = None

    def resolve(self, scenario: str | None = None) -> ModelBindings:
        scenario = self.scenario if scenario is None else scenario
        runtime = self.config.runtime
        if runtime is None:
            return self.recipe.default_model_bindings(scenario)
        # The scenario's own model is served by this Reef too, so a rollout reaches it through the same route.
        binding = self.recipe.bind_model_metadata(ModelBinding.from_runtime(runtime))
        served = self.recipe.served_through_service(binding, scenario)
        return ModelBindings(served=served, named=dict.fromkeys(self.recipe.models, served))


@dataclass(frozen=True)
class HarnessEvolutionRecipe(Recipe):
    """Harness evolution of one adapter's harness tree by one method.

    Values: ``method`` is the deployment's ``Method``. ``evolution`` and
    ``evaluation`` are the parsed sections. ``scorer`` scores each rollout,
    and ``promoter`` picks the prompts that promotion adds: each is the
    plugin that ``evaluation`` names, or the instance that a recipe built in
    Python receives, and ``None`` when there is neither. ``seed`` is the first
    tree of a new scenario. ``models`` and ``model_metadata`` are the parsed
    ``evolution`` entries. ``executor`` runs the rollouts;
    ``worker_executor`` places the rollout workers (``execution.evolution``).
    The model under test is the recipe's inference runtime. It reaches the
    method as ``tools.models.served``, and the adapter's ``model_binding``
    templates write it into each evaluation rollout, so neither the seed nor
    a kept tree ever carries a provider.
    Purpose: a deployment names it as its recipe; the service builds it once
    from the deployment config, and ``build`` makes each scenario's trainer
    over a ``HarnessBackend``.
    """

    method: Method
    evolution: EvolutionSettings
    evaluation: EvaluationSettings
    seed: HarnessTree = field(default_factory=lambda: HarnessTree({}))
    scorer: RolloutScorer | None = None
    promoter: Promoter | None = None
    models: Mapping[str, ModelBinding] = field(default_factory=dict)
    model_metadata: Mapping[str, ModelMetadata] = field(default_factory=dict)
    executor: ProgramExecutor = field(default_factory=lambda: build_executor(None))
    worker_executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    model_name: str | None = None
    #: Where this Reef answers inference: evaluation rollouts sample the release it serves through it.
    served_endpoint: ServedEndpoint | None = None
    config_sections: ClassVar[tuple[str, ...]] = ("evolution",)

    name: str = field(default="harness_evolution", kw_only=True)
    scenario_model: ModelConfig | None = field(default=None, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.method, Method):
            raise ValueError("harness evolution requires a Method")
        if self.scorer is None and self.evaluation.policy != "none":
            raise ValueError("harness evolution requires a scorer unless the evaluation policy is none")
        try:
            descriptor = get_adapter(self.evolution.adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        seed_problems = validate_tree(self.seed, descriptor).problems
        if seed_problems:
            raise RecipeConfigError(f"evolution.seed fails validation: {'; '.join(seed_problems)}")
        directories = self.evaluation.task_manifest is not None or self.evaluation.imported_tasks
        if directories and not descriptor.is_prompt_task_directory:
            raise RecipeConfigError(
                f"evaluation tasks from a manifest or the imported tasks are task directories, and adapter "
                f"{descriptor.name!r} takes a prompt (terminus takes a task directory)"
            )
        if self.evaluation.promote_failures and descriptor.is_prompt_task_directory:
            raise RecipeConfigError(
                "evaluation.promote_failures adds prompts, and this adapter plays task directories"
            )
        if issubclass(self.stages().sample2job.plugin_class, TrainTaskRollouts):
            # An offline step plays the imported train tasks, which are task directories, and scores each rollout.
            if not descriptor.is_prompt_task_directory:
                raise RecipeConfigError(
                    f"offline rollouts play task directories, and adapter {descriptor.name!r} takes a prompt "
                    "(terminus takes a task directory)"
                )
            if self.scorer is None:
                raise RecipeConfigError("offline rollouts score each train task, so they need evaluation.scorer")
        if self.scorer is not None:
            evaluation_selection(self.scorer, self.worker_executor)
        # A task manifest is read now, so a deployment with an unreadable one refuses to start.
        self.configured_tasks()

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
                else RolloutScorer.from_setting(evaluation.scorer, setting="evolution.evaluation.scorer")
            )
            promoter = (
                None
                if evaluation.promote is None
                else Promoter.from_setting(evaluation.promote, setting="evolution.evaluation.promote")
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
            "promoter": promoter,
            "models": models,
            "model_metadata": model_metadata,
            "executor": executor,
            "worker_executor": worker_executor,
            "model_name": model_name if isinstance(model_name, str) and model_name else None,
        }

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> HarnessEvolutionRecipe:
        return replace(self, served_endpoint=endpoint)

    def model_binding(self, scenario: str | None = None) -> ModelBinding:
        """The served model's endpoint.

        With the service known and a scenario named, this Reef's own
        evaluation route for that scenario: a rollout then samples the
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

    def default_model_bindings(self, scenario: str | None = None) -> ModelBindings:
        return ModelBindings(served=self.model_binding(scenario), named=dict(self.models))

    def model_bindings(self, scenario: str | None = None) -> ModelBindings:
        """The scenario's model override, or the recipe's served and named models."""
        if self.scenario_model is not None:
            return ScenarioModels(self.scenario_model, self).resolve(scenario)
        return self.default_model_bindings(scenario)

    def build_surface(self, scenario: str) -> Surface:
        model = self.model_name or (self.runtime.model_path if self.runtime is not None else None)
        # Only a provider proxy has a dialect; a local engine serves Chat Completions.
        api = self.runtime.api if isinstance(self.runtime, InferenceProxyRuntime) else "openai"
        client_models = self.evolution.client_models
        override = self.scenario_model.runtime if self.scenario_model is not None else None
        if override is not None:
            model = override.model_path
            api = override.api
            client_models = ()
        if self.evolution.adapter == "codex" and self.runtime is not None:
            selected = self.bind_model_metadata(ModelBinding.from_runtime(override or self.runtime, model=model))
            served_metadata = selected.metadata
        else:
            served_metadata = None
        return create_harness_surface(
            get_adapter(self.evolution.adapter),
            layout=HarnessLayout(get_adapter(self.evolution.adapter)),
            served_model=model if isinstance(model, str) and model else None,
            served_api=api,
            model_metadata=self.model_metadata,
            served_metadata=served_metadata,
            client_models=client_models,
        )

    def base_artifact_source(self) -> BaseArtifactSource:
        """The seed tree as the base harness, which a new scenario serves and the first step builds on.

        The files are the seed tree's files; the empty seed makes a base
        harness with no files. The input is the seed tree's checksum, so a
        scenario opens only on a base made from the same seed.
        """
        return BaseArtifactSource(files=dict(self.seed.files), inputs={"seed": self.seed.checksum()})

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's proposals, history and step records: what a delete archives beside the record store."""
        return (self.state_root(scenario),)

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
        if releases is None:
            raise RecipeConfigError("harness evolution reads its head tree from the scenario's committed releases")
        backend = self.build_backend(scenario, releases, samples)
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=self.build_pipeline,
            candidate_backend=backend,
            candidate_evaluator=self.build_policy(backend),
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
        self, scenario: str, releases: ComponentReleases, samples: TrainerSamples | None
    ) -> HarnessBackend:
        """The scenario's backend: the binary installed, the history, the evaluator, the tasks and the records."""
        descriptor = get_adapter(self.evolution.adapter)
        # Each component of a composed scenario keeps its own proposals, history and records.
        root = self.state_root(scenario) / releases.component
        try:
            history = History.open(root / HISTORY_DIR)
        except OSError as exc:
            raise RecipeConfigError(f"evolution.state_dir cannot hold {root}: {exc}") from exc
        models = self.model_bindings(scenario)
        evaluator = None if self.scorer is None else self.build_evaluator(descriptor, models, history, self.scorer)
        eval_samples = None
        if self.evaluation.imported_tasks:
            if samples is None:
                raise RecipeConfigError(
                    "evolution.evaluation.imported_tasks needs the sample store of a scenario session"
                )
            eval_samples = samples
        return HarnessBackend(
            descriptor=descriptor,
            releases=releases,
            method=self.method,
            evaluator=evaluator,
            tasks=EvaluationTasks(
                seed_tasks=self.configured_tasks(),
                eval_samples=eval_samples,
                promote_failures=self.evaluation.promote_failures,
                promoter=self.promoter,
                max_promoted=self.evaluation.max_promoted_tasks,
                max_promoted_per_client=self.evaluation.max_promoted_per_client,
            ),
            history=history,
            models=models,
            model_resolver=(
                None if self.scenario_model is None else ScenarioModels(self.scenario_model, self, scenario)
            ),
            limits=StepLimits(
                max_steps=self.evolution.max_steps,
                max_failure_streak=self.evolution.max_failure_streak,
                max_model_calls_per_step=self.evolution.max_model_calls_per_step,
                max_rollouts_per_step=self.evolution.max_rollouts_per_step,
                recheck_every=self.evaluation.recheck_every,
            ),
            records=StepRecord(root / STEPS_DIR),
            inbox=ProposalInbox(root / PROPOSALS_DIR, self.evolution.max_pending_proposals),
        )

    def build_evaluator(
        self, descriptor: AdapterDescriptor, models: ModelBindings, history: History, scorer: RolloutScorer
    ) -> HarnessRolloutEvaluator:
        """The rollout evaluator: the adapter's binary installed, the worker pool placed, ``scorer`` bound."""
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
        return HarnessRolloutEvaluator(
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
            rollout_timeout_s=self.evaluation.rollout_timeout_s,
            rollout_repeats=self.evaluation.repeats,
            executor=executor,
            forbid_residue=self.evaluation.forbid_residue,
            worker_selection=selection,
            worker_requirements=requirements,
            history=history,
        )


__all__ = ["EvolutionSettings", "HarnessEvolutionRecipe", "ScenarioModels", "plugin_name"]
