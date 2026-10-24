"""The harness evolution recipe: the one recipe contract every harness evolution method extends.

``HarnessEvolutionRecipe`` reads the ``evolution`` config section and builds
one ``HarnessBackend`` per scenario, with the evaluator, the selection plugin
and the surface around it. A method is a subclass that changes defaults, never
the backend. Every method defines the same parts, in the same files of its
package::

    recipe.py       the HarnessEvolutionRecipe subclass: its name, its config
                    fields, and evolution_spec()
    proposer.py     its Proposer
    scorer.py       its EpisodeScorer, when the method scores episodes its own way
    selection.py    its SelectionPlugin, when it selects its own way
    processor.py    its record stages, when it reads records its own way

``evolution_spec()`` returns the method's default proposer, scorer and
selection, the counterpart of ``WeightTrainingRecipe.training_spec()``; a
deployment's ``evolution.propose``, ``evolution.evaluate`` and
``evolution.selection`` replace each one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

from reef.core.errors import ReefError
from reef.core.model_metadata import ModelMetadata
from reef.core.plugins import PluginConfigError, PluginReference
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import TaskSplitError, manifest_task_paths
from reef.evaluation.evaluators import BackendAlwaysSelectPlugin, SelectionPlugin
from reef.evaluation.harness import (
    EpisodeScorer,
    FloorPlugin,
    HarnessEpisodeEvaluator,
    ScoreComparisonPlugin,
    evaluation_selection,
)
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings, ModelBindingsResolver
from reef.harness.episodes.vendor_install import AdapterProgram
from reef.harness.tree import HarnessTree, TreeKind, validate_tree
from reef.inference.http import InferenceProxyRuntime
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe, ServedEndpoint
from reef.recipe.errors import RecipeConfigError
from reef.record2record import ReportRecord2Record
from reef.record2task.traces import TraceTasks
from reef.runtime.executor.config import ExecutorSettings, role_executor_settings
from reef.sandbox.config import build_executor
from reef.sandbox.e2b import E2BExecutor, deployment_owner
from reef.sandbox.process import EpisodeExecutor, LocalExecutor, SandboxExecutor, SandboxUnavailable
from reef.storage.records import RecordStore
from reef.storage.tasks import TaskStorage
from reef.surface.base import Surface
from reef.surface.harnesses import create_harness_surface
from reef.task2batch.traces import TraceBatches
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import AgentExecution, HarnessBackend, StepLimits
from reef.train.harness_backend.contract import Promoter, Proposer
from reef.train.harness_backend.inbox import ProposalInbox
from reef.train.harness_backend.records import StepRecord
from reef.train.harness_backend.tasks import EvaluationTasks
from reef.train.pipeline import PipelineSpec
from reef.train.trainer import Trainer

#: The built-in ``evolution.selection`` names.
SELECTION_PLUGINS: dict[str, type[SelectionPlugin]] = {
    "score_comparison": ScoreComparisonPlugin,
    "floor": FloorPlugin,
    "always": BackendAlwaysSelectPlugin,
}


def proposer_agent_settings(
    section: Any, environ: Mapping[str, str], *, proposals_dir: Path = Path(".reef/proposals")
) -> tuple[EpisodeExecutor | None, float, float]:
    """``evolution.proposer_agent`` as the agent's executor and its two timeouts; no section runs no agent.

    ``sandbox: bwrap`` jails the agent and gives it the internet but no host port
    but the gateway's, and refuses to start where bwrap or pasta is missing;
    ``sandbox: e2b`` runs it in an E2B cloud sandbox (``e2b_api_key``, else
    ``E2B_API_KEY``; ``e2b_template``, else the harness's pinned binary, built on
    first use) that reaches the gateway through a tunnel and nothing else of the
    host, and carries the deployment that ``proposals_dir`` names, so a later
    start stops the sandboxes a stopped run left; ``sandbox: none`` runs it
    unisolated with the service's privileges, and must be chosen. Left empty
    (and ``REEF_PROPOSER_SANDBOX`` unset), the agent is jailed where the host
    can, and off (the text proposer answers requests) where it cannot.
    """
    if section is None:
        return None, 1800.0, 300.0
    if not isinstance(section, Mapping):
        raise RecipeConfigError("evolution.proposer_agent must be a mapping")
    timeouts = []
    for key, default in (("timeout_s", 1800.0), ("trial_timeout_s", 300.0)):
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RecipeConfigError(f"evolution.proposer_agent.{key} must be a positive number")
        timeouts.append(float(value))
    sandbox = str(section.get("sandbox") or environ.get("REEF_PROPOSER_SANDBOX") or "").strip()
    if sandbox == "none":
        logging.getLogger(__name__).warning(
            "evolution.proposer_agent.sandbox is none: the agent proposer runs with the service's own privileges and "
            "full network access, fed text from clients; use it only where you trust every client"
        )
        return LocalExecutor(), timeouts[0], timeouts[1]
    if sandbox == "e2b":
        try:
            remote = E2BExecutor.from_config(
                {key: section[key] for key in ("e2b_api_key", "e2b_api_key_env", "e2b_template") if key in section},
                environ,
            )
            remote = replace(remote, timeout_s=timeouts[0], owner=deployment_owner(proposals_dir))
            remote.preflight()
        except SandboxUnavailable as exc:
            raise RecipeConfigError(f"evolution.proposer_agent.sandbox is e2b, but {exc}") from exc
        return remote, timeouts[0], timeouts[1]
    if sandbox not in ("", "bwrap"):
        raise RecipeConfigError("evolution.proposer_agent.sandbox must be 'bwrap', 'e2b' or 'none'")
    executor = SandboxExecutor(network="isolated")
    try:
        executor.preflight()
    except SandboxUnavailable as exc:
        if sandbox == "bwrap":
            raise RecipeConfigError(f"evolution.proposer_agent.sandbox is bwrap, but {exc}") from exc
        logging.getLogger(__name__).warning(
            "the agent proposer is off: %s. Requests are answered by the text proposer; set "
            "evolution.proposer_agent.sandbox: none to run the agent without isolation",
            exc,
        )
        return None, timeouts[0], timeouts[1]
    return executor, timeouts[0], timeouts[1]


@dataclass(frozen=True)
class ScenarioModels(ModelBindingsResolver):
    config: ModelConfig
    recipe: HarnessEvolutionRecipe
    #: The scenario the bindings serve; a call naming none resolves for it.
    scenario: str | None = None

    def resolve(self, scenario: str | None = None) -> ModelBindings:
        scenario = self.scenario if scenario is None else scenario
        runtime = self.config.runtime
        if runtime is None:
            return self.recipe.default_model_bindings(scenario)
        # The scenario's own model is served by this Reef too, so an episode reaches it through the same route.
        binding = self.recipe.bind_model_metadata(ModelBinding.from_runtime(runtime))
        served = self.recipe.served_through_service(binding, scenario)
        return ModelBindings(served=served, named=dict.fromkeys(self.recipe.models, served))


def selection_reference(value: object) -> PluginReference:
    """``evolution.selection``: a built-in name, or a :class:`SelectionPlugin` class reference."""
    if isinstance(value, str) and value in SELECTION_PLUGINS:
        return PluginReference(SELECTION_PLUGINS[value])
    try:
        return SelectionPlugin.reference(value, setting="evolution.selection")
    except PluginConfigError as exc:
        raise RecipeConfigError(f"{exc}; built-in selections: {', '.join(SELECTION_PLUGINS)}") from exc


def require_task_directories(evolution: Mapping[str, Any], setting: str) -> None:
    """Refuse ``evolution.<setting>``, whose tasks are directories, with an adapter that takes a prompt.

    ``promote_failures`` is refused too: it adds prompts, which an evaluation of task directories cannot run.
    """
    adapter_name = str(evolution.get("adapter", "pi"))
    try:
        descriptor = get_adapter(adapter_name)
    except DescriptorError as exc:
        raise RecipeConfigError(str(exc)) from exc
    if not descriptor.is_prompt_task_directory:
        raise RecipeConfigError(
            f"evolution.{setting} needs an adapter that takes a task directory, not a prompt; "
            f"{adapter_name!r} takes a prompt (terminus takes a task directory)"
        )
    if evolution.get("promote_failures", False):
        raise RecipeConfigError("evolution.promote_failures adds prompts to an evaluation whose tasks are directories")


@dataclass(frozen=True)
class HarnessEvolutionSpec:
    """The parts a harness evolution method binds by default.

    Values: ``proposer`` and ``scorer`` reference the method's ``Proposer`` and
    ``EpisodeScorer`` classes, or are ``None`` when the deployment must name
    them in ``evolution.propose`` and ``evolution.evaluate``. ``selection``
    references the ``SelectionPlugin`` used when ``evolution.selection`` is
    absent.
    Purpose: a method recipe returns it from ``evolution_spec()``; the config
    parsing fills each part from it, and a deployment's own setting replaces it.
    """

    proposer: PluginReference | None = None
    scorer: PluginReference | None = None
    selection: PluginReference = field(default_factory=lambda: PluginReference(ScoreComparisonPlugin))


@dataclass(frozen=True)
class HarnessEvolutionRecipe(Recipe):
    """Harness evolution of one adapter's harness tree: the recipe every harness method extends.

    Config shape (the ``evolution`` section): ``adapter`` (a name
    ``reef.harness.adapters.get_adapter`` resolves); ``seed``, a directory
    holding the first tree (see ``reef.harness.tree.layout``), which becomes
    the base release of a new scenario; ``tasks``, the prompts each step
    evaluates (or ``task_manifest`` with ``tasks_root``, or
    ``eval_task_store``, whose eval split each step reads); ``propose``,
    ``evaluate``, ``selection`` and ``promote``, which replace the method's
    ``evolution_spec()`` parts. The other keys tune the episodes
    (``binary``, ``episode_timeout_s``, ``episode_repeats``,
    ``forbid_residue``), the step limits and publication (``max_steps``,
    ``max_failure_streak``, ``max_model_calls_per_step``, ``recheck_every``,
    ``max_rejected_history``, ``publish``, ``review_kinds``), the grown task
    set (``promote_failures``, ``max_promoted_tasks``,
    ``max_promoted_per_client``), the proposal inbox (``proposals_dir``,
    ``max_pending_proposals``), the step record (``step_record_dir``), an
    agent proposer's isolation (``proposer_agent``), and the method's
    auxiliary models (``models``). Worker placement lives under
    ``execution.evolution``.

    The model under test is the recipe's inference runtime. It reaches the
    proposer as ``context.models.served``, and the adapter's
    ``model_binding`` templates write it into each evaluation episode, so
    neither the seed nor a published tree ever carries a provider.
    """

    propose: Proposer
    score_episode: EpisodeScorer
    tasks: tuple[str, ...]
    #: A task store whose eval split each step evaluates on, in place of ``tasks``.
    eval_task_store: str | None = None
    adapter: str = "pi"
    binary: str | None = None
    episode_timeout_s: float = 600.0
    episode_repeats: int = 1
    #: The traces in one evolution step; the trainer makes a batch when the pipeline holds this many.
    batch_size: int = 1
    forbid_residue: bool = False
    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    executor: EpisodeExecutor = field(default_factory=lambda: build_executor(None))
    promote_failures: bool = False
    max_promoted_tasks: int = 50
    max_promoted_per_client: int = 5
    promote: Promoter | None = None
    recheck_every: int = 0
    max_rejected_history: int = 25
    min_win_margin: int = 0
    floor_score: float = 1.0
    publish: str = "auto"
    review_kinds: tuple[TreeKind, ...] = ()
    #: Models an installed client may switch to besides the served one, rendered into its config.
    client_models: tuple[str, ...] = ()
    #: The first tree of a new scenario: ``evolution.seed`` read as a directory.
    seed: HarnessTree = field(default_factory=lambda: HarnessTree({}))
    model_name: str | None = None
    models: Mapping[str, ModelBinding] = field(default_factory=dict)
    model_metadata: Mapping[str, ModelMetadata] = field(default_factory=dict)
    #: Where this Reef answers inference: evaluation episodes sample the release it serves through it.
    served_endpoint: ServedEndpoint | None = None
    selection: PluginReference = field(default_factory=lambda: PluginReference(ScoreComparisonPlugin), repr=False)
    #: Default proposal inbox root, with one directory per scenario.
    proposals_dir: str = ".reef/proposals"
    max_pending_proposals: int = 8
    step_record_dir: str | None = None
    worker_executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    #: The isolation an agent proposer runs under (``evolution.proposer_agent``); ``None`` runs no agent.
    agent_executor: EpisodeExecutor | None = None
    agent_timeout_s: float = 1800.0
    agent_trial_timeout_s: float = 300.0
    config_sections: ClassVar[tuple[str, ...]] = ("evolution",)

    name: str = field(default="harness_evolution", kw_only=True)
    scenario_model: ModelConfig | None = field(default=None, repr=False, kw_only=True)

    def with_model_config(self, config: ModelConfig) -> HarnessEvolutionRecipe:
        super().with_model_config(config)
        return replace(self, scenario_model=config)

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=ReportRecord2Record, record2task=TraceTasks, task2batch=TraceBatches)

    @property
    def harness_adapter(self) -> str | None:
        return self.adapter

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            seed_problems = validate_tree(self.seed, get_adapter(self.adapter)).problems
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        if seed_problems:
            raise RecipeConfigError(f"evolution.seed fails validation: {'; '.join(seed_problems)}")
        if not isinstance(self.propose, Proposer):
            raise ValueError("harness evolution requires a Proposer")
        if not isinstance(self.score_episode, EpisodeScorer):
            raise ValueError("harness evolution requires an EpisodeScorer")
        if not self.tasks and self.eval_task_store is None:
            raise ValueError("harness evolution requires tasks or an eval task store")
        evaluation_selection(self.score_episode, self.worker_executor)
        if self.episode_timeout_s <= 0:
            raise ValueError("episode_timeout_s must be positive")
        if self.episode_repeats < 1:
            raise ValueError("episode_repeats must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("recheck_every", self.recheck_every),
            ("max_rejected_history", self.max_rejected_history),
            ("min_win_margin", self.min_win_margin),
        ):
            if value < 0:
                raise ValueError(f"{label} must be at least 0 (0 disables the limit)")
        if self.floor_score <= 0:
            raise ValueError("floor_score must be positive")
        if self.publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")
        if not isinstance(self.selection, PluginReference) or not issubclass(
            self.selection.plugin_class, SelectionPlugin
        ):
            raise ValueError("selection must reference a SelectionPlugin class")
        if not isinstance(self.proposals_dir, str) or not self.proposals_dir.strip():
            raise ValueError("proposals_dir must be a non-empty path")
        if isinstance(self.max_pending_proposals, bool) or self.max_pending_proposals < 1:
            raise ValueError("max_pending_proposals must be an integer of at least 1")
        if self.step_record_dir is not None and (
            not isinstance(self.step_record_dir, str) or not self.step_record_dir
        ):
            raise ValueError("step_record_dir must be a non-empty path when set")

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("harness evolution requires an 'evolution' config section")
        tasks = evolution.get("tasks")
        manifest_path = evolution.get("task_manifest")
        tasks_root = evolution.get("tasks_root")
        eval_task_store = evolution.get("eval_task_store")
        if eval_task_store is not None:
            if tasks is not None or manifest_path is not None or tasks_root is not None:
                raise RecipeConfigError(
                    "evolution.eval_task_store cannot be set with evolution.tasks or evolution.task_manifest"
                )
            if not isinstance(eval_task_store, str) or not eval_task_store:
                raise RecipeConfigError("evolution.eval_task_store must be the name of a task store")
            require_task_directories(evolution, "eval_task_store")
            # The tasks are the store's eval split, which build reads from the scenario session.
            tasks = []
        elif manifest_path is not None:
            if tasks is not None:
                raise RecipeConfigError("evolution.tasks and evolution.task_manifest cannot both be set")
            if (
                not isinstance(manifest_path, str)
                or not manifest_path
                or not isinstance(tasks_root, str)
                or not tasks_root
            ):
                raise RecipeConfigError(
                    "evolution.task_manifest and evolution.tasks_root must both be non-empty paths"
                )
            require_task_directories(evolution, "task_manifest")
            try:
                task_paths = manifest_task_paths(
                    Path(manifest_path).expanduser(), Path(tasks_root).expanduser(), "eval"
                )
            except TaskSplitError as exc:
                raise RecipeConfigError(str(exc)) from exc
            if not task_paths:
                raise RecipeConfigError(f"evolution.task_manifest {manifest_path} names no eval tasks")
            tasks = [str(path) for path in task_paths]
        elif tasks_root is not None:
            raise RecipeConfigError("evolution.tasks_root is only read with evolution.task_manifest")
        elif not isinstance(tasks, Sequence) or isinstance(tasks, str) or not tasks:
            raise RecipeConfigError(
                "evolution.tasks must be a non-empty list of prompts, or set evolution.task_manifest with "
                "evolution.tasks_root, or evolution.eval_task_store"
            )
        binary = evolution.get("binary")
        if binary is not None and (not isinstance(binary, str) or not binary):
            raise RecipeConfigError("evolution.binary must be a non-empty string when set")
        timeout = evolution.get("episode_timeout_s", 600.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise RecipeConfigError("evolution.episode_timeout_s must be a positive number")
        repeats = evolution.get("episode_repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise RecipeConfigError("evolution.episode_repeats must be an integer of at least 1")
        batch_size = evolution.get("batch_size", 1)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise RecipeConfigError("evolution.batch_size must be an integer of at least 1")
        forbid_residue = evolution.get("forbid_residue", False)
        if not isinstance(forbid_residue, bool):
            raise RecipeConfigError("evolution.forbid_residue must be a boolean")
        try:
            executor = build_executor(evolution, environ=values)
        except ReefError as exc:
            raise RecipeConfigError(str(exc)) from exc
        budget_defaults = {
            "max_steps": 0,
            "max_failure_streak": 0,
            "max_model_calls_per_step": 0,
            "recheck_every": 0,
            "max_rejected_history": 25,
            "min_win_margin": 0,
        }
        budgets: dict[str, int] = {}
        for label, default in budget_defaults.items():
            value = evolution.get(label, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RecipeConfigError(f"evolution.{label} must be an integer of at least 0 (0 disables the limit)")
            budgets[label] = value
        promote_failures = evolution.get("promote_failures", False)
        if not isinstance(promote_failures, bool):
            raise RecipeConfigError("evolution.promote_failures must be a boolean")
        max_promoted_tasks = evolution.get("max_promoted_tasks", 50)
        if isinstance(max_promoted_tasks, bool) or not isinstance(max_promoted_tasks, int) or max_promoted_tasks < 0:
            raise RecipeConfigError("evolution.max_promoted_tasks must be an integer of at least 0")
        per_client = evolution.get("max_promoted_per_client", 5)
        if isinstance(per_client, bool) or not isinstance(per_client, int) or per_client < 0:
            raise RecipeConfigError("evolution.max_promoted_per_client must be an integer of at least 0")
        seed_dir = evolution.get("seed")
        if seed_dir is not None and (not isinstance(seed_dir, str) or not seed_dir.strip()):
            raise RecipeConfigError("evolution.seed must be the path of a harness tree directory")
        try:
            seed = HarnessTree({}) if seed_dir is None else HarnessTree.read(Path(seed_dir).expanduser())
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.seed: {exc}") from exc
        spec = cls.evolution_spec()
        selection_setting = evolution.get("selection")
        selection = spec.selection if selection_setting is None else selection_reference(selection_setting)
        if budgets["min_win_margin"]:
            if selection_setting not in (None, "score_comparison"):
                raise RecipeConfigError("evolution.min_win_margin applies only to the score_comparison selection")
            selection = PluginReference(ScoreComparisonPlugin, {"min_win_margin": budgets["min_win_margin"]})
        floor_score = evolution.get("floor_score", 1.0)
        if isinstance(floor_score, bool) or not isinstance(floor_score, (int, float)) or floor_score <= 0:
            raise RecipeConfigError("evolution.floor_score must be a positive number")
        if "floor_score" in evolution:
            if selection_setting != "floor":
                raise RecipeConfigError("evolution.floor_score applies only to the floor selection")
            selection = PluginReference(FloorPlugin, {"floor_score": float(floor_score)})
        # A recheck compares two trees; the floor evaluates one.
        if selection_setting == "floor" and budgets["recheck_every"]:
            raise RecipeConfigError("evolution.recheck_every does not apply to the floor selection")
        publish = evolution.get("publish", "auto")
        if publish not in ("auto", "review"):
            raise RecipeConfigError("evolution.publish must be 'auto' or 'review'")
        review_kinds = evolution.get("review_kinds", ())
        if isinstance(review_kinds, str) or not isinstance(review_kinds, Sequence):
            raise RecipeConfigError("evolution.review_kinds must be a list of tree kind names")
        if not all(kind in tuple(TreeKind) for kind in review_kinds):
            raise RecipeConfigError(f"evolution.review_kinds must name tree kinds: {', '.join(TreeKind)}")
        client_models = evolution.get("client_models", ())
        if isinstance(client_models, str) or not isinstance(client_models, Sequence):
            raise RecipeConfigError("evolution.client_models must be a list of model names")
        if not all(isinstance(name, str) and name for name in client_models):
            raise RecipeConfigError("evolution.client_models must be a list of model names")
        adapter = str(evolution.get("adapter", "pi"))
        model = settings.get("model")
        model_name = model.get("path") if isinstance(model, Mapping) else None
        named = evolution.get("models") or {}
        if not isinstance(named, Mapping):
            raise RecipeConfigError("evolution.models must map a name to a model section (url, model, api_key_env)")
        metadata_config = evolution.get("model_metadata", {})
        if not isinstance(metadata_config, Mapping):
            raise RecipeConfigError("evolution.model_metadata must map model names to metadata")
        model_metadata: dict[str, ModelMetadata] = {}
        for name, section in metadata_config.items():
            if not isinstance(name, str) or not name.strip():
                raise RecipeConfigError("evolution.model_metadata requires non-empty model names")
            try:
                model_metadata[name] = ModelMetadata.from_config(section)
            except ValueError as exc:
                raise RecipeConfigError(f"evolution.model_metadata.{name}: {exc}") from exc
        models: dict[str, ModelBinding] = {}
        for name, section in named.items():
            if not isinstance(name, str) or not name or not isinstance(section, Mapping):
                raise RecipeConfigError(
                    "evolution.models must map a name to a model section (url, model, api_key_env)"
                )
            try:
                models[name] = ModelBinding.from_config(section, values, where=f"evolution.models.{name}")
            except ValueError as exc:
                raise RecipeConfigError(str(exc)) from exc
        if "served" in models:
            raise RecipeConfigError("evolution.models may not name a model 'served'; that is the model under test")
        proposals_dir = evolution.get("proposals_dir", cls.proposals_dir)
        if not isinstance(proposals_dir, str) or not proposals_dir.strip():
            raise RecipeConfigError("evolution.proposals_dir must be a non-empty path")
        max_pending = evolution.get("max_pending_proposals", 8)
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise RecipeConfigError("evolution.max_pending_proposals must be an integer of at least 1")
        step_record_dir = evolution.get("step_record_dir")
        if step_record_dir is not None and (not isinstance(step_record_dir, str) or not step_record_dir.strip()):
            raise RecipeConfigError("evolution.step_record_dir must be a non-empty path when set")
        try:
            worker_executor = role_executor_settings(settings, "evolution")
            scorer = EpisodeScorer.from_setting(
                evolution.get("evaluate"), setting="evolution.evaluate", default=spec.scorer
            )
            propose = Proposer.from_setting(
                evolution.get("propose"), setting="evolution.propose", default=spec.proposer
            )
            promote = (
                Promoter.from_setting(evolution["promote"], setting="evolution.promote")
                if "promote" in evolution
                else None
            )
        except (TypeError, ValueError) as exc:
            raise RecipeConfigError(str(exc)) from exc
        agent_executor, agent_timeout_s, agent_trial_timeout_s = proposer_agent_settings(
            evolution.get("proposer_agent"), values, proposals_dir=Path(proposals_dir.strip())
        )
        return {
            **super()._recipe_kwargs(settings, values),
            "agent_executor": agent_executor,
            "agent_timeout_s": agent_timeout_s,
            "agent_trial_timeout_s": agent_trial_timeout_s,
            "proposals_dir": proposals_dir.strip(),
            "max_pending_proposals": max_pending,
            "propose": propose,
            "promote": promote,
            "score_episode": scorer,
            "worker_executor": worker_executor,
            "tasks": tuple(str(task) for task in tasks),
            "eval_task_store": eval_task_store,
            "adapter": adapter,
            "binary": binary,
            "episode_timeout_s": float(timeout),
            "episode_repeats": repeats,
            "batch_size": batch_size,
            "forbid_residue": forbid_residue,
            **budgets,
            "floor_score": float(floor_score),
            "executor": executor,
            "promote_failures": promote_failures,
            "max_promoted_tasks": max_promoted_tasks,
            "max_promoted_per_client": per_client,
            "publish": publish,
            "review_kinds": tuple(TreeKind(kind) for kind in review_kinds),
            "client_models": tuple(client_models),
            "seed": seed,
            "model_name": model_name if isinstance(model_name, str) and model_name else None,
            "models": models,
            "model_metadata": model_metadata,
            "selection": selection,
            "step_record_dir": None if step_record_dir is None else step_record_dir.strip(),
        }

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> HarnessEvolutionRecipe:
        return replace(self, served_endpoint=endpoint)

    def model_binding(self, scenario: str | None = None) -> ModelBinding:
        """The served model's endpoint.

        With the service known and a scenario named, this Reef's own
        evaluation route for that scenario: an episode then samples the
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
        if self.adapter == "codex":
            return binding.with_metadata(self.model_metadata.get(binding.model))
        return binding

    def served_through_service(self, binding: ModelBinding, scenario: str | None) -> ModelBinding:
        """``binding`` routed through this Reef's evaluation route for ``scenario``, when the service is known.

        A component of a composed release names itself in the route, so the
        release's hooks for the component a candidate replaces stay out.
        """
        if self.served_endpoint is None or scenario is None:
            return binding
        # The name is free form: quoted as the wrapper quotes it, so a slash or a space stays one segment.
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
        client_models = self.client_models
        override = self.scenario_model.runtime if self.scenario_model is not None else None
        if override is not None:
            model = override.model_path
            api = override.api
            client_models = ()
        if self.adapter == "codex" and self.runtime is not None:
            selected = self.bind_model_metadata(ModelBinding.from_runtime(override or self.runtime, model=model))
            served_metadata = selected.metadata
        else:
            served_metadata = None
        return create_harness_surface(
            get_adapter(self.adapter),
            served_model=model if isinstance(model, str) and model else None,
            served_api=api,
            model_metadata=self.model_metadata,
            served_metadata=served_metadata,
            client_models=client_models,
        )

    def base_artifact_files(self) -> Mapping[str, str] | None:
        """The seed tree's files: a new scenario's base release, which the first step builds on."""
        return dict(self.seed.files) or None

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's proposal inbox and its step records: what a delete archives beside the record store."""
        dirs = [self.proposals_path(scenario)]
        if self.step_record_dir is not None:
            dirs.append(Path(self.step_record_dir).expanduser().resolve() / scenario)
        return tuple(dirs)

    def proposals_path(self, scenario: str) -> Path:
        """The scenario's proposal inbox: ``proposals_dir`` made absolute, one directory per scenario under it."""
        return Path(self.proposals_dir).expanduser().resolve() / scenario

    @classmethod
    def evolution_spec(cls) -> HarnessEvolutionSpec:
        """The method's default proposer, scorer and selection; a method recipe overrides this.

        The base names no proposer or scorer, so a deployment of the base
        recipe names both, and it selects by score comparison.
        """
        return HarnessEvolutionSpec()

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        releases: ComponentReleases | None = None,
    ) -> Trainer:
        if releases is None:
            raise RecipeConfigError("harness evolution reads its head tree from the scenario's committed releases")
        backend = self.build_backend(scenario, releases, task_storage)
        if self.training_mode != "auto" and not self.propose.reads_requests:
            raise RecipeConfigError(
                f"harness evolution in training_mode={self.training_mode!r} requires a proposer "
                "whose reads_requests is true"
            )
        selection_class = SelectionPlugin.referenced_class(self.selection)
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: self.build_pipeline(
                replace(context, manual_enabled=self.propose.reads_requests)
            ),
            candidate_backend=backend,
            candidate_evaluator=selection_class.from_config(self.selection.config, candidate_backend=backend),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
            batch_size=self.batch_size,
        )

    def build_backend(
        self, scenario: str, releases: ComponentReleases, task_storage: TaskStorage | None
    ) -> HarnessBackend:
        """The scenario's backend: the binary installed, the evaluator, the tasks, the inbox and the record."""
        descriptor = get_adapter(self.adapter)
        program = AdapterProgram(descriptor)
        executor, binary = self.executor.for_program(program, binary=self.binary)
        executor.preflight()
        models = self.model_bindings(scenario)
        selection, requirements = evaluation_selection(self.score_episode, self.worker_executor)
        evaluator = HarnessEpisodeEvaluator(
            descriptor=descriptor,
            scorer=self.score_episode,
            models=models,
            binary=binary,
            episode_timeout_s=self.episode_timeout_s,
            episode_repeats=self.episode_repeats,
            executor=executor,
            forbid_residue=self.forbid_residue,
            worker_selection=selection,
            worker_requirements=requirements,
        )
        task_store = None
        if self.eval_task_store is not None:
            if task_storage is None:
                raise RecipeConfigError("evolution.eval_task_store needs the task storage of a scenario session")
            task_store = task_storage.open(self.eval_task_store)
        agent = None
        if self.agent_executor is not None:
            agent_executor, agent_binary = self.agent_executor.for_program(program, binary=self.binary)
            agent_executor.preflight()
            agent = AgentExecution(agent_executor, agent_binary, self.agent_timeout_s, self.agent_trial_timeout_s)
        # One recipe serves many scenarios: each scenario's steps record under their own absolute directory.
        record_root = None if self.step_record_dir is None else Path(self.step_record_dir).expanduser().resolve()
        return HarnessBackend(
            descriptor=descriptor,
            releases=releases,
            proposer=self.propose,
            evaluator=evaluator,
            tasks=EvaluationTasks(
                seed_tasks=self.tasks,
                task_store=task_store,
                promote_failures=self.promote_failures,
                promoter=self.promote,
                max_promoted=self.max_promoted_tasks,
                max_promoted_per_client=self.max_promoted_per_client,
            ),
            models=models,
            model_resolver=(
                None if self.scenario_model is None else ScenarioModels(self.scenario_model, self, scenario)
            ),
            limits=StepLimits(
                max_steps=self.max_steps,
                max_failure_streak=self.max_failure_streak,
                max_model_calls_per_step=self.max_model_calls_per_step,
                recheck_every=self.recheck_every,
                max_rejected_history=self.max_rejected_history,
                publish=self.publish,
                review_kinds=frozenset(self.review_kinds),
            ),
            records=StepRecord(None if record_root is None else record_root / scenario),
            inbox=ProposalInbox(self.proposals_path(scenario), self.max_pending_proposals),
            agent=agent,
        )
