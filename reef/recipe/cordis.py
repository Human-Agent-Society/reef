"""Harness evolution recipe: boots the composition mutation loop from config.

``CordisRecipe`` boots the loop from a config yaml. ``propose`` and
``evaluate`` are Python callables, named as dotted ``module:attribute``
references in YAML or passed directly when registering from code; ``propose``
returns one ``Mutation``, a sequence of them (one composite proposal under
one selection decision), a ``StepProposal`` (the mutations plus notes the
step records), or ``None``. ``CordisBackend`` owns the
mutation/render/episode/scoring phases; the recipe composes that evaluator
and its selection policy into the candidate evaluator executed by ``Trainer``.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

from reef.core.errors import ReefError
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import TaskSplitError, manifest_task_paths
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.episodes.e2b import E2BExecutor, deployment_owner
from reef.harness.episodes.executor import (
    EpisodeExecutor,
    LocalExecutor,
    SandboxExecutor,
    SandboxUnavailable,
    build_executor,
)
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings, ModelBindingsResolver
from reef.harness.episodes.requests import request_entries
from reef.harness.episodes.version_check import version_check_entry
from reef.harness.tree.render import render_composition
from reef.inference.http import InferenceProxyRuntime
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe, ServedEndpoint
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.runtime.executor.config import ExecutorSettings, WorkerResources, executor_settings, role_executor_settings
from reef.storage.records import RecordStore
from reef.surface.base import Surface
from reef.surface.harnesses import create_harness_surface
from reef.train.backend import STALE_RESULT_POLICIES
from reef.train.cordis_backend.backend import (
    CordisBackend,
    FloorPluginFactory,
    ScoreComparisonPluginFactory,
    tree_files,
)
from reef.train.cordis_backend.execution import evaluation_selection, legacy_worker_settings
from reef.train.cordis_backend.processor import CordisProcessor, RecordDrivenTraceProcessor
from reef.train.cordis_backend.strategies import (
    EpisodeScorer,
    Promoter,
    Proposer,
    resolve_episode_scorer,
    resolve_promoter,
    resolve_proposer,
)
from reef.train.evaluation.evaluators import AlwaysSelectPluginFactory, CandidatePluginFactory
from reef.train.trainer import Trainer

_CANDIDATE_PLUGIN_FACTORIES: dict[str, CandidatePluginFactory] = {
    "score_comparison": ScoreComparisonPluginFactory(),
    "floor": FloorPluginFactory(),
    "always": AlwaysSelectPluginFactory(),
}


def proposer_agent_settings(section: Any, environ: Mapping[str, str]) -> tuple[EpisodeExecutor | None, float, float]:
    """``evolution.proposer_agent`` as the agent's executor and its two timeouts; no section runs no agent.

    ``sandbox: bwrap`` jails the agent and gives it the internet but no host port
    but the gateway's, and refuses to start where bwrap or pasta is missing;
    ``sandbox: e2b`` runs it in an E2B cloud sandbox (``e2b_api_key``, else
    ``E2B_API_KEY``; ``e2b_template``, else the harness's pinned binary, built on
    first use) that reaches the gateway through a tunnel and nothing else of the
    host; ``sandbox: none`` runs it unisolated with the service's privileges, and must
    be chosen. Left empty (and ``REEF_PROPOSER_SANDBOX`` unset), the agent is
    jailed where the host can, and off (the text proposer answers requests)
    where it cannot.
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
        remote = E2BExecutor(
            api_key=str(section.get("e2b_api_key") or environ.get("E2B_API_KEY") or "").strip(),
            template=str(section.get("e2b_template") or "").strip(),
            timeout_s=timeouts[0],
        )
        try:
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
class _ScenarioModels(ModelBindingsResolver):
    config: ModelConfig
    recipe: CordisRecipe
    #: The scenario the bindings serve; a call naming none resolves for it.
    scenario: str | None = None

    def resolve(self, scenario: str | None = None) -> ModelBindings:
        scenario = self.scenario if scenario is None else scenario
        runtime = self.config.runtime
        if runtime is None:
            return self.recipe.default_model_bindings(scenario)
        # The scenario's own model is served by this Reef too, so an episode reaches it through the same route.
        served = self.recipe.served_through_service(ModelBinding.from_runtime(runtime), scenario)
        return ModelBindings(served=served, named=dict.fromkeys(self.recipe.models, served))


def _resolve_callable(value: Any, what: str) -> Any:
    """A callable as-is, or a dotted ``module:attribute`` reference to one."""
    if callable(value):
        return value
    if isinstance(value, str) and ":" in value:
        module_name, _, attribute = value.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise RecipeConfigError(f"cannot import {what} {value!r}: {exc}") from exc
        if callable(resolved):
            return resolved
    raise RecipeConfigError(f"{what} must be a callable or a dotted 'module:attribute' reference")


def _resolve_seed_entries(value: str) -> Sequence[Mapping[str, Any]]:
    """The entry sequence a dotted ``module:attribute`` seed reference names."""
    module_name, _, attribute = value.partition(":")
    try:
        resolved = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RecipeConfigError(f"cannot import evolution.seed reference {value!r}: {exc}") from exc
    if isinstance(resolved, str) or not isinstance(resolved, Sequence):
        raise RecipeConfigError(f"evolution.seed reference {value!r} must name a sequence of entry option mappings")
    if not all(isinstance(entry, Mapping) for entry in resolved):
        raise RecipeConfigError(f"evolution.seed reference {value!r} must name a sequence of entry option mappings")
    return resolved


def _resolve_candidate_plugin(value: Any) -> CandidatePluginFactory:
    if isinstance(value, str) and value in _CANDIDATE_PLUGIN_FACTORIES:
        return _CANDIDATE_PLUGIN_FACTORIES[value]
    resolved = value
    if isinstance(value, str) and ":" in value:
        module_name, _, attribute = value.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise RecipeConfigError(f"cannot import evolution.selection {value!r}: {exc}") from exc
    if isinstance(resolved, type) and issubclass(resolved, CandidatePluginFactory):
        resolved = resolved()
    if isinstance(resolved, CandidatePluginFactory):
        return resolved
    raise RecipeConfigError(
        "evolution.selection must be a built-in name or a dotted reference to a "
        "candidate-evaluation plugin factory inheriting CandidatePluginFactory"
    )


@dataclass(frozen=True)
class CordisRecipe(Recipe):
    """The harness evolution loop as a bootable recipe class.

    Config shape (the ``evolution`` section): ``adapter`` (a name
    ``reef.harness.adapters.get_adapter`` resolves), ``propose`` and
    ``evaluate`` (callables or dotted references), ``tasks`` (the episode
    prompts scored per step), optional ``binary`` (a path to the harness
    binary; left unset, backend construction installs the adapter descriptor's
    pinned version through the vendor's own channel, into the same prefix a
    client's install script uses - see :mod:`reef.harness.episodes.vendor_install`),
    optional ``seed`` (a list of entry options - id, name, config -
    loaded into the composition tree on first boot, where an item may also
    be a dotted ``module:attribute`` naming a sequence of them; a recovered
    algorithm state always wins over the seed), optional ``selection`` (the
    candidate-selection policy: ``score_comparison``, the default; ``floor``,
    which runs the candidate alone and selects it when every task scores at
    least ``floor_score``, default ``1.0``; ``always``;
    or a dotted reference to a ``CandidatePluginFactory`` subclass or instance),
    optional ``step_record_dir`` (a directory under which every scenario's
    steps write the proposer's model calls, the parsed proposal and each evaluation
    episode's trajectory files, so the decision is reconstructible; off by
    default),
    optional ``client_models`` (further model names the installed client
    may switch to; the install script renders them into its config beside
    the served model, which stays the default),
    and optional ``version_check``
    (``true`` appends the adapter's shipped update notice extension to the
    seed, so every pulled tree tells its user at startup when it is behind
    the channel head; adapters without a shipped extension refuse boot),
    optional ``requests`` (``true`` appends the adapter's shipped harness requests
    extension and its pi extension API skill to the seed after the notice,
    so a session can ask for a harness change from the TUI and the method
    reads the API reference before it writes an extension; same refusal for
    adapters without one),
    the agent proposal inbox: ``proposals_dir`` (default
    ``.reef/proposals``, one directory per scenario under it, created when
    the first proposal arrives) and ``max_pending_proposals`` (default 8,
    the number of admitted proposals a scenario holds before the route
    refuses more).

    Worker placement/count/resources live under ``execution.evolution``, not
    the business configuration. Legacy ``episode_workers`` remains an alias.

    The model under test is the recipe's inference runtime - the
    deployment's ``reef.upstream_url`` / ``reef.upstream_model`` (and
    ``reef.upstream_api`` for a Responses or Anthropic-style provider). It
    reaches ``propose`` as ``models.served`` and is rendered into each
    evaluation episode through the adapter's ``model_binding`` template, so
    the seed carries no provider nodes and neither does the published tree; a
    client points its own harness at Reef. A method's auxiliary models - a
    stronger proposer, a judge - are declared under ``evolution.models`` and
    reach ``propose`` as ``models["name"]``::

        evolution:
          models:
            teacher:
              url: https://api.openai.com
              model: gpt-4o
              api_key_env: OPENAI_API_KEY   # the key stays out of the file
              api: openai                   # default; or responses / anthropic

    A seed names the baseline nodes the first mutation is measured against::

        evolution:
          seed:
            - id: answer-style
              name: skill
              config:
                name: answer-style
                text: |
                  # answer-style
                  Put the final answer alone on the last line.
    """

    propose: Proposer
    score_episode: EpisodeScorer
    tasks: tuple[str, ...]
    adapter: str = "pi"
    binary: str | None = None
    episode_timeout_s: float = 600.0
    episode_repeats: int = 1
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
    review_kinds: tuple[str, ...] = ()
    #: Models an installed client may switch to besides the served one, rendered into its config.
    client_models: tuple[str, ...] = ()
    seed: tuple[Mapping[str, Any], ...] = ()
    model_name: str | None = None
    models: Mapping[str, ModelBinding] = field(default_factory=dict)
    #: Where this Reef answers inference: evaluation episodes sample the release it serves through it.
    served_endpoint: ServedEndpoint | None = None
    #: What a result becomes when another component's commit replaced its base while it was evaluated:
    #: merged onto the release served now, evaluated again, or refused and proposed again.
    on_stale: str = "merge"
    candidate_plugin: CandidatePluginFactory = field(default_factory=ScoreComparisonPluginFactory, repr=False)
    episode_workers: int | None = None  # Deprecated Python compatibility alias.
    #: Default proposal inbox root, with one directory per scenario.
    proposals_dir: str = ".reef/proposals"
    max_pending_proposals: int = 8
    step_record_dir: str | None = None
    worker_executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    worker_gpus: float | None = None
    #: The isolation an agent proposer runs under (``evolution.proposer_agent``); ``None`` runs no agent.
    agent_executor: EpisodeExecutor | None = None
    agent_timeout_s: float = 1800.0
    agent_trial_timeout_s: float = 300.0
    config_sections: ClassVar[tuple[str, ...]] = ("evolution",)

    batch_size: int = config_field(1)
    batch_policy: str = config_field("reports")
    name: str = field(default="harness_evolve", kw_only=True)
    scenario_model: ModelConfig | None = field(default=None, repr=False, kw_only=True)

    def with_model_config(self, config: ModelConfig) -> CordisRecipe:
        super().with_model_config(config)
        return replace(self, scenario_model=config)

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.propose, Proposer):
            raise ValueError("harness evolution requires a Proposer")
        if not isinstance(self.score_episode, EpisodeScorer):
            raise ValueError("harness evolution requires an EpisodeScorer")
        if not self.tasks:
            raise ValueError("harness evolution requires tasks")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        settings = legacy_worker_settings(self.worker_executor, self.episode_workers, self.worker_gpus)
        _, requirements = evaluation_selection(self.score_episode, None, settings)
        object.__setattr__(self, "worker_executor", settings)
        object.__setattr__(self, "episode_workers", requirements.workers)
        if self.batch_policy not in ("reports", "records"):
            raise ValueError("batch_policy must be 'reports' or 'records'")
        if self.episode_timeout_s <= 0:
            raise ValueError("episode_timeout_s must be positive")
        if self.episode_repeats < 1:
            raise ValueError("episode_repeats must be at least 1")
        if self.on_stale not in STALE_RESULT_POLICIES:
            raise ValueError(f"on_stale must be one of {STALE_RESULT_POLICIES}")
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
        if not isinstance(self.candidate_plugin, CandidatePluginFactory):
            raise ValueError("candidate_plugin must be a CandidatePluginFactory instance")
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
            raise RecipeConfigError("harness_evolve requires an 'evolution' config section")
        tasks = evolution.get("tasks")
        manifest_path = evolution.get("task_manifest")
        tasks_root = evolution.get("tasks_root")
        if manifest_path is not None:
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
            adapter_name = str(evolution.get("adapter", "pi"))
            try:
                descriptor = get_adapter(adapter_name)
            except DescriptorError as exc:
                raise RecipeConfigError(str(exc)) from exc
            if not descriptor.is_prompt_task_directory:
                raise RecipeConfigError(
                    f"evolution.task_manifest needs an adapter that takes a task directory, not a prompt; "
                    f"{adapter_name!r} takes a prompt (terminus takes a task directory)"
                )
            if evolution.get("promote_failures", False):
                raise RecipeConfigError(
                    "evolution.promote_failures adds prompts to an evaluation whose tasks are directories"
                )
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
                "evolution.tasks must be a non-empty list of prompts, or set evolution.task_manifest with evolution.tasks_root"
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
        on_stale = evolution.get("on_stale", "merge")
        if on_stale not in STALE_RESULT_POLICIES:
            raise RecipeConfigError(f"evolution.on_stale must be one of {STALE_RESULT_POLICIES}")
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
        seed = evolution.get("seed")
        if seed is None:
            seed = ()
        elif not isinstance(seed, Sequence) or isinstance(seed, str):
            raise RecipeConfigError("evolution.seed must be a list of entry option mappings")
        entries: list[Mapping[str, Any]] = []
        for entry in seed:
            # A dotted reference names a shipped sequence, such as the native harness's seed tools and hook.
            if isinstance(entry, str) and ":" in entry:
                entries.extend(_resolve_seed_entries(entry))
            elif isinstance(entry, Mapping):
                entries.append(entry)
            else:
                raise RecipeConfigError("evolution.seed entries must be entry option mappings or dotted references")
        seed = entries
        if "acceptance" in evolution:
            raise RecipeConfigError("evolution.acceptance was removed; configure evolution.selection")
        selection = evolution.get("selection", "score_comparison")
        candidate_plugin = _resolve_candidate_plugin(selection)
        if budgets["min_win_margin"]:
            if selection != "score_comparison":
                raise RecipeConfigError("evolution.min_win_margin applies only to the score_comparison selection")
            candidate_plugin = ScoreComparisonPluginFactory(min_win_margin=budgets["min_win_margin"])
        floor_score = evolution.get("floor_score", 1.0)
        if isinstance(floor_score, bool) or not isinstance(floor_score, (int, float)) or floor_score <= 0:
            raise RecipeConfigError("evolution.floor_score must be a positive number")
        if "floor_score" in evolution:
            if selection != "floor":
                raise RecipeConfigError("evolution.floor_score applies only to the floor selection")
            candidate_plugin = FloorPluginFactory(floor_score=float(floor_score))
        # A recheck compares two trees; the floor evaluates one.
        if selection == "floor" and budgets["recheck_every"]:
            raise RecipeConfigError("evolution.recheck_every does not apply to the floor selection")
        publish = evolution.get("publish", "auto")
        if publish not in ("auto", "review"):
            raise RecipeConfigError("evolution.publish must be 'auto' or 'review'")
        review_kinds = evolution.get("review_kinds", ())
        if isinstance(review_kinds, str) or not isinstance(review_kinds, Sequence):
            raise RecipeConfigError("evolution.review_kinds must be a list of node kind names")
        if not all(isinstance(kind, str) and kind for kind in review_kinds):
            raise RecipeConfigError("evolution.review_kinds must be a list of node kind names")
        client_models = evolution.get("client_models", ())
        if isinstance(client_models, str) or not isinstance(client_models, Sequence):
            raise RecipeConfigError("evolution.client_models must be a list of model names")
        if not all(isinstance(name, str) and name for name in client_models):
            raise RecipeConfigError("evolution.client_models must be a list of model names")
        adapter = str(evolution.get("adapter", "pi"))
        version_check = evolution.get("version_check", False)
        if not isinstance(version_check, bool):
            raise RecipeConfigError("evolution.version_check must be a boolean")
        if version_check:
            try:
                seed = (*seed, version_check_entry(adapter))
            except DescriptorError as exc:
                raise RecipeConfigError(str(exc)) from exc
        requests = evolution.get("requests", False)
        if not isinstance(requests, bool):
            raise RecipeConfigError("evolution.requests must be a boolean")
        if requests:
            try:
                seed = (*seed, *request_entries(adapter))
            except DescriptorError as exc:
                raise RecipeConfigError(str(exc)) from exc
        model = settings.get("model")
        model_name = model.get("path") if isinstance(model, Mapping) else None
        named = evolution.get("models") or {}
        if not isinstance(named, Mapping):
            raise RecipeConfigError("evolution.models must map a name to a model section (url, model, api_key_env)")
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
        # A ``${VAR}`` interpolation arrives as text, so a digit string counts.
        raw_workers = evolution.get("episode_workers")
        if isinstance(raw_workers, str) and raw_workers.strip().isdigit():
            raw_workers = int(raw_workers)
        if "episode_workers" in evolution and (
            isinstance(raw_workers, bool) or not isinstance(raw_workers, int) or raw_workers < 1
        ):
            raise RecipeConfigError("evolution.episode_workers must be a positive integer")
        episode_workers = raw_workers
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
            role_settings = role_executor_settings(settings, "evolution")
            if "worker_executor" in evolution and (
                role_settings.workers is not None or role_settings.resources != WorkerResources()
            ):
                raise ValueError(
                    "remove deprecated evolution.worker_executor when execution.evolution defines workers/resources"
                )
            worker_executor = (
                executor_settings(settings, evolution["worker_executor"])
                if "worker_executor" in evolution
                else role_settings
            )
            resources = evolution.get("worker_resources", {})
            if not isinstance(resources, Mapping) or set(resources) - {"num_gpus"}:
                raise ValueError("evolution.worker_resources accepts only num_gpus")
            worker_gpus = resources.get("num_gpus")
            if any(key in evolution for key in ("episode_workers", "worker_executor", "worker_resources")):
                warnings.warn(
                    "evolution.episode_workers/worker_executor/worker_resources are deprecated; "
                    "use execution.evolution.backend/workers/resources",
                    DeprecationWarning,
                    stacklevel=3,
                )
            worker_executor = legacy_worker_settings(worker_executor, episode_workers, worker_gpus)
            scorer = resolve_episode_scorer(evolution.get("evaluate"))
            evaluation_selection(scorer, episode_workers, worker_executor, worker_gpus)
        except (TypeError, ValueError) as exc:
            raise RecipeConfigError(str(exc)) from exc
        agent_executor, agent_timeout_s, agent_trial_timeout_s = proposer_agent_settings(
            evolution.get("proposer_agent"), values
        )
        if isinstance(agent_executor, E2BExecutor):
            # The deployment's own state directory names its sandboxes, so its first sandbox can stop the ones a
            # previous run left, which a stop mid run never closed.
            agent_executor = replace(agent_executor, owner=deployment_owner(Path(proposals_dir.strip())))
        return {
            "agent_executor": agent_executor,
            "agent_timeout_s": agent_timeout_s,
            "agent_trial_timeout_s": agent_trial_timeout_s,
            "proposals_dir": proposals_dir.strip(),
            "max_pending_proposals": max_pending,
            "propose": resolve_proposer(evolution.get("propose")),
            "promote": resolve_promoter(evolution["promote"]) if "promote" in evolution else None,
            "score_episode": scorer,
            "worker_executor": worker_executor,
            "worker_gpus": worker_gpus,
            "tasks": tuple(str(task) for task in tasks),
            "adapter": adapter,
            "binary": binary,
            "episode_timeout_s": float(timeout),
            "episode_repeats": repeats,
            "on_stale": on_stale,
            "forbid_residue": forbid_residue,
            **budgets,
            "floor_score": float(floor_score),
            "executor": executor,
            "promote_failures": promote_failures,
            "max_promoted_tasks": max_promoted_tasks,
            "max_promoted_per_client": per_client,
            "publish": publish,
            "review_kinds": tuple(review_kinds),
            "client_models": tuple(client_models),
            "seed": tuple(seed),
            "model_name": model_name if isinstance(model_name, str) and model_name else None,
            "models": models,
            "candidate_plugin": candidate_plugin,
            "episode_workers": episode_workers,
            "step_record_dir": None if step_record_dir is None else step_record_dir.strip(),
        }

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> CordisRecipe:
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
        return self.served_through_service(binding, scenario)

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
            return _ScenarioModels(self.scenario_model, self).resolve(scenario)
        return self.default_model_bindings(scenario)

    def build_surface(self, scenario: str) -> Surface:
        model = self.model_name or getattr(self.runtime, "model_path", None)
        # Only a provider proxy has a dialect; a local engine serves Chat Completions.
        api = self.runtime.api if isinstance(self.runtime, InferenceProxyRuntime) else "openai"
        client_models = self.client_models
        override = self.scenario_model.runtime if self.scenario_model is not None else None
        if override is not None:
            model = override.model_path
            api = override.api
            client_models = ()
        return create_harness_surface(
            seed_entries=tuple(dict(entry) for entry in self.seed),
            served_model=model if isinstance(model, str) and model else None,
            served_api=api,
            client_models=client_models,
        )

    def base_artifact_files(self) -> Mapping[str, str] | None:
        """The seed rendered for the adapter, with its entries list where the adapter carries one: a fresh scenario serves it before any step publishes."""
        if not self.seed:
            return None
        descriptor = get_adapter(self.adapter)
        nodes = tuple((str(entry["name"]), entry.get("config")) for entry in self.seed if not entry.get("disabled"))
        return {**render_composition(nodes, descriptor), **tree_files(descriptor, self.seed)}

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's proposal inbox and its step records: what a delete archives beside the record store."""
        dirs = [self.proposals_path(scenario)]
        if self.step_record_dir is not None:
            dirs.append(Path(self.step_record_dir).expanduser().resolve() / scenario)
        return tuple(dirs)

    def proposals_path(self, scenario: str) -> Path:
        """The scenario's proposal inbox: ``proposals_dir`` made absolute, one directory per scenario under it."""
        return Path(self.proposals_dir).expanduser().resolve() / scenario

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        kwargs = self._backend_kwargs(scenario)
        if self.scenario_model is not None:
            # Frozen again at every step, for this scenario: the bindings then follow the service's route.
            kwargs["model_resolver"] = _ScenarioModels(self.scenario_model, self, scenario)
        # One recipe serves many scenarios, so each scenario's steps record under their own directory; absolute,
        # so the path a commit record names resolves from any working directory.
        if kwargs["step_record_dir"] is not None:
            kwargs["step_record_dir"] = Path(kwargs["step_record_dir"]).expanduser().resolve() / scenario
        candidate_backend = CordisBackend(**kwargs, proposals_dir=self.proposals_path(scenario))
        return self._build_trainer(
            scenario,
            records,
            candidate_backend,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )

    def _backend_kwargs(self, scenario: str | None = None) -> dict[str, Any]:
        """Arguments shared by the stock backend and recipe specializations."""
        return {
            "descriptor": get_adapter(self.adapter),
            "propose": self.propose,
            "score_episode": self.score_episode,
            "tasks": self.tasks,
            "models": self.model_bindings(scenario),
            "on_stale": self.on_stale,
            "binary": self.binary,
            "episode_timeout_s": self.episode_timeout_s,
            "episode_repeats": self.episode_repeats,
            "forbid_residue": self.forbid_residue,
            "executor": self.executor,
            "max_steps": self.max_steps,
            "max_failure_streak": self.max_failure_streak,
            "max_model_calls_per_step": self.max_model_calls_per_step,
            "promote_failures": self.promote_failures,
            "max_promoted_tasks": self.max_promoted_tasks,
            "max_promoted_per_client": self.max_promoted_per_client,
            "promote": self.promote,
            "recheck_every": self.recheck_every,
            "max_rejected_history": self.max_rejected_history,
            "publish": self.publish,
            "review_kinds": self.review_kinds,
            "seed": self.seed,
            "max_pending_proposals": self.max_pending_proposals,
            "step_record_dir": self.step_record_dir,
            "worker_executor": self.worker_executor,
            "agent_executor": self.agent_executor,
            "agent_timeout_s": self.agent_timeout_s,
            "agent_trial_timeout_s": self.agent_trial_timeout_s,
        }

    def _build_trainer(
        self,
        scenario: str,
        records: RecordStore,
        candidate_backend: CordisBackend,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger | None,
    ) -> Trainer:
        if self.training_mode != "auto" and not self.propose.reads_requests:
            raise RecipeConfigError(
                f"harness evolution in training_mode={self.training_mode!r} requires a proposer "
                "that accepts the 'requests' keyword"
            )
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: (
                RecordDrivenTraceProcessor if self.batch_policy == "records" else CordisProcessor
            )(
                context.with_config(
                    {
                        "batch_size": self.batch_size,
                        "manual_enabled": self.propose.reads_requests,
                    }
                )
            ),
            candidate_backend=candidate_backend,
            candidate_evaluator=self.candidate_plugin.build(candidate_backend),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
        )
