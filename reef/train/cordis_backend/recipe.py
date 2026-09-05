"""Harness evolution recipe: boots the composition mutation loop from config.

``CordisRecipe`` boots the loop from a config yaml. ``propose`` and
``evaluate`` are Python callables, named as dotted ``module:attribute``
references in YAML or passed directly when registering from code; ``propose``
returns one ``Mutation``, a sequence of them (one composite proposal under
one selection decision), or ``None``. ``CordisBackend`` owns the
mutation/render/episode/scoring phases; the recipe composes that evaluator
and its selection policy into the candidate evaluator executed by ``Trainer``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.errors import ReefError
from reef.core.reports import ScoredRolloutReport
from reef.harness.adapters import get_adapter
from reef.harness.descriptor import DescriptorError
from reef.harness.executor import EpisodeExecutor, build_executor
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.harness.render import render_composition
from reef.harness.version_check import version_check_entry
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.records import RecordStore
from reef.surface.base import Surface
from reef.surface.harnesses import create_harness_surface
from reef.train.cordis_backend.backend import CordisBackend, ScoreComparisonSelector, tree_files
from reef.train.cordis_backend.processor import CordisProcessor, RecordDrivenTraceProcessor
from reef.train.cordis_backend.strategies import (
    EpisodeScorer,
    Promoter,
    Proposer,
    resolve_episode_scorer,
    resolve_promoter,
    resolve_proposer,
)
from reef.train.evaluation.contracts import CandidateSelector
from reef.train.evaluation.evaluators import AlwaysSelect, DefaultCandidateEvaluationPlugin
from reef.train.trainer import Trainer

_CANDIDATE_SELECTORS: dict[str, CandidateSelector] = {
    "score_comparison": ScoreComparisonSelector(),
    "always": AlwaysSelect(),
}
#: Where a scenario's proposal inbox lands when the recipe names no directory: beside the service's other state.
DEFAULT_PROPOSALS_DIR = ".reef/proposals"


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


def _resolve_candidate_selector(value: Any) -> CandidateSelector:
    if isinstance(value, str) and value in _CANDIDATE_SELECTORS:
        return _CANDIDATE_SELECTORS[value]
    resolved = value
    if isinstance(value, str) and ":" in value:
        module_name, _, attribute = value.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise RecipeConfigError(f"cannot import evolution.selection {value!r}: {exc}") from exc
    if callable(getattr(resolved, "decide", None)):
        return resolved
    raise RecipeConfigError(
        "evolution.selection must provide decide(candidate, evaluation) or be a dotted reference to one"
    )


@dataclass(frozen=True)
class CordisRecipe(Recipe):
    """The harness evolution loop as a bootable recipe class.

    Config shape (the ``evolution`` section): ``adapter`` (a name
    ``reef.harness.adapters.get_adapter`` resolves), ``propose`` and
    ``evaluate`` (callables or dotted references), ``tasks`` (the episode
    prompts scored per step), optional ``binary`` (a path overriding the
    adapter's binary name - the seam tests drive a fake harness through),
    optional ``seed`` (a list of entry options - id, name, config -
    loaded into the composition tree on first boot, where an item may also
    be a dotted ``module:attribute`` naming a sequence of them; a recovered
    algorithm state always wins over the seed), optional ``selection`` (the
    candidate-selection policy: ``score_comparison``, the default; ``always``;
    or a dotted reference to an object implementing ``decide``), optional
    ``episode_workers`` (how many evaluation episodes run at once, default
    one; a large task set is one wave instead of a long turn-taking pass),
    optional ``version_check``
    (``true`` appends the adapter's shipped update notice extension to the
    seed, so every pulled tree tells its user at startup when it is behind
    the channel head; adapters without a shipped extension refuse boot),
    and the agent proposal inbox: ``proposals_dir`` (default
    ``.reef/proposals``, one directory per scenario under it, created when
    the first proposal arrives) and ``max_pending_proposals`` (default 8,
    the number of admitted proposals a scenario holds before the route
    refuses more).

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
    publish: str = "auto"
    review_kinds: tuple[str, ...] = ()
    seed: tuple[Mapping[str, Any], ...] = ()
    model_name: str | None = None
    models: Mapping[str, ModelBinding] = field(default_factory=dict)
    candidate_selector: CandidateSelector = field(default_factory=ScoreComparisonSelector, repr=False)
    episode_workers: int = 1
    proposals_dir: str = DEFAULT_PROPOSALS_DIR
    max_pending_proposals: int = 8
    batch_size: int = config_field(1)
    max_score: float = config_field(0.0)
    batch_policy: str = config_field("reports")
    name: str = field(default="harness_evolve", kw_only=True)

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport

    def __post_init__(self) -> None:
        if not isinstance(self.propose, Proposer):
            raise ValueError("harness evolution requires a Proposer")
        if not isinstance(self.score_episode, EpisodeScorer):
            raise ValueError("harness evolution requires an EpisodeScorer")
        if not self.tasks:
            raise ValueError("harness evolution requires tasks")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.episode_workers < 1:
            raise ValueError("episode_workers must be positive")
        if self.batch_policy not in ("reports", "records"):
            raise ValueError("batch_policy must be 'reports' or 'records'")
        if self.episode_timeout_s <= 0:
            raise ValueError("episode_timeout_s must be positive")
        if self.episode_repeats < 1:
            raise ValueError("episode_repeats must be at least 1")
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
        if self.publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")
        if not callable(getattr(self.candidate_selector, "decide", None)):
            raise ValueError("candidate_selector must provide decide(candidate, evaluation)")
        if not isinstance(self.proposals_dir, str) or not self.proposals_dir.strip():
            raise ValueError("proposals_dir must be a non-empty path")
        if isinstance(self.max_pending_proposals, bool) or self.max_pending_proposals < 1:
            raise ValueError("max_pending_proposals must be an integer of at least 1")

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("harness_evolve requires an 'evolution' config section")
        tasks = evolution.get("tasks")
        if not isinstance(tasks, Sequence) or isinstance(tasks, str) or not tasks:
            raise RecipeConfigError("evolution.tasks must be a non-empty list of prompts")
        binary = evolution.get("binary")
        if binary is not None and (not isinstance(binary, str) or not binary):
            raise RecipeConfigError("evolution.binary must be a non-empty string when set")
        timeout = evolution.get("episode_timeout_s", 600.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise RecipeConfigError("evolution.episode_timeout_s must be a positive number")
        repeats = evolution.get("episode_repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise RecipeConfigError("evolution.episode_repeats must be an integer of at least 1")
        forbid_residue = evolution.get("forbid_residue", False)
        if not isinstance(forbid_residue, bool):
            raise RecipeConfigError("evolution.forbid_residue must be a boolean")
        try:
            executor = build_executor(evolution)
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
        candidate_selector = _resolve_candidate_selector(selection)
        if budgets["min_win_margin"]:
            if selection != "score_comparison":
                raise RecipeConfigError("evolution.min_win_margin applies only to the score_comparison selection")
            candidate_selector = ScoreComparisonSelector(min_win_margin=budgets["min_win_margin"])
        publish = evolution.get("publish", "auto")
        if publish not in ("auto", "review"):
            raise RecipeConfigError("evolution.publish must be 'auto' or 'review'")
        review_kinds = evolution.get("review_kinds", ())
        if isinstance(review_kinds, str) or not isinstance(review_kinds, Sequence):
            raise RecipeConfigError("evolution.review_kinds must be a list of node kind names")
        if not all(isinstance(kind, str) and kind for kind in review_kinds):
            raise RecipeConfigError("evolution.review_kinds must be a list of node kind names")
        adapter = str(evolution.get("adapter", "pi"))
        version_check = evolution.get("version_check", False)
        if not isinstance(version_check, bool):
            raise RecipeConfigError("evolution.version_check must be a boolean")
        if version_check:
            try:
                seed = (*seed, version_check_entry(adapter))
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
        raw_workers = evolution.get("episode_workers", 1)
        if isinstance(raw_workers, str) and raw_workers.strip().isdigit():
            raw_workers = int(raw_workers)
        if isinstance(raw_workers, bool) or not isinstance(raw_workers, int) or raw_workers < 1:
            raise RecipeConfigError("evolution.episode_workers must be a positive integer")
        episode_workers = raw_workers
        proposals_dir = evolution.get("proposals_dir", DEFAULT_PROPOSALS_DIR)
        if not isinstance(proposals_dir, str) or not proposals_dir.strip():
            raise RecipeConfigError("evolution.proposals_dir must be a non-empty path")
        max_pending = evolution.get("max_pending_proposals", 8)
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise RecipeConfigError("evolution.max_pending_proposals must be an integer of at least 1")
        return {
            "proposals_dir": proposals_dir.strip(),
            "max_pending_proposals": max_pending,
            "propose": resolve_proposer(evolution.get("propose")),
            "promote": resolve_promoter(evolution["promote"]) if "promote" in evolution else None,
            "score_episode": resolve_episode_scorer(evolution.get("evaluate")),
            "tasks": tuple(str(task) for task in tasks),
            "adapter": adapter,
            "binary": binary,
            "episode_timeout_s": float(timeout),
            "episode_repeats": repeats,
            "forbid_residue": forbid_residue,
            **budgets,
            "executor": executor,
            "promote_failures": promote_failures,
            "max_promoted_tasks": max_promoted_tasks,
            "max_promoted_per_client": per_client,
            "publish": publish,
            "review_kinds": tuple(review_kinds),
            "seed": tuple(seed),
            "model_name": model_name if isinstance(model_name, str) and model_name else None,
            "models": models,
            "candidate_selector": candidate_selector,
            "episode_workers": episode_workers,
        }

    def model_binding(self) -> ModelBinding:
        """The served model's endpoint, derived from the recipe's runtime."""
        if self.runtime is None:
            raise RecipeConfigError(
                "harness evolution requires an inference runtime: set reef.upstream_url (and reef.upstream_model) "
                "in the deployment config"
            )
        try:
            return ModelBinding.from_runtime(self.runtime, model=self.model_name)
        except ValueError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def model_bindings(self) -> ModelBindings:
        """What ``propose`` receives: the served model plus ``evolution.models``."""
        return ModelBindings(served=self.model_binding(), named=dict(self.models))

    def build_surface(self, scenario: str) -> Surface:
        model = self.model_name or getattr(self.runtime, "model_path", None)
        return create_harness_surface(
            seed_entries=tuple(dict(entry) for entry in self.seed),
            served_model=model if isinstance(model, str) and model else None,
        )

    def base_artifact_files(self) -> Mapping[str, str] | None:
        """The seed rendered for the adapter, with its entries list where the adapter carries one: a fresh scenario serves it before any step publishes."""
        if not self.seed:
            return None
        descriptor = get_adapter(self.adapter)
        nodes = tuple((str(entry["name"]), entry.get("config")) for entry in self.seed if not entry.get("disabled"))
        return {**render_composition(nodes, descriptor), **tree_files(descriptor, self.seed)}

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
        training_backend = CordisBackend(**self._backend_kwargs(), proposals_dir=self.proposals_path(scenario))
        return self._build_trainer(
            scenario,
            records,
            training_backend,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )

    def _backend_kwargs(self) -> dict[str, Any]:
        """Arguments shared by the stock backend and recipe specializations."""
        return {
            "descriptor": get_adapter(self.adapter),
            "propose": self.propose,
            "score_episode": self.score_episode,
            "tasks": self.tasks,
            "models": self.model_bindings(),
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
            "episode_workers": self.episode_workers,
            "max_pending_proposals": self.max_pending_proposals,
        }

    def _build_trainer(
        self,
        scenario: str,
        records: RecordStore,
        training_backend: CordisBackend,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger | None,
    ) -> Trainer:
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: (
                RecordDrivenTraceProcessor(context.with_config({"batch_size": self.batch_size}))
                if self.batch_policy == "records"
                else CordisProcessor(context.with_config({"batch_size": self.batch_size, "max_score": self.max_score}))
            ),
            training_backend=training_backend,
            candidate_evaluator=DefaultCandidateEvaluationPlugin(training_backend, self.candidate_selector),
            algorithm_state=algorithm_state,
            report_type=self.report_type,
            experiment_logger=experiment_logger,
        )
