"""Configuration and per-scenario binding for the Meta-Harness method.

``MetaHarnessRecipe`` reuses ``CordisRecipe`` for trace batching, model
bindings, adapter resolution, paired episodes, scoring, and publication.  It
only binds the history-aware proposal policy and its population selector.

Example method block::

    implementation: recipes.meta_harness.recipe:MetaHarnessRecipe
    evolution:
      adapter: pi
      evaluate: harness.scoring:score_episode
      tasks: ["validation task one", "validation task two"]
      models:
        proposer: {url: https://api.openai.com, model: gpt-5, api_key_env: OPENAI_API_KEY}
      meta_harness:
        archive: ${REEF_WORK}/meta-harness
        mode: full_history
        max_candidates: 20
        max_target_episodes: 200
        max_nodes: 32

The archive path is a post-commit JSON mirror.  Reef's algorithm state is
always authoritative, including after restart.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

from reef.harness.adapters import get_adapter
from reef.observability import ExperimentLogger
from reef.recipe.errors import RecipeConfigError
from reef.records import RecordStore
from reef.train.cordis_backend import CordisRecipe
from reef.train.cordis_backend.strategies import Mutation, Proposer, resolve_proposer
from reef.train.trainer import Trainer
from reef.train.types import TraceSample

from .backend import POPULATION_STATE_KEY, MetaHarnessBackend
from .method import SEARCH_MODES, MetaHarnessProposer, MetaHarnessSelector
from .population import PopulationStore


class _UnboundProposer(Proposer):
    """Placeholder replaced with a per-scenario, population-bound proposer."""

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TraceSample, ...],
        models: Any,
        *,
        manifest: Any = None,
        rejected: Sequence[Mapping[str, Any]] = (),
    ) -> Mutation | Sequence[Mutation] | None:
        raise RecipeConfigError("the Meta-Harness proposer is bound by MetaHarnessRecipe.build")


class _UnboundSelector:
    def decide(self, candidate: Any, evaluation: Any) -> Any:
        raise RecipeConfigError("the Meta-Harness selector is bound by MetaHarnessRecipe.build")


@dataclass(frozen=True)
class MetaHarnessRecipe(CordisRecipe):
    """Full-history population search over adapter-neutral Reef compositions."""

    _: KW_ONLY
    archive_dir: Path
    mode: str = "full_history"
    proposer_model: str = "proposer"
    kinds: tuple[str, ...] = ()
    max_candidates: int = 0
    max_target_episodes: int = 0
    max_nodes: int = 32
    name: str = "meta_harness"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"evolution.meta_harness.mode must be one of {SEARCH_MODES}")
        if not self.proposer_model:
            raise ValueError("evolution.meta_harness.model must not be empty")
        for label, value in (
            ("max_candidates", self.max_candidates),
            ("max_target_episodes", self.max_target_episodes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"evolution.meta_harness.{label} must be a non-negative integer")
        if isinstance(self.max_nodes, bool) or not isinstance(self.max_nodes, int) or self.max_nodes < 1:
            raise ValueError("evolution.meta_harness.max_nodes must be a positive integer")

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("meta_harness requires an 'evolution' config section")
        if "selection" in evolution:
            raise RecipeConfigError("Meta-Harness owns evolution.selection so population and serving commit together")
        supplied = dict(evolution)
        supplied.setdefault("propose", _UnboundProposer())
        supplied["selection"] = _UnboundSelector()
        kwargs = super()._recipe_kwargs({**settings, "evolution": supplied}, values)

        block = evolution.get("meta_harness")
        if not isinstance(block, Mapping):
            raise RecipeConfigError("meta_harness requires an 'evolution.meta_harness' config section")
        archive = block.get("archive")
        if not isinstance(archive, str) or not archive.strip():
            raise RecipeConfigError("evolution.meta_harness.archive must be a directory path string")
        mode = block.get("mode", "full_history")
        if mode not in SEARCH_MODES:
            raise RecipeConfigError(f"evolution.meta_harness.mode must be one of {SEARCH_MODES}")
        model = block.get("model", "proposer")
        if not isinstance(model, str) or not model.strip():
            raise RecipeConfigError("evolution.meta_harness.model must be a non-empty string")
        kinds = block.get("components", ())
        if isinstance(kinds, str) or not isinstance(kinds, Sequence):
            raise RecipeConfigError("evolution.meta_harness.components must be a list of Reef node kinds")
        if not all(isinstance(kind, str) and kind for kind in kinds):
            raise RecipeConfigError("evolution.meta_harness.components must be a list of Reef node kinds")
        return {
            **kwargs,
            "archive_dir": Path(archive.strip()),
            "mode": str(mode),
            "proposer_model": model.strip(),
            "kinds": tuple(kinds),
            "max_candidates": _non_negative_int(block, "max_candidates", 0),
            "max_target_episodes": _non_negative_int(block, "max_target_episodes", 0),
            "max_nodes": _positive_int(block, "max_nodes", 32),
        }

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        store = PopulationStore(scenario_population_path(self.archive_dir, scenario))
        population_state = None if algorithm_state is None else algorithm_state.get(POPULATION_STATE_KEY)
        if population_state is not None:
            if not isinstance(population_state, Mapping):
                raise RecipeConfigError(f"{POPULATION_STATE_KEY} algorithm state must be a mapping")
            store.restore_committed(population_state)
            # A crash can land after the Reef commit and before the derived
            # mirror.  Recovery heals it from the committed state.
            store.persist()

        descriptor = get_adapter(self.adapter)
        propose = self.propose
        if isinstance(propose, _UnboundProposer):
            propose = resolve_proposer(
                MetaHarnessProposer(
                    store=store,
                    descriptor=descriptor,
                    tasks=self.tasks,
                    episode_repeats=self.episode_repeats,
                    mode=self.mode,
                    model=self.proposer_model,
                    kinds=self.kinds,
                    max_candidates=self.max_candidates,
                    max_target_episodes=self.max_target_episodes,
                    max_nodes=self.max_nodes,
                )
            )
        bound = dataclasses.replace(self, propose=propose, candidate_selector=MetaHarnessSelector(store))
        backend = MetaHarnessBackend(population_store=store, **bound._backend_kwargs())
        return bound._build_trainer(
            scenario,
            records,
            backend,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )


def scenario_population_path(directory: Path, scenario: str) -> Path:
    """Map any scenario name to one path-safe file directly under ``directory``."""
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    return Path(directory) / f"{key}.json"


def _non_negative_int(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecipeConfigError(f"evolution.meta_harness.{key} must be a non-negative integer")
    return value


def _positive_int(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecipeConfigError(f"evolution.meta_harness.{key} must be a positive integer")
    return value


__all__ = ["MetaHarnessRecipe", "scenario_population_path"]
