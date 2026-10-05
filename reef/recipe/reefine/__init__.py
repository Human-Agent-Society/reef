"""Reefine: built-in harness refinement from a user's training instructions.

``ReefineRecipe`` binds the served-model proposer to the shared Cordis loop.
It accepts the same ``evolution`` settings as ``CordisRecipe``; the shipped
``reefine`` profile supplies the health task and seed. Custom tasks should
also supply their own ``evolution.evaluate`` scorer. The bundled scorer only
recognizes the profile's health task: the prompt, or on an adapter that plays
a Harbor task directory (terminus) the ``health`` task directory beside this
module, which the recipe runs in the prompt's place.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.episodes.requests import ships_requests
from reef.harness.episodes.version_check import ships_version_check
from reef.observability import ExperimentLogger
from reef.recipe.config_fields import config_field
from reef.recipe.cordis import CordisRecipe, _ScenarioModels
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine.agent import AgentProposer
from reef.recipe.reefine.evaluation import evaluation_factory
from reef.recipe.reefine.evolution import EXTENSION_ADAPTER, HEALTH_TASK_DIRECTORY
from reef.recipe.reefine.multimodal import MultimodalProvider, MultimodalSettings, ProviderRelay
from reef.runtime.interfaces import MultimodalRelay
from reef.storage.records import RecordStore
from reef.train.reefine.backend import ReefineBackend
from reef.train.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(CordisRecipe):
    """Refine a pi harness once per instruction, with extensions held for review.

    Configuration defaults enable harness requests and update notices, and
    answer a request with the agent proposer (:mod:`reef.recipe.reefine.agent`)
    where the host can isolate it (``evolution.proposer_agent``).
    ``evolution.multimodal`` names a gateway for images, embeddings, speech and
    decisions: Reef relays the harness's calls to it with its key, unrecorded,
    and the agent's trials reach the same one.
    Selection defaults to independent Reefine evaluation: request behavior,
    health, protected tasks and a fresh model review must all pass. Explicit
    legacy selection policies retain their existing scoring behavior.
    """

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")
    #: ``evolution.multimodal``: the gateway images, embeddings, speech and decisions go to, for the harness's
    #: extensions at run time and for the agent proposer's trials; ``None`` offers no multimodal calls.
    multimodal: MultimodalSettings | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.propose, AgentProposer):
            # The agent's trials reach the same provider the harness will at run time; the text proposer
            # writes for the adapter the deployment serves.
            object.__setattr__(self, "propose", AgentProposer(self.multimodal_provider, adapter=self.adapter))

    @property
    def multimodal_provider(self) -> MultimodalProvider | None:
        """The configured provider, its key the chat upstream's when the upstream is the same gateway."""
        if self.multimodal is None:
            return None
        binding = self.model_binding() if self.runtime is not None else None
        return self.multimodal.provider(
            None if binding is None else binding.base_url, None if binding is None else binding.api_key
        )

    @property
    def multimodal_relay(self) -> MultimodalRelay | None:
        provider = self.multimodal_provider
        return None if provider is None else ProviderRelay(provider)

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution", {})
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        adapter = str(evolution.get("adapter", "pi"))
        defaults = {
            # A request goes to the agent proposer where the host can jail it; otherwise to the text proposer.
            "propose": "reef.recipe.reefine.agent:propose",
            "proposer_agent": {},
            "evaluate": "reef.recipe.reefine.evolution:evaluate",
            "requests": True,
            "version_check": True,
            "review_kinds": ["code_extension"],
            "selection": "reefine",
        }
        merged = {**defaults, **evolution}
        try:
            descriptor = get_adapter(adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        tasks = merged.get("tasks")
        if descriptor.is_prompt_task_directory and isinstance(tasks, Sequence) and not isinstance(tasks, str):
            # The adapter's prompt is a Harbor task directory, so the health prompt runs as the same check in
            # that form, scored by its verifier.
            merged["tasks"] = [HEALTH_TASK_DIRECTORY if str(task).startswith("[health]") else task for task in tasks]
        # The /reefine command and the update notice are entries the adapter ships; on an adapter that ships
        # one of them not, the profile runs without it: a request still arrives through reef-<adapter> evolve,
        # and reef-<adapter> update installs a release. An adapter Reef installs nothing for (terminus) has no
        # wrapper: a request comes through POST /reef/train and the published tree through GET /reef/harness.
        installs = descriptor.install is not None
        channel = "requests come through POST /reef/train; GET /reef/harness serves a published tree"
        # Warnings, so the startup log shows them, as the docs say.
        if merged.get("requests") is True and not ships_requests(adapter):
            merged["requests"] = False
            logger.warning(
                "adapter %r ships no /reefine command (requests off); %s",
                adapter,
                f"requests come through reef-{adapter} evolve" if installs else channel,
            )
        if merged.get("version_check") is True and not ships_version_check(adapter):
            merged["version_check"] = False
            logger.warning(
                "adapter %r ships no update notice (version_check off); %s",
                adapter,
                f"reef-{adapter} update installs a release" if installs else channel,
            )
        if adapter != EXTENSION_ADAPTER:
            # The agent proposer answers requests on pi alone (reef.recipe.reefine.agent); on another adapter the text
            # proposer does, so no agent is built, jailed or warned about.
            merged["proposer_agent"] = None
        selection = merged["selection"]
        evaluation = merged.pop("evaluation", None)
        if selection == "reefine":
            merged.setdefault("on_stale", "reevaluate")
            if merged["on_stale"] == "merge":
                raise RecipeConfigError("Reefine evaluation requires on_stale: reevaluate or refuse")
            merged["selection"] = "floor"
        kwargs = super()._recipe_kwargs({**settings, "evolution": merged}, values)
        if selection == "reefine":
            factory = evaluation_factory(evaluation, tuple(kwargs["tasks"]))
            if factory.settings.reviewer_model != "served" and factory.settings.reviewer_model not in kwargs["models"]:
                raise RecipeConfigError("evaluation.reviewer_model must name served or an evolution.models binding")
            kwargs["candidate_plugin"] = factory
        try:
            kwargs["multimodal"] = MultimodalSettings.from_config(evolution.get("multimodal"), values)
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.{exc}") from exc
        return kwargs

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, object] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        kwargs = self._backend_kwargs(scenario)
        if self.scenario_model is not None:
            kwargs["model_resolver"] = _ScenarioModels(self.scenario_model, self, scenario, self.models)
        if kwargs["step_record_dir"] is not None:
            from pathlib import Path

            kwargs["step_record_dir"] = Path(kwargs["step_record_dir"]).expanduser().resolve() / scenario
        backend = ReefineBackend(**kwargs, proposals_dir=self.proposals_path(scenario))
        return self._build_trainer(
            scenario, records, backend, algorithm_state=algorithm_state, experiment_logger=experiment_logger
        )
