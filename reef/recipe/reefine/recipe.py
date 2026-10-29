"""Reefine's recipe: the harness evolution recipe with Reefine's defaults.

``ReefineRecipe`` answers a person's training instructions, one step per
instruction (``training_mode: manual``). Its ``evolution_spec()`` names the
agent proposer, the health task scorer, the floor selection and the seed
beside this module. The floor runs the candidate alone on the profile's
health task and publishes it when every task scores at least
``evolution.floor_score``: it checks that the tree still works, not that the
requested change does; the step's design and review notes and the person
judge that. By default, a person activates a tree that changes an extension
(the ``activation`` rule's ``review_kinds``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.plugins import PluginReference
from reef.evaluation.harness import FloorPlugin
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe, HarnessEvolutionSpec
from reef.recipe.reefine.agent import AgentProposer
from reef.recipe.reefine.multimodal import MultimodalProvider, MultimodalSettings, ProviderRelay
from reef.recipe.reefine.proposer import EXTENSION_ADAPTER
from reef.recipe.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.runtime.interfaces import MultimodalRelay

#: The first tree of a new Reefine scenario: the starter skill.
SEED_DIRECTORY = Path(__file__).with_name("seed")


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(HarnessEvolutionRecipe):
    """Refine a harness once per instruction, with extensions held for review.

    ``evolution.multimodal`` names a gateway for images, embeddings, speech
    and decisions: Reef relays the harness's calls to it with its key,
    unrecorded, and the agent's trials reach the same one. Override
    ``evolution.selection`` to compare scores against the current release,
    and ``training-mode`` to learn from reports too.
    """

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")
    #: ``evolution.multimodal``: the gateway images, embeddings, speech and decisions go to, for the harness's
    #: extensions at run time and for the agent proposer's trials; ``None`` offers no multimodal calls.
    multimodal: MultimodalSettings | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.propose, AgentProposer):
            # The agent's trials reach the same provider the harness will at run time.
            object.__setattr__(self, "propose", AgentProposer(self.multimodal_provider))

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
    def evolution_spec(cls) -> HarnessEvolutionSpec:
        return HarnessEvolutionSpec(
            proposer=PluginReference(AgentProposer),
            scorer=PluginReference(AnswerScorer),
            selection=PluginReference(FloorPlugin),
            seed=SEED_DIRECTORY,
        )

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution", {})
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        adapter = str(evolution.get("adapter", "pi"))
        # The agent runs where the host can jail it.
        merged = {"proposer_agent": {}, **evolution}
        try:
            descriptor = get_adapter(adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        tasks = merged.get("tasks")
        if descriptor.is_prompt_task_directory and isinstance(tasks, Sequence) and not isinstance(tasks, str):
            # The adapter's prompt is a Harbor task directory, so the health prompt runs as the same check in
            # that form, scored by its verifier.
            merged["tasks"] = [HEALTH_TASK_DIRECTORY if str(task).startswith("[health]") else task for task in tasks]
        if adapter != EXTENSION_ADAPTER:
            # The agent answers requests on pi alone; on another adapter the text proposer does, so no agent is
            # built, jailed or warned about.
            merged["proposer_agent"] = None
        # An extension runs in the harness's process with the user's privileges, so by default a person activates a
        # tree that changes one.
        activation = settings.get("activation", {"module": "always", "config": {"review_kinds": ["code_extension"]}})
        kwargs = super()._recipe_kwargs({**settings, "evolution": merged, "activation": activation}, values)
        try:
            kwargs["multimodal"] = MultimodalSettings.from_config(evolution.get("multimodal"), values)
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.{exc}") from exc
        return kwargs


__all__ = ["SEED_DIRECTORY", "ReefineRecipe"]
