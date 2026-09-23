"""Reefine: built-in harness refinement from a user's training instructions.

``ReefineRecipe`` binds the served-model proposer to the shared Cordis loop.
It accepts the same ``evolution`` settings as ``CordisRecipe``; the shipped
``reefine`` profile supplies the health task and seed. Custom tasks should
also supply their own ``evolution.evaluate`` scorer. The bundled scorer only
recognizes the profile's health task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.recipe.config_fields import config_field
from reef.recipe.cordis import CordisRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine.agent import AgentProposer
from reef.recipe.reefine.multimodal import MultimodalProvider, MultimodalSettings, ProviderRelay
from reef.runtime.interfaces import MultimodalRelay


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(CordisRecipe):
    """Refine a pi harness once per instruction, with extensions held for review.

    Configuration defaults enable harness requests and update notices, and
    answer a request with the agent proposer (:mod:`reef.recipe.reefine.agent`)
    where the host can isolate it (``evolution.proposer_agent``).
    ``evolution.multimodal`` names a gateway for images, embeddings, speech and
    decisions: Reef relays the harness's calls to it with its key, unrecorded,
    and the agent's trials reach the same one.
    Selection is ``floor``: the evaluation runs the candidate alone on the
    profile's health task and publishes it when every task scores at least
    ``evolution.floor_score``. The floor checks that the tree still works
    (the model binding, the tools, the extensions load), not that the
    requested change does; the step's design and review notes and the
    person judge that. Override ``evolution.selection`` to compare scores
    against the current release, and ``training-mode`` to learn from
    reports too.
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
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution", {})
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        defaults = {
            # A request goes to the agent proposer where the host can jail it; otherwise to the text proposer.
            "propose": "reef.recipe.reefine.agent:propose",
            "proposer_agent": {},
            "evaluate": "reef.recipe.reefine.evolution:evaluate",
            "requests": True,
            "version_check": True,
            # What runs on the person's machine waits for their review: an extension, and a settings change
            # (hooks are shell commands).
            "review_kinds": ["code_extension", "config"],
            "selection": "floor",
        }
        kwargs = super()._recipe_kwargs({**settings, "evolution": {**defaults, **evolution}}, values)
        try:
            kwargs["multimodal"] = MultimodalSettings.from_config(evolution.get("multimodal"), values)
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.{exc}") from exc
        return kwargs
