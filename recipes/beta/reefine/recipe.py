"""Reefine's recipe: the harness evolution recipe with Reefine's defaults, its agent and its multimodal gateway.

``ReefineRecipe`` answers a person's training instructions, one step per
instruction (``training_mode: manual``), with ``ReefineMethod``. It merges
Reefine's defaults into the deployment's ``evolution`` section: the method,
the seed beside this module, and an evaluation that runs the candidate alone
on the deployment's tasks, scored by ``AnswerScorer``, and keeps it when every
task meets the floor. On the health task the floor checks that the tree still
works, not that the requested change does; the step's design and review and
the person judge that. By default, a person activates a tree that changes an
extension (the ``activation`` rule's ``review_kinds``).

``evolution.agent`` says where the coding agent that answers a request on
pi runs (``reef.train.harness_backend.agent.settings``); Reefine adds
``trial_timeout_s`` to it, the limit of each trial, and runs its agent on the
harness's own binary unless the section names another. ``multimodal`` is
Reefine's own key: the gateway for images, embeddings, speech and decisions
(``recipes.beta.reefine.multimodal``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from recipes.beta.reefine.agent import ReefineAgent
from recipes.beta.reefine.answer import EXTENSION_ADAPTER
from recipes.beta.reefine.method import ReefineMethod
from recipes.beta.reefine.multimodal import MultimodalProvider, MultimodalSettings, ProviderRelay
from recipes.beta.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.tree import TreeKind
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.runtime.interfaces import MultimodalRelay
from reef.storage.samples import TrainerSamples
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import HarnessBackend

#: The first tree of a new Reefine scenario: the starter skill.
SEED_DIRECTORY = Path(__file__).with_name("seed")
#: Reefine's defaults for the ``evolution`` section and its ``evaluation`` section; the deployment's keys win.
EVOLUTION_DEFAULTS: Mapping[str, Any] = {
    "method": f"{ReefineMethod.__module__}:{ReefineMethod.__qualname__}",
    "seed": str(SEED_DIRECTORY),
}
#: Reefine's ``activation`` section when the deployment names none. An extension runs in the harness's process with
#: the user's privileges, so a person activates a tree that changes one.
ACTIVATION_DEFAULT: Mapping[str, Any] = {
    "module": "always",
    "config": {"review_kinds": [TreeKind.CODE_EXTENSION.value]},
}
EVALUATION_DEFAULTS: Mapping[str, Any] = {
    "policy": "floor",
    "scorer": f"{AnswerScorer.__module__}:{AnswerScorer.__qualname__}",
}


#: How long one trial of the agent may take when ``evolution.agent.trial_timeout_s`` says nothing.
DEFAULT_TRIAL_TIMEOUT_S = 300.0


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(HarnessEvolutionRecipe):
    """Refine a harness once per instruction; a person activates a tree that changes an extension.

    Values: the fields of ``HarnessEvolutionRecipe``, read with Reefine's
    defaults merged in. ``multimodal`` is ``evolution.multimodal`` read,
    ``None`` for no gateway: Reef relays the harness's image, embedding,
    speech and decision calls to it with its key, unrecorded, and the
    agent's trials reach the same one. ``agent`` is ``evolution.agent``
    read, ``None`` when no agent answers requests (off pi, or where the host
    cannot jail it), and ``trial_timeout_s`` > 0 bounds each of its trials.
    Purpose: a deployment names it as its recipe; it builds each scenario's
    backend with ``ReefineMethod``. Override ``evolution.evaluation.policy``
    to compare scores against the head tree, and ``training-mode`` to
    learn from reports too.
    """

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")
    multimodal: MultimodalSettings | None = None
    trial_timeout_s: float = DEFAULT_TRIAL_TIMEOUT_S

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
        section = settings.get("evolution")
        if not isinstance(section, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        evaluation = section.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            raise RecipeConfigError("evolution.evaluation must be a mapping")
        adapter = str(section.get("adapter", EXTENSION_ADAPTER))
        try:
            descriptor = get_adapter(adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        evaluation = {**EVALUATION_DEFAULTS, **evaluation}
        tasks = evaluation.get("tasks")
        if descriptor.is_prompt_task_directory and isinstance(tasks, Sequence) and not isinstance(tasks, str):
            # The adapter's prompt is a Harbor task directory, so the health prompt runs as the same check in
            # that form, scored by its verifier.
            evaluation["tasks"] = [
                HEALTH_TASK_DIRECTORY if str(task).startswith("[health]") else task for task in tasks
            ]
        own = {key: value for key, value in section.items() if key not in ("agent", "multimodal")}
        # The agent answers requests on pi alone; on another adapter the served model does, so no agent is
        # built, jailed or warned about. On pi it runs unless the section is null, on the harness's own binary.
        agent = section.get("agent", {}) if adapter == EXTENSION_ADAPTER else None
        trial_timeout_s: Any = DEFAULT_TRIAL_TIMEOUT_S
        if agent is not None:
            if not isinstance(agent, Mapping):
                raise RecipeConfigError("evolution.agent must be a mapping")
            if agent.get("adapter", EXTENSION_ADAPTER) != EXTENSION_ADAPTER:
                raise RecipeConfigError(f"Reefine's agent is {EXTENSION_ADAPTER}: evolution.agent.adapter must be it")
            trial_timeout_s = agent.get("trial_timeout_s", DEFAULT_TRIAL_TIMEOUT_S)
            if (
                isinstance(trial_timeout_s, bool)
                or not isinstance(trial_timeout_s, (int, float))
                or trial_timeout_s <= 0
            ):
                raise RecipeConfigError("evolution.agent.trial_timeout_s must be a positive number")
            binary = section.get("binary")
            agent = {**({"binary": binary} if binary else {}), **agent}
        kwargs = super()._recipe_kwargs(
            {
                **settings,
                "evolution": {**EVOLUTION_DEFAULTS, **own, "agent": agent, "evaluation": evaluation},
                "activation": settings.get("activation", ACTIVATION_DEFAULT),
            },
            values,
        )
        try:
            kwargs["multimodal"] = MultimodalSettings.from_config(section.get("multimodal"), values)
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.{exc}") from exc
        kwargs["trial_timeout_s"] = float(trial_timeout_s)
        return kwargs

    def build_backend(
        self, scenario: str, releases: ComponentReleases, samples: TrainerSamples | None
    ) -> HarnessBackend:
        """The scenario's backend. With an agent configured, its Reefine method runs the agent, bound here to pi
        once per scenario as the evaluator is: the binary is resolved, and installed when missing, on the Reef
        host, or named in a cloud sandbox's image."""
        if self.agent is None or not isinstance(self.method, ReefineMethod):
            return super().build_backend(scenario, releases, samples)
        method = ReefineMethod(ReefineAgent(self.agent.bind(), self.trial_timeout_s, self.multimodal_provider))
        return HarnessEvolutionRecipe.build_backend(replace(self, method=method), scenario, releases, samples)


__all__ = ["DEFAULT_TRIAL_TIMEOUT_S", "SEED_DIRECTORY", "ReefineRecipe"]
